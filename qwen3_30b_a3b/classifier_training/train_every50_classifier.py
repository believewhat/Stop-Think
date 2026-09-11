#!/usr/bin/env python3
"""Train policy-aligned every-50 open-math ESTAR classifiers.

The input is either one merged JSONL file or the four/sixteen independent
collector shards.  Each JSONL row represents one qid and contains the natural
full rollout plus every-50-token counterfactual answer probes.  This trainer is
deliberately strict about the temporal grid and the ordered feature contract so
an apparently successful model cannot be trained on a distribution that the
online policy will never see.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import sqlite3
import time
from collections import Counter, deque
from datetime import datetime, timezone
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from math_verify import parse as math_parse, verify as math_verify
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from math_cluster_features import FEATURES as CLUSTER_FEATURES
from math_equivalence_frozen import grade_answer_sympy, mathd_normalize_answer


FEATURES = tuple(CLUSTER_FEATURES)
TARGET_FINAL = "final_consistency"
TARGET_GOLD = "gold_safe"
TARGETS = (TARGET_FINAL, TARGET_GOLD)
LABEL_COLUMNS = {
    TARGET_FINAL: "label_final_consistency",
    TARGET_GOLD: "label_gold_safe",
}
PROBABILITY_COLUMNS = {
    TARGET_FINAL: "probability_final_consistency",
    TARGET_GOLD: "probability_gold_safe",
}
MODEL_FILENAMES = {
    TARGET_FINAL: "classifier_final_consistency.joblib",
    TARGET_GOLD: "classifier_gold_safe.joblib",
}
TRAINER_SCHEMA = "qwen30a3b_math_every50_cluster22_dual_lgbm_v1"
LABEL_VERSION = "cached_math_equivalence_probe_to_full_and_gold_v1"
EQUIVALENCE_CACHE_SCHEMA = "math_equivalence_sqlite_v2"
EQUIVALENCE_VERIFIER_VERSION = "math_verify_then_frozen_grade_answer_sympy_v1"


class EquivalenceTimeout(Exception):
    """A single symbolic equivalence check exceeded its time budget."""


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise EquivalenceTimeout()


def answer_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = mathd_normalize_answer(text)
    return str(normalized or "").strip()


def _slow_equivalent(prediction: str, reference: str, timeout_seconds: float) -> bool:
    """Use the same symbolic equivalence stack as the previous math trainer."""
    if not prediction or not reference:
        return False
    can_alarm = bool(
        timeout_seconds > 0
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
    )
    old_handler: Any = None
    if can_alarm:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        parsed_prediction = math_parse(f"\\boxed{{{prediction}}}")
        parsed_reference = math_parse(f"\\boxed{{{reference}}}")
        if (
            parsed_prediction
            and parsed_reference
            and bool(math_verify(parsed_prediction, parsed_reference))
        ):
            return True
        return bool(grade_answer_sympy(prediction, reference))
    except EquivalenceTimeout:
        # Preserve the old False label while allowing the supervising process to
        # record that the symbolic verifier actually timed out.
        raise
    except Exception:
        return False
    finally:
        if can_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


class MathEquivalenceCache:
    """Cache symbolic checks by normalized (prediction, reference) pair."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 0.25,
        verifier: Callable[[str, str, float], bool] = _slow_equivalent,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.verifier = verifier
        self.values: dict[tuple[str, str], bool] = {}
        self.hits = 0
        self.misses = 0
        self.exact_matches = 0
        self.empty_rejections = 0

    def prepare_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        universe_metadata: Mapping[str, str] | None = None,
    ) -> None:
        """Resolve every unique non-trivial pair before row labels are built.

        The base implementation is deliberately serial for lightweight tests and
        callers that inject a verifier.  Production uses the disk-backed parallel
        subclass below.
        """
        unique_pairs = sorted(
            {
                (str(prediction).strip(), str(reference).strip())
                for prediction, reference in pairs
                if str(prediction).strip()
                and str(reference).strip()
                and str(prediction).strip() != str(reference).strip()
            }
        )
        for prediction, reference in unique_pairs:
            if (prediction, reference) not in self.values:
                self.equivalent(
                    prediction,
                    reference,
                    prediction_key=prediction,
                    reference_key=reference,
                )

    def equivalent(
        self,
        prediction: Any,
        reference: Any,
        *,
        prediction_key: str | None = None,
        reference_key: str | None = None,
    ) -> bool:
        normalized_prediction = (
            answer_key(prediction) if prediction_key is None else str(prediction_key).strip()
        )
        normalized_reference = (
            answer_key(reference) if reference_key is None else str(reference_key).strip()
        )
        if not normalized_prediction or not normalized_reference:
            self.empty_rejections += 1
            return False
        if normalized_prediction == normalized_reference:
            self.exact_matches += 1
            return True
        cache_key = (normalized_prediction, normalized_reference)
        if cache_key in self.values:
            self.hits += 1
            return self.values[cache_key]
        self.misses += 1
        try:
            result = bool(
                self.verifier(
                    normalized_prediction,
                    normalized_reference,
                    self.timeout_seconds,
                )
            )
        except EquivalenceTimeout:
            # This is the legacy label semantics: a timed-out symbolic check is
            # a negative equivalence result, not a trainer failure.
            result = False
        self.values[cache_key] = result
        return result

    def diagnostics(self) -> dict[str, int | float]:
        attempted = self.hits + self.misses
        return {
            "cached_pairs": len(self.values),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate_nonexact": self.hits / attempted if attempted else 0.0,
            "normalized_exact_matches": self.exact_matches,
            "empty_rejections": self.empty_rejections,
        }


def _equivalence_worker_main(
    slot: int,
    connection: Any,
    verifier: Callable[[str, str, float], bool],
    timeout_seconds: float,
) -> None:
    """Persistent worker supervised and hard-killed one task at a time."""
    # Some parser failures print directly rather than raising cleanly.  With 64
    # workers this can otherwise flood the trainer log and become an I/O bottleneck.
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
    finally:
        os.close(null_fd)
    thread_environment = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for name in thread_environment:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        limiter: Any = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    if limiter is not None:
        limiter.__enter__()
    try:
        connection.send(("ready", slot, None, None, None))
        while True:
            try:
                task = connection.recv()
            except EOFError:
                return
            if task is None:
                return
            pair_index, prediction, reference = task
            connection.send(("started", slot, pair_index, None, None))
            started = time.monotonic()
            try:
                value = bool(verifier(prediction, reference, timeout_seconds))
                status = "ok"
                error_text = ""
            except EquivalenceTimeout:
                value = False
                status = "timeout"
                error_text = ""
            except BaseException as error:
                # Unexpected worker errors must never be silently converted into
                # negative labels.  The parent retries and ultimately fails.
                value = False
                status = "error"
                error_text = f"{type(error).__name__}: {error}"
            elapsed = time.monotonic() - started
            connection.send(
                ("result", slot, pair_index, value, (status, elapsed, error_text))
            )
    finally:
        if limiter is not None:
            limiter.__exit__(None, None, None)
        connection.close()


def _multiprocessing_context(start_method: str) -> Any:
    if start_method != "auto":
        return mp.get_context(start_method)
    available = mp.get_all_start_methods()
    # Gamma is Linux: fork avoids importing SymPy/LightGBM independently in 64
    # workers.  Spawn remains the portable fallback for local Windows tests.
    return mp.get_context("fork" if "fork" in available else "spawn")


def parallel_verify_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    verifier: Callable[[str, str, float], bool],
    timeout_seconds: float,
    workers: int,
    hard_timeout_grace_seconds: float,
    worker_retries: int,
    worker_max_tasks: int,
    worker_start_timeout_seconds: float,
    start_method: str,
    on_result: Callable[[tuple[str, str], bool, str, float], None],
    worker_entrypoint: Callable[..., None] = _equivalence_worker_main,
) -> dict[str, int | float | str]:
    """Verify sorted unique pairs with a process-level hard timeout.

    Each slot receives at most one pair.  A normal SIGALRM timeout is reported by
    the child; if native symbolic code ignores Python signals, the parent kills
    that entire worker at the hard deadline and replaces it.  Thus one pathological
    expression can never stall the run or poison later work.
    """
    if workers < 1:
        raise ValueError("equivalence workers must be positive")
    if timeout_seconds <= 0:
        raise ValueError("equivalence timeout must be positive")
    if hard_timeout_grace_seconds < 0:
        raise ValueError("hard timeout grace must be nonnegative")
    if worker_retries < 0:
        raise ValueError("worker retries must be nonnegative")
    if worker_max_tasks < 1:
        raise ValueError("worker max tasks must be positive")
    if worker_start_timeout_seconds <= 0:
        raise ValueError("worker start timeout must be positive")
    if not pairs:
        return {
            "scheduled_pairs": 0,
            "completed_pairs": 0,
            "worker_count": 0,
            "start_method": start_method,
            "wall_seconds": 0.0,
        }

    context = _multiprocessing_context(start_method)
    actual_start_method = context.get_start_method()
    connections: dict[int, Any] = {}
    processes: dict[int, Any] = {}
    active: dict[int, dict[str, Any]] = {}
    awaiting_ready: dict[int, float] = {}
    generation_tasks: Counter[int] = Counter()
    pending = deque(range(len(pairs)))
    counters: Counter[str] = Counter()
    retry_counts: Counter[int] = Counter()
    startup_retry_counts: Counter[int] = Counter()
    completed = 0
    wall_started = time.monotonic()
    hard_limit = timeout_seconds + hard_timeout_grace_seconds

    def spawn(slot: int) -> None:
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=worker_entrypoint,
            args=(slot, child_connection, verifier, timeout_seconds),
            name=f"math-equivalence-{slot:02d}",
        )
        process.daemon = True
        try:
            process.start()
        except BaseException:
            parent_connection.close()
            child_connection.close()
            raise
        child_connection.close()
        connections[slot] = parent_connection
        processes[slot] = process
        awaiting_ready[slot] = time.monotonic()
        generation_tasks[slot] = 0

    def stop(slot: int, *, terminate: bool, request_exit: bool = False) -> None:
        process = processes.pop(slot, None)
        connection = connections.pop(slot, None)
        awaiting_ready.pop(slot, None)
        if request_exit and connection is not None and process is not None:
            if process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
        if process is not None:
            if terminate and process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            if process.is_alive():
                raise RuntimeError(
                    f"Could not terminate equivalence worker slot={slot} pid={process.pid}"
                )
        if connection is not None:
            connection.close()

    def assign(slot: int) -> bool:
        if not pending:
            stop(slot, terminate=False, request_exit=True)
            return False
        pair_index = pending.popleft()
        prediction, reference = pairs[pair_index]
        connections[slot].send((pair_index, prediction, reference))
        active[slot] = {
            "pair_index": pair_index,
            "started_at": None,
            "assigned_at": time.monotonic(),
        }
        return True

    worker_count = min(workers, len(pairs))
    thread_environment = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    saved_thread_environment = {name: os.environ.get(name) for name in thread_environment}
    for name in thread_environment:
        os.environ[name] = "1"

    def restore_thread_environment() -> None:
        for name, value in saved_thread_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def retry_or_fail(slot: int, pair_index: int, reason: str) -> None:
        attempts = retry_counts[pair_index]
        active.pop(slot, None)
        stop(slot, terminate=True)
        if attempts >= worker_retries:
            pair = pairs[pair_index]
            raise RuntimeError(
                "Unexpected equivalence worker failure after "
                f"{attempts + 1} attempt(s); pair_index={pair_index}, "
                f"pair={pair!r}, reason={reason}"
            )
        retry_counts[pair_index] += 1
        counters[f"retry_{reason}"] += 1
        pending.appendleft(pair_index)
        spawn(slot)

    def retry_startup_or_fail(slot: int, reason: str) -> None:
        attempts = startup_retry_counts[slot]
        stop(slot, terminate=True)
        if attempts >= worker_retries:
            raise RuntimeError(
                "Equivalence worker failed before ready after "
                f"{attempts + 1} attempt(s); slot={slot}, reason={reason}"
            )
        startup_retry_counts[slot] += 1
        counters[f"retry_startup_{reason}"] += 1
        spawn(slot)

    try:
        for slot in range(worker_count):
            spawn(slot)
        while completed < len(pairs):
            messages: list[tuple[Any, ...]] = []
            ready = wait_connections(list(connections.values()), timeout=0.05)
            for connection in ready:
                while connection.poll():
                    try:
                        messages.append(connection.recv())
                    except (EOFError, OSError):
                        break

            for kind, slot, pair_index, value, detail in messages:
                if kind == "ready":
                    awaiting_ready.pop(slot, None)
                    startup_retry_counts[slot] = 0
                    if slot not in active:
                        assign(slot)
                    continue
                current = active.get(slot)
                if current is None or current["pair_index"] != pair_index:
                    # A result queued just before a timed-out worker was killed is
                    # stale; the parent's hard-timeout decision is authoritative.
                    continue
                if kind == "started":
                    current["started_at"] = time.monotonic()
                    continue
                if kind != "result":
                    continue
                status, elapsed, error_text = detail
                if status == "error":
                    retry_or_fail(
                        slot,
                        int(pair_index),
                        f"worker_error:{error_text}",
                    )
                    continue
                if status not in {"ok", "timeout"}:
                    retry_or_fail(slot, int(pair_index), f"unknown_status:{status}")
                    continue
                if float(elapsed) > timeout_seconds:
                    value = False
                    status = "deadline_exceeded"
                pair = pairs[pair_index]
                on_result(pair, bool(value), str(status), float(elapsed))
                counters[str(status)] += 1
                completed += 1
                del active[slot]
                generation_tasks[slot] += 1
                if generation_tasks[slot] >= worker_max_tasks and pending:
                    counters["worker_recycles"] += 1
                    stop(slot, terminate=False, request_exit=True)
                    spawn(slot)
                else:
                    assign(slot)

            now = time.monotonic()
            for slot, spawned_at in list(awaiting_ready.items()):
                process = processes[slot]
                if not process.is_alive():
                    retry_startup_or_fail(slot, "worker_crash")
                    continue
                if now - spawned_at > worker_start_timeout_seconds:
                    retry_startup_or_fail(slot, "worker_start_timeout")

            for slot, current in list(active.items()):
                process = processes[slot]
                if not process.is_alive():
                    pair_index = int(current["pair_index"])
                    retry_or_fail(slot, pair_index, "worker_crash")
                    continue
                started_at = current["started_at"]
                if started_at is not None and now - float(started_at) > hard_limit:
                    pair_index = int(current["pair_index"])
                    elapsed = now - float(started_at)
                    on_result(pairs[pair_index], False, "hard_timeout", elapsed)
                    counters["hard_timeout"] += 1
                    completed += 1
                    del active[slot]
                    stop(slot, terminate=True)
                    spawn(slot)
                    continue
                # This is distinct from the pre-ready watchdog above: the worker
                # was healthy, received a task, but did not acknowledge its start.
                if (
                    started_at is None
                    and now - float(current["assigned_at"])
                    > worker_start_timeout_seconds
                ):
                    pair_index = int(current["pair_index"])
                    retry_or_fail(slot, pair_index, "worker_start_timeout")
    finally:
        for slot in list(processes):
            stop(slot, terminate=True)
        restore_thread_environment()

    return {
        "scheduled_pairs": len(pairs),
        "completed_pairs": completed,
        "worker_count": worker_count,
        "start_method": actual_start_method,
        "wall_seconds": time.monotonic() - wall_started,
        "worker_retries": worker_retries,
        "worker_max_tasks": worker_max_tasks,
        "worker_start_timeout_seconds": worker_start_timeout_seconds,
        **{f"status_{name}": int(count) for name, count in sorted(counters.items())},
    }


class ParallelMathEquivalenceCache(MathEquivalenceCache):
    """Disk-backed, resumable, globally deduplicated parallel equivalence cache."""

    def __init__(
        self,
        *,
        cache_path: Path,
        progress_path: Path | None = None,
        timeout_seconds: float = 0.25,
        workers: int = 64,
        checkpoint_every: int = 256,
        hard_timeout_grace_seconds: float = 0.75,
        worker_retries: int = 1,
        worker_max_tasks: int = 1_000,
        worker_start_timeout_seconds: float = 30.0,
        start_method: str = "auto",
        verifier: Callable[[str, str, float], bool] = _slow_equivalent,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds, verifier=verifier)
        if checkpoint_every < 1:
            raise ValueError("equivalence checkpoint interval must be positive")
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path = (
            Path(progress_path)
            if progress_path is not None
            else self.cache_path.with_suffix(self.cache_path.suffix + ".progress.json")
        )
        self.workers = int(workers)
        self.checkpoint_every = int(checkpoint_every)
        self.hard_timeout_grace_seconds = float(hard_timeout_grace_seconds)
        self.worker_retries = int(worker_retries)
        self.worker_max_tasks = int(worker_max_tasks)
        self.worker_start_timeout_seconds = float(worker_start_timeout_seconds)
        self.start_method = str(start_method)
        self.status_counts: Counter[str] = Counter()
        self.resumed_pairs = 0
        self.computed_this_run = 0
        self.parallel_run: dict[str, Any] = {}
        self._universe_metadata: dict[str, str] = {}
        self._connection = sqlite3.connect(str(self.cache_path))
        quick_check = self._connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            self._connection.close()
            raise ValueError(
                f"Equivalence cache failed SQLite quick_check: {quick_check!r}"
            )
        integrity_check = self._connection.execute("PRAGMA integrity_check").fetchone()
        if integrity_check != ("ok",):
            self._connection.close()
            raise ValueError(
                f"Equivalence cache failed SQLite integrity_check: {integrity_check!r}"
            )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS results (
                   prediction TEXT NOT NULL,
                   reference TEXT NOT NULL,
                   equivalent INTEGER NOT NULL CHECK(equivalent IN (0, 1)),
                   status TEXT NOT NULL,
                   elapsed_seconds REAL NOT NULL,
                   PRIMARY KEY (prediction, reference)
               )"""
        )
        expected_metadata = {
            "schema": EQUIVALENCE_CACHE_SCHEMA,
            "verifier_version": EQUIVALENCE_VERIFIER_VERSION,
            "label_version": LABEL_VERSION,
            "timeout_seconds": format(self.timeout_seconds, ".17g"),
        }
        observed_metadata = dict(
            self._connection.execute("SELECT key, value FROM metadata")
        )
        observed_core = {
            key: observed_metadata.get(key) for key in expected_metadata
        }
        if observed_metadata and observed_core != expected_metadata:
            raise ValueError(
                "Equivalence cache metadata mismatch; use a fresh --equivalence-cache: "
                f"observed={observed_metadata!r} expected={expected_metadata!r}"
            )
        if not observed_metadata:
            self._connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(expected_metadata.items()),
            )
            observed_metadata = dict(expected_metadata)
        allowed_metadata = set(expected_metadata) | {
            "pair_universe_sha256",
            "pair_universe_count",
            "input_sha256",
        }
        unknown_metadata = set(observed_metadata) - allowed_metadata
        if unknown_metadata:
            raise ValueError(
                f"Equivalence cache has unknown metadata keys: {sorted(unknown_metadata)}"
            )
        self._universe_metadata = {
            key: value
            for key, value in observed_metadata.items()
            if key not in expected_metadata
        }
        rows = self._connection.execute(
            "SELECT prediction, reference, equivalent, status, elapsed_seconds "
            "FROM results "
            "ORDER BY prediction, reference"
        ).fetchall()
        allowed_statuses = {
            "ok",
            "timeout",
            "deadline_exceeded",
            "hard_timeout",
        }
        for prediction, reference, equivalent, status, elapsed in rows:
            if status not in allowed_statuses:
                raise ValueError(f"Invalid equivalence cache status: {status!r}")
            if not math.isfinite(float(elapsed)) or float(elapsed) < 0:
                raise ValueError(
                    f"Invalid equivalence cache elapsed_seconds: {elapsed!r}"
                )
            if status != "ok" and bool(equivalent):
                raise ValueError(
                    f"Timed-out equivalence cache row cannot be True: {status!r}"
                )
            if status == "ok" and float(elapsed) > self.timeout_seconds:
                raise ValueError(
                    "Equivalence cache contains an ok result beyond the semantic "
                    f"deadline: {elapsed!r}"
                )
            self.values[(str(prediction), str(reference))] = bool(equivalent)
            self.status_counts[str(status)] += 1
        self.resumed_pairs = len(rows)
        self._connection.commit()

    def _write_progress(
        self,
        *,
        total_unique_pairs: int,
        pending_pairs: int,
        complete: bool,
        state: str,
    ) -> None:
        payload = {
            "schema": EQUIVALENCE_CACHE_SCHEMA,
            "cache_path": str(self.cache_path),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": bool(complete),
            "state": state,
            "total_unique_nonexact_pairs": int(total_unique_pairs),
            "cached_pairs": int(len(self.values)),
            "resumed_pairs": int(self.resumed_pairs),
            "computed_this_run": int(self.computed_this_run),
            "pending_pairs": int(pending_pairs),
            "workers": int(self.workers),
            "timeout_seconds": float(self.timeout_seconds),
            "hard_timeout_grace_seconds": float(self.hard_timeout_grace_seconds),
            "worker_retries": int(self.worker_retries),
            "worker_max_tasks": int(self.worker_max_tasks),
            "worker_start_timeout_seconds": float(
                self.worker_start_timeout_seconds
            ),
            "status_counts": dict(sorted(self.status_counts.items())),
        }
        atomic_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            self.progress_path,
        )

    def prepare_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        universe_metadata: Mapping[str, str] | None = None,
    ) -> None:
        unique_pairs = sorted(
            {
                (str(prediction).strip(), str(reference).strip())
                for prediction, reference in pairs
                if str(prediction).strip()
                and str(reference).strip()
                and str(prediction).strip() != str(reference).strip()
            }
        )
        if universe_metadata is None or not str(
            universe_metadata.get("input_sha256", "")
        ).strip():
            raise ValueError(
                "Parallel cache requires input_sha256 universe metadata"
            )
        expected_universe_metadata = {
            "pair_universe_sha256": pair_universe_sha256(unique_pairs),
            "pair_universe_count": str(len(unique_pairs)),
            "input_sha256": str(universe_metadata["input_sha256"]),
        }
        if self._universe_metadata:
            if self._universe_metadata != expected_universe_metadata:
                raise ValueError(
                    "Equivalence cache pair/input universe mismatch; use a fresh "
                    f"--equivalence-cache: observed={self._universe_metadata!r} "
                    f"expected={expected_universe_metadata!r}"
                )
        else:
            if self.values:
                raise ValueError(
                    "Equivalence cache has results but no pair-universe metadata"
                )
            self._connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(expected_universe_metadata.items()),
            )
            self._connection.commit()
            self._universe_metadata = expected_universe_metadata
        extra_cached_pairs = set(self.values) - set(unique_pairs)
        if extra_cached_pairs:
            sample = sorted(extra_cached_pairs)[:3]
            raise ValueError(
                "Equivalence cache contains pairs outside the bound universe; "
                f"sample={sample!r}"
            )
        missing = [pair for pair in unique_pairs if pair not in self.values]
        self.misses += len(missing)
        self._write_progress(
            total_unique_pairs=len(unique_pairs),
            pending_pairs=len(missing),
            complete=not missing,
            state="complete" if not missing else "running",
        )

        since_commit = 0

        def save_result(
            pair: tuple[str, str], value: bool, status: str, elapsed: float
        ) -> None:
            nonlocal since_commit
            self.values[pair] = bool(value)
            self.status_counts[status] += 1
            self.computed_this_run += 1
            since_commit += 1
            self._connection.execute(
                "INSERT OR REPLACE INTO results"
                "(prediction, reference, equivalent, status, elapsed_seconds) "
                "VALUES (?, ?, ?, ?, ?)",
                (pair[0], pair[1], int(bool(value)), status, float(elapsed)),
            )
            if since_commit >= self.checkpoint_every:
                self._connection.commit()
                since_commit = 0
                self._write_progress(
                    total_unique_pairs=len(unique_pairs),
                    pending_pairs=len(missing) - self.computed_this_run,
                    complete=False,
                    state="running",
                )

        try:
            self.parallel_run = parallel_verify_pairs(
                missing,
                verifier=self.verifier,
                timeout_seconds=self.timeout_seconds,
                workers=self.workers,
                hard_timeout_grace_seconds=self.hard_timeout_grace_seconds,
                worker_retries=self.worker_retries,
                worker_max_tasks=self.worker_max_tasks,
                worker_start_timeout_seconds=self.worker_start_timeout_seconds,
                start_method=self.start_method,
                on_result=save_result,
            )
        except BaseException:
            self._connection.commit()
            self._write_progress(
                total_unique_pairs=len(unique_pairs),
                pending_pairs=max(0, len(missing) - self.computed_this_run),
                complete=False,
                state="failed",
            )
            raise
        else:
            self._connection.commit()
        self._write_progress(
            total_unique_pairs=len(unique_pairs),
            pending_pairs=0,
            complete=True,
            state="complete",
        )

    def equivalent(
        self,
        prediction: Any,
        reference: Any,
        *,
        prediction_key: str | None = None,
        reference_key: str | None = None,
    ) -> bool:
        normalized_prediction = (
            answer_key(prediction) if prediction_key is None else str(prediction_key).strip()
        )
        normalized_reference = (
            answer_key(reference) if reference_key is None else str(reference_key).strip()
        )
        if not normalized_prediction or not normalized_reference:
            self.empty_rejections += 1
            return False
        if normalized_prediction == normalized_reference:
            self.exact_matches += 1
            return True
        cache_key = (normalized_prediction, normalized_reference)
        if cache_key not in self.values:
            raise KeyError(
                "Nonexact pair was not globally prepared before label construction: "
                f"{cache_key!r}"
            )
        self.hits += 1
        return self.values[cache_key]

    def diagnostics(self) -> dict[str, Any]:
        output = super().diagnostics()
        output.update(
            {
                "schema": EQUIVALENCE_CACHE_SCHEMA,
                "cache_path": str(self.cache_path),
                "resumed_pairs": self.resumed_pairs,
                "computed_this_run": self.computed_this_run,
                "workers": self.workers,
                "checkpoint_every": self.checkpoint_every,
                "hard_timeout_grace_seconds": self.hard_timeout_grace_seconds,
                "worker_retries": self.worker_retries,
                "worker_max_tasks": self.worker_max_tasks,
                "worker_start_timeout_seconds": self.worker_start_timeout_seconds,
                "universe_metadata": dict(self._universe_metadata),
                "status_counts": dict(sorted(self.status_counts.items())),
                "parallel_run": self.parallel_run,
            }
        )
        return output

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_schema_sha256() -> str:
    return hashlib.sha256(
        json.dumps(list(FEATURES), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pair_universe_sha256(pairs: Sequence[tuple[str, str]]) -> str:
    """Hash a sorted directional pair universe without separator ambiguity."""
    digest = hashlib.sha256()
    for prediction, reference in pairs:
        for value in (prediction, reference):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _semantic_generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only per-shard selection fields before cross-shard comparison."""
    canonical = copy.deepcopy(dict(config))
    for section_name in ("selection", "shard"):
        section = canonical.get(section_name)
        if isinstance(section, dict):
            for field in (
                "shard_index",
                "selected_qids_sha256",
                "ordered_selected_qids_sha256",
                "row_count",
                "qid_count",
                "ordered_qids_sha256",
                "start_index",
                "end_index",
            ):
                section.pop(field, None)
    canonical.pop("shard_index", None)
    return canonical


def _validate_generation_config(config: Mapping[str, Any], *, location: str) -> None:
    model = config.get("model")
    model_path = model.get("path") if isinstance(model, dict) else model
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError(f"{location}: generation_config.model is empty")
    if Path(model_path).name != "Qwen3-30B-A3B":
        raise ValueError(f"{location}: expected Qwen3-30B-A3B generation model")

    feature_order: Any = config.get("feature_order")
    if feature_order is None and isinstance(config.get("features"), dict):
        feature_order = config["features"].get("order")
    if feature_order is None and isinstance(config.get("probe"), dict):
        feature_order = config["probe"].get("feature_order")
    if feature_order is not None and tuple(feature_order) != FEATURES:
        raise ValueError(
            f"{location}: generation feature order differs from cluster22 contract"
        )

    stride_candidates = (
        config.get("probe_stride_tokens"),
        config.get("probe_every_tokens"),
        config.get("every_n_tokens"),
        config.get("probe", {}).get("stride_tokens")
        if isinstance(config.get("probe"), dict)
        else None,
        config.get("probe", {}).get("every_tokens")
        if isinstance(config.get("probe"), dict)
        else None,
        config.get("probe", {}).get("token_step")
        if isinstance(config.get("probe"), dict)
        else None,
    )
    declared_strides = [int(value) for value in stride_candidates if value is not None]
    if declared_strides and any(value != 50 for value in declared_strides):
        raise ValueError(f"{location}: generation config is not every-50")


def expand_input_paths(values: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value.is_dir():
            candidates = sorted(value.glob("*.jsonl"))
            if not candidates:
                raise FileNotFoundError(f"No JSONL files in input directory: {value}")
            paths.extend(candidates)
        else:
            paths.append(value)
    resolved = [path.resolve() for path in paths]
    if len(resolved) not in {1, 4, 16}:
        raise ValueError(
            "--inputs must resolve to one merged JSONL or exactly 4/16 shard JSONLs; "
            f"found {len(resolved)}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("--inputs contains duplicate paths")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    return resolved


def _finite_feature_values(features: Any, *, location: str) -> dict[str, float]:
    if not isinstance(features, dict):
        raise TypeError(f"{location}: features must be an ordered JSON object")
    observed_order = tuple(features.keys())
    if observed_order != FEATURES:
        raise ValueError(
            f"{location}: feature order mismatch; observed={observed_order} "
            f"expected={FEATURES}"
        )
    output: dict[str, float] = {}
    for name in FEATURES:
        try:
            value = float(features[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{location}: nonnumeric feature {name!r}") from error
        if not math.isfinite(value):
            raise ValueError(f"{location}: nonfinite feature {name!r}")
        output[name] = value
    return output


def _full_rollout_valid(main: Mapping[str, Any], full_key: str) -> bool:
    if "full_valid" in main:
        return bool(main["full_valid"] and full_key)
    finish_reason = str(main.get("finish_reason") or "").strip().lower()
    return bool(
        full_key
        and main.get("close_think_found") is True
        and main.get("box_closed") is True
        and finish_reason in {"stop", "eos"}
    )


def _validate_probe_grid(probes: Sequence[Any], *, location: str) -> None:
    for position, probe in enumerate(probes, 1):
        if not isinstance(probe, dict):
            raise TypeError(f"{location}: probe {position} is not an object")
        probe_index = probe.get("probe_index")
        step_tokens = probe.get("step_tokens")
        if isinstance(probe_index, bool) or int(probe_index or 0) != position:
            raise ValueError(
                f"{location}: expected probe_index={position}, observed={probe_index!r}"
            )
        expected_step = 50 * position
        if isinstance(step_tokens, bool) or int(step_tokens or 0) != expected_step:
            raise ValueError(
                f"{location}: every-50 grid violation at probe {position}; "
                f"expected step_tokens={expected_step}, observed={step_tokens!r}"
            )


def load_records(
    paths: Sequence[Path],
    *,
    equivalence: MathEquivalenceCache,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    def scan(*, build_rows: bool, collect_pairs: bool) -> dict[str, Any]:
        step_rows: list[dict[str, Any]] = []
        qid_rows: list[dict[str, Any]] = []
        pairs: set[tuple[str, str]] = set()
        seen_qids: set[str] = set()
        generation_semantic_sha: str | None = None
        generation_fingerprints: set[str] = set()
        shard_summaries: list[dict[str, Any]] = []

        for path in paths:
            stat_before = path.stat()
            path_sha = file_sha256(path)
            path_qids = 0
            path_probes = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise ValueError(f"{path}:{line_number}: blank JSONL record")
                    obj = json.loads(line)
                    location = f"{path}:{line_number}"
                    if not isinstance(obj, dict):
                        raise TypeError(f"{location}: record is not an object")
                    qid = str(obj.get("qid") or "").strip()
                    if not qid:
                        raise ValueError(f"{location}: empty qid")
                    if qid in seen_qids:
                        raise ValueError(f"{location}: duplicate qid {qid!r}")
                    seen_qids.add(qid)
                    path_qids += 1

                    generation_config = obj.get("generation_config")
                    fingerprint = obj.get("generation_config_sha256")
                    if not isinstance(generation_config, dict):
                        raise TypeError(
                            f"{location}: generation_config is not an object"
                        )
                    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                        raise ValueError(
                            f"{location}: invalid generation_config_sha256"
                        )
                    if config_sha256(generation_config) != fingerprint:
                        raise ValueError(
                            f"{location}: generation_config_sha256 mismatch"
                        )
                    _validate_generation_config(generation_config, location=location)
                    semantic_sha = config_sha256(
                        _semantic_generation_config(generation_config)
                    )
                    if generation_semantic_sha is None:
                        generation_semantic_sha = semantic_sha
                    elif semantic_sha != generation_semantic_sha:
                        raise ValueError(
                            f"{location}: mixed semantic generation contracts across shards"
                        )
                    generation_fingerprints.add(fingerprint)

                    main = obj.get("main")
                    if not isinstance(main, dict):
                        raise TypeError(f"{location}: main is not an object")
                    full_answer = str(main.get("full_answer") or "").strip()
                    full_key = str(main.get("full_answer_key") or "").strip()
                    if full_answer and not full_key:
                        full_key = answer_key(full_answer)
                    gold_answer = str(obj.get("gold_answer") or "").strip()
                    gold_key = answer_key(gold_answer)
                    full_valid = _full_rollout_valid(main, full_key)
                    full_gold_eligible = bool(full_valid and full_key and gold_key)
                    if (
                        collect_pairs
                        and full_gold_eligible
                        and full_key != gold_key
                    ):
                        pairs.add((full_key, gold_key))
                    cot_tokens = int(main.get("cot_tokens") or 0)
                    if cot_tokens < 0:
                        raise ValueError(f"{location}: negative main.cot_tokens")

                    probes = obj.get("probes")
                    if not isinstance(probes, list):
                        raise TypeError(f"{location}: probes is not a list")
                    _validate_probe_grid(probes, location=location)
                    path_probes += len(probes)
                    full_correct = int(
                        build_rows
                        and full_gold_eligible
                        and equivalence.equivalent(
                            full_key,
                            gold_key,
                            prediction_key=full_key,
                            reference_key=gold_key,
                        )
                    )
                    if build_rows:
                        qid_rows.append(
                            {
                                "qid": qid,
                                "problem": str(obj.get("problem") or ""),
                                "gold_answer": gold_answer,
                                "gold_answer_key": gold_key,
                                "full_answer": full_answer,
                                "full_answer_key": full_key,
                                "full_valid": int(full_valid),
                                "full_correct_gold": full_correct,
                                "cot_tokens": cot_tokens,
                                "probe_count": len(probes),
                                "generation_config_sha256": fingerprint,
                            }
                        )

                    for position, probe in enumerate(probes, 1):
                        probe_location = f"{location}:qid={qid!r}:probe={position}"
                        feature_values = _finite_feature_values(
                            probe.get("features"), location=probe_location
                        )
                        probe_answer = str(probe.get("probe_answer") or "").strip()
                        probe_key_raw = probe.get("answer_key")
                        if not isinstance(probe_key_raw, str):
                            raise TypeError(
                                f"{probe_location}: answer_key must be a string"
                            )
                        probe_key = probe_key_raw.strip()
                        if (
                            "box_closed" not in probe
                            or type(probe["box_closed"]) is not bool
                        ):
                            raise TypeError(
                                f"{probe_location}: box_closed must be boolean"
                            )
                        if (
                            "eligible" not in probe
                            or type(probe["eligible"]) is not bool
                        ):
                            raise TypeError(
                                f"{probe_location}: eligible must be boolean"
                            )
                        expected_eligible = bool(probe["box_closed"] and probe_key)
                        if probe["eligible"] is not expected_eligible:
                            raise ValueError(
                                f"{probe_location}: eligible disagrees with "
                                "box_closed/answer_key"
                            )

                        final_pair_eligible = bool(
                            expected_eligible and full_valid and probe_key and full_key
                        )
                        gold_pair_eligible = bool(
                            expected_eligible and probe_key and gold_key
                        )
                        if collect_pairs:
                            if final_pair_eligible and probe_key != full_key:
                                pairs.add((probe_key, full_key))
                            if gold_pair_eligible and probe_key != gold_key:
                                pairs.add((probe_key, gold_key))
                        if build_rows:
                            final_label = int(
                                final_pair_eligible
                                and equivalence.equivalent(
                                    probe_key,
                                    full_key,
                                    prediction_key=probe_key,
                                    reference_key=full_key,
                                )
                            )
                            gold_label = int(
                                gold_pair_eligible
                                and equivalence.equivalent(
                                    probe_key,
                                    gold_key,
                                    prediction_key=probe_key,
                                    reference_key=gold_key,
                                )
                            )
                            row: dict[str, Any] = {
                                "qid": qid,
                                "probe_index": position,
                                "step_tokens": 50 * position,
                                "probe_answer": probe_answer,
                                "probe_answer_key": probe_key,
                                "eligible": int(expected_eligible),
                                "box_closed": int(probe["box_closed"]),
                                "full_valid": int(full_valid),
                                "full_correct_gold": full_correct,
                                "label_final_consistency": final_label,
                                "label_gold_safe": gold_label,
                            }
                            row.update(feature_values)
                            step_rows.append(row)

            stat_after = path.stat()
            if (
                stat_before.st_size != stat_after.st_size
                or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            ):
                raise RuntimeError(f"Input changed while scanning: {path}")
            shard_summaries.append(
                {
                    "path": str(path),
                    "sha256": path_sha,
                    "qids": path_qids,
                    "probes": path_probes,
                }
            )

        return {
            "step_rows": step_rows,
            "qid_rows": qid_rows,
            "pairs": pairs,
            "qid_count": len(seen_qids),
            "generation_semantic_sha256": generation_semantic_sha,
            "generation_config_sha256_values": sorted(generation_fingerprints),
            "inputs": shard_summaries,
        }

    # Pass 1 validates the complete corpus and retains only the ~unique symbolic
    # pair universe.  Crucially, the 1.258M feature rows do not exist when the 64
    # fork workers are created, avoiding massive copy-on-write pressure.
    first_pass = scan(build_rows=False, collect_pairs=True)
    ordered_pairs = sorted(first_pass["pairs"])
    input_identity = config_sha256(
        {
            "inputs": [
                {
                    "ordinal": index,
                    "sha256": item["sha256"],
                    "qids": item["qids"],
                    "probes": item["probes"],
                }
                for index, item in enumerate(first_pass["inputs"])
            ]
        }
    )
    equivalence.prepare_pairs(
        ordered_pairs,
        universe_metadata={"input_sha256": input_identity},
    )

    # Pass 2 preserves the original file/qid/probe order and performs only exact
    # checks or lookups in the completed cache while constructing feature rows.
    second_pass = scan(build_rows=True, collect_pairs=False)
    for field in (
        "qid_count",
        "generation_semantic_sha256",
        "generation_config_sha256_values",
        "inputs",
    ):
        if second_pass[field] != first_pass[field]:
            raise RuntimeError(f"Input contract changed between scan passes: {field}")

    steps = pd.DataFrame(second_pass["step_rows"])
    qids = pd.DataFrame(second_pass["qid_rows"])
    if qids.empty:
        raise ValueError("No qid records loaded")
    if steps.empty:
        raise ValueError("No every-50 probes loaded")
    if tuple(name for name in FEATURES if name in steps.columns) != FEATURES:
        raise ValueError("Loaded dataframe lost the ordered cluster22 feature contract")
    feature_matrix = steps.loc[:, list(FEATURES)].to_numpy(dtype=float)
    if feature_matrix.shape[1] != len(FEATURES) or not np.isfinite(feature_matrix).all():
        raise ValueError("Loaded feature matrix is not finite cluster22")
    metadata = {
        "generation_semantic_sha256": first_pass["generation_semantic_sha256"],
        "generation_config_sha256_values": first_pass[
            "generation_config_sha256_values"
        ],
        "inputs": first_pass["inputs"],
        "input_sha256": input_identity,
        "pair_universe_count": len(ordered_pairs),
        "pair_universe_sha256": pair_universe_sha256(ordered_pairs),
        "equivalence_cache": equivalence.diagnostics(),
    }
    return steps, qids, metadata


def stable_qid_split(
    qids: Iterable[str], *, seed: int, validation_fraction: float
) -> tuple[set[str], set[str]]:
    unique = sorted(set(str(qid) for qid in qids))
    if len(unique) < 2:
        raise ValueError("At least two qids are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    def rank(qid: str) -> bytes:
        return hashlib.sha256(f"{seed}\0{qid}".encode("utf-8")).digest()

    ranked = sorted(unique, key=lambda qid: (rank(qid), qid))
    validation_count = max(1, min(len(ranked) - 1, round(len(ranked) * validation_fraction)))
    validation = set(ranked[:validation_count])
    training = set(ranked[validation_count:])
    if training & validation or training | validation != set(unique):
        raise AssertionError("qid split is not a partition")
    return training, validation


def make_model(seed: int, n_jobs: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def _safe_metric(function: Callable[..., Any], *args: Any) -> float | None:
    try:
        return float(function(*args))
    except ValueError:
        return None


def row_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "rows": int(labels.size),
        "positive_rate": float(labels.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": _safe_metric(roc_auc_score, labels, probabilities),
        "average_precision": _safe_metric(
            average_precision_score, labels, probabilities
        ),
    }


def evaluate_policy(
    validation_steps: pd.DataFrame,
    validation_qids: pd.DataFrame,
    *,
    target: str,
    probability_column: str,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    label_column = LABEL_COLUMNS[target]
    ordered = validation_steps.sort_values(["qid", "step_tokens"]).copy()
    hits = ordered[
        (ordered["eligible"] == 1) & (ordered[probability_column] >= threshold)
    ]
    first_hits = {
        str(qid): group.iloc[0]
        for qid, group in hits.groupby("qid", sort=False)
    }

    decisions: list[dict[str, Any]] = []
    for qid_row in validation_qids.sort_values("qid").to_dict("records"):
        qid = str(qid_row["qid"])
        hit = first_hits.get(qid)
        stopped = hit is not None
        if stopped:
            selected_gold_correct = int(hit["label_gold_safe"])
            selected_matches_full = int(hit["label_final_consistency"])
            selected_tokens = int(hit["step_tokens"])
            selected_answer = str(hit["probe_answer"])
            selected_probability = float(hit[probability_column])
            selected_target_correct = int(hit[label_column])
            probe_index: int | None = int(hit["probe_index"])
        else:
            selected_gold_correct = int(qid_row["full_correct_gold"])
            selected_matches_full = int(qid_row["full_valid"])
            selected_tokens = int(qid_row["cot_tokens"])
            selected_answer = str(qid_row["full_answer"])
            selected_probability = float("nan")
            selected_target_correct = (
                int(qid_row["full_valid"])
                if target == TARGET_FINAL
                else int(qid_row["full_correct_gold"])
            )
            probe_index = None
        decisions.append(
            {
                "qid": qid,
                "target": target,
                "stopped": int(stopped),
                "probe_index": probe_index,
                "selected_tokens": selected_tokens,
                "full_cot_tokens": int(qid_row["cot_tokens"]),
                "probability": selected_probability,
                "selected_answer": selected_answer,
                "selected_target_correct": selected_target_correct,
                "selected_matches_full": selected_matches_full,
                "selected_correct_gold": selected_gold_correct,
                "full_correct_gold": int(qid_row["full_correct_gold"]),
                "harm": int(
                    int(qid_row["full_correct_gold"]) == 1
                    and selected_gold_correct == 0
                ),
                "rescue": int(
                    int(qid_row["full_correct_gold"]) == 0
                    and selected_gold_correct == 1
                ),
            }
        )
    frame = pd.DataFrame(decisions)
    stopped = frame[frame["stopped"] == 1]
    full_mean = float(frame["full_cot_tokens"].mean())
    selected_mean = float(frame["selected_tokens"].mean())
    baseline_accuracy = float(frame["full_correct_gold"].mean())
    selected_accuracy = float(frame["selected_correct_gold"].mean())
    metrics = {
        "qids": int(len(frame)),
        "threshold": float(threshold),
        "coverage": float(frame["stopped"].mean()),
        "stopped_qids": int(frame["stopped"].sum()),
        "no_hit_fallback_qids": int((frame["stopped"] == 0).sum()),
        "earliest_hit_target_precision": (
            float(stopped["selected_target_correct"].mean()) if len(stopped) else None
        ),
        "earliest_hit_matches_full": (
            float(stopped["selected_matches_full"].mean()) if len(stopped) else None
        ),
        "earliest_hit_accuracy_gold": (
            float(stopped["selected_correct_gold"].mean()) if len(stopped) else None
        ),
        "baseline_full_accuracy_gold": baseline_accuracy,
        "policy_accuracy_gold": selected_accuracy,
        "policy_accuracy_delta_pp": 100.0 * (selected_accuracy - baseline_accuracy),
        "harm_count": int(frame["harm"].sum()),
        "rescue_count": int(frame["rescue"].sum()),
        "mean_full_cot_tokens": full_mean,
        "mean_policy_tokens": selected_mean,
        "mean_token_reduction": full_mean - selected_mean,
        "relative_token_reduction": (
            (full_mean - selected_mean) / full_mean if full_mean > 0 else None
        ),
    }
    return metrics, frame


def atomic_joblib_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    joblib.dump(value, temporary)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20_260_722)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--expected-qids", type=int, default=10_000)
    parser.add_argument("--equivalence-timeout-seconds", type=float, default=0.25)
    parser.add_argument("--equivalence-workers", type=int, default=64)
    parser.add_argument("--equivalence-cache", type=Path)
    parser.add_argument("--equivalence-progress", type=Path)
    parser.add_argument("--equivalence-checkpoint-every", type=int, default=256)
    parser.add_argument(
        "--equivalence-hard-timeout-grace-seconds", type=float, default=0.75
    )
    parser.add_argument("--equivalence-worker-retries", type=int, default=1)
    parser.add_argument("--equivalence-worker-max-tasks", type=int, default=1_000)
    parser.add_argument(
        "--equivalence-worker-start-timeout-seconds", type=float, default=30.0
    )
    parser.add_argument(
        "--equivalence-start-method",
        choices=("auto", "fork", "forkserver", "spawn"),
        default="auto",
    )
    parser.add_argument("--backbone-model", default="")
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if args.equivalence_workers < 1:
        raise ValueError("equivalence-workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = expand_input_paths(args.inputs)
    equivalence_cache_path = (
        args.equivalence_cache
        if args.equivalence_cache is not None
        else args.output_dir / "math_equivalence_cache.sqlite3"
    )
    equivalence = ParallelMathEquivalenceCache(
        cache_path=equivalence_cache_path,
        progress_path=args.equivalence_progress,
        timeout_seconds=args.equivalence_timeout_seconds,
        workers=args.equivalence_workers,
        checkpoint_every=args.equivalence_checkpoint_every,
        hard_timeout_grace_seconds=args.equivalence_hard_timeout_grace_seconds,
        worker_retries=args.equivalence_worker_retries,
        worker_max_tasks=args.equivalence_worker_max_tasks,
        worker_start_timeout_seconds=args.equivalence_worker_start_timeout_seconds,
        start_method=args.equivalence_start_method,
    )
    steps, qids, input_metadata = load_records(
        input_paths, equivalence=equivalence
    )
    if args.expected_qids and len(qids) != args.expected_qids:
        raise ValueError(
            f"Expected {args.expected_qids} qids, loaded {len(qids)}"
        )
    if steps.duplicated(["qid", "probe_index"]).any():
        raise ValueError("Duplicate (qid, probe_index) after loading")

    train_qids, validation_qids = stable_qid_split(
        qids["qid"], seed=args.seed, validation_fraction=args.validation_fraction
    )
    qids = qids.copy()
    qids["split"] = qids["qid"].map(
        lambda qid: "validation" if str(qid) in validation_qids else "train"
    )
    steps = steps.copy()
    steps["split"] = steps["qid"].map(
        lambda qid: "validation" if str(qid) in validation_qids else "train"
    )

    validation_step_predictions = steps[steps["split"] == "validation"].copy()
    validation_qid_frame = qids[qids["split"] == "validation"].copy()
    target_metrics: dict[str, Any] = {}
    policy_frames: list[pd.DataFrame] = []
    final_models: dict[str, LGBMClassifier] = {}

    for target in TARGETS:
        label_column = LABEL_COLUMNS[target]
        if target == TARGET_FINAL:
            usable_mask = (steps["eligible"] == 1) & (steps["full_valid"] == 1)
        else:
            usable_mask = steps["eligible"] == 1
        usable = steps[usable_mask].copy()
        training = usable[usable["split"] == "train"].copy()
        validation = usable[usable["split"] == "validation"].copy()
        if training[label_column].nunique() != 2:
            raise ValueError(f"{target}: training split does not contain both classes")
        if validation[label_column].nunique() != 2:
            raise ValueError(f"{target}: validation split does not contain both classes")

        holdout_model = make_model(args.seed, args.n_jobs)
        holdout_model.fit(training.loc[:, list(FEATURES)], training[label_column])
        validation_probabilities = holdout_model.predict_proba(
            validation.loc[:, list(FEATURES)]
        )[:, 1]
        row_summary = row_metrics(
            validation[label_column].to_numpy(dtype=int),
            validation_probabilities,
            threshold=args.threshold,
        )

        probability_column = PROBABILITY_COLUMNS[target]
        validation_step_predictions[probability_column] = holdout_model.predict_proba(
            validation_step_predictions.loc[:, list(FEATURES)]
        )[:, 1]
        policy_summary, policy_frame = evaluate_policy(
            validation_step_predictions,
            validation_qid_frame,
            target=target,
            probability_column=probability_column,
            threshold=args.threshold,
        )
        policy_frames.append(policy_frame)

        final_model = make_model(args.seed, args.n_jobs)
        final_model.fit(usable.loc[:, list(FEATURES)], usable[label_column])
        final_models[target] = final_model
        target_metrics[target] = {
            "label_column": label_column,
            "usable_qids": int(usable["qid"].nunique()),
            "usable_rows": int(len(usable)),
            "train_qids": int(training["qid"].nunique()),
            "train_rows": int(len(training)),
            "train_label_counts": {
                "0": int((training[label_column] == 0).sum()),
                "1": int((training[label_column] == 1).sum()),
            },
            "validation_qids_with_usable_rows": int(validation["qid"].nunique()),
            "validation_rows": int(len(validation)),
            "validation_label_counts": {
                "0": int((validation[label_column] == 0).sum()),
                "1": int((validation[label_column] == 1).sum()),
            },
            "row": row_summary,
            "policy_earliest_hit": policy_summary,
        }

    input_sha = {
        str(item["path"]): str(item["sha256"])
        for item in input_metadata["inputs"]
    }
    for target, model in final_models.items():
        package = {
            "model": model,
            "feats": list(FEATURES),
            "feature_names": tuple(FEATURES),
            "tag": TRAINER_SCHEMA,
            "feature_schema": "math_cluster22_every50_v1",
            "feature_schema_sha256": feature_schema_sha256(),
            "target": target,
            "label_version": LABEL_VERSION,
            "threshold": float(args.threshold),
            "deployment_threshold": float(args.threshold),
            "probe_stride_tokens": 50,
            "seed": int(args.seed),
            "training_qids": int(
                steps.loc[
                    (steps["eligible"] == 1)
                    & ((steps["full_valid"] == 1) if target == TARGET_FINAL else True),
                    "qid",
                ].nunique()
            ),
            "backbone_model": args.backbone_model,
            "generation_semantic_sha256": input_metadata[
                "generation_semantic_sha256"
            ],
            "input_sha256": input_sha,
        }
        atomic_joblib_dump(package, args.output_dir / MODEL_FILENAMES[target])

    policy_decisions = pd.concat(policy_frames, ignore_index=True)
    validation_step_predictions.to_parquet(
        args.output_dir / "validation_probe_predictions.parquet", index=False
    )
    policy_decisions.to_csv(
        args.output_dir / "validation_policy_decisions.csv", index=False
    )
    qids.to_csv(args.output_dir / "qid_summary.csv", index=False)
    atomic_text(
        json.dumps(list(FEATURES), indent=2) + "\n",
        args.output_dir / "feature_columns.json",
    )

    summary = {
        "schema": TRAINER_SCHEMA,
        "label_version": LABEL_VERSION,
        "backbone_model": args.backbone_model,
        "probe_stride_tokens": 50,
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "feature_schema": "math_cluster22_every50_v1",
        "feature_schema_sha256": feature_schema_sha256(),
        "features": list(FEATURES),
        "all_qids": int(len(qids)),
        "all_probe_rows": int(len(steps)),
        "train_qids": int(len(train_qids)),
        "validation_qids": int(len(validation_qids)),
        "full_valid_qids": int(qids["full_valid"].sum()),
        "natural_full_accuracy_gold": float(qids["full_correct_gold"].mean()),
        "input": input_metadata,
        "equivalence_cache_final": equivalence.diagnostics(),
        "targets": target_metrics,
        "models": {
            target: str(args.output_dir / MODEL_FILENAMES[target])
            for target in TARGETS
        },
    }
    atomic_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        args.output_dir / "metrics.json",
    )
    equivalence.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
