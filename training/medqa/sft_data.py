"""Completion-only tokenization shared with the production SFT trainer."""
from __future__ import annotations
import collections
import json
from pathlib import Path
from typing import Any
import torch

STOP = "<stop>"
STOP_ID = 151669

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completion_content(row: dict[str, Any], name: str, index: int) -> str:
    """Return the assistant text, never a serialized messages container."""
    messages = row.get("completion")
    if not isinstance(messages, list) or len(messages) != 1:
        raise RuntimeError(f"invalid completion messages {name}:{index}")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RuntimeError(f"invalid completion role {name}:{index}")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise RuntimeError(f"invalid completion content {name}:{index}")
    if content.startswith("[{'role':") or content.startswith('[{"role":'):
        raise RuntimeError(f"serialized completion wrapper {name}:{index}")
    return content


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)
        result: dict[str, list[list[int]]] = {
            "input_ids": [], "attention_mask": [], "labels": []
        }
        for item in batch:
            padding = width - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [self.pad_id] * padding)
            result["attention_mask"].append(item["attention_mask"] + [0] * padding)
            result["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def prepare(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int, name: str
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    result: list[dict[str, list[int]]] = []
    histogram: collections.Counter[int] = collections.Counter()
    maximum = completion_labels = 0
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        completion = completion_content(row, name, index)
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        expected_stops = int(row["stop_label_count"])
        if completion_ids.count(STOP_ID) != expected_stops:
            raise RuntimeError(f"stop tokenization drift {name}:{index}")
        if tokenizer.eos_token_id is not None:
            completion_ids.append(int(tokenizer.eos_token_id))
        input_ids = list(prompt_ids) + completion_ids
        if len(input_ids) > max_length:
            raise RuntimeError(f"sequence too long {name}:{index}={len(input_ids)}")
        histogram[expected_stops] += 1
        maximum = max(maximum, len(input_ids))
        completion_labels += len(completion_ids)
        result.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + list(completion_ids),
        })
    return result, {
        "rows": len(rows),
        "stop_count_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "stop_labels": sum(k * v for k, v in histogram.items()),
        "completion_labels": completion_labels,
        "max_sequence_tokens": maximum,
    }
