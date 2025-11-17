# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import AsyncGenerator, Mapping
from copy import copy
from typing import Any, Optional, Union

import numpy as np

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.envs import VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device, cdiv
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import AsyncMPClient, DPAsyncMPClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.output_processor import (OutputProcessor,
                                             RequestOutputCollector)
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import (StatLoggerBase, StatLoggerFactory,
                                     setup_default_loggers)
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from contextlib import contextmanager
logger = init_logger(__name__)


import asyncio
import re
import numpy as np
from typing import Optional, List, Dict, Any
from uuid import uuid4
from vllm.v1.engine.probe_features_vllm import (
    compute_probe_slot_from_vllm_steps,
    SeqFeatureTracker, cum_top_onehot
)

import inspect, time
from uuid import uuid4



import re, time, inspect
import numpy as np
from typing import Optional, Dict, Any, List
from uuid import uuid4

from vllm.v1.engine.probe_features_vllm import (
    compute_probe_slot_from_vllm_steps,
    SeqFeatureTracker,
    cum_top_onehot,
    compute_openqa_slot_from_vllm_steps,   # NEW
    OpenSeqFeatureTracker,                 # NEW
)
from vllm.sampling_params import SamplingParams

class AsyncLLM(EngineClient):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.log_requests = log_requests
        self.log_stats = log_stats

        # Set up stat loggers; independent set for each DP rank.
        self.stat_loggers: list[list[StatLoggerBase]] = setup_default_loggers(
            vllm_config=vllm_config,
            log_stats=self.log_stats,
            engine_num=vllm_config.parallel_config.data_parallel_size,
            custom_stat_loggers=stat_loggers,
        )

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            lora_config=vllm_config.lora_config)

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            mm_registry=mm_registry,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=self.log_stats)

        # EngineCore (starts the engine in background process).
        core_client_class = AsyncMPClient if (
            vllm_config.parallel_config.data_parallel_size
            == 1) else DPAsyncMPClient

        self.engine_core = core_client_class(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )
        if self.stat_loggers:
            for stat_logger in self.stat_loggers[0]:
                stat_logger.log_engine_initialized()
        self.output_handler: Optional[asyncio.Task] = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
    ) -> "AsyncLLM":
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=not disable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

        if handler := getattr(self, "output_handler", None):
            handler.cancel()

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        if self.errored:
            raise EngineDeadError()

        assert isinstance(params, SamplingParams), \
            "Pooling is not supported in V1"

        # Create a new output collector for the request.
        queue = RequestOutputCollector(output_kind=params.output_kind)

        # Convert Input --> Request.
        prompt_str, request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            tokenization_kwargs, trace_headers, prompt_adapter_request,
            priority)

        if params.n == 1:
            await self._add_request(request, prompt_str, None, 0, queue)
            return queue

        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request_id, params)
        for idx in range(params.n):
            request_id, params = parent_request.get_child_info(idx)
            child_request = request if idx == params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params
            await self._add_request(child_request, prompt_str, parent_request,
                                    idx, queue)
        return queue

    async def _add_request(self, request: EngineCoreRequest,
                           prompt: Optional[str],
                           parent_req: Optional[ParentRequest], index: int,
                           queue: RequestOutputCollector):

        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index,
                                          queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            self._run_output_handler()

            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
                tokenization_kwargs=tokenization_kwargs,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        stat_loggers = self.stat_loggers if log_stats else None

        async def output_handler():
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = IterationStats() if (
                        log_stats and num_outputs) else None

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                        slices = (outputs.outputs, )
                    else:
                        slices = np.array_split(
                            outputs.outputs,
                            cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE))

                    for i, outputs_slice in enumerate(slices):
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats)
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        if i + 1 < len(slices):
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        await engine_core.abort_requests_async(
                            processed_outputs.reqs_to_abort)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if stat_loggers:
                        assert outputs.scheduler_stats is not None
                        AsyncLLM._record_stats(
                            stat_loggers[outputs.engine_index],
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    async def abort(self, request_id: str) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = self.output_processor.abort_requests((request_id, ))
        await self.engine_core.abort_requests_async(request_ids)

        if self.log_requests:
            logger.info("Aborted request %s.", request_id)

    @staticmethod
    def _record_stats(
        stat_loggers: list[StatLoggerBase],
        scheduler_stats: SchedulerStats,
        iteration_stats: Optional[IterationStats],
    ):
        """static so that it can be used from the output_handler task
        without a circular ref to AsyncLLM."""
        for stat_logger in stat_loggers:
            stat_logger.record(scheduler_stats=scheduler_stats,
                               iteration_stats=iteration_stats)

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ):
        raise ValueError("Not Supported on V1 yet.")

    async def get_vllm_config(self) -> VllmConfig:
        return self.vllm_config

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_decoding_config(self):
        raise ValueError("Not Supported on V1 yet.")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.processor.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.tokenizer.get_lora_tokenizer(lora_request)

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs=None,
        model_output=None,
    ) -> None:
        for loggers in self.stat_loggers:
            for stat_logger in loggers:
                stat_logger.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")

    async def start_profile(self) -> None:
        await self.engine_core.profile_async(True)

    async def stop_profile(self) -> None:
        await self.engine_core.profile_async(False)

    async def reset_mm_cache(self) -> None:
        self.processor.mm_registry.reset_processor_cache()
        self.processor.mm_input_cache_client.reset()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        if device == Device.CPU:
            raise ValueError("Not supported on CPU.")
        await self.engine_core.reset_prefix_cache_async()

    async def sleep(self, level: int = 1) -> None:
        await self.engine_core.sleep_async(level)

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        await self.engine_core.wake_up_async(tags)

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(self,
                             method: str,
                             timeout: Optional[float] = None,
                             args: tuple = (),
                             kwargs: Optional[dict] = None):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs)

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    def _lp_step_to_pairs(self, step_obj) -> list[tuple[str, float]]:
        """
        把 vLLM 的“一步 top-logprobs”统一转成 [(decoded_token, logprob), ...]（按 rank 升序）。
        支持两种形态：
          - dict 形态: {token_id: Logprob(...)}
          - list 形态: [Logprob(...), ...]
        """
        pairs: list[tuple[str, float]] = []
        try:
            if hasattr(step_obj, "items"):
                items = list(step_obj.items())
                items.sort(key=lambda kv: getattr(kv[1], "rank", 10**9))
                for _, obj in items:
                    txt = (getattr(obj, "decoded_token", "") or "").strip()
                    lp = float(getattr(obj, "logprob", float("-inf")))
                    pairs.append((txt, lp))
                return pairs

            if isinstance(step_obj, (list, tuple)):
                items = list(step_obj)
                items.sort(key=lambda obj: getattr(obj, "rank", 10**9))
                for obj in items:
                    txt = (getattr(obj, "decoded_token", "") or "").strip()
                    lp = float(getattr(obj, "logprob", float("-inf")))
                    pairs.append((txt, lp))
                return pairs
        except Exception:
            pass
        return pairs

    # ===========================
    # 2) 兼容 vLLM finished 属性 / 方法
    # ===========================
    def _lp_step_to_pairs(self, step_obj) -> list[tuple[str, float]]:
        """
        把 vLLM 的“一步 top-logprobs”统一转成 [(decoded_token, logprob), ...]（按 rank 升序）。
        支持两种形态：
          - dict 形态: {token_id: Logprob(...)}
          - list 形态: [Logprob(...), ...]
        """
        pairs: list[tuple[str, float]] = []
        try:
            # dict 形态: {token_id: Logprob(...)}
            if hasattr(step_obj, "items"):
                items = list(step_obj.items())
                items.sort(key=lambda kv: getattr(kv[1], "rank", 10**9))
                for _, obj in items:
                    txt = (getattr(obj, "decoded_token", "") or "").strip()
                    lp = float(getattr(obj, "logprob", float("-inf")))
                    pairs.append((txt, lp))
                return pairs

            # list/tuple 形态: [Logprob(...), ...]
            if isinstance(step_obj, (list, tuple)):
                items = list(step_obj)
                items.sort(key=lambda obj: getattr(obj, "rank", 10**9))
                for obj in items:
                    txt = (getattr(obj, "decoded_token", "") or "").strip()
                    lp = float(getattr(obj, "logprob", float("-inf")))
                    pairs.append((txt, lp))
                return pairs
        except Exception:
            pass
        return pairs

    # ===========================
    # 2) 兼容 vLLM finished 属性 / 方法
    # ===========================
    def _is_finished_output(self, o) -> tuple[bool, Optional[str]]:
        """
        vLLM 有的版本是 o.finished (bool)，有的是 o.finished() (方法)。
        这里统一成 (is_finished: bool, reason: Optional[str])。
        """
        fin_attr = getattr(o, "finished", False)
        if callable(fin_attr):
            try:
                is_finished = bool(fin_attr())
            except TypeError:
                # 万一签名有变，就退回当成属性用
                is_finished = bool(fin_attr)
        else:
            is_finished = bool(fin_attr)

        reason = getattr(o, "finish_reason", None)
        if reason is None:
            # 有的版本只写在 stop_reason 上
            reason = getattr(o, "stop_reason", None)

        return is_finished, reason

    # ===========================
    # 3) 一次性短请求（probe 内部复用）
    # ===========================
    async def _run_once(
        self,
        *,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
        stop_on_first_finish: bool = True,
        tokenization_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        以一次性短请求的形式跑到 max_tokens 或 stop 条件，然后返回最终文本与 logprobs。

        额外返回：
          - num_generated_tokens：本段实际生成的 token 数
          - finish_reason：模型的终止原因（如 "stop" / "length" / None）
        """
        rid = request_id or f"once-{uuid4().hex[:8]}"
        text_total = ""
        steps_logprobs = []
        finished = False
        num_generated_tokens = 0
        finish_reason = None

        async for out in self.generate(
            prompt,
            sampling_params,
            request_id=rid,
            tokenization_kwargs=tokenization_kwargs,
        ):
            o = out.outputs[0]

            if o.text:
                # vLLM 流是累计文本
                text_total = o.text
            if o.logprobs:
                # 同理是累计步；最后统一取整段
                steps_logprobs = o.logprobs

            if o.token_ids is not None:
                num_generated_tokens = len(o.token_ids)

            is_finished, fin_reason = self._is_finished_output(o)
            if is_finished:
                finished = True
                finish_reason = fin_reason
                if stop_on_first_finish:
                    break

        return {
            "text": text_total,
            "steps_logprobs": steps_logprobs,
            "finished": finished,
            "num_generated_tokens": int(num_generated_tokens),
            "finish_reason": finish_reason,
        }

    # ===========================
    # 4) 正则 & openQA boxed 抽取（MATH 用 brace-balanced 方式）
    # ===========================
    _FA_BLOCK = re.compile(
        r"<\s*final_answer\s*>(.*?)<\s*/\s*final_answer\s*>",
        flags=re.I | re.S,
    )
    _FA_OPEN = re.compile(r"<\s*final_answer\b", flags=re.I)
    _FA_CLOSE = re.compile(r"</\s*final_answer\s*>", flags=re.I)
    _THINK_CLOSE = re.compile(r"</\s*think\s*>", flags=re.I)

    _BOXED_RE = re.compile(r"\\boxed\s*\{\s*(.*?)\s*\}", flags=re.S)

    # ---- brace-balanced 版 boxed 提取（复刻你 offline 的逻辑）----
    def _last_boxed_only_string(self, string: str) -> Optional[str]:
        """返回最后一个 \\boxed{...} 或 \\fbox{...}（包含外层大括号），用 brace 计数避免截断 \\frac 之类。"""
        if not isinstance(string, str):
            return None
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            ch = string[i]
            if ch == "{":
                num_left_braces_open += 1
            if ch == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            return None
        return string[idx:right_brace_idx + 1]

    def _remove_boxed_braces(self, s: str) -> Optional[str]:
        """从 '\\boxed{...}' 或 '\\fbox{...}' 中去掉外层大括号，只保留内部内容。"""
        try:
            if not isinstance(s, str):
                return None
            if s.startswith("\\boxed{"):
                return s[len("\\boxed{"):-1]
            if s.startswith("\\fbox{"):
                return s[len("\\fbox{"):-1]
            return None
        except Exception:
            return None

    def _extract_answer_key(self, text: str) -> str:
        """
        openQA / MATH 场景下用于构造 answer_key：
        - 优先用 brace-balanced 的方式，从最后一个 \\boxed{...} / \\fbox{...} 中抽内部；
        - 若失败，再退回简单正则 / <final_answer>；
        - 最后兜底返回整段 strip。
        """
        if not isinstance(text, str):
            return ""

        # 1) 优先：brace-balanced 查找最后一个 \boxed / \fbox
        chunk = self._last_boxed_only_string(text)
        if chunk is not None:
            inner = self._remove_boxed_braces(chunk)
            if inner is not None:
                return inner.strip()

        # 2) 次优：简单正则（不保证嵌套严格正确）
        all_boxes = self._BOXED_RE.findall(text)
        if all_boxes:
            return all_boxes[-1].strip()

        # 3) 再退：<final_answer>...</final_answer> 内部，如有，再看内部是否有 \boxed
        mfa = self._FA_BLOCK.search(text)
        if mfa:
            inner = mfa.group(1).strip()
            inner_boxes = self._BOXED_RE.findall(inner)
            if inner_boxes:
                return inner_boxes[-1].strip()
            return inner

        # 4) 兜底
        return text.strip()

    # ===========================
    # 5) probe 一次（区分 closeqa / openqa）
    # ===========================
    async def _probe_once(
        self,
        *,
        base_prompt: str,
        think_accum: str,
        suffix: str,
        probe_max_steps: int,
        topk: int,
        qa_mode: str,
    ) -> dict:
        """
        串行探针： base_prompt + think_accum + suffix （closeQA）或固定 openQA 模板。
        不与主解码并发，不共享同一条长流；依赖全局 enable_prefix_caching=True 自动命中 KV 前缀。
        返回 steps_logprobs 已转换为 list[list[(decoded_token, logprob)]]。
        """
        qa_mode = (qa_mode or "closeqa").lower()
        if qa_mode == "openqa":
            # openQA：忽略传入 suffix，统一固定为 "</think> \\boxed{"，
            # 不再用 "}" 作为 stop，只用 max_tokens/EOS 控制长度。
            probe_prompt = f"{base_prompt}{think_accum} </think> \\boxed{{"
            stop_seqs: List[str] = []
        else:
            # closeQA：保持原有 suffix + </final_answer> 停
            probe_prompt = f"{base_prompt}{think_accum}{suffix}"
            stop_seqs = ["</final_answer>"]

        sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(probe_max_steps),
            logprobs=int(topk),
            stop=stop_seqs,
        )

        res = await self._run_once(
            prompt=probe_prompt,
            sampling_params=sp,
            stop_on_first_finish=True,
            tokenization_kwargs=None,
        )

        raw_steps = res.get("steps_logprobs") or []
        steps_pairs: list[list[tuple[str, float]]] = []
        for step in raw_steps:
            p = self._lp_step_to_pairs(step)
            if p:
                steps_pairs.append(p)

        return {
            "steps_logprobs": steps_pairs,
            "probe_text": res.get("text") or "",
        }

    # ===========================
    # 6) SamplingParams 辅助
    # ===========================
    def _sp_from_like(self, sp_like):
        if isinstance(sp_like, SamplingParams):
            return sp_like
        if isinstance(sp_like, dict):
            return SamplingParams(**sp_like)
        try:
            return SamplingParams(**dict(sp_like))
        except Exception:
            pass
        raise TypeError(
            f"sampling_params must be dict or SamplingParams, got {type(sp_like)}"
        )

    # ===========================
    # 7) 主函数：generate_with_checks
    # ===========================
    async def generate_with_checks(
        self,
        *,
        prompt: str,  # 包含 <system> + user + assistant<think> 开场
        sampling_params_main,  # dict / SamplingParams
        probe_max_steps: int = 4,
        check_interval: int = 50,
        topk: int = 8,
        classifier_callable=None,  # (feats_dict)->(prob, letter_hint)
        threshold: float = 0.95,
        request_id: Optional[str] = None,
        enable_early_stop: bool = True,
        qa_mode: str = "closeqa",
    ) -> dict:
        """
        单流版本（只有一条主流）+ 间隔 probe + 可选早停。

        closeQA:
          - 可以用 <final_answer>...</final_answer> 做硬停；
          - probe 特征用于 ABCD bucket + classifier 早停。

        openQA (MATH):
          - 完全不依赖 <final_answer>，只看 \boxed{...}；
          - probe 时 prompt 固定到 "</think> \\boxed{"，特征从 boxed 轨迹里提；
          - 早停时直接返回标准形式 "\\boxed{...}"。
        """
        qa_mode = (qa_mode or "closeqa").lower()
        if qa_mode not in ("closeqa", "openqa"):
            raise ValueError(
                f"qa_mode must be 'closeqa' or 'openqa', got {qa_mode!r}"
            )

        if enable_early_stop and classifier_callable is None:
            raise ValueError(
                "enable_early_stop=True 但未提供 classifier_callable；"
                "如果不想使用分类器早停，请将 enable_early_stop 设为 False。"
            )

        use_classifier_for_logging = classifier_callable is not None
        use_classifier_for_early_stop = (
            enable_early_stop and classifier_callable is not None
        )

        sp_main = self._sp_from_like(sampling_params_main)
        # 主流不取 logprobs，避免巨量 Python 列表化拷贝
        try:
            sp_main.logprobs = None
        except Exception:
            pass
        # 给主流一个安全上限（由外部决定最大生成量）
        if not getattr(sp_main, "max_tokens", None):
            sp_main.max_tokens = 4096

        # Tracker：closeQA / openQA 不同
        if qa_mode == "closeqa":
            tracker: Any = SeqFeatureTracker(W=5, K_recent=3)
        else:
            tracker = OpenSeqFeatureTracker(W=5, K_recent=3)

        # 用来收集所有 probe 的中间信息
        probe_records: list[dict[str, Any]] = []
        last_probe_prob: float = 0.0

        # ===== Round-0 baseline probe（空 think）=====
        base_probe = await self._probe_once(
            base_prompt=prompt,
            think_accum="",
            suffix=" </think> <final_answer>",
            probe_max_steps=probe_max_steps,
            topk=topk,
            qa_mode=qa_mode,
        )

        if qa_mode == "closeqa":
            # 选择题：ABCD 桶聚合
            slot0 = compute_probe_slot_from_vllm_steps(
                steps_logprobs=base_probe["steps_logprobs"],
                topk=int(topk),
                prefer_letter=None,
                probe_text=base_probe["probe_text"] or "",
            )
            feats0 = tracker.update_with_step_vals(slot0["vals"])
            feats0.update(cum_top_onehot(feats0.get("cum_top")))
            answer_key0 = None
        else:
            # openQA / math：用 1D L_sum + 轨迹特征，
            # answer_key 从“fake boxed”中提取
            slot0 = compute_openqa_slot_from_vllm_steps(
                steps_logprobs=base_probe["steps_logprobs"],
                topk=int(topk),
            )
            probe_text0 = base_probe["probe_text"] or ""
            fake_box0 = f"\\boxed{{{probe_text0}}}"
            answer_key0 = self._extract_answer_key(fake_box0)
            feats0 = tracker.update_with_slot(slot0, answer_key=answer_key0)

        if use_classifier_for_logging:
            prob0, letter0 = classifier_callable(feats0)
        else:
            prob0, letter0 = None, None

        if isinstance(prob0, (int, float)):
            last_probe_prob = float(prob0)
        else:
            last_probe_prob = 0.0

        probe_records.append(
            {
                "step": 0,
                "kind": "baseline",
                "probe_text": base_probe["probe_text"],
                "steps_logprobs": base_probe["steps_logprobs"],
                "slot": slot0,
                "feats": feats0,
                "classifier_prob": float(prob0) if prob0 is not None else None,
                "classifier_letter": letter0,
                "early_stop_triggered": False,
                "qa_mode": qa_mode,
            }
        )

        # ========= 单条主生成流 =========
        main_rid = request_id or f"main-{uuid4().hex[:8]}"""
        agen = self.generate(prompt, sp_main, request_id=main_rid)

        think_accum = ""
        seen_close = False       # </think> 是否出现过
        seen_final_open = False  # <final_answer> 是否出现过（仅 closeQA 有意义）
        total_tokens = 0
        last_probe_at = 0
        finished = False

        try:
            async for out in agen:
                o = out.outputs[0]

                # 累计文本（vLLM 是累计式）
                new_text = (o.text or "")
                if len(new_text) > len(think_accum):
                    delta = new_text[len(think_accum):]
                    think_accum = new_text
                else:
                    delta = ""

                if delta:
                    if self._THINK_CLOSE.search(delta):
                        seen_close = True
                    if self._FA_OPEN.search(delta):
                        seen_final_open = True

                # token_ids 是“本次补全”的累计长度
                if o.token_ids is not None:
                    total_tokens = len(o.token_ids)

                # ================= closeQA 硬停 =================
                if qa_mode == "closeqa":
                    # 硬停 A：THINK 中出现完整 <final_answer>...</final_answer>
                    mfin = self._FA_BLOCK.search(think_accum)
                    if mfin:
                        inside = mfin.group(1)
                        finished = True
                        await self.abort(main_rid)
                        return {
                            "final_text": f"<final_answer>{inside}</final_answer>",
                            "final_cause": "think_final",
                            "step_tokens": total_tokens,
                            "probe_prob": float(last_probe_prob),
                            "probe_records": probe_records,
                        }

                    # 硬停 B：出现 </final_answer>（视为已作答，做一次 hard probe）
                    if (self._FA_CLOSE.search(delta) is not None) or (
                        self._FA_CLOSE.search(think_accum) is not None
                    ):
                        if not seen_close:
                            suffix = " </think> <final_answer>"
                        elif not seen_final_open:
                            suffix = " <final_answer>"
                        else:
                            suffix = ""
                        probe = await self._probe_once(
                            base_prompt=prompt,
                            think_accum=think_accum,
                            suffix=suffix,
                            probe_max_steps=max(1, probe_max_steps // 2),
                            topk=topk,
                            qa_mode=qa_mode,
                        )

                        slot_hard = compute_probe_slot_from_vllm_steps(
                            steps_logprobs=probe["steps_logprobs"],
                            topk=int(topk),
                            prefer_letter=tracker.current_cum_top_letter(),
                            probe_text=probe["probe_text"] or "",
                        )
                        feats_hard = tracker.update_with_step_vals(
                            slot_hard["vals"]
                        )
                        feats_hard.update(
                            cum_top_onehot(feats_hard.get("cum_top"))
                        )
                        letter_hard = (
                            slot_hard.get("probe_letter")
                            or tracker.current_cum_top_letter()
                            or "A"
                        )
                        final_text = f"<final_answer>{letter_hard}</final_answer>"

                        probe_records.append(
                            {
                                "step": int(total_tokens),
                                "kind": "hard_stop",
                                "probe_text": probe["probe_text"],
                                "steps_logprobs": probe["steps_logprobs"],
                                "slot": slot_hard,
                                "feats": feats_hard,
                                "classifier_prob": None,
                                "classifier_letter": None,
                                "early_stop_triggered": False,
                                "qa_mode": qa_mode,
                            }
                        )

                        finished = True
                        await self.abort(main_rid)
                        return {
                            "final_text": final_text,
                            "final_cause": "think_final",
                            "step_tokens": total_tokens,
                            "probe_prob": 1.0,
                            "probe_records": probe_records,
                        }

                # ================= 自然结束（open/close 共用） =================
                is_finished, fin_reason = self._is_finished_output(o)
                if is_finished:
                    finished = True
                    finish_reason = fin_reason or "finished"
                    return {
                        "final_text": think_accum,
                        "final_cause": finish_reason,
                        "step_tokens": total_tokens,
                        "probe_prob": float(last_probe_prob),
                        "probe_records": probe_records,
                    }

                # ================= 达到检查步长 -> interval probe =================
                if total_tokens - last_probe_at >= int(check_interval):
                    last_probe_at = total_tokens

                    if qa_mode == "closeqa":
                        if not seen_close:
                            suffix = " </think> <final_answer>"
                        elif not seen_final_open:
                            suffix = " <final_answer>"
                        else:
                            suffix = ""
                    else:
                        # openQA：suffix 无意义（_probe_once 内部会忽略）
                        suffix = ""

                    probe = await self._probe_once(
                        base_prompt=prompt,
                        think_accum=think_accum,
                        suffix=suffix,
                        probe_max_steps=probe_max_steps,
                        topk=topk,
                        qa_mode=qa_mode,
                    )

                    if qa_mode == "closeqa":
                        slot = compute_probe_slot_from_vllm_steps(
                            steps_logprobs=probe["steps_logprobs"],
                            topk=int(topk),
                            prefer_letter=tracker.current_cum_top_letter(),
                            probe_text=probe["probe_text"] or "",
                        )
                        feats = tracker.update_with_step_vals(slot["vals"])
                        feats.update(cum_top_onehot(feats.get("cum_top")))
                        answer_key = None
                    else:
                        # openQA / MATH interval probe：fake boxed + brace-balanced 抽取
                        slot = compute_openqa_slot_from_vllm_steps(
                            steps_logprobs=probe["steps_logprobs"],
                            topk=int(topk),
                        )
                        probe_text_i = probe["probe_text"] or ""
                        fake_box_i = f"\\boxed{{{probe_text_i}}}"
                        answer_key = self._extract_answer_key(fake_box_i)
                        feats = tracker.update_with_slot(
                            slot, answer_key=answer_key
                        )

                    feats["step"] = int(total_tokens)

                    if use_classifier_for_logging:
                        prob, letter_hint = classifier_callable(feats)
                    else:
                        prob, letter_hint = None, None

                    if isinstance(prob, (int, float)):
                        last_probe_prob = float(prob)

                    record = {
                        "step": int(total_tokens),
                        "kind": "interval",
                        "probe_text": probe["probe_text"],
                        "steps_logprobs": probe["steps_logprobs"],
                        "slot": slot,
                        "feats": feats,
                        "classifier_prob": float(prob)
                        if prob is not None
                        else None,
                        "classifier_letter": letter_hint,
                        "early_stop_triggered": False,
                        "qa_mode": qa_mode,
                    }

                    # ===== 仅当允许 early_stop 时，才用 classifier+threshold 终止 =====
                    elig_flag = slot.get("early_stop_elig", qa_mode == "openqa")
                    if (
                        use_classifier_for_early_stop
                        and elig_flag
                        and (prob is not None)
                        and float(prob) >= float(threshold)
                    ):
                        if qa_mode == "closeqa":
                            letter = (
                                slot.get("probe_letter")
                                or letter_hint
                                or "A"
                            )
                            final_text = (
                                f"<final_answer>{letter}</final_answer>"
                            )
                        else:
                            # openQA：早停时，只关心 boxed 内部，不使用 <final_answer>
                            probe_text_es = probe["probe_text"] or ""
                            fake_box_es = f"\\boxed{{{probe_text_es}}}"
                            inner = self._extract_answer_key(fake_box_es)
                            final_text = f"\\boxed{{{inner}}}"

                        record["early_stop_triggered"] = True
                        probe_records.append(record)

                        finished = True
                        await self.abort(main_rid)
                        return {
                            "final_text": final_text,
                            "final_cause": "early_stop",
                            "step_tokens": total_tokens,
                            "probe_prob": float(prob),
                            "probe_records": probe_records,
                        }
                    else:
                        probe_records.append(record)

                    await asyncio.sleep(0)

            # 流走到头：统一视作自然结束
            return {
                "final_text": think_accum,
                "final_cause": "finished",
                "step_tokens": total_tokens,
                "probe_prob": float(last_probe_prob),
                "probe_records": probe_records,
            }
        finally:
            # 任何异常/提前返回都确保主流被关闭并 abort，杜绝悬挂序列
            try:
                await agen.aclose()
            except Exception:
                pass
            if not finished:
                try:
                    await self.abort(main_rid)
                except Exception:
                    pass
