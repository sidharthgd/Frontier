"""
Frontier as the RL environment's engine: one workload trace replayed through
Frontier's simulator with the vLLM v1 scheduler in priority mode, so that the
RL hook (frontier/rl/scheduler.py) orders admission. frontier_env.FrontierEnv
builds a WorkloadSpec per episode and runs it.

- WorkloadSpec: the replayed trace and the hardware/engine settings;
  a100_llama31_8b / PRESETS name the vLLM-validated A100 setup.
- make_simulation_config: a Frontier SimulationConfig, through Frontier's CLI.
- _shared_predictors: one execution-time predictor per engine setting per
  process (building one takes ~1 min), optionally with rounded inputs.
- fast_forest: installed on import (FRONTIER_FAST_FOREST=0 turns it off).
"""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from frontier.rl import fast_forest
from frontier.simulator import Simulator


@dataclass
class WorkloadSpec:
    """A workload trace to replay (a CSV with arrived_at, num_prefill_tokens,
    num_decode_tokens and optionally session_id, block_hash_ids, think_time,
    as frontier_env.FrontierEnv.workload_trace writes it) and the engine."""

    name: str
    replay_trace: Optional[str] = None
    num_requests: int = 0  # informational: the trace's row count
    arrivals: str = "replay"  # the only kind
    replay_time_scale: float = 1.0
    max_tokens: int = 4096
    # hardware and engine
    model: str = "llama2_7b_dense_example"
    device: str = "h200"
    network_device: str = "h200_dgx"
    tensor_parallel: int = 1
    max_tokens_in_batch: int = 2048
    batch_size_cap: int = 256
    num_blocks: Optional[int] = None
    enable_prefix_caching: bool = False
    # fixed time per batch instead of the profiled predictor (plumbing tests)
    dummy_exec_ms: Optional[float] = None
    # Frontier's per-step CPU overhead model (off by default in Frontier)
    model_cpu_overhead: bool = True
    # Round every predictor input to this many significant bits (5 = within
    # ~3%), so that prediction memo hits across episodes; None = exact.
    # Approximate, like Vidur's 64-token KV granularity; see rl/README.
    prediction_rounding_bits: Optional[int] = None
    # decode CUDA graphs as vLLM runs them: none | piecewise | full_decode_only
    decode_cuda_graph_mode: str = "none"
    # context limit (vLLM --max-model-len) and the predictor's lookup range
    # (prediction_max_tokens_per_request None: max(max_tokens, 4096))
    max_model_len: Optional[int] = None
    prediction_max_tokens_per_request: Optional[int] = None
    prediction_max_batch_size: Optional[int] = None
    # any other frontier.main flags, e.g. ["--enable_thinking_mode", ...]
    extra_args: List[str] = field(default_factory=list)


def a100_llama31_8b(**overrides) -> "WorkloadSpec":
    """
    A100 80GB PCIe + Llama-3.1-8B as validated against vLLM 0.10.2
    (data/profiling/compute/a100/llama3.1-8b/README.md): FlashInfer, 2048-token
    steps, 256 sequences, 128k context, 453,088-token KV cache, piecewise CUDA
    graphs, no CPU-overhead tables.
    """
    spec = dict(A100_LLAMA31_8B)
    spec.update(overrides)
    return WorkloadSpec(**spec)


# a100_llama31_8b's settings (also a named preset for the training stack's
# `frontier: {preset: a100_llama31_8b}`)
A100_LLAMA31_8B = dict(
    name="a100_llama31_8b",
    model="llama3.1-8b", device="a100", network_device="a100_dgx",
    max_tokens_in_batch=2048, batch_size_cap=256, num_blocks=28318,
    decode_cuda_graph_mode="piecewise", model_cpu_overhead=False,
    max_model_len=131072, prediction_max_tokens_per_request=131072,
    prediction_max_batch_size=256, max_tokens=131072,
)
PRESETS = {"a100_llama31_8b": A100_LLAMA31_8B}


def simulation_args(
    spec: WorkloadSpec, seed: int, output_dir: str, log_level: str = "error"
) -> List[str]:
    args = [
        "--log_level", log_level,
        "--seed", str(seed),
        "--sys_arch", "co-location",
        "--simulation_mode", "online",
        "--cc_backend_config_type", "analytical",
        "--replica_config_model_name", spec.model,
        "--replica_config_device", spec.device,
        "--replica_config_network_device", spec.network_device,
        "--replica_config_attn_tensor_parallel_size", str(spec.tensor_parallel),
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_scheduling_policy", "priority",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--vllm_v1_scheduler_config_max_tokens_in_batch", str(spec.max_tokens_in_batch),
        "--vllm_v1_scheduler_config_batch_size_cap", str(spec.batch_size_cap),
        "--decode_cuda_graph_mode", spec.decode_cuda_graph_mode,
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        str(max(spec.max_tokens_in_batch, 1024)),
        # the predictor's own request-length limit (default 4096) must cover
        # the workload's requests, e.g. 131072 for 128k-context traces
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request",
        str(
            spec.prediction_max_tokens_per_request
            if spec.prediction_max_tokens_per_request is not None
            else max(spec.max_tokens, 4096)
        ),
        "--metrics_config_output_dir", output_dir,
        "--no-metrics_config_write_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]
    args += [
        "--no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling"
        if spec.model_cpu_overhead
        else "--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling"
    ]
    if spec.dummy_exec_ms is not None:
        args += [
            "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
            "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
            str(spec.dummy_exec_ms),
        ]
    else:
        args += ["--no-random_forrest_execution_time_predictor_config_enable_dummy_mode"]
    if spec.num_blocks is not None:
        args += [
            "--vllm_v1_scheduler_config_num_blocks_mode", "explicit",
            "--vllm_v1_scheduler_config_num_blocks", str(spec.num_blocks),
        ]
    if spec.enable_prefix_caching:
        args += ["--vllm_v1_scheduler_config_enable_prefix_caching"]
    if spec.max_model_len is not None:
        args += ["--vllm_v1_scheduler_config_max_model_len", str(spec.max_model_len)]
    if spec.prediction_max_batch_size is not None:
        args += [
            "--random_forrest_execution_time_predictor_config_prediction_max_batch_size",
            str(spec.prediction_max_batch_size),
        ]

    assert spec.replay_trace, "WorkloadSpec needs replay_trace"
    args += [
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", spec.replay_trace,
        "--trace_request_generator_config_time_scale_factor", str(spec.replay_time_scale),
        "--trace_request_generator_config_max_tokens", str(spec.max_tokens),
        "--trace_request_generator_config_seed", str(seed),
    ]
    return args + list(spec.extra_args)


def make_simulation_config(
    spec: WorkloadSpec, seed: int, output_dir: str, log_level: str = "error"
):
    """A SimulationConfig built through Frontier's own CLI parser."""
    from frontier.config import SimulationConfig

    argv = sys.argv
    sys.argv = ["frontier.main"] + simulation_args(spec, seed, output_dir, log_level)
    try:
        return SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = argv


@contextmanager
def _working_dir(path: str) -> Iterator[None]:
    # Frontier resolves data/ and cache/ relative to the working directory
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


# Building a predictor re-derives its lookup tables from the cached sklearn
# models: about 50 s for the H200 Llama-2-7B profile, against seconds for the
# event loop. It depends only on the hardware and engine settings, so it is
# built once per process for each and handed to every later Simulator, which
# also keeps its memo of past predictions warm.
_PREDICTORS: Dict[Tuple, Dict[Any, Any]] = {}
_WORKLOAD_FIELDS = {"name", "num_requests", "arrivals", "replay_trace", "replay_time_scale"}


def _engine_key(spec: WorkloadSpec) -> Tuple:
    return tuple(
        (k, tuple(v) if isinstance(v, list) else v)
        for k, v in sorted(vars(spec).items())
        if k not in _WORKLOAD_FIELDS
    )


def _round_bits(value: Any, bits: int) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        return value
    value = float(value)
    if value == 0.0 or not np.isfinite(value):
        return value
    mantissa, exponent = np.frexp(value)
    return float(np.ldexp(np.round(mantissa * (1 << bits)) / (1 << bits), exponent))


def _round_predictor_inputs(predictor: Any, bits: int) -> None:
    """Round the features of every model lookup, so the memo is reused."""
    for name in ("_get_prediction_for_features", "_get_on_demand_prediction"):
        method = getattr(predictor, name, None)
        if method is None:
            continue

        def rounded(model_name, features, *args, _method=method, **kwargs):
            features = {k: _round_bits(v, bits) for k, v in features.items()}
            return _method(model_name, features, *args, **kwargs)

        setattr(predictor, name, rounded)


@contextmanager
def _shared_predictors(key: Tuple) -> Iterator[None]:
    from frontier.execution_time_predictor.execution_time_predictor_registry import (
        ExecutionTimePredictorRegistry,
    )

    cache = _PREDICTORS.setdefault(key, {})
    original = ExecutionTimePredictorRegistry.__dict__["get"]

    def get(cls, *args, **kwargs):
        cluster_type = kwargs.get("cluster_type")
        if cluster_type not in cache:
            cache[cluster_type] = original.__func__(cls, *args, **kwargs)
            bits = dict(key).get("prediction_rounding_bits")
            if bits is not None:
                _round_predictor_inputs(cache[cluster_type], bits)
        predictor = cache[cluster_type]
        # the only per-Simulator object it holds
        predictor._cc_backend = kwargs.get("cc_backend", predictor._cc_backend)
        return predictor

    ExecutionTimePredictorRegistry.get = classmethod(get)
    try:
        yield
    finally:
        ExecutionTimePredictorRegistry.get = original


# bit-identical single-row forest predictions, ~10x faster episodes
if os.environ.get("FRONTIER_FAST_FOREST", os.environ.get("VIDUR_FRONTIER_FAST_FOREST", "1")) != "0":
    fast_forest.install()

FRONTIER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _RLSimulator(Simulator):
    """Frontier's simulator without the end-of-run metric dump."""

    def _write_output(self) -> None:
        pass
