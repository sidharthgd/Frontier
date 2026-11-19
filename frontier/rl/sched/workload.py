"""
Workloads for the RL stack: the configs and request generators it used from
Vidur, copied so the generated requests stay exactly what they were (Frontier's
own request generators differ, e.g. in how uniform lengths round).

    config = make_simulation_config(qps=20, num_blocks=4000)
    requests = generate_requests(config.request_generator_config)

FrontierEnv (frontier_env.py) writes the requests for a workload seed to a CSV
that Frontier replays. Three kinds of workload (make_request_generator_config):

    SyntheticRequestGeneratorConfig    Poisson arrivals, lengths shuffled from a trace
    TraceWindowRequestGeneratorConfig  a seed-picked window of a trace's real arrivals
    SessionTraceRequestGeneratorConfig agent sessions with block hash ids for the
                                       prefix cache (frontier/trace_prep/sessions.py)

SimulationConfig and its parts keep the attribute paths of Vidur's config
(config.cluster_config.replica_scheduler_config.block_size, ...) with only the
fields FrontierEnv reads.
"""

import functools
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from frontier.rl.env import FRONTIER_ROOT
from frontier.rl.scoring import CONCURRENT_OVERPROVISION
from frontier.utils.random import set_seeds

# --- configs -------------------------------------------------------------------


@dataclass
class PoissonRequestIntervalGeneratorConfig:
    seed: int = 42
    qps: float = 0.5


@dataclass
class TraceRequestLengthGeneratorConfig:
    seed: int = 42
    max_tokens: int = 4096
    trace_file: str = "data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv"
    prefill_scale_factor: float = 1
    decode_scale_factor: float = 1


@dataclass
class SyntheticRequestGeneratorConfig:
    seed: int = 42
    length_generator_config: TraceRequestLengthGeneratorConfig = field(
        default_factory=TraceRequestLengthGeneratorConfig
    )
    interval_generator_config: PoissonRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )
    num_requests: Optional[int] = 128

    def __post_init__(self):
        self.max_tokens = self.length_generator_config.max_tokens


@dataclass
class TraceWindowRequestGeneratorConfig:
    """
    Replays num_requests consecutive requests of an arrival trace, starting at
    a row picked by the seed, so each seed is a different real window. qps
    rescales time so the whole trace averages this rate (windows keep their
    local burstiness); None replays the native rate.
    """

    seed: int = 42
    trace_file: str = "data/processed_traces/splitwise_conv.csv"
    num_requests: int = 300
    qps: Optional[float] = None
    # per request, clipped proportionally
    max_tokens: int = 4096


@dataclass
class SessionTraceRequestGeneratorConfig:
    """
    num_requests calls at qps sampled from a session trace with block hashes
    (frontier/trace_prep: whole sessions with their own call gaps, overlaid at
    random phases; the seed picks sessions and phases). Requests carry
    session_id and block_hash_ids for the prefix cache.
    """

    seed: int = 42
    # session trace CSV (frontier.trace_prep), or synthetic:<preset> / synthetic:random
    trace_file: str = ""
    num_requests: int = 300
    qps: float = 1.0
    # drop calls whose prompt + output exceed this
    max_tokens: int = 131072
    # scale each session's gaps between calls
    session_time_scale: float = 1.0
    # mean think time between turns for traces without times
    turn_gap: float = 30.0
    # stationary (sessions already running) or start
    session_phase: str = "stationary"
    # a session's next call waits for its previous one to complete
    session_closed_loop: bool = True
    # seconds of warm-up calls before the num_requests scored ones, simulated
    # but left out of rewards and metrics
    session_warmup: float = 0.0
    # if > 0, a closed system of this many agents running sessions back to back
    # (frontier.trace_prep.sessions.sample_concurrent_workload) instead of a qps
    # window; num_requests calls are scored, the first arriving after
    # session_warmup seconds of simulated time
    session_concurrency: int = 0
    # with session_concurrency: mean gap before an agent's next session
    session_gap: float = 30.0


@dataclass
class RlVllmSchedulerConfig:
    """The vLLM scheduler settings FrontierEnv passes to Frontier (defaults:
    vLLM V1 on A100 with the OpenAI API server)."""

    batch_size_cap: int = 256
    block_size: int = 16
    watermark_blocks_fraction: float = 0.0
    # None: the KV cache Vidur's memory planner gave (see vidur_num_blocks)
    num_blocks: Optional[int] = None
    max_num_batched_tokens: int = 2048
    long_prefill_token_threshold: int = 0
    num_priority_levels: int = 16
    tie_break: str = "arrival"
    decision_mode: str = "every_batch"
    rerank_interval_s: float = 0.1

    def __post_init__(self):
        assert self.tie_break in (
            "arrival",
            "previous_order",
        ), f"Unknown tie_break {self.tie_break}"
        assert self.decision_mode in (
            "every_batch",
            "arrival_and_periodic",
        ), f"Unknown decision_mode {self.decision_mode}"
        assert self.rerank_interval_s > 0


@dataclass
class ReplicaConfig:
    """The model and device Vidur sized the KV cache for (vidur_num_blocks)."""

    model_name: str = "meta-llama/Meta-Llama-3-8B"
    device: str = "a100"
    network_device: str = "a100_pairwise_nvlink"


@dataclass
class ClusterConfig:
    num_replicas: int = 1
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    replica_scheduler_config: RlVllmSchedulerConfig = field(
        default_factory=RlVllmSchedulerConfig
    )


@dataclass
class MetricsConfig:
    output_dir: str = "simulator_output"


@dataclass
class SimulationConfig:
    seed: int = 42
    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)
    request_generator_config: object = field(
        default_factory=SyntheticRequestGeneratorConfig
    )
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)


# KV cache blocks Vidur's memory planner gave a replica, by (model, device,
# max_tokens, block_size), for workloads that leave num_blocks unset. These
# came from vidur.rl.frontier_env.vidur_num_blocks (Vidur commit 62e7b81);
# Frontier runs hold the same number of tokens.
VIDUR_NUM_BLOCKS = {
    ("meta-llama/Meta-Llama-3-8B", "a100", 4096, 16): 30208,
    ("meta-llama/Meta-Llama-3-8B", "a100", 8192, 16): 30208,
    ("meta-llama/Meta-Llama-3-8B", "a100", 131072, 16): 24576,
    ("meta-llama/Meta-Llama-3-8B", "h100", 4096, 16): 30208,
    ("meta-llama/Meta-Llama-3-8B", "h100", 8192, 16): 30208,
    ("meta-llama/Meta-Llama-3-8B", "h100", 131072, 16): 24576,
}


def vidur_num_blocks(config: SimulationConfig) -> int:
    """The KV cache size for this config: num_blocks if set, else the size
    Vidur's memory planner gave it (VIDUR_NUM_BLOCKS)."""
    scheduler_config = config.cluster_config.replica_scheduler_config
    if scheduler_config.num_blocks:
        return int(scheduler_config.num_blocks)
    replica_config = config.cluster_config.replica_config
    key = (
        replica_config.model_name,
        replica_config.device,
        int(config.request_generator_config.max_tokens),
        int(scheduler_config.block_size),
    )
    if key not in VIDUR_NUM_BLOCKS:
        raise ValueError(
            f"no Vidur KV cache size for {key}: set the workload's num_blocks "
            f"(known: {sorted(VIDUR_NUM_BLOCKS)})"
        )
    return VIDUR_NUM_BLOCKS[key]


# --- generators ------------------------------------------------------------------


class WorkloadRequest:
    """A generated request: arrival and token counts (session traces add
    session_id, block_hash_ids, think_time, ids_cover_output, warmup)."""

    def __init__(self, arrived_at: float, num_prefill_tokens: int, num_decode_tokens: int):
        self.arrived_at = arrived_at
        self.num_prefill_tokens = num_prefill_tokens
        self.num_decode_tokens = num_decode_tokens


class _TraceLengths:
    """Request lengths from a trace, clipped to max_tokens and shuffled by the
    seed (Vidur's TraceRequestLengthGenerator)."""

    def __init__(self, config: TraceRequestLengthGeneratorConfig):
        self.config = config
        self.trace_df = pd.read_csv(config.trace_file)

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        )
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)

        # deduct the diff tokens from the prefill and decode tokens proportionally
        prefill_tokens_ratio = self.trace_df["num_prefill_tokens"] / total_tokens
        decode_tokens_ratio = self.trace_df["num_decode_tokens"] / total_tokens

        self.trace_df["num_prefill_tokens"] -= (
            np.ceil(diff_tokens * prefill_tokens_ratio)
        ).astype(int)

        self.trace_df["num_decode_tokens"] -= (
            np.ceil(diff_tokens * decode_tokens_ratio)
        ).astype(int)

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        assert all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= self.config.max_tokens
        )
        assert all(self.trace_df["num_prefill_tokens"] > 0)
        assert all(self.trace_df["num_decode_tokens"] > 0)

        # randomly shuffle the df based on the seed
        self.trace_df = self.trace_df.sample(frac=1, random_state=self.config.seed)
        self.next_request_idx = 0

    def get_next_num_tokens(self):
        if self.next_request_idx >= len(self.trace_df):
            return None, None

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return (
            row["num_prefill_tokens"],
            row["num_decode_tokens"],
        )


class _PoissonIntervals:
    """Vidur's PoissonRequestIntervalGenerator (Python's random, capped at
    three mean intervals)."""

    def __init__(self, config: PoissonRequestIntervalGeneratorConfig):
        self.qps = config.qps
        self.std = 1.0 / self.qps
        self.max_interval = self.std * 3.0

    def get_next_inter_request_time(self) -> float:
        next_interval = -math.log(1.0 - random.random()) / self.qps
        next_interval = min(next_interval, self.max_interval)

        return next_interval


def _synthetic_requests(config: SyntheticRequestGeneratorConfig) -> List[WorkloadRequest]:
    lengths = _TraceLengths(config.length_generator_config)
    intervals = _PoissonIntervals(config.interval_generator_config)
    assert config.num_requests
    set_seeds(config.seed)

    requests = []
    current_time = 0
    for _ in range(config.num_requests):
        arrived_at = current_time + intervals.get_next_inter_request_time()
        prefill_tokens, decode_tokens = lengths.get_next_num_tokens()
        request = WorkloadRequest(
            arrived_at=arrived_at,
            num_prefill_tokens=int(prefill_tokens),
            num_decode_tokens=int(decode_tokens),
        )
        current_time = request.arrived_at
        requests.append(request)

    # sort requests by arrival time
    requests.sort(key=lambda x: x.arrived_at)
    return requests


@functools.lru_cache(maxsize=8)
def _load_trace(trace_file: str, max_tokens: int) -> pd.DataFrame:
    trace_df = pd.read_csv(trace_file).sort_values("arrived_at", kind="stable")
    prefill = trace_df["num_prefill_tokens"].to_numpy(dtype=float)
    decode = trace_df["num_decode_tokens"].to_numpy(dtype=float)

    # same as _TraceLengths: deduct any excess over max_tokens from prefill and
    # decode proportionally, keeping at least one of each
    total = prefill + decode
    excess = np.clip(total - max_tokens, 0, None)
    prefill = np.clip(prefill - np.ceil(excess * prefill / total), 1, None)
    decode = np.clip(decode - np.ceil(excess * decode / total), 1, None)

    return pd.DataFrame(
        {
            "arrived_at": trace_df["arrived_at"].to_numpy(dtype=float),
            "num_prefill_tokens": prefill.astype(int),
            "num_decode_tokens": decode.astype(int),
        }
    )


def _trace_window_requests(config: TraceWindowRequestGeneratorConfig) -> List[WorkloadRequest]:
    """A window of consecutive requests from an arrival trace (e.g. the Azure
    LLM inference traces), keeping the real arrival pattern and the real pairing
    of arrival times with request lengths."""
    trace_df = _load_trace(config.trace_file, config.max_tokens)
    assert config.num_requests <= len(trace_df), (
        f"{config.trace_file} has only {len(trace_df)} requests"
    )
    arrived_at = trace_df["arrived_at"].to_numpy()
    native_qps = (len(arrived_at) - 1) / (arrived_at[-1] - arrived_at[0])
    time_scale_factor = native_qps / config.qps if config.qps else 1.0

    rng = np.random.default_rng(config.seed)
    num_requests = config.num_requests
    start = int(rng.integers(0, len(trace_df) - num_requests + 1))
    window = trace_df.iloc[start : start + num_requests]

    arrived_at = window["arrived_at"].to_numpy()
    arrived_at = (arrived_at - arrived_at[0]) * time_scale_factor
    return [
        WorkloadRequest(
            arrived_at=float(t),
            num_prefill_tokens=int(p),
            num_decode_tokens=int(d),
        )
        for t, p, d in zip(
            arrived_at,
            window["num_prefill_tokens"],
            window["num_decode_tokens"],
        )
    ]




def _session_requests(config: SessionTraceRequestGeneratorConfig) -> List[WorkloadRequest]:
    """A window of calls sampled from a session trace by
    frontier/trace_prep/sessions.py. Each request also gets session_id and
    block_hash_ids (runs text, frontier/utils/block_hash_ids.py) for the
    prefix cache."""
    from frontier.trace_prep import sessions

    assert config.trace_file, "session_trace needs a trace_file"
    if config.session_concurrency:
        # closed system: over-provisioned calls; FrontierEnv scores the first
        # num_requests arriving after the warm-up and stops there
        workload = sessions.sample_named_workload(
            config.trace_file,
            CONCURRENT_OVERPROVISION * config.num_requests,
            config.qps,
            config.seed,
            config.max_tokens,
            concurrency=config.session_concurrency,
            session_time_scale=config.session_time_scale,
            turn_gap=config.turn_gap,
            session_gap=config.session_gap,
        )
    else:
        workload = sessions.sample_named_workload(
            config.trace_file,
            config.num_requests,
            config.qps,
            config.seed,
            config.max_tokens,
            session_time_scale=config.session_time_scale,
            turn_gap=config.turn_gap,
            phase=config.session_phase,
            closed_loop=config.session_closed_loop,
            **({"warmup": config.session_warmup} if config.session_warmup else {}),
        )
    requests = []
    for row in workload.itertuples(index=False):
        request = WorkloadRequest(
            arrived_at=float(row.arrived_at),
            num_prefill_tokens=int(row.num_prefill_tokens),
            num_decode_tokens=int(row.num_decode_tokens),
        )
        request.session_id = int(row.session_id)
        request.block_hash_ids = str(row.block_hash_ids)
        # the ids name the output's blocks too (frontier/trace_prep)
        request.ids_cover_output = int(getattr(row, "ids_cover_output", 0) or 0)
        think_time = getattr(row, "think_time", float("nan"))
        request.think_time = None if think_time != think_time else float(think_time)
        # warm-up call (session_warmup): simulated, not scored
        request.warmup = int(getattr(row, "warmup", 0) or 0)
        requests.append(request)
    return requests


def generate_requests(config) -> List[WorkloadRequest]:
    """The requests of a request generator config."""
    if isinstance(config, SyntheticRequestGeneratorConfig):
        return _synthetic_requests(config)
    if isinstance(config, TraceWindowRequestGeneratorConfig):
        return _trace_window_requests(config)
    if isinstance(config, SessionTraceRequestGeneratorConfig):
        return _session_requests(config)
    raise TypeError(f"unknown request generator config {type(config).__name__}")


def set_workload_seed(request_generator_config, seed: int) -> None:
    # arrival times use the generator seed; trace lengths are shuffled with the
    # length generator seed
    request_generator_config.seed = seed
    for name in ("length_generator_config", "interval_generator_config"):
        sub_config = getattr(request_generator_config, name, None)
        if sub_config is not None:
            sub_config.seed = seed


# --- workload configs ------------------------------------------------------------

# trace names make_simulation_config accepts in place of a path (relative to
# the Frontier root)
TRACE_FILES = {
    # Azure LLM inference trace 2023, conversation / code (via Splitwise);
    # these have arrival times, so they also work with arrivals="trace"
    "azure_conv": "./data/processed_traces/splitwise_conv.csv",
    "azure_code": "./data/processed_traces/splitwise_code.csv",
    # first-turn lengths, Llama-3 tokenizer (prepare_traces.py)
    "sharegpt": "./data/processed_traces/sharegpt_llama3_tokenizer.csv",
    "lmsys": "./data/processed_traces/lmsys_llama3_tokenizer.csv",
    "arxiv": "./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv",
}

# Session traces with block hashes for prefix caching (frontier/trace_prep),
# for arrivals="sessions". Paths are under the Frontier root.
SESSION_TRACE_FILES = {
    "freeinference": "data/processed_traces/freeinference_2026-08-30_128k.csv",
    "tracelab": "data/processed_traces/tracelab_coding_128k.csv",
    "sharegpt_multiturn": "data/processed_traces/sharegpt_multiturn_llama3_4k.csv",
}


def _trace_file(name: str) -> str:
    if name not in TRACE_FILES:
        return name
    return os.path.normpath(os.path.join(FRONTIER_ROOT, TRACE_FILES[name]))


def _session_trace_file(name: str) -> str:
    if name not in SESSION_TRACE_FILES:
        return name
    return os.path.join(FRONTIER_ROOT, SESSION_TRACE_FILES[name])


def make_request_generator_config(
    trace_file: str = "azure_conv",
    arrivals: str = "poisson",
    qps: Optional[float] = 10.0,
    num_requests: int = 300,
    max_tokens: int = 4096,
    seed: int = 42,
    session_time_scale: float = 1.0,
    turn_gap: float = 30.0,
    session_phase: str = "stationary",
    session_closed_loop: bool = True,
    session_warmup: float = 0.0,
    session_concurrency: int = 0,
    session_gap: float = 30.0,
):
    """
    The workload: request lengths from `trace_file` (a TRACE_FILES name or a
    path) and either
      arrivals="poisson": Poisson arrivals at qps, lengths shuffled by seed;
      arrivals="trace":   a seed-picked window of num_requests consecutive
                          requests replayed with their real arrival times,
                          time scaled so the whole trace averages qps
                          (None = native rate);
      arrivals="sessions": num_requests calls at qps from a session trace
                          (SESSION_TRACE_FILES name or path), whole sessions
                          with their own gaps and block hashes, for Frontier's
                          prefix cache (frontier/trace_prep/sessions.py).
    """
    if arrivals == "sessions":
        assert qps, "arrivals=sessions needs a qps"
        return SessionTraceRequestGeneratorConfig(
            seed=seed,
            trace_file=_session_trace_file(trace_file),
            num_requests=num_requests,
            qps=qps,
            max_tokens=max_tokens,
            session_time_scale=session_time_scale,
            turn_gap=turn_gap,
            session_phase=session_phase,
            session_closed_loop=session_closed_loop,
            session_warmup=session_warmup,
            session_concurrency=session_concurrency,
            session_gap=session_gap,
        )
    trace_file = _trace_file(trace_file)
    if arrivals == "trace":
        return TraceWindowRequestGeneratorConfig(
            seed=seed,
            trace_file=trace_file,
            num_requests=num_requests,
            qps=qps,
            max_tokens=max_tokens,
        )
    assert arrivals == "poisson", f"unknown arrivals {arrivals!r}"
    return SyntheticRequestGeneratorConfig(
        seed=seed,
        num_requests=num_requests,
        length_generator_config=TraceRequestLengthGeneratorConfig(
            seed=seed,
            trace_file=trace_file,
            max_tokens=max_tokens,
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(
            seed=seed,
            qps=qps,
        ),
    )


def make_simulation_config(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    device: str = "a100",
    trace_file: str = "azure_conv",
    arrivals: str = "poisson",
    max_tokens: int = 4096,
    qps: Optional[float] = 10.0,
    num_requests: int = 300,
    batch_size_cap: int = 256,
    max_num_batched_tokens: int = 2048,
    num_blocks: Optional[int] = None,
    num_priority_levels: int = 16,
    tie_break: str = "arrival",
    decision_mode: str = "every_batch",
    rerank_interval_s: float = 0.1,
    seed: int = 42,
    output_dir: str = "simulator_output",
    session_time_scale: float = 1.0,
    turn_gap: float = 30.0,
    session_phase: str = "stationary",
    session_closed_loop: bool = True,
    session_warmup: float = 0.0,
    session_concurrency: int = 0,
    session_gap: float = 30.0,
) -> SimulationConfig:
    """
    Single-replica config with the workload from make_request_generator_config
    (Poisson arrivals with trace request lengths by default). num_blocks=None
    takes the KV cache size Vidur gave model_name on device
    (vidur_num_blocks); set it lower to force memory pressure and preemption.
    """
    return SimulationConfig(
        seed=seed,
        cluster_config=ClusterConfig(
            num_replicas=1,
            replica_config=ReplicaConfig(
                model_name=model_name,
                device=device,
                network_device=f"{device}_pairwise_nvlink",
            ),
            replica_scheduler_config=RlVllmSchedulerConfig(
                batch_size_cap=batch_size_cap,
                max_num_batched_tokens=max_num_batched_tokens,
                num_blocks=num_blocks,
                num_priority_levels=num_priority_levels,
                tie_break=tie_break,
                decision_mode=decision_mode,
                rerank_interval_s=rerank_interval_s,
            ),
        ),
        request_generator_config=make_request_generator_config(
            trace_file=trace_file,
            arrivals=arrivals,
            qps=qps,
            num_requests=num_requests,
            max_tokens=max_tokens,
            seed=seed,
            session_time_scale=session_time_scale,
            turn_gap=turn_gap,
            session_phase=session_phase,
            session_closed_loop=session_closed_loop,
            session_warmup=session_warmup,
            session_concurrency=session_concurrency,
            session_gap=session_gap,
        ),
        metrics_config=MetricsConfig(output_dir=output_dir),
    )
