"""Training workloads with prefix reuse, sampled from a session trace.

    workload = sample_session_workload(load_session_trace(path), num_requests=500, qps=4, seed=0)

A real agentic trace runs at well under 1 qps over days, with sessions that
last hours and call the model every few seconds to minutes. Scaling its time
up to a simulator's load would squeeze tool and think time too, until a
session's next call arrives before its previous one has finished. Instead,
the sampler overlays many sessions, each keeping its own call gaps:

- The window is W = num_requests / qps seconds.
- Sessions are drawn without replacement, with probability proportional to
  duration + W (the chance a session of a stationary server overlaps the
  window). Each is placed at a uniformly random phase, its first call at
  start ~ U(-duration, W), and contributes its calls that land in [0, W).
  Sessions are added until num_requests calls land, and the first
  num_requests are kept. The calls are then, in expectation, a uniform
  sample of the trace's calls placed uniformly in time.

The window is a stationary slice of a server holding about
qps * (mean gap between a session's calls) sessions at once, so the mix of
early (short) and late (long) prompts is the trace's. A session that was
already running at t=0 sends its first call in the window with a cold cache,
as a server that just restarted would see it.

Trace times are kept (scaled by session_time_scale). Traces without
arrived_at (ShareGPT) get think-time gaps from a seeded exponential with mean
turn_gap seconds, floored at min_turn_gap.

closed_loop=True (the default) makes sessions closed loop in the simulator:
each session's first call in the window arrives at its sampled time, and
each later call carries think_time, the trace's gap between the previous
call completing and this one starting (completed_at; arrival gaps when the
trace lacks it). Frontier holds such a call until the previous call of its
session completes and releases it think_time later, so a session never has
two calls in flight, while different sessions still arrive independently.
The sampled times of later calls then only order the window; under load the
achieved call rate falls below qps, as a real agent's would.

phase="start" starts every session within the window instead
(start ~ U(0, W)), which over-weights the start of sessions.

The output is a session trace (frontier/trace_prep/common.py) with
session_ids renumbered from 0; Frontier replays it with
--request_generator_config_type trace_replay and
--vllm_v1_scheduler_config_enable_prefix_caching.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Sessions:
    frame: pd.DataFrame  # sorted by session, step
    bounds: np.ndarray  # rows of session i: bounds[i]:bounds[i+1]
    has_arrivals: bool


@functools.lru_cache(maxsize=4)
def load_session_trace(path: str, max_tokens: Optional[int] = None) -> _Sessions:
    """A session trace grouped by session; calls over max_tokens dropped."""
    return sessions_from_frame(pd.read_csv(path, dtype={"block_hash_ids": str}), max_tokens, path)


def sessions_from_frame(frame: pd.DataFrame, max_tokens: Optional[int] = None, path: str = "frame") -> _Sessions:
    missing = {"session_id", "step", "num_prefill_tokens", "num_decode_tokens", "block_hash_ids"} - set(frame)
    if missing:
        raise ValueError(f"{path} is not a session trace: missing {sorted(missing)}")
    if max_tokens is not None:
        frame = frame[frame["num_prefill_tokens"] + frame["num_decode_tokens"] <= max_tokens]
    frame = frame.sort_values(["session_id", "step"], kind="stable").reset_index(drop=True)
    starts = np.flatnonzero(np.r_[True, frame["session_id"].to_numpy()[1:] != frame["session_id"].to_numpy()[:-1]])
    bounds = np.r_[starts, len(frame)]
    has_arrivals = "arrived_at" in frame and frame["arrived_at"].notna().all()
    return _Sessions(frame, bounds, bool(has_arrivals))


def sample_session_workload(
    sessions: _Sessions,
    num_requests: int,
    qps: float,
    seed: int = 0,
    session_time_scale: float = 1.0,
    turn_gap: float = 30.0,
    min_turn_gap: float = 1.0,
    phase: str = "stationary",
    closed_loop: bool = True,
    warmup: float = 0.0,
) -> pd.DataFrame:
    """warmup > 0 prepends warmup seconds of calls, flagged in a "warmup"
    column, before the num_requests scored ones: the server starts empty, so
    without it the sessions already running when the window opens all arrive
    with cold prefix caches at once. Consumers simulate warm-up calls but
    leave them out of metrics and rewards."""
    assert num_requests > 0 and qps > 0 and warmup >= 0
    assert phase in ("stationary", "start"), phase
    rng = np.random.default_rng(seed)
    frame = sessions.frame
    bounds = sessions.bounds
    num_sessions = len(bounds) - 1
    # calls are drawn over [0, warmup + num_requests / qps); the first warmup
    # seconds are warm-up
    window = num_requests / qps + warmup
    gaps, offsets = _call_gaps(sessions, rng, turn_gap, min_turn_gap)
    offsets = offsets * session_time_scale
    durations = offsets[bounds[1:] - 1]
    return _sample_window(
        sessions, rng, gaps, offsets, durations, num_requests, window, warmup,
        session_time_scale, phase, closed_loop,
    )


def _call_gaps(sessions: _Sessions, rng, turn_gap: float, min_turn_gap: float):
    """Per call: the gap before it (think time: from the previous call's
    completion when the trace has it, else from its arrival; 0 for a
    session's first call) and its time from its session's first call."""
    frame = sessions.frame
    bounds = sessions.bounds
    # each call's time from its session's first call
    if sessions.has_arrivals:
        arrivals = frame["arrived_at"].to_numpy(dtype=float)
        offsets = arrivals - np.repeat(arrivals[bounds[:-1]], np.diff(bounds))
        # gap before each call: from the previous call's completion when the
        # trace has it, else from its arrival
        gaps = np.r_[0.0, np.diff(arrivals)]
        if "completed_at" in frame:
            completed = frame["completed_at"].to_numpy(dtype=float)
            after_completion = arrivals[1:] - completed[:-1]
            known = ~np.isnan(after_completion)
            gaps[1:][known] = after_completion[known]
        gaps = np.maximum(gaps, 0.0)
    else:
        gaps = np.maximum(rng.exponential(turn_gap, len(frame)), min_turn_gap)
        gaps[bounds[:-1]] = 0.0
        offsets = np.cumsum(gaps)
        offsets -= np.repeat(offsets[bounds[:-1]], np.diff(bounds))
    return gaps, offsets


def _sample_window(
    sessions: _Sessions, rng, gaps, offsets, durations, num_requests, window, warmup,
    session_time_scale, phase, closed_loop,
) -> pd.DataFrame:
    frame = sessions.frame
    bounds = sessions.bounds
    num_sessions = len(bounds) - 1
    # In a stationary server, sessions overlapping the window are drawn in
    # proportion to duration + window (sessions starting in (-duration, W)),
    # which makes the window's calls a uniform sample of the trace's calls.
    # Weighted order without replacement: Gumbel top-k.
    if phase == "stationary":
        keys = np.log(durations + window) + rng.gumbel(size=num_sessions)
        order = np.argsort(-keys, kind="stable")
    else:
        order = rng.permutation(num_sessions)

    picked: List[np.ndarray] = []
    times: List[np.ndarray] = []
    session_ids: List[np.ndarray] = []
    landed = 0
    for new_id, session in enumerate(order):
        lo, hi = bounds[session], bounds[session + 1]
        low = -durations[session] if phase == "stationary" else 0.0
        at = rng.uniform(low, window) + offsets[lo:hi]
        inside = (at >= 0) & (at < window)
        if not inside.any():
            continue
        picked.append(np.arange(lo, hi)[inside])
        times.append(at[inside])
        session_ids.append(np.full(int(inside.sum()), new_id))
        landed += int((inside & (at >= warmup)).sum())
        if landed >= num_requests:
            break
    else:
        raise ValueError(
            f"the trace's {num_sessions} sessions give only {landed} "
            f"calls in a {window:.0f} s window; lower num_requests or raise qps"
        )

    rows = np.concatenate(picked)
    columns = ["step", "num_prefill_tokens", "num_decode_tokens", "block_hash_ids"]
    if "ids_cover_output" in frame:
        columns.append("ids_cover_output")
    out = frame.iloc[rows][columns].reset_index(drop=True)
    ids = np.concatenate(session_ids)
    out.insert(0, "session_id", ids)
    out.insert(0, "arrived_at", np.concatenate(times))
    if closed_loop:
        # a session's calls in the window are consecutive rows of the trace;
        # its first one arrives at its sampled time, each later one think_time
        # after the previous one completes in the simulation
        think = gaps[rows] * session_time_scale
        think[np.r_[True, ids[1:] != ids[:-1]]] = np.nan
        out["think_time"] = think
    out = out.sort_values(["arrived_at", "session_id", "step"], kind="stable")
    if warmup > 0:
        warm = out["arrived_at"].to_numpy() < warmup
        out = pd.concat([out[warm], out[~warm].head(num_requests)])
        out["warmup"] = np.r_[np.ones(int(warm.sum()), dtype=int), np.zeros(len(out) - int(warm.sum()), dtype=int)]
    else:
        out = out.head(num_requests)
    # renumber sessions in order of first arrival
    out["session_id"] = pd.factorize(out["session_id"])[0]
    out["arrived_at"] -= out["arrived_at"].iloc[0]
    return out.reset_index(drop=True)


def sample_concurrent_workload(
    sessions: _Sessions,
    num_calls: int,
    concurrency: int,
    seed: int = 0,
    session_time_scale: float = 1.0,
    turn_gap: float = 30.0,
    min_turn_gap: float = 1.0,
    session_gap: float = 30.0,
    stagger: float = 60.0,
) -> pd.DataFrame:
    """A closed system of `concurrency` agents: each agent ("slot") runs
    whole sessions back to back, closed loop, starting the next session
    Exp(session_gap) seconds after its previous one's last call completes, so
    the number of sessions in flight stays constant and the load is set by
    concurrency rather than by a call rate (the achieved rate is whatever the
    simulated server sustains). Slots start at U(0, stagger) seconds.

    session_id is the slot (Frontier chains a session_id's calls through
    think_time); "trace_session" numbers the sessions. Sessions are drawn in
    random order without replacement and dealt to the slot with the fewest
    calls until every slot has num_calls / concurrency (each slot is cut at
    twice that). Arrival times
    after a slot's first call only order its calls: consumers should
    over-provision num_calls and score by simulated time."""
    assert num_calls > 0 and concurrency > 0
    rng = np.random.default_rng(seed)
    frame = sessions.frame
    bounds = sessions.bounds
    num_sessions = len(bounds) - 1
    gaps, _ = _call_gaps(sessions, rng, turn_gap, min_turn_gap)

    slot_rows: List[List[np.ndarray]] = [[] for _ in range(concurrency)]
    slot_calls = np.zeros(concurrency, dtype=int)
    order = list(rng.permutation(num_sessions))
    placed = 0
    per_slot = -(-num_calls // concurrency)
    while slot_calls.min() < per_slot:
        if not order:
            # reusing a session would share its block hashes across slots
            raise ValueError(f"the trace's {num_sessions} sessions give only {placed} calls; lower num_calls")
        session = order.pop()
        slot = int(np.argmin(slot_calls))
        rows = np.arange(bounds[session], bounds[session + 1])
        slot_rows[slot].append(rows)
        slot_calls[slot] += len(rows)
        placed += len(rows)

    parts = []
    trace_session = 0
    for slot, chunks in enumerate(slot_rows):
        if not chunks:
            continue
        start = rng.uniform(0.0, stagger)
        thinks, numbers = [], []
        for k, rows in enumerate(chunks):
            think = gaps[rows] * session_time_scale
            think[0] = np.nan if k == 0 else rng.exponential(session_gap)
            thinks.append(think)
            numbers.append(np.full(len(rows), trace_session))
            trace_session += 1
        rows = np.concatenate(chunks)[: 2 * per_slot]
        part = frame.iloc[rows][["num_prefill_tokens", "num_decode_tokens", "block_hash_ids"]
                                + (["ids_cover_output"] if "ids_cover_output" in frame else [])]
        part = part.reset_index(drop=True)
        # placeholder times, increasing within the slot: Frontier orders a
        # session's calls by arrival and releases each after its predecessor
        part.insert(0, "arrived_at", start + np.arange(len(rows)) * 1e-3)
        part.insert(1, "session_id", slot)
        part["think_time"] = np.concatenate(thinks)[: len(rows)]
        part["trace_session"] = np.concatenate(numbers)[: len(rows)]
        parts.append(part)
    out = pd.concat(parts).sort_values(["arrived_at", "session_id"], kind="stable")
    out["arrived_at"] -= out["arrived_at"].iloc[0]
    return out.reset_index(drop=True)


def sample_named_workload(
    trace: str,
    num_requests: int,
    qps: float,
    seed: int = 0,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """sample_session_workload from a session trace file, or from synthetic
    sessions generated for this seed when trace is "synthetic:<preset>" or
    "synthetic:random" (frontier/trace_prep/synthetic.py; random also scales
    qps, or concurrency, by its drawn load multiplier)."""
    import dataclasses
    import os

    from frontier.trace_prep import synthetic

    # concurrency=N: sample_concurrent_workload (num_requests calls, qps unused)
    concurrency = kwargs.pop("concurrency", None)
    if concurrency:
        concurrent_kwargs = {
            k: v for k, v in kwargs.items() if k in ("session_time_scale", "turn_gap", "session_gap")
        }
    if not synthetic.is_synthetic(trace):
        sessions = load_session_trace(os.path.abspath(trace), max_tokens)
        if concurrency:
            return sample_concurrent_workload(sessions, num_requests, concurrency, seed, **concurrent_kwargs)
        return sample_session_workload(sessions, num_requests, qps, seed, **kwargs)
    params, load = synthetic.synthetic_params(trace, seed)
    while concurrency:
        # random's load multiplier scales the number of agents
        sessions = sessions_from_frame(synthetic.generate_session_frame(params, seed), max_tokens)
        try:
            return sample_concurrent_workload(
                sessions, num_requests, max(1, int(round(concurrency * load))), seed, **concurrent_kwargs
            )
        except ValueError:
            params = dataclasses.replace(params, num_sessions=params.num_sessions * 2)
    while True:
        sessions = sessions_from_frame(synthetic.generate_session_frame(params, seed), max_tokens)
        try:
            return sample_session_workload(sessions, num_requests, qps * load, seed, **kwargs)
        except ValueError:
            # too few sessions land in the window: generate more
            params = dataclasses.replace(params, num_sessions=params.num_sessions * 2)


def write_session_workload(path: str, workload: pd.DataFrame) -> None:
    import os

    temporary = f"{path}.{os.getpid()}.tmp"
    workload.to_csv(temporary, index=False, float_format="%.6f")
    os.replace(temporary, path)
