"""
RL environment on Frontier (frontier/rl). Uses dummy execution times, so it
needs no profiling data or trained predictors.

    python -m pytest tests/rl
"""

import numpy as np
import pytest

from frontier.rl.agents import baseline_agent
from frontier.rl.env import BaseAgent, FrontierSchedulingEnv, OracleAgent, WorkloadSpec
from frontier.rl.observation import build_observation
from frontier.rl.reward import RewardConfig

# tight KV memory so admission blocks and running requests get preempted
SPEC = WorkloadSpec(
    name="tiny", num_requests=120, qps=30, prefill_tokens=600, decode_tokens=300,
    dummy_exec_ms=0.05, num_blocks=400,
)
SEEDS = (0, 1)


@pytest.fixture(scope="module")
def env():
    return FrontierSchedulingEnv(SPEC, RewardConfig(weights={"e2e": 1.0}))


def _timings(requests):
    # ids come from a global counter, so they differ between simulations
    return sorted(
        (r.arrived_at, r.prefill_completed_at, r.completed_at, r.get_total_preemption_count())
        for r in requests
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_equal_priorities_match_stock_frontier(env, seed):
    stock = env.run_stock(seed)
    env.run_episode(baseline_agent("fcfs"), seed)
    rl = env.last_simulator
    assert _timings(rl._all_requests) == _timings(stock._all_requests)
    # the workload actually exercises preemption
    assert sum(r.get_total_preemption_count() for r in stock._all_requests) > 0


def test_seeds_give_different_workloads():
    spec = WorkloadSpec(
        name="seeded", num_requests=30, qps=10,
        length_trace="data/processed_traces/splitwise_conv.csv", dummy_exec_ms=0.05,
    )
    env = FrontierSchedulingEnv(spec)
    a = env.run_episode(baseline_agent("fcfs"), 0).requests
    b = env.run_episode(baseline_agent("fcfs"), 1).requests
    a2 = env.run_episode(baseline_agent("fcfs"), 0).requests
    lengths = lambda rs: [(r.arrived_at, r.num_prefill_tokens, r.num_decode_tokens) for r in rs]
    assert lengths(a) == lengths(a2)
    assert lengths(a) != lengths(b)


def test_ranking_changes_the_schedule(env):
    fcfs = env.run_episode(baseline_agent("fcfs"), 0).summary
    sjf = env.run_episode(baseline_agent("sjf_prefill"), 0).summary
    rnd = env.run_episode(baseline_agent("random", seed=3), 0).summary
    assert fcfs["mean_e2e"] != rnd["mean_e2e"] or fcfs["mean_e2e"] != sjf["mean_e2e"]


def test_only_effective_decisions_are_kept(env):
    episode = env.run_episode(baseline_agent("random", seed=0), 0)
    assert episode.transitions
    assert all(t.admitted_ids for t in episode.transitions)
    assert episode.num_dropped_decisions > 0
    times = [t.time for t in episode.transitions]
    assert times == sorted(times)


def test_reward_components_sum_to_request_totals(env):
    episode = env.run_episode(baseline_agent("random", seed=1), 0)
    start = episode.transitions[0].time
    expected = sum(
        max(0.0, r.completed_at - max(r.arrived_at, start)) for r in episode.requests
    )
    got = sum(t.reward_components["e2e"] for t in episode.transitions)
    assert got == pytest.approx(expected, rel=1e-9)
    assert episode.total_reward == pytest.approx(-got, rel=1e-9)


class _OutputLengthGuard:
    def __init__(self, request):
        object.__setattr__(self, "_r", request)

    def __getattr__(self, name):
        assert "decode_tokens" not in name and "total_tokens" not in name, name
        return getattr(object.__getattribute__(self, "_r"), name)


def test_observation_never_reads_output_length(env):
    class Checking(OracleAgent):
        checked = 0

        def act_oracle(self, obs, waiting, scheduler):
            guarded = [_OutputLengthGuard(r) for r in waiting]
            build_observation(obs.time, guarded, scheduler, obs.num_priority_levels)
            Checking.checked += 1
            return np.zeros(len(waiting), dtype=np.int64), {}

    env.run_episode(Checking(), 0)
    assert Checking.checked > 0


def test_learned_agents_are_held_to_the_priority_levels(env):
    class TooHigh(BaseAgent):
        def act(self, obs):
            return np.full(obs.num_requests, obs.num_priority_levels), {}

    with pytest.raises(AssertionError):
        env.run_episode(TooHigh(), 0)


def _session_replay(tmp_path, think_time):
    """6 sessions of 4 calls, all stamped at t=0 (open loop: a pile-up)."""
    import pandas as pd

    rows = []
    for session in range(6):
        for step in range(4):
            rows.append({
                "arrived_at": 0.0, "session_id": session, "step": step,
                "num_prefill_tokens": 400, "num_decode_tokens": 50,
                "block_hash_ids": f"{session * 100}-{session * 100 + 24}",
                "think_time": think_time if step else None,
            })
    path = tmp_path / "sessions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _session_records(env):
    episode = env.run_episode(baseline_agent("fcfs"), seed=0)
    by_session = {}
    for record in episode.requests:
        by_session.setdefault(record.session_id, []).append(record)
    return episode, [sorted(calls, key=lambda r: r.arrived_at) for calls in by_session.values()]


def test_closed_loop_sessions_wait_for_their_previous_call(tmp_path):
    spec = WorkloadSpec(name="sessions", arrivals="replay", replay_trace=_session_replay(tmp_path, 0.5),
                        dummy_exec_ms=0.05, num_blocks=400)
    episode, sessions = _session_records(FrontierSchedulingEnv(spec))
    assert len(episode.requests) == 24 and all(r.completed_at is not None for r in episode.requests)
    for calls in sessions:
        assert calls[0].arrived_at == 0.0
        for previous, call in zip(calls, calls[1:]):
            # released exactly think_time after the previous call completes
            assert call.arrived_at == pytest.approx(previous.completed_at + 0.5)
    # different sessions are independent: their first calls arrive together
    assert sum(calls[0].arrived_at == 0.0 for calls in sessions) == 6


def test_open_loop_rows_without_think_time_keep_trace_arrivals(tmp_path):
    spec = WorkloadSpec(name="sessions", arrivals="replay", replay_trace=_session_replay(tmp_path, None),
                        dummy_exec_ms=0.05, num_blocks=400)
    episode, _ = _session_records(FrontierSchedulingEnv(spec))
    assert all(r.arrived_at == 0.0 for r in episode.requests)


def test_session_workloads_replay_closed_loop_with_prefix_caching(tmp_path):
    import pandas as pd

    trace = []
    for session in range(40):
        for step in range(6):
            trace.append({
                "arrived_at": 1000.0 * session + 3.0 * step, "completed_at": 1000.0 * session + 3.0 * step + 1.0,
                "session_id": session, "step": step, "num_prefill_tokens": 16 * (20 + 10 * step),
                "num_decode_tokens": 20,
                "block_hash_ids": f"{session * 1000}-{session * 1000 + 19 + 10 * step}",
            })
    path = tmp_path / "trace.csv"
    pd.DataFrame(trace).to_csv(path, index=False)
    spec = WorkloadSpec(name="sessions", arrivals="sessions", session_trace=str(path), num_requests=60,
                        qps=2.0, dummy_exec_ms=0.05, num_blocks=2000, enable_prefix_caching=True)
    env = FrontierSchedulingEnv(spec, output_dir=str(tmp_path / "out"))
    episode, sessions = _session_records(env)
    assert len(episode.requests) == 60
    for calls in sessions:
        for previous, call in zip(calls, calls[1:]):
            assert call.arrived_at == pytest.approx(previous.completed_at + 2.0)  # 3 s apart, 1 s long
    # later calls reuse the session's earlier prompt blocks
    assert sum(r.num_cached_prefill_tokens for r in episode.requests) > 0


@pytest.mark.parametrize("cover_output", [False, True])
def test_output_blocks_are_cached_when_trace_ids_cover_them(tmp_path, cover_output):
    import pandas as pd

    # call 1's prompt is call 0's prompt (20 blocks) + its output (64 tokens,
    # 4 blocks) + 2 new blocks; call 0's ids name its output's blocks too
    rows = [
        {"arrived_at": 0.0, "session_id": 0, "step": 0, "num_prefill_tokens": 320,
         "num_decode_tokens": 64, "block_hash_ids": "0-23", "think_time": None},
        {"arrived_at": 0.0, "session_id": 0, "step": 1, "num_prefill_tokens": 416,
         "num_decode_tokens": 16, "block_hash_ids": "0-26", "think_time": 0.5},
    ]
    frame = pd.DataFrame(rows)
    if cover_output:
        frame["ids_cover_output"] = 1
    path = tmp_path / "trace.csv"
    frame.to_csv(path, index=False)
    spec = WorkloadSpec(name="out", arrivals="replay", replay_trace=str(path), dummy_exec_ms=0.05,
                        num_blocks=400, enable_prefix_caching=True)
    episode, sessions = _session_records(FrontierSchedulingEnv(spec, output_dir=str(tmp_path / "o")))
    second = sessions[0][1]
    # vLLM caches the output's full blocks: the next call hits prompt + output,
    # less the block of the last output token, which is sampled but never
    # written to the cache
    assert second.num_cached_prefill_tokens == ((320 + 64 - 1) // 16 * 16 if cover_output else 20 * 16)


def test_restarts_can_hit_their_own_cached_output(tmp_path):
    import pandas as pd

    # a KV cache too small for every call's prompt + output: decode victims
    # restart and look up their prompt and generated blocks
    rows = []
    for session in range(8):
        for step in range(3):
            prompt = 400 + 300 * step
            rows.append({
                "arrived_at": 0.0, "session_id": session, "step": step,
                "num_prefill_tokens": prompt, "num_decode_tokens": 300,
                "block_hash_ids": f"{session * 1000}-{session * 1000 + (prompt + 300) // 16 - 1}",
                "think_time": 0.1 if step else None, "ids_cover_output": 1,
            })
    path = tmp_path / "trace.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    spec = WorkloadSpec(name="restarts", arrivals="replay", replay_trace=str(path), dummy_exec_ms=0.05,
                        num_blocks=160, enable_prefix_caching=True)
    episode, _ = _session_records(FrontierSchedulingEnv(spec, output_dir=str(tmp_path / "o")))
    assert all(r.completed_at is not None for r in episode.requests)
    assert sum(r.num_preemptions for r in episode.requests) > 0
