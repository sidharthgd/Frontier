"""
Scheduling rules of the RL environment (frontier/rl/frontier_env.py):

- SJF job size (history.job_size via frontier_env._OracleRequest), including a
  victim preempted after prefill, which recomputes its prompt and kept output;
- a lone waiting request is ranked but is no decision
  (frontier/rl/scheduler.py MIN_WAITING_TO_RECORD);
- closed-loop agent workloads score each agent's quota after the warm-up
  (frontier/rl/scoring.py).

Run from the repo root:
    python -m pytest tests/rl/test_scheduling_rules.py -v
"""

import tempfile
from types import SimpleNamespace

import pytest

from frontier.rl.frontier_env import _OracleRequest
from frontier.rl.history import job_size
from frontier.rl.prefix_cache import prefill_target_tokens
from frontier.rl.scheduler import MIN_WAITING_TO_RECORD


def _request(prompt, processed, recomputing, remaining_decode):
    return SimpleNamespace(
        num_prefill_tokens=prompt,
        num_processed_tokens=processed,
        is_recomputing=recomputing,
        remaining_decode_tokens=remaining_decode,
    )


@pytest.mark.parametrize(
    "request_, cached, work, total",
    [
        # never admitted: the prompt less its cache hit
        (_request(10_000, 0, False, 500), 6_000, 4_000, 4_500),
        # preempted during prefill: restarts its prompt
        (_request(10_000, 0, False, 500), 0, 10_000, 10_500),
        # preempted after prefill with 300 of 800 output tokens: recomputes
        # prompt + kept output, less what the cache still holds of them
        (_request(10_000, 10_300, True, 500), 9_000, 1_300, 1_800),
        # its whole recompute span cached but the last block
        (_request(10_000, 10_300, True, 500), 10_288, 12, 512),
    ],
)
def test_sjf_job_sizes(request_, cached, work, total):
    oracle = _OracleRequest(request_, cached)
    assert oracle.num_prefill_tokens == prefill_target_tokens(request_)
    assert job_size(oracle, "prefill_tokens", 2048) == work
    assert job_size(oracle, "total_tokens", 2048) == total


# --- episodes ------------------------------------------------------------------

SEED = 3
N, AGENTS, WARMUP = 48, 16, 300


@pytest.fixture(scope="module")
def env():
    import os

    from frontier.rl.observation import ObservationConfig
    from frontier.rl.reward import RewardConfig
    from frontier.rl.rollout import WorkloadConfig
    from frontier.rl.workload import _session_trace_file

    if not os.path.exists(_session_trace_file("tracelab")):
        pytest.skip("tracelab session trace not built (frontier/trace_prep)")
    return WorkloadConfig(
        name="tracelab_c16", trace="tracelab", arrivals="sessions", num_requests=N,
        num_blocks=28318, gap_metric=False,
        simulation={"max_tokens": 131072, "session_concurrency": AGENTS, "session_warmup": WARMUP},
        frontier={"preset": "a100_llama31_8b", "prediction_rounding_bits": 5,
                  "enable_prefix_caching": True},
    ).make_env(RewardConfig(), ObservationConfig(prefix_cache=True), 16, tempfile.mkdtemp())


def test_closed_loop_episodes_score_each_agents_quota(env):
    import pandas as pd

    from frontier.rl.agents import baseline_agent

    episode = env.run_episode(baseline_agent("sjf_prefill_tokens_age200"), workload_seed=SEED)
    # the scored calls: N, spread evenly over the agents, none in the warm-up
    assert len(episode.requests) == N
    trace = pd.read_csv(env.workload_trace(SEED))
    scored = trace.iloc[sorted(r.id for r in episode.requests)]
    assert (scored["warmup"] == 0).all()
    assert scored["session_id"].value_counts().max() <= -(-N // AGENTS)
    assert len(episode.transitions) > 0
    # decisions are rankings of at least MIN_WAITING_TO_RECORD requests
    assert min(t.obs.num_requests for t in episode.transitions) >= MIN_WAITING_TO_RECORD


def test_lone_requests_are_ranked(env):
    from frontier.rl.agents import FCFSAgent

    agent = FCFSAgent()
    sizes = []
    act = agent.act

    def counting(obs, act=act):
        sizes.append(obs.num_requests)
        return act(obs)

    agent.act = counting
    episode = env.run_episode(agent, workload_seed=SEED)
    assert 1 in sizes  # the policy was asked about a lone request
    assert all(t.obs.num_requests >= MIN_WAITING_TO_RECORD for t in episode.transitions)
