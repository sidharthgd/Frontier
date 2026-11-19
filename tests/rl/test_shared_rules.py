"""
The rules both RL stacks share (frontier/rl: FrontierSchedulingEnv, and
frontier/rl/sched: FrontierEnv), and that they give identical episodes:

- SJF job size (frontier/rl/observation.py job_size_tokens), including a
  victim preempted after prefill, which recomputes its prompt and kept output;
- a lone waiting request is ranked but is no decision
  (frontier/rl/scheduler.py MIN_WAITING_TO_RECORD);
- closed-loop agent workloads score each agent's quota after the warm-up
  (frontier/rl/scoring.py) and replay byte-identical workload files.

Run from the repo root:
    python -m pytest tests/rl/test_shared_rules.py -v
"""

import tempfile
from types import SimpleNamespace

import pytest

from frontier.rl.observation import job_size_tokens, prefill_target_tokens, prefill_work_tokens
from frontier.rl.scheduler import MIN_WAITING_TO_RECORD
from frontier.rl.sched.frontier_env import _OracleRequest
from frontier.rl.sched.history import job_size


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
def test_both_stacks_size_jobs_the_same(request_, cached, work, total):
    assert prefill_work_tokens(request_, cached) == work
    assert job_size_tokens(request_, cached, with_output=False) == work
    assert job_size_tokens(request_, cached, with_output=True) == total
    # the training stack's SJF baselines (sjf_prefill_tokens, sjf_total_tokens)
    oracle = _OracleRequest(request_, cached)
    assert oracle.num_prefill_tokens == prefill_target_tokens(request_)
    assert job_size(oracle, "prefill_tokens", 2048) == work
    assert job_size(oracle, "total_tokens", 2048) == total


# --- episodes ------------------------------------------------------------------

SEED = 3
N, AGENTS, WARMUP = 48, 16, 300


def _envs():
    from frontier.rl.env import FrontierSchedulingEnv, a100_llama31_8b
    from frontier.rl.sched.observation import ObservationConfig
    from frontier.rl.sched.reward import RewardConfig
    from frontier.rl.sched.rollout import WorkloadConfig
    from frontier.rl.sched.workload import _session_trace_file

    mine = WorkloadConfig(
        name="tracelab_c16", trace="tracelab", arrivals="sessions", num_requests=N,
        num_blocks=28318, gap_metric=False,
        simulation={"max_tokens": 131072, "session_concurrency": AGENTS, "session_warmup": WARMUP},
        frontier={"preset": "a100_llama31_8b", "prediction_rounding_bits": 5,
                  "enable_prefix_caching": True},
    ).make_env(RewardConfig(), ObservationConfig(prefix_cache=True), 16, tempfile.mkdtemp())
    sids = FrontierSchedulingEnv(
        a100_llama31_8b(
            name="tracelab_c16", arrivals="sessions", session_trace=_session_trace_file("tracelab"),
            num_requests=N, session_concurrency=AGENTS, session_warmup=WARMUP,
            prediction_rounding_bits=5, enable_prefix_caching=True,
            # the scheduler settings FrontierEnv passes (vLLM V1: no watermark)
            extra_args=["--vllm_v1_scheduler_config_block_size", "16",
                        "--vllm_v1_scheduler_config_watermark_blocks_fraction", "0.0"],
        ),
        output_dir=tempfile.mkdtemp(),
    )
    return mine, sids


@pytest.fixture(scope="module")
def envs():
    import os

    from frontier.rl.sched.workload import _session_trace_file

    if not os.path.exists(_session_trace_file("tracelab")):
        pytest.skip("tracelab session trace not built (frontier/trace_prep)")
    return _envs()


def test_both_stacks_replay_the_same_workload(envs):
    from frontier.rl.env import session_workload_trace

    mine, sids = envs
    with open(mine.workload_trace(SEED), "rb") as a, open(
        session_workload_trace(sids.spec, SEED, sids._output_dir), "rb"
    ) as b:
        assert a.read() == b.read()


def test_both_stacks_give_identical_episodes(envs):
    from frontier.rl.agents import baseline_agent as sids_agent
    from frontier.rl.sched.agents import baseline_agent as my_agent

    mine, sids = envs
    episode = mine.run_episode(my_agent("sjf_prefill_tokens_age200"), workload_seed=SEED)
    reference = sids.run_episode(sids_agent("sjf_prefill_age200"), seed=SEED)
    # the scored calls: each agent's quota after the warm-up
    assert len(episode.requests) == len(reference.requests) == N
    assert {r.id: (r.prefill_completed_at, r.completed_at) for r in episode.requests} == {
        r.id: (r.first_token_at, r.completed_at) for r in reference.requests
    }
    assert len(episode.transitions) == len(reference.transitions) > 0
    for key in ("mean_ttft", "p99_ttft", "mean_e2e", "p99_e2e"):
        assert episode.summary[key] == reference.summary[key], key
    # decisions are rankings of at least MIN_WAITING_TO_RECORD requests
    assert min(t.obs.num_requests for t in episode.transitions) >= MIN_WAITING_TO_RECORD
    assert min(t.obs.num_requests for t in reference.transitions) >= MIN_WAITING_TO_RECORD


def test_lone_requests_are_ranked(envs):
    from frontier.rl.agents import FCFSAgent as SidsFCFS
    from frontier.rl.sched.agents import FCFSAgent as MyFCFS

    mine, sids = envs
    for env, agent, run in (
        (mine, MyFCFS(), lambda e, a: e.run_episode(a, workload_seed=SEED)),
        (sids, SidsFCFS(), lambda e, a: e.run_episode(a, seed=SEED)),
    ):
        sizes = []
        act = agent.act

        def counting(obs, act=act):
            sizes.append(obs.num_requests)
            return act(obs)

        agent.act = counting
        episode = run(env, agent)
        assert 1 in sizes  # the policy was asked about a lone request
        assert all(t.obs.num_requests >= MIN_WAITING_TO_RECORD for t in episode.transitions)
