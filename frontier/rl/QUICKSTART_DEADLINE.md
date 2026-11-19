# Trying the RL scheduling env with the TTFT deadline

Branch: `agent-sessions-local` (fork `sidharthgd/Frontier`). Python 3.11; same packages as `environment.yml` plus torch, pandas, scikit-learn.

## 1. Data (not in git)

The env replays TraceLab coding-agent sessions. Convert the raw trace once:

    python -m frontier.rl.agent_trace data/raw_traces/syfi_coding_trace.jsonl.gz rl_runs/agent_traces/tracelab_calls.csv

(raw trace: `uw-syfi/TraceLab`, ~100 MB). The A100 + Llama-3.1-8B profile it simulates is in `data/profiling/compute/a100/`.
The first simulation builds the timing models and caches them under `cache/` (slow once, minutes).

## 2. What the deadline is

`env.run_episode(agent, seed, deadline_s=D)`: any call that has waited at least `D` seconds for its first token
goes ahead of every other call (oldest first), whatever the agent chose. In training, `D = deadline_c x` the p99 TTFT that
FCFS gets on the same (load, seed). The policy sees it through the inputs `time_to_deadline_frac`, `overdue`,
`log_uncached_steps` (per call) and `log_deadline_s`, `oldest_wait_over_deadline`, `overdue_fraction` (global);
see `frontier/rl/policy_input.py`. `deadline_s=0` makes any policy behave exactly like FCFS (tested).

## 3. One episode with a deadline

    from frontier.rl.train import TrainConfig, _env, _WORKER
    from frontier.rl.agents import baseline_agent

    config = TrainConfig(output_dir="x")
    _WORKER["config"] = config
    env = _env(config, 12)                      # 12 concurrent sessions (also try 16, 24)
    fcfs = env.run_episode(baseline_agent("fcfs"), 0)
    D = fcfs.summary["p99_ttft"]                # deadline = FCFS's own p99 TTFT (c = 1)
    free = env.run_episode(baseline_agent("sjf_uncached"), 0)
    held = env.run_episode(baseline_agent("sjf_uncached"), 0, deadline_s=D)
    for name, ep in (("sjf_uncached", free), ("sjf_uncached + deadline", held)):
        s = ep.summary
        print(name, "mean session serving %+.1f%%" % (100 * (s["mean_session_serving"] / fcfs.summary["mean_session_serving"] - 1)),
              "p99 TTFT %+.1f%%" % (100 * (s["p99_ttft"] / D - 1)))

One seed is noisy (p99 intervals are tens of points); compare policies over >= 8 seeds, as `agent_compare` does.

## 4. Tests and a training run

    python -m pytest tests/rl                   # 38 tests, ~2 min
    python -m frontier.rl.train --output-dir rl_runs/train/try --set method='"ppo"' imitation_iterations=0 \
        deadline_c=1.0 seed_pool=8 ttft_sq_weight=0.01 ttft_ref_weight=0.0 ppo_iterations=40 num_workers=10

Add `resume=true` to continue a run from `<output-dir>/state.pt`. The reward weights above are NOT tuned yet.
Each PPO iteration simulates ~24 episodes (minutes each on one core), so use as many cores as you can (`num_workers`).

Rule baselines under a deadline: `python -m frontier.rl.agent_compare --help` (policy names like `sjf_uncached_deadline120`).
