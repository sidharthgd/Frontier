"""
Learned request scheduling on Frontier's vLLM v1 scheduler: priority policies
trained by imitation (DAgger on oracle SJF baselines) and PPO. This is the one
RL stack; it was ported from the Vidur repo (branch frontier-backend,
vidur/rl), which is now a frozen reference.

    python -m frontier.rl.imitation --config configs/rl/frontier_concurrency.yaml
    python -m frontier.rl.train --config configs/rl/frontier_concurrency.yaml \\
        --set init_checkpoint=rl_runs/frontier_concurrency/best.pt
    python -m frontier.rl.plot_cdfs --checkpoint <best.pt> \\
        --workloads configs/rl/frontier_concurrency_cdf_workloads.yaml ...
    python -m frontier.rl.baselines --workload '{name: q, qps: 20, num_blocks: 4000}'

scripts/rl/run_frontier_*.sh run whole experiments. The package imports
nothing itself, so frontier.rl.scheduler stays light.

Layers:
- scheduler.py: the RL hook in Frontier's vLLM v1 scheduler (PriorityPolicy).
- engine.py: Frontier simulator setup (WorkloadSpec, presets, shared predictors).
- frontier_env.py: FrontierEnv, the environment: workloads (workload.py),
  features (features.py, observation.py, prefix_cache.py), scoring
  (scoring.py) and episodes; env.py has the records and reference runs.
- reward.py, agents.py / teacher.py, model.py / torch_agent.py, imitation.py,
  ppo.py / train.py, rollout.py, evaluate.py, plot_cdfs.py: training.
"""
