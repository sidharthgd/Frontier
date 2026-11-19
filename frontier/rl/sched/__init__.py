"""
Learned request scheduling on Frontier's vLLM v1 scheduler: priority policies
trained by imitation (DAgger on oracle SJF baselines) and PPO, ported from the
Vidur RL stack (Vidur repo, branch frontier-backend, vidur/rl). Episodes run in
Frontier through frontier/rl/scheduler.py's priority hook (frontier_env.py).

    python -m frontier.rl.sched.imitation --config configs/rl/frontier_concurrency.yaml
    python -m frontier.rl.sched.train --config configs/rl/frontier_concurrency.yaml \
        --set init_checkpoint=rl_runs/frontier_concurrency/best.pt
    python -m frontier.rl.sched.plot_cdfs --checkpoint <best.pt> \
        --workloads configs/rl/frontier_concurrency_cdf_workloads.yaml ...
    python -m frontier.rl.sched.baselines --workload '{name: q, qps: 20, num_blocks: 4000}'

scripts/rl/run_frontier_*.sh run whole experiments. Next to it,
frontier/rl/{env,agents,reward,observation}.py are a separate, lighter RL
environment (FrontierSchedulingEnv) with its own observation and reward; both
share the scheduler hook and the prefix-cache views in frontier/rl.
"""

from frontier.rl.sched.action import admission_action_bounds
from frontier.rl.sched.agents import (
    BaseAgent,
    FCFSAgent,
    OracleAgent,
    OracleSJFAgent,
    RandomAgent,
    baseline_agent,
)
from frontier.rl.sched.env import (
    Episode,
    SchedulingEnv,
    Transition,
    make_simulation_config,
)
from frontier.rl.sched.features import (
    GLOBAL_FEATURES,
    RUNNING_FEATURES,
    WAITING_FEATURES,
    global_feature,
    running_feature,
    waiting_feature,
)
from frontier.rl.sched.observation import Observation, ObservationConfig
from frontier.rl.sched.reward import (
    REWARD_COMPONENTS,
    RewardConfig,
    RewardContext,
    episode_cost,
    episode_cost_components,
    per_request_reward_components,
    per_request_rewards,
    window_thresholds,
)

__all__ = [
    "admission_action_bounds",
    "BaseAgent",
    "FCFSAgent",
    "RandomAgent",
    "OracleAgent",
    "OracleSJFAgent",
    "Episode",
    "SchedulingEnv",
    "Transition",
    "make_simulation_config",
    "GLOBAL_FEATURES",
    "RUNNING_FEATURES",
    "WAITING_FEATURES",
    "global_feature",
    "running_feature",
    "waiting_feature",
    "Observation",
    "ObservationConfig",
    "REWARD_COMPONENTS",
    "RewardConfig",
    "RewardContext",
    "episode_cost",
    "episode_cost_components",
    "window_thresholds",
    "baseline_agent",
    "per_request_reward_components",
    "per_request_rewards",
]
