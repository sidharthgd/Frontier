"""
Run a policy trained in our Vidur fork (vidur/rl_runs/*/best.pt) inside
Frontier, to see whether it transfers to new workloads:

    python -m frontier.rl.vidur_transfer \\
        --checkpoint ../vidur/rl_runs/imitation_sjf_age400_ppo/best.pt \\
        --output-dir rl_runs/vidur_transfer

frontier/rl/model.py is a copy of Vidur's network, so the weights load as
they are. VidurCheckpointAgent rebuilds the inputs Vidur's default
ObservationConfig gave it (vidur/rl/features.py), in the same order and with
the same transforms: 14 waiting features (with each request's decision
history), 7 running features and 39 global ones (counts, free memory, and
mean/min/max of queue length, free memory, batch tokens and batch time over
the last 10/100/1000 scheduling steps). It decides with rank decoding, as
Vidur evaluated it, in Vidur's every_batch mode (which is what Frontier's hook
does: rank the whole queue at every scheduling step).
"""

import argparse
import os
from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional

import numpy as np
import torch

from frontier.rl.env import OracleAgent
from frontier.rl.model import ModelConfig, SchedulerModel, collate, full_actions, modeled_indices
from frontier.rl.policy_input import PolicyInput

WINDOWS = (10, 100, 1000)
WINDOW_METRICS = ("queue_length", "free_block_fraction", "batch_num_tokens", "batch_time")
# (transform, scale) per metric, as vidur/rl/features.py registers them
WINDOW_TRANSFORMS = {
    "queue_length": ("log1p", 1.0),
    "free_block_fraction": ("none", 1.0),
    "batch_num_tokens": ("log1p", 1.0),
    "batch_time": ("log1p", 100.0),
}


@dataclass
class _History:
    num_decisions_seen: int = 0
    times_passed_over: int = 0
    num_overtaken_by: int = 0
    last_rank_percentile: Optional[float] = None


def _percentiles(values) -> np.ndarray:
    n = len(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / max(1, n - 1)


class VidurCheckpointAgent(OracleAgent):
    """Reads scheduler state (not output lengths) the way Vidur's env did."""

    def __init__(self, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = SchedulerModel(ModelConfig(**state["model_config"])).eval()
        self.model.load_state_dict(state["model"])
        self._history: Dict[int, _History] = {}
        self._windows = {m: deque(maxlen=max(WINDOWS)) for m in WINDOW_METRICS}

    # -- state ------------------------------------------------------------

    def _record(self, metric: str, value: float) -> None:
        self._windows[metric].append(float(value))

    def on_batch_end(self, batch) -> None:
        self._record("batch_num_tokens", batch.total_num_tokens)
        self._record("batch_time", batch.completed_at - batch.scheduled_at)

    def on_admission(self, waiting, priorities, admitted) -> None:
        order = sorted(range(len(waiting)), key=lambda i: (priorities[i], waiting[i].arrived_at, waiting[i].id))
        admitted_ids = {r.id for r in admitted}
        admitted_arrivals = [r.arrived_at for r in admitted]
        for position, i in enumerate(order):
            request = waiting[i]
            history = self._history.setdefault(request.id, _History())
            history.num_decisions_seen += 1
            if admitted:
                history.last_rank_percentile = position / max(1, len(order) - 1)
            if request.id in admitted_ids or not admitted:
                continue
            history.times_passed_over += 1
            history.num_overtaken_by += sum(a > request.arrived_at for a in admitted_arrivals)

    def _window_features(self) -> List[float]:
        out = []
        for metric in WINDOW_METRICS:
            transform, scale = WINDOW_TRANSFORMS[metric]
            values = list(self._windows[metric])
            for size in WINDOWS:
                recent = values[-size:]
                stats = (np.mean(recent), np.min(recent), np.max(recent)) if recent else (0.0, 0.0, 0.0)
                for v in stats:
                    v = float(v) * scale
                    out.append(np.log1p(v) if transform == "log1p" else v)
        return out

    # -- acting ------------------------------------------------------------

    def act_oracle(self, obs, waiting, scheduler):
        self._record("queue_length", len(waiting))
        self._record("free_block_fraction", scheduler.num_free_blocks / scheduler.num_blocks)
        block_size = scheduler.block_size
        free = max(1, scheduler.num_free_blocks)
        histories = [self._history.get(r.id, _History()) for r in waiting]
        blocks = np.array([ceil(r.num_prefill_tokens / block_size) for r in waiting], float)
        w = np.stack([
            np.log1p([r.num_prefill_tokens for r in waiting]),
            np.log1p([obs.time - r.arrived_at for r in waiting]),
            [r.get_total_preemption_count() for r in waiting],  # num_restarts
            np.log1p(blocks),
            np.log1p(blocks / free),
            _percentiles([r.arrived_at for r in waiting]),
            [h.num_decisions_seen == 0 for h in histories],
            np.log1p([h.times_passed_over for h in histories]),
            np.log1p([h.num_overtaken_by for h in histories]),
            [h.last_rank_percentile or 0.0 for h in histories],
            [h.last_rank_percentile is not None for h in histories],
            np.zeros(len(waiting)),  # current_priority: unset in every_batch mode
            np.zeros(len(waiting)),  # has_priority
            np.ones(len(waiting)),  # to_rank
        ], axis=1).astype(np.float32)

        running = scheduler.running_requests
        allocation = scheduler._allocation_map
        r = np.stack([
            np.log1p([q.num_prefill_tokens for q in running]),
            np.log1p([q.num_processed_decode_tokens for q in running]),
            [min(q.num_processed_tokens, q.num_prefill_tokens) / q.num_prefill_tokens for q in running],
            [float(q.is_prefill_complete) for q in running],
            np.log1p([allocation.get(q.id, 0) for q in running]),
            np.log1p([obs.time - q.arrived_at for q in running]),
            np.log1p([obs.time - q.prefill_completed_at if q.is_prefill_complete else 0.0 for q in running]),
        ], axis=1).astype(np.float32) if running else np.zeros((0, 7), np.float32)

        g = np.array([
            np.log1p(len(waiting)),
            np.log1p(len(running)),
            scheduler.num_free_blocks / scheduler.num_blocks,
            *self._window_features(),
        ], np.float32)

        pi = PolicyInput(obs.num_priority_levels, obs.request_ids, obs.arrival_times, w, r, g)
        modeled = modeled_indices(pi, self.model.config.max_queue_len)
        with torch.no_grad():
            dist = self.model(collate([pi], self.model.config, [modeled]), with_value=False).dist
            expected = (dist.level_log_probs()[0].exp() * torch.arange(dist.num_levels)).sum(-1)
        keys = expected[: len(modeled)].numpy()
        order = np.lexsort((pi.request_ids[modeled], pi.arrival_times[modeled], keys))
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(len(order))
        levels = ranks * dist.num_levels // max(1, len(order))
        return full_actions(pi, modeled, levels), {}


def run_task(args):
    checkpoint, num_sessions, seed, policies, calls_per_session = args
    from frontier.rl.agent_trace import SessionWorkload
    from frontier.rl.agents import baseline_agent
    from frontier.rl.env import FrontierSchedulingEnv, a100_llama31_8b

    workload = SessionWorkload(
        "rl_runs/agent_traces/tracelab_calls.csv", num_sessions=num_sessions,
        calls_per_session=calls_per_session, session_rate=num_sessions / 60.0,
    )
    env = FrontierSchedulingEnv(a100_llama31_8b(
        name=f"agents_{num_sessions}", sessions=workload,
        enable_prefix_caching=True, prediction_rounding_bits=5,
    ))
    rows = []
    for policy in policies:
        agent = VidurCheckpointAgent(checkpoint) if policy == "vidur_policy" else baseline_agent(policy, seed)
        summary = env.run_episode(agent, seed).summary
        rows.append({"workload": f"sessions{num_sessions}", "seed": seed, "policy": policy, **summary})
    print(f"done sessions={num_sessions} seed={seed}", flush=True)
    return rows


def main() -> None:
    import multiprocessing as mp

    import pandas as pd

    from frontier.rl.agent_compare import summarize

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loads", type=int, nargs="*", default=[12, 16, 24])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--policies", nargs="*",
                        default=["fcfs", "vidur_policy", "sjf_prefill_age400", "sjf_total_age400", "sjf_uncached"])
    parser.add_argument("--calls-per-session", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = os.path.abspath(args.checkpoint)
    tasks = [(checkpoint, n, s, args.policies, args.calls_per_session) for n in args.loads for s in args.seeds]
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        rows = [r for task_rows in pool.imap_unordered(run_task, tasks) for r in task_rows]
    order = {f"sessions{n}": i for i, n in enumerate(args.loads)}
    table = pd.DataFrame(rows).sort_values(
        ["workload", "seed"], key=lambda c: c.map(order) if c.name == "workload" else c
    )
    table.to_csv(os.path.join(args.output_dir, "episodes.csv"), index=False)
    summary = summarize(table)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
