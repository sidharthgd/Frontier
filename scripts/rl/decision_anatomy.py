"""
What the RL policy's rankings do at each scheduler call, for one checkpoint
on the frontier_concurrency eval workloads:

    python scripts/rl/decision_anatomy.py --config configs/rl/frontier_concurrency_long.yaml \
        --checkpoint rl_runs/.../best.pt --output-dir rl_runs/x/decisions \
        --seeds 1000000 ... --variants base skip_exact skip_sjf --workers 48

Every call of the scheduler hook (frontier/rl/scheduler.py: every non-empty
waiting queue, each iteration) is classified before the policy runs, from
state a vLLM scheduler has at that point:

  blocked  no request can be admitted: no token budget left after the running
           requests, or the running batch is at max_num_seqs
  nofit    no waiting request's first chunk fits the free KV blocks, so
           admission stops at whichever is first
  single   one waiting request, and it fits: there is no order; its priority
           only matters later, for preemption
  allfit   every waiting request fits this iteration in any order (their
           uncached prefill fits the token budget, the running cap has room,
           and the KV pool has their blocks): order changes nothing about this
           batch, only the priorities kept for preemption
  decide   everything else: the ranking can change the batch

blocked and nofit calls admit nothing whatever the ranking, and a ranking that
admits nothing is undone (frontier_env._BridgePolicy.on_admission), so
skipping them is exact. single and allfit calls set priorities that later
decide preemption victims.

Variants: base runs the policy on every call (the evaluated behavior) and
times its parts; skip_exact skips blocked and nofit calls; skip_<fallback>
also skips single and allfit calls, giving their requests a fallback priority
(mid: the middle level; sjf: uncached-prefill rank cut into the levels).
Request histories, which the features read, are updated as if the policy had
run.

Writes decisions.csv (one row per call: category, queue/batch state, what was
admitted, wall time of the cache lookups, features and forward pass),
queue.csv (base only: every waiting request at `decide` calls, with its
level, whether its first chunk fits on its own, and whether it was admitted) and
requests.csv (one row per scored request, as ttft_tail_anatomy.py).
"""

import argparse
import os
import sys
import time
from math import ceil

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from frontier.rl.preload import pool_context  # noqa: E402
from frontier.rl.reward import RewardConfig  # noqa: E402
from frontier.rl.train import load_config  # noqa: E402

_STATE = {}


class _Log:
    def __init__(self, variant):
        self.variant = variant
        self.fallback = variant.split("_", 1)[1] if variant.startswith("skip_") else None
        self.skip = {"base": (), "skip_exact": ("blocked", "nofit")}.get(
            variant, ("blocked", "nofit", "single", "allfit"))
        self.decisions = []
        self.queue = []


def _patch():
    """Instrumented copies of _BridgePolicy.get_priorities / on_admission
    (frontier/rl/frontier_env.py), same behavior plus the log and the skip."""
    from frontier.rl import frontier_env as fe
    from frontier.rl.prefix_cache import prefill_target_tokens, prefix_cache_view

    original_schedule = fe.FrontierFeatureScheduler._schedule_waiting_requests
    original_admission = fe._BridgePolicy.on_admission

    def schedule_waiting(self, token_budget):
        self._rl_budget = int(token_budget)
        return original_schedule(self, token_budget)

    def get_priorities(self, waiting, scheduler):
        log = _STATE["log"]
        self._pending = None
        self._row = None
        self._qrows = None
        if self.score_window is not None:
            for r in waiting:
                self.score_window.saw(r)
        if not waiting:
            return None
        n = len(waiting)
        record = n >= fe.MIN_WAITING_TO_RECORD
        t0 = time.perf_counter()
        cache_states = [prefix_cache_view(scheduler, r) for r in waiting]
        t1 = time.perf_counter()

        budget = scheduler._rl_budget
        running = len(scheduler._running_requests)
        cap_room = scheduler.batch_size_cap - running
        bs = scheduler.block_size
        targets = [prefill_target_tokens(r) for r in waiting]
        uncached = np.array([max(t - s[0], 0) for t, s in zip(targets, cache_states)])
        need = sum(ceil(t / bs) - s[0] // bs for t, s in zip(targets, cache_states))
        idle = sum(s[1] for s in cache_states)
        free = scheduler._kv_cache_manager.block_pool.get_num_free_blocks()
        # each request's first chunk on its own (KVCacheManager.can_allocate_slots:
        # new blocks <= free blocks less the idle blocks its hit takes off the queue)
        chunk = [max(min(u, budget), 1) for u in uncached]
        fits = np.array([
            ceil((s[0] + c) / bs) - s[0] // bs <= free - s[1]
            for s, c in zip(cache_states, chunk)])
        if budget <= 0 or cap_room <= 0:
            category = "blocked"
        elif not fits.any():
            category = "nofit"
        elif n == 1:
            category = "single"
        elif n <= cap_room and uncached.sum() <= budget and need <= free - idle:
            category = "allfit"
        else:
            category = "decide"
        sjf_first = int(np.lexsort(([r.id for r in waiting], [r.arrived_at for r in waiting], uncached))[0])
        t2 = time.perf_counter()
        skipped = category in log.skip
        row = dict(
            time=float(scheduler._current_schedule_time), n=n, running=running,
            budget=budget, cap_room=cap_room, free_blocks=free, need_blocks=need,
            uncached_sum=int(uncached.sum()), category=category, skipped=skipped,
            n_fit=int(fits.sum()), sjf_first_fits=bool(fits[sjf_first]), first_fits=np.nan,
            t_cache=t1 - t0, t_check=t2 - t1, t_obs=np.nan, t_model=np.nan,
        )
        log.decisions.append(row)
        self._row = (row, scheduler, record)
        if skipped:
            if category in ("blocked", "nofit"):
                return None
            levels = self._num_levels
            if log.fallback == "mid":
                return np.full(n, levels // 2, dtype=np.int64)
            order = np.lexsort((
                [r.id for r in waiting], [r.arrived_at for r in waiting], uncached))
            ranks = np.empty(n, dtype=np.int64)
            ranks[order] = np.arange(n)
            return ranks * levels // n

        rank_mask = [True] * n
        view = fe._SchedulerView(scheduler, self._num_levels, self.first_token_at, rank_mask)
        obs = fe.build_observation(
            [fe._view(r, True, self.first_token_at, s) for r, s in zip(waiting, cache_states)],
            view, self._observation_config,
        )
        t3 = time.perf_counter()
        action, info = fe._act(self._agent, obs, waiting, view)
        t4 = time.perf_counter()
        row["t_obs"], row["t_model"] = t3 - t2, t4 - t3
        first = int(np.lexsort(([r.id for r in waiting], [r.arrived_at for r in waiting], action))[0])
        row["first_fits"] = bool(fits[first])
        assert action.shape == (n,)
        assert action.min() >= 0 and action.max() < self._num_levels, action
        self._pending = (
            obs, action, info, [(r.arrived_at, r.id) for r in waiting],
            [(r, r.priority) for r in waiting], scheduler, record,
        )
        if log.variant == "base" and category == "decide":
            now = obs.time
            self._qrows = [
                dict(time=now, req=r.id, level=int(a), uncached=int(u), target=int(t), fits=bool(f),
                     cached=int(s[0]), idle_blocks=int(s[1]), evict_distance=int(s[2]),
                     waited=now - r.arrived_at, restarts=r.get_total_preemption_count(),
                     session=-1 if r.session_id is None else int(r.session_id),
                     decode=int(r.num_decode_tokens), arrived=r.arrived_at)
                for r, a, u, t, s, f in zip(waiting, action, uncached, targets, cache_states, fits)
            ]
        return action

    def on_admission(self, admitted, boundary):
        log = _STATE["log"]
        held = getattr(self, "_row", None)
        self._row = None
        if held is not None:
            row, scheduler, record = held
            row["n_admitted"] = len(admitted)
            row["boundary"] = boundary is not None
            if row["skipped"] and record:
                # the history update the policy's ranking would have made
                self._update_history(scheduler, scheduler._ranked_waiting, admitted)
            if self._qrows:
                ids = {r.id for r in admitted}
                decision = len(log.decisions) - 1
                for q in self._qrows:
                    q["admitted"] = q["req"] in ids
                    q["decision"] = decision
                log.queue.extend(self._qrows)
            self._qrows = None
        original_admission(self, admitted, boundary)

    fe.FrontierFeatureScheduler._schedule_waiting_requests = schedule_waiting
    fe._BridgePolicy.get_priorities = get_priorities
    fe._BridgePolicy.on_admission = on_admission


def _init(config_path, checkpoint):
    import torch

    from frontier.rl.train import load_checkpoint

    torch.set_num_threads(1)
    config = load_config(config_path, [])
    _patch()
    _STATE.update(
        workloads={w.name: w for w in config.eval_workload_configs()},
        observation=config.observation_config(),
        levels=config.num_priority_levels,
        model=load_checkpoint(checkpoint)["model_object"],
        envs={},
    )


def _run(task):
    from frontier.rl.torch_agent import TorchAgent

    workload, seed, variant, sim_dir = task
    s = _STATE
    if workload not in s["envs"]:
        s["envs"][workload] = s["workloads"][workload].make_env(
            RewardConfig(), s["observation"], s["levels"], sim_dir)
    env = s["envs"][workload]
    s["log"] = log = _Log(variant)
    start = time.perf_counter()
    episode = env.run_episode(TorchAgent(s["model"], seed, "argmax"), seed,
                              reward_config=RewardConfig())
    wall = time.perf_counter() - start
    scored = [r for r in episode.requests if r.completed]
    first = min(r.arrived_at for r in episode.requests)
    tag = dict(workload=workload, seed=seed, variant=variant)
    decisions = [dict(tag, scored_window=d["time"] >= first, **d) for d in log.decisions]
    queue = [dict(tag, **q) for q in log.queue if q["time"] >= first]
    requests = [
        dict(tag, ttft=r.prefill_completed_at - r.arrived_at, e2e=r.completed_at - r.arrived_at,
             restarts=r.num_restarts, prompt=r.num_prefill_tokens,
             cached=r.num_cached_prefill_tokens, decode=r.num_decode_tokens,
             session=-1 if r.session_id is None else r.session_id, arrived=r.arrived_at,
             episode_wall_s=wall)
        for r in scored
    ]
    return decisions, queue, requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--variants", nargs="+", default=["base", "skip_exact", "skip_sjf"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config, [])
    paths = {k: os.path.join(args.output_dir, f"{k}.csv") for k in ("decisions", "queue", "requests")}
    done = set()
    if os.path.exists(paths["requests"]):
        old = pd.read_csv(paths["requests"], usecols=["workload", "seed", "variant"])
        done = set(map(tuple, old.drop_duplicates().itertuples(index=False)))
    sim_dir = os.path.join(args.output_dir, "sim")
    tasks = [
        (w.name, s, v, sim_dir)
        for v in args.variants for w in config.eval_workload_configs() for s in args.seeds
        if (w.name, s, v) not in done
    ]
    if not tasks:
        return
    ctx = pool_context(config.eval_workload_configs(), os.path.join(args.output_dir, "preload"))
    with ctx.Pool(min(args.workers, len(tasks)), _init, (args.config, args.checkpoint)) as pool:
        for i, result in enumerate(pool.imap_unordered(_run, tasks)):
            # requests last: a task counts as done once its requests are written
            for key, rows in zip(("decisions", "queue", "requests"), result):
                if rows:
                    header = not os.path.exists(paths[key])
                    pd.DataFrame(rows).to_csv(paths[key], mode="a", header=header, index=False)
            print(f"done {i + 1}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
