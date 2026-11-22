"""
Worker pools that share one copy of the execution-time predictors.

Each spawned worker builds its own predictors, and for the a100_llama31_8b
preset their random forests take ~7 GB per process, so RAM, not cores, caps
the worker count. pool_context() returns a forkserver context whose server
imports this module with the workloads in FRONTIER_RL_PRELOAD_WORKLOADS: it
builds their predictors once (frontier.rl.engine keeps them per process and
engine setting), freezes the heap out of the cyclic GC, and every worker
forks from it, sharing the forests copy-on-write. The forkserver never runs
torch ops, so forking it is safe; workers still call torch.set_num_threads(1).

The forkserver is per parent process and starts with its first pool, so all
pools of one process share the workloads of the first pool_context() call.
"""

import gc
import multiprocessing as mp
import os
import pickle
import tempfile
from typing import Sequence

ENV_VAR = "FRONTIER_RL_PRELOAD_WORKLOADS"


def build_predictors(workloads: Sequence, output_dir: str) -> None:
    from frontier.rl.observation import ObservationConfig
    from frontier.rl.reward import RewardConfig

    for workload in workloads:
        env = workload.make_env(RewardConfig(), ObservationConfig(), 16, output_dir)
        env.build_predictors()


def pool_context(workloads: Sequence, output_dir: str):
    """A multiprocessing context for pools that run episodes of `workloads`."""
    os.makedirs(output_dir, exist_ok=True)
    if ENV_VAR not in os.environ:
        fd, path = tempfile.mkstemp(prefix="preload_", suffix=".pkl", dir=output_dir)
        with os.fdopen(fd, "wb") as f:
            pickle.dump((list(workloads), output_dir), f)
        os.environ[ENV_VAR] = path
    context = mp.get_context("forkserver")
    context.set_forkserver_preload([__name__])
    return context


def _preload() -> None:
    path = os.environ.get(ENV_VAR)
    if not path or mp.current_process().name != "MainProcess":
        return
    with open(path, "rb") as f:
        workloads, output_dir = pickle.load(f)
    build_predictors(workloads, output_dir)
    gc.collect()
    gc.freeze()


# the forkserver imports this module; a parent importing it has no ENV_VAR yet
if os.environ.get(ENV_VAR) and os.environ.get("FRONTIER_RL_PRELOAD_DONE") != "1":
    os.environ["FRONTIER_RL_PRELOAD_DONE"] = "1"
    _preload()
