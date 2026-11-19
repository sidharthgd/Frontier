"""
Which calls an episode scores on a closed-loop agent workload
(session_concurrency: N agents running sessions back to back), shared by
FrontierSchedulingEnv (env.py) and the training stack's FrontierEnv
(sched/frontier_env.py):

- warm-up: calls arriving before warmup_s are simulated, so the server's
  queue and prefix cache are warm, but not scored;
- per-agent quota: each agent's (session_id's) first per_agent calls after
  the warm-up are scored, a fixed share per agent, so a policy that starves
  an agent cannot shrink that agent's share of the score (starved agents send
  fewer calls);
- stop: the simulation ends once every scored call has finished, so the
  drain after the load ends (few requests, fast) is not scored either.

Workloads over-provision calls (CONCURRENT_OVERPROVISION per scored call) so
the quota is always reachable; everything outside the scored set is warm-up
or spill-over and carries no reward.
"""

import warnings
from typing import Any, Dict, Optional, Set

# calls generated per scored call (warm-up included); the simulation stops
# once the scored ones finish, so extras are cheap
CONCURRENT_OVERPROVISION = 4


class ScoreWindow:
    def __init__(self, simulator, warmup_s: float, per_agent: int, num_agents: int) -> None:
        self._simulator = simulator
        self.warmup_s = float(warmup_s)
        self.per_agent = int(per_agent)
        self.num_agents = int(num_agents)
        self._arrived: Dict[int, Any] = {}
        # set once every scored call has finished (the simulator then stops)
        self.scored_ids: Optional[Set[int]] = None

    @classmethod
    def for_workload(cls, simulator, warmup_s: float, num_requests: int, num_agents: int):
        """num_requests scored calls over num_agents agents (rounded up per agent)."""
        return cls(simulator, warmup_s, -(-int(num_requests) // int(num_agents)), num_agents)

    @property
    def num_scored(self) -> int:
        return self.per_agent * self.num_agents

    def saw(self, request) -> None:
        """Record a request the scheduler has seen (waiting or in a batch)."""
        self._arrived.setdefault(request.id, request)

    def window_ids(self) -> Set[int]:
        """Each agent's first per_agent calls arriving after the warm-up, of
        those seen so far."""
        late = [r for r in self._arrived.values() if float(r.arrived_at) >= self.warmup_s]
        late.sort(key=lambda r: (float(r.arrived_at), r.id))
        taken: Dict[Any, int] = {}
        ids = set()
        for r in late:
            if taken.get(r.session_id, 0) < self.per_agent:
                taken[r.session_id] = taken.get(r.session_id, 0) + 1
                ids.add(r.id)
        return ids

    def maybe_stop(self, completed) -> None:
        """Stop the simulator once the window is full and every call in it
        is in `completed` (ids)."""
        ids = self.window_ids()
        if len(ids) == self.num_scored and all(i in completed for i in ids):
            self.scored_ids = ids
            self._simulator._terminate = True

    def final_ids(self) -> Set[int]:
        """The scored calls of a finished episode (warning if the trace ran
        out before the window filled)."""
        scored = self.scored_ids or self.window_ids()
        if len(scored) < self.num_scored:
            warnings.warn(
                f"session_concurrency: only {len(scored)} of {self.num_scored} "
                f"calls arrived after the warm-up; raise CONCURRENT_OVERPROVISION"
            )
        return scored
