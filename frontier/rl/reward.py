"""
Per-transition reward components, as in vidur/rl/reward.py. Each component is
an integral over simulated time, so summed over the episode it equals a
per-request metric summed over requests:

    e2e             ∫ #(requests in system) dt          = Σ e2e
    e2e_normalized  ∫ Σ_in-system 1/output_tokens dt    = Σ e2e / output tokens
    ttft            ∫ #(requests without a first token) = Σ TTFT
    e2e_hinge       ∫ #(in system, age > e2e_threshold)  = Σ max(0, e2e - τ)
    ttft_hinge      ∫ #(no first token, age > ttft_threshold)

Transition k is credited with what accrues between its decision and the next
kept one (the last runs to the end of the episode). What accrued before the
first decision is not attributable and is left out. Here the integrals are
computed from request timings after the episode, which gives the same numbers
as integrating online between events.

Which components to use and how to weight them is the trainer's call: reward =
-Σ weight * component.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from frontier.rl.env import Episode

COMPONENTS = ("e2e", "e2e_normalized", "ttft", "e2e_hinge", "ttft_hinge")


@dataclass
class RewardConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {"e2e": 1.0})
    e2e_threshold: Optional[float] = None
    ttft_threshold: Optional[float] = None

    def __post_init__(self):
        unknown = set(self.weights) - set(COMPONENTS)
        assert not unknown, f"unknown reward components {unknown}"

    def combine(self, components: Dict[str, float]) -> float:
        return -sum(self.weights.get(k, 0.0) * v for k, v in components.items())


def _overlap(starts: np.ndarray, ends: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """[num_intervals, num_items] length of [start, end) inside each [edge_k, edge_k+1)."""
    lo = np.maximum(edges[:-1, None], starts[None, :])
    hi = np.minimum(edges[1:, None], ends[None, :])
    return np.clip(hi - lo, 0.0, None)


def transition_components(
    episode: "Episode", config: RewardConfig
) -> List[Dict[str, float]]:
    if not episode.transitions:
        return []
    edges = np.array([t.time for t in episode.transitions] + [episode.end_time])
    end = episode.end_time
    reqs = episode.requests
    arrived = np.array([r.arrived_at for r in reqs])
    done = np.array([r.completed_at if r.completed_at is not None else end for r in reqs])
    first = np.array(
        [r.first_token_at if r.first_token_at is not None else end for r in reqs]
    )
    output = np.array([max(1, r.num_decode_tokens) for r in reqs], dtype=float)

    in_system = _overlap(arrived, done, edges)
    waiting_first = _overlap(arrived, first, edges)
    out = {
        "e2e": in_system.sum(axis=1),
        "e2e_normalized": (in_system / output[None, :]).sum(axis=1),
        "ttft": waiting_first.sum(axis=1),
    }
    for name, threshold, stop in (
        ("e2e_hinge", config.e2e_threshold, done),
        ("ttft_hinge", config.ttft_threshold, first),
    ):
        if threshold is None:
            out[name] = np.zeros(len(edges) - 1)
        else:
            out[name] = _overlap(arrived + threshold, np.maximum(stop, arrived + threshold), edges).sum(axis=1)
    return [{k: float(v[i]) for k, v in out.items()} for i in range(len(edges) - 1)]
