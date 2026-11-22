"""Paired seed-bootstrap of PPO checkpoints against heuristics at c10 (final_eval18):
90% interval of the relative difference in e2e p50 / p99 / mean and per-agent span p50.

    .venv/bin/python rl_runs/chameleon/final_eval_headtohead.py > rl_runs/chameleon/final_eval18/headtohead.txt
"""
import os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_eval18")
R = "ckpt:rl_runs/chameleon/"
A = {"pure e2e+1h @20": R + "ppo_pure_e2e_w1/best.pt",
     "pure e2e+1h s1 @40": R + "ppo_pure_e2e_w1_seed1/best.pt",
     "pure e2e+0.02sq @40": R + "ppo_pure_e2e_sq0.02/best.pt"}
H = {"plain SJF": "sjf_prefill_tokens", "age200": "sjf_prefill_tokens_age200",
     "rel30": "sjf_prefill_tokens_rel30"}
d = pd.read_csv(os.path.join(D, "requests.csv"),
                usecols=["workload", "seed", "policy", "e2e", "session", "arrived"])
d = d[d.policy.isin(list(A.values()) + list(H.values())) & d.workload.str.endswith("c10")]
d["done"] = d.arrived + d.e2e
seeds = np.sort(d.seed.unique())
rng = np.random.default_rng(0)
idx = rng.integers(0, len(seeds), (1000, len(seeds)))
stats = {"e2e p50": lambda e, s: np.median(e), "e2e p99": lambda e, s: np.quantile(e, .99),
         "e2e mean": lambda e, s: e.mean(), "agent span p50": lambda e, s: np.median(s)}


def per_seed(x):
    out = []
    for sd in seeds:
        y = x[x.seed == sd]
        g = y.groupby("session")
        out.append((y.e2e.to_numpy(), (g.done.max() - g.arrived.min()).to_numpy()))
    return out


for w in ["tracelab_c10", "freeinference_c10"]:
    x = d[d.workload == w]
    ps = {p: per_seed(x[x.policy == p]) for p in list(A.values()) + list(H.values())}
    print(f"==== {w}: PPO vs heuristic, relative difference [90% paired seed bootstrap]; negative = PPO faster")
    for an, ap in A.items():
        for hn, hp in H.items():
            cells = []
            for sn, f in stats.items():
                def val(p, ii):
                    e = np.concatenate([ps[p][i][0] for i in ii])
                    s = np.concatenate([ps[p][i][1] for i in ii])
                    return f(e, s)
                point = val(ap, range(len(seeds))) / val(hp, range(len(seeds))) - 1
                b = np.array([val(ap, ii) / val(hp, ii) - 1 for ii in idx])
                lo, hi = np.quantile(b, [.05, .95])
                cells.append(f"{sn} {100*point:+.1f}% [{100*lo:+.0f},{100*hi:+.0f}]")
            print(f"  {an:20s} vs {hn:9s}: " + "; ".join(cells))
