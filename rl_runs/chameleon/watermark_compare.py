"""Watermark SJF (watermark18/, scripts/rl/ttft_tail_anatomy.py wmsjf<f>) and
cache-protecting SJF (cpsjf18/, cpsjf) against
FCFS, plain SJF and pure e2e+1h @20 from final_eval18 on the same seeds:
e2e mean/p50/p95/p99 and TTFT p95/p99/p100 change vs FCFS, cache hit rate,
restarted share, and paired-bootstrap e2e changes vs plain SJF and PPO.

    .venv/bin/python rl_runs/chameleon/watermark_compare.py > rl_runs/chameleon/watermark18/tables.txt
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
f18 = os.path.join(ROOT, "final_eval18", "requests.csv")
cols = ["workload", "seed", "policy", "ttft", "e2e", "prompt", "cached", "restarts"]
a = pd.read_csv(f18 if os.path.exists(f18) else f18 + ".gz", usecols=cols)
PPO = "ckpt:rl_runs/chameleon/ppo_pure_e2e_w1/best.pt"
a = a[a.policy.isin(["fcfs", "sjf_prefill_tokens", "sjf_prefill_tokens_age200", PPO])]
wm = pd.concat([pd.read_csv(os.path.join(ROOT, r, "requests.csv"), usecols=cols) for r in ("watermark18", "cpsjf18")])
# only seeds with every workload done for every watermark policy
full = wm.groupby(["seed"]).apply(lambda g: g.groupby(["workload", "policy"]).ngroups).loc[lambda s: s == s.max()].index
d = pd.concat([a, wm])
d = d[d.seed.isin(full)]
NAME = {"fcfs": "FCFS", "sjf_prefill_tokens": "plain SJF", "sjf_prefill_tokens_age200": "age200",
        PPO: "PPO e2e+1h @20", "wmsjf0.1": "SJF + 10% KV watermark", "wmsjf0.2": "SJF + 20% KV watermark",
        "cpsjf": "cache-protecting SJF"}
seeds = np.sort(d.seed.unique())
boot = np.random.default_rng(0).integers(0, len(seeds), (1000, len(seeds)))


def paired(x, y, stat):
    xs = [x[x.seed == s].e2e.to_numpy() for s in seeds]
    ys = [y[y.seed == s].e2e.to_numpy() for s in seeds]
    pt = stat(np.concatenate(xs)) / stat(np.concatenate(ys)) - 1
    bs = [stat(np.concatenate([xs[i] for i in ix])) / stat(np.concatenate([ys[i] for i in ix])) - 1 for ix in boot]
    lo, hi = np.quantile(bs, [0.05, 0.95])
    return f"{100 * pt:+.1f}% [{100 * lo:+.0f},{100 * hi:+.0f}]"


print(f"{len(seeds)} seeds")
for w in ["tracelab_c10", "freeinference_c10", "tracelab_c8", "freeinference_c8"]:
    x = d[d.workload == w]
    f = x[x.policy == "fcfs"]
    print(f"\n{w}: change vs FCFS — e2e mean / p50 / p95 / p99 | TTFT p95 / p99 / p100 (s) | hit% restarted%")
    for p, name in NAME.items():
        y = x[x.policy == p]
        if not len(y):
            continue
        e = [y.e2e.mean() / f.e2e.mean() - 1] + [y.e2e.quantile(q) / f.e2e.quantile(q) - 1 for q in (0.5, 0.95, 0.99)]
        print(f"  {name:24s} " + " / ".join(f"{100 * v:+5.1f}%" for v in e)
              + f" | {y.ttft.quantile(.95):5.1f} / {y.ttft.quantile(.99):5.0f} / {y.ttft.max():5.0f}"
              + f" | {100 * y.cached.sum() / y.prompt.sum():.1f} {100 * (y.restarts > 0).mean():.1f}")
    for p in ("wmsjf0.1", "wmsjf0.2", "cpsjf"):
        y = x[x.policy == p]
        if not len(y):
            continue
        for ref in ("sjf_prefill_tokens", PPO):
            r = x[x.policy == ref]
            print(f"  {NAME[p]} vs {NAME[ref]}: e2e mean {paired(y, r, np.mean)}, p50 {paired(y, r, np.median)}, "
                  f"p95 {paired(y, r, lambda v: np.quantile(v, .95))}")
