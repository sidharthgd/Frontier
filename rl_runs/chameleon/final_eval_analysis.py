"""Tables for RESULTS.md from final_eval18/requests.csv (18 seeds x 4 eval workloads,
one row per completed request; scripts/rl/ttft_tail_anatomy.py).

    .venv/bin/python rl_runs/chameleon/final_eval_analysis.py > rl_runs/chameleon/final_eval18/tables.txt

Per workload, pooled over seeds: TTFT p50/p99/p100 and requests past FCFS's worst
TTFT; e2e mean/p50/p99/p100 with the change vs FCFS and a seed-bootstrap 90%
interval for the p50/p99 changes; per-session totals (span = first arrival to
last completion, and summed e2e)
per agent: session_id is the agent (frontier/rl/scoring.py) and each agent's
scored calls are a fixed quota (30 at c10, 38 at c8), so the span is the time
the agent takes to finish its share of work; 180 agents per c10 workload, so
p50 / p90 / max rather than p99; cache hit rate, uncached prefill and
preemption restarts.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "final_eval18")
W = ["tracelab_c8", "freeinference_c8", "tracelab_c10", "freeinference_c10"]
SHORT = {"tracelab_c8": "TL c8", "freeinference_c8": "FI c8",
         "tracelab_c10": "TL c10", "freeinference_c10": "FI c10"}
BASE = {"fcfs": "FCFS", "sjf_prefill_tokens": "plain SJF",
        "sjf_prefill_tokens_age200": "age200", "sjf_prefill_tokens_rel30": "rel30"}
RUN = {"imitation": "imitation", "ppo_w0.01": "imit+PPO h0.01", "ppo_w0.02": "imit+PPO h0.02",
       "ppo_w0.04": "imit+PPO h0.04", "ppo_w0.01_lr3e-5": "imit+PPO h0.01 lr3e-5",
       "ppo_w0.02_lr3e-5": "imit+PPO h0.02 lr3e-5", "ppo_imit_e2e_w2": "imit+PPO e2e+2h",
       "ppo_pure_e2e_w1": "pure e2e+1h", "ppo_pure_e2e_w1_seed1": "pure e2e+1h s1",
       "ppo_pure_e2e_w2": "pure e2e+2h", "ppo_pure_e2e_w6": "pure e2e+6h",
       "ppo_pure_e2e_sq0.01": "pure e2e+0.01sq", "ppo_pure_e2e_sq0.02": "pure e2e+0.02sq",
       "ppo_pure_e2e_sq0.05": "pure e2e+0.05sq"}

iters = {}
for line in open(os.path.join(D, "checkpoint_iterations.txt")):
    path, it, _ = line.split()
    iters[path] = int(it)


def label(policy):
    if policy in BASE:
        return BASE[policy]
    path = policy[len("ckpt:"):]
    run, f = path.split("/")[-2:]
    if f == "cdfs_iter10.pt":
        return f"{RUN[run]} @10"
    kind = f[:-3]
    return f"{RUN[run]} @{iters.get(path, '?')} {kind}"


d = pd.read_csv(os.path.join(D, "requests.csv"))
d["label"] = d.policy.map(label)
d["uncached"] = d.prompt - d.cached
d["done_at"] = d.arrived + d.e2e
labels = list(dict.fromkeys(d.label))
order = [BASE[k] for k in BASE] + sorted(l for l in labels if l not in BASE.values())
order = [l for l in order if l in labels]
n_seeds = d.seed.nunique()
print(f"rows {len(d)}, seeds {n_seeds}, policies {len(labels)}")
cnt = d.groupby(["workload", "label"]).size().unstack(0)
print("requests per (policy, workload) - should be equal:", {w: sorted(set(cnt[w])) for w in cnt})

rng = np.random.default_rng(0)
seeds = np.sort(d.seed.unique())
B = 1000
boot_idx = rng.integers(0, len(seeds), size=(B, len(seeds)))


def by_seed(x, col):
    return [x[x.seed == s][col].to_numpy() for s in seeds]


def boot_change(per_seed, ref_seed, q):
    """90% interval of pooled-quantile change vs FCFS, resampling seeds (paired)."""
    out = np.empty(B)
    for b in range(B):
        idx = boot_idx[b]
        a = np.concatenate([per_seed[i] for i in idx])
        r = np.concatenate([ref_seed[i] for i in idx])
        out[b] = np.quantile(a, q) / np.quantile(r, q) - 1
    return np.quantile(out, [0.05, 0.95])


def pct(a, b):
    return f"{100 * (a / b - 1):+.0f}%"


summary = []
for w in W:
    x = d[d.workload == w]
    f = x[x.label == "FCFS"]
    fmax = f.ttft.max()
    f_seed = by_seed(f, "e2e")
    sess = x[x.session >= 0].groupby(["label", "seed", "session"]).agg(
        span=("done_at", "max"), start=("arrived", "min"), busy=("e2e", "sum"))
    sess["span"] -= sess.start
    print(f"\n==== {SHORT[w]}  ({len(f)} requests per policy, {n_seeds} seeds)")
    print(f"{'policy':34s} {'TTFT p50/p99/p100':>20s} {'>FCFSmax':>8s} | "
          f"{'e2e mean':>14s} {'e2e p50':>22s} {'e2e p99':>22s} {'p100':>5s} | "
          f"{'agent span p50/p90/max':>22s} {'agent busy p50/p90':>18s} | hit%  uncachedM/ep restart%")
    fs = sess.loc["FCFS"]
    for l in order:
        y = x[x.label == l]
        if len(y) != len(f):  # not every seed finished (partial file)
            continue
        t, e = y.ttft.to_numpy(), y.e2e.to_numpy()
        p50, p99 = np.median(e), np.quantile(e, 0.99)
        fp50, fp99 = np.median(f.e2e), np.quantile(f.e2e, 0.99)
        ys = by_seed(y, "e2e")
        c50 = boot_change(ys, f_seed, 0.5) if l != "FCFS" else (0, 0)
        c99 = boot_change(ys, f_seed, 0.99) if l != "FCFS" else (0, 0)
        s = sess.loc[l]
        row = dict(workload=w, label=l, ttft_p50=np.median(t), ttft_p99=np.quantile(t, .99),
                   ttft_p100=t.max(), over=(t > fmax).sum(), e2e_mean=e.mean(), e2e_p50=p50,
                   e2e_p99=p99, e2e_p100=e.max(), d50=p50 / fp50 - 1, d99=p99 / fp99 - 1,
                   d50_lo=c50[0], d50_hi=c50[1], d99_lo=c99[0], d99_hi=c99[1],
                   dmean=e.mean() / f.e2e.mean() - 1,
                   span_p50=s.span.median(), span_p90=s.span.quantile(.9), span_max=s.span.max(),
                   busy_p50=s.busy.median(), busy_p90=s.busy.quantile(.9),
                   dspan90=s.span.quantile(.9) / fs.span.quantile(.9) - 1,
                   dspanmax=s.span.max() / fs.span.max() - 1,
                   agents=len(s),
                   dspan50=s.span.median() / fs.span.median() - 1,
                   dbusy50=s.busy.median() / fs.busy.median() - 1,
                   dbusy90=s.busy.quantile(.9) / fs.busy.quantile(.9) - 1,
                   hit=100 * y.cached.sum() / y.prompt.sum(),
                   uncached=y.uncached.sum() / 1e6 / n_seeds,
                   restarted=100 * (y.restarts > 0).mean())
        summary.append(row)
        print(f"{l:34s} {row['ttft_p50']:5.1f}/{row['ttft_p99']:6.1f}/{row['ttft_p100']:5.0f} "
              f"{row['over']:8d} | {row['e2e_mean']:6.1f} {pct(e.mean(), f.e2e.mean()):>6s} "
              f"{p50:6.1f} {pct(p50, fp50):>5s} [{100*c50[0]:+.0f},{100*c50[1]:+.0f}] "
              f"{p99:6.0f} {pct(p99, fp99):>5s} [{100*c99[0]:+.0f},{100*c99[1]:+.0f}] {e.max():5.0f} | "
              f"{row['span_p50']:6.0f} {pct(row['span_p50'], fs.span.median()):>4s}/{row['span_p90']:5.0f}/{row['span_max']:5.0f}  {row['busy_p50']:6.0f}/{row['busy_p90']:6.0f}    | "
              f"{row['hit']:4.1f} {row['uncached']:6.2f} {row['restarted']:6.1f}")

s = pd.DataFrame(summary)
s.to_csv(os.path.join(D, "summary.csv"), index=False)
print("\n==== ranking by e2e p50 change vs FCFS, then p99 (top 8 per workload)")
for w in W:
    z = s[s.workload == w].sort_values("d50").head(8)
    print(SHORT[w] + ": " + "; ".join(
        f"{r.label} {100*r.d50:+.1f}%/{100*r.d99:+.1f}%" for r in z.itertuples()))
print("\n==== per-agent completion time (fixed quota of scored calls): change vs FCFS in p50 / p90 / max span")
for w in W:
    z = s[s.workload == w].sort_values("dspan50").head(8)
    print(SHORT[w] + f" ({int(s[s.workload == w].agents.iloc[0])} agents): " + "; ".join(
        f"{r.label} {100*r.dspan50:+.1f}%/{100*r.dspan90:+.1f}%/{100*r.dspanmax:+.1f}%" for r in z.itertuples()))
print("\n==== best.pt vs latest.pt (e2e p50 / p99 / TTFT p100), per run, c10 TL | FI")
for run in RUN.values():
    rows = s[s.label.str.startswith(run + " @") & s.workload.str.endswith("c10")]
    for kind in ("best", "latest"):
        r = rows[rows.label.str.endswith(kind)]
        if len(r):
            print(f"{r.label.iloc[0]:38s} " + " | ".join(
                f"{100*q.d50:+.1f}% {100*q.d99:+.1f}% {q.ttft_p100:.0f}s" for q in r.itertuples()))
