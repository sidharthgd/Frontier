"""Tables from decisions18/ (scripts/rl/decision_anatomy.py, pure e2e+1h @20
best.pt, 18 eval seeds): how often the policy's ranking can matter, what
skipping the rest costs and saves, and how its rankings differ from plain
(uncached-prefill) SJF.

    .venv/bin/python rl_runs/chameleon/decision_analysis.py > rl_runs/chameleon/decisions18/tables.txt
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "decisions18")
W = ["tracelab_c10", "freeinference_c10", "tracelab_c8", "freeinference_c8"]
SHORT = {"tracelab_c8": "TL c8", "freeinference_c8": "FI c8",
         "tracelab_c10": "TL c10", "freeinference_c10": "FI c10"}
CATS = ["blocked", "nofit", "single", "allfit", "decide"]


def load(name):
    p = os.path.join(D, name)
    return pd.read_csv(p if os.path.exists(p) else p + ".gz")


dec = load("decisions.csv")
req = load("requests.csv")
W = [w for w in W if w in set(dec.workload)]
base = dec[(dec.variant == "base") & dec.scored_window]

print("==== 1. scheduler calls by category (base policy, scored window, all seeds pooled)")
print("blocked: no token budget / batch at max_num_seqs; nofit: no waiting request's first chunk fits the free KV;")
print("single: 1 waiting (fits); allfit: all waiting fit in any order;")
print("decide: order can change the batch. 'policy calls/s' = decide calls per simulated second.")
print(f"{'workload':8s} {'calls':>8s} " + " ".join(f"{c:>8s}" for c in CATS)
      + f" {'decide/s':>9s} {'mean n|decide':>14s} | ex post at decide: none / some / all admitted")
for w in W:
    x = base[base.workload == w]
    frac = x.category.value_counts(normalize=True)
    dx = x[x.category == "decide"]
    span = x.groupby("seed").time.agg(lambda t: t.max() - t.min()).sum()
    none = (dx.n_admitted == 0).mean()
    allin = (dx.n_admitted == dx.n).mean()
    print(f"{SHORT[w]:8s} {len(x):8d} " + " ".join(f"{100 * frac.get(c, 0):7.1f}%" for c in CATS)
          + f" {len(dx) / span:9.2f} {dx.n.mean():14.1f} | {100 * none:5.1f}% / {100 * (1 - none - allin):5.1f}% / {100 * allin:5.1f}%")
    # the a-priori checks must agree with what admission then did
    bad_allfit = x[(x.category == "allfit") & (x.n_admitted != x.n)]
    bad_blocked = x[x.category.isin(["blocked", "nofit"]) & (x.n_admitted != 0)]
    if len(bad_allfit) or len(bad_blocked):
        print(f"   check violations: allfit not all admitted {len(bad_allfit)}, blocked/nofit admitted {len(bad_blocked)}")
    held = dx[(dx.n_admitted == 0)]
    print(f"   decide calls admitting nothing: {len(held)}; policy's first request did not fit in "
          f"{100 * (held.first_fits == False).mean():.1f}% of them, though SJF's first would have fit in "
          f"{100 * ((held.first_fits == False) & held.sjf_first_fits).mean():.1f}% (all decide calls: policy first "
          f"fits {100 * dx.first_fits.mean():.1f}%, SJF first fits {100 * dx.sjf_first_fits.mean():.1f}%)")

print("\n==== 2. policy cost per call (wall time on the Chameleon node's CPU, one thread)")
print("t_cache: prefix-cache lookups (vLLM does these at admission anyway); t_obs: feature building; "
      "t_model: transformer forward. ms.")
for w in W:
    x = base[base.workload == w]
    ran = x.t_model.notna()
    pol = (x.t_obs + x.t_model)[ran]
    gaps = x.groupby("seed").time.diff().dropna()
    gaps = gaps[gaps > 0]
    dx = x[x.category == "decide"]
    cost = x.t_obs + x.t_model
    total_all = cost.sum()
    exact = cost[~x.category.isin(["blocked", "nofit"])].sum()
    total_decide = cost[x.category == "decide"].sum()
    span = x.groupby("seed").time.agg(lambda t: t.max() - t.min()).sum()
    print(f"{SHORT[w]:8s} per call: cache {1e3 * x.t_cache.mean():.2f}  check {1e3 * x.t_check.mean():.2f}  "
          f"obs {1e3 * x.t_obs.mean():.2f}  model {1e3 * x.t_model.mean():.2f} (obs+model p99 {1e3 * pol.quantile(.99):.2f}) "
          f"| median sim gap between calls {1e3 * gaps.median():.1f} ms\n         policy ms per simulated second: "
          f"every call {1e3 * total_all / span:.1f}, skipping blocked+nofit {1e3 * exact / span:.1f} "
          f"({100 * (1 - exact / total_all):.0f}% saved), decide only {1e3 * total_decide / span:.1f} "
          f"({100 * (1 - total_decide / total_all):.0f}% saved)")

print("\n==== 3. skipping non-decide calls: e2e / TTFT vs base (pooled quantiles; % change, paired seeds)")
keys = ["workload", "seed", "variant"]
for w in W:
    r = req[req.workload == w]
    b = r[r.variant == "base"]
    line = []
    for v in sorted(set(r.variant) - {"base"}):
        y = r[r.variant == v]
        common = set(b.seed) & set(y.seed)
        bb, yy = b[b.seed.isin(common)], y[y.seed.isin(common)]
        parts = [f"{q}: " + "/".join(f"{100 * (yy[c].quantile(p) / bb[c].quantile(p) - 1):+.1f}%"
                                     for p in (0.5, 0.95, 0.99, 1.0)) for c, q in (("e2e", "e2e"), ("ttft", "TTFT"))]
        same = (np.isclose(bb.sort_values(["seed", "arrived"]).e2e.to_numpy(),
                           yy.sort_values(["seed", "arrived"]).e2e.to_numpy()).mean()
                if len(bb) == len(yy) else float("nan"))
        line.append(f"{v} ({len(common)} seeds, mean e2e {100 * (yy.e2e.mean() / bb.e2e.mean() - 1):+.1f}%, "
                    f"{100 * same:.0f}% of calls identical e2e) " + "  ".join(parts))
    print(f"{SHORT[w]:8s} " + "\n         ".join(line))
print("(quantiles p50/p95/p99/p100)")

q = load("queue.csv") if os.path.exists(os.path.join(D, "queue.csv")) or os.path.exists(
    os.path.join(D, "queue.csv.gz")) else None
if q is None:
    sys.exit()
q = q[q.variant == "base"].copy()
q["k"] = q.groupby(["workload", "seed", "decision"]).admitted.transform("sum")
q["n"] = q.groupby(["workload", "seed", "decision"]).req.transform("size")
print("\n==== 4. policy vs plain SJF (uncached-prefill order) at decide calls")
print("partial = admitted some but not all. same set: the k admitted are the k smallest uncached prefills;")
print("top-1: its first-ranked request is a smallest-uncached one; tau: Kendall tau-b of level vs uncached size")
records = {}
for w in W:
    x = q[q.workload == w]
    same, top1, taus, inv_pairs, top1_all = [], [], [], [], []
    for (seed, decision), g in x.groupby(["seed", "decision"], sort=False):
        pol = g.sort_values(["level", "arrived", "req"])
        sjf_min = g.uncached.min()
        top1_all.append(pol.uncached.iloc[0] == sjf_min)
        if g.uncached.nunique() > 1 and g.level.nunique() > 1:
            taus.append(kendalltau(g.level, g.uncached).statistic)
        k, n = int(g.k.iloc[0]), len(g)
        if not 0 < k < n:
            continue
        sjf = g.sort_values(["uncached", "arrived", "req"])
        same.append(set(g.req[g.admitted]) == set(sjf.req.iloc[:k]))
        top1.append(pol.uncached.iloc[0] == sjf_min)
        a, na = g[g.admitted], g[~g.admitted]
        for row in a.itertuples():
            for o in na[na.uncached < row.uncached].itertuples():
                inv_pairs.append(dict(
                    ratio=(row.uncached + 1) / (o.uncached + 1), d_waited=row.waited - o.waited,
                    waited_more=row.waited > o.waited, cached_more=row.cached > o.cached,
                    hit_frac_more=row.cached / max(row.target, 1) > o.cached / max(o.target, 1),
                    closer_evict=(row.evict_distance >= 0) and (o.evict_distance < 0 or row.evict_distance < o.evict_distance),
                    a_idle=row.idle_blocks > 0, a_restart=row.restarts > 0, o_restart=o.restarts > 0,
                    o_fits=o.fits, shorter_decode=row.decode < o.decode,
                    a_uncached=row.uncached, o_uncached=o.uncached))
    records[w] = pd.DataFrame(inv_pairs)
    nd = x.groupby(["seed", "decision"]).ngroups
    print(f"{SHORT[w]:8s} decide calls {nd:7d} (mean queue {x.groupby(['seed', 'decision']).size().mean():.1f}): "
          f"top-1 is smallest uncached {100 * np.mean(top1_all):5.1f}%, tau median {np.median(taus):+.2f} mean {np.mean(taus):+.2f} "
          f"| partial {len(same)}: same set as SJF {100 * np.mean(same):5.1f}%, top-1 {100 * np.mean(top1):5.1f}%")

print("\n==== 5. where the policy departs from SJF at partial calls: pairs (A admitted, O left waiting, A's uncached > O's)")
print("share of pairs where A ... (the policy sees waited, cached tokens, idle blocks, eviction distance, restarts; "
      "it does not see output length)")
cols = [("waited_more", "waited longer"), ("cached_more", "more cached tokens"),
        ("hit_frac_more", "higher hit fraction"), ("a_idle", "idle cached prefix"),
        ("closer_evict", "nearer eviction"), ("a_restart", "is a restart"),
        ("o_restart", "(O restart)"), ("o_fits", "(O fit on its own)"), ("shorter_decode", "shorter output")]
for w in W:
    p = records[w]
    if p.empty:
        continue
    print(f"{SHORT[w]:8s} pairs {len(p):6d}, uncached A/O median {p.a_uncached.median():.0f}/{p.o_uncached.median():.0f} "
          f"(ratio {p.ratio.median():.1f}x), A waited {p.d_waited.median():+.1f}s more (median) | "
          + "  ".join(f"{label} {100 * p[c].mean():.0f}%" for c, label in cols))

print("\n==== 6. what predicts the policy's order: logistic regression on 'i ranked ahead of j' over request pairs")
print("with different levels at decide calls (pair orientation randomized, no intercept; 5-fold CV accuracy).")
print("standardized coefficients: positive = a larger value moves the request ahead")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
names = ["log uncached", "log cached", "log waited", "restarts", "idle prefix", "log evict dist", "fits now"]
for w in W:
    x = q[q.workload == w]
    rows = []
    for _, g in x.groupby(["seed", "decision"], sort=False):
        if len(g) < 2:
            continue
        lv = g.level.to_numpy()
        i, j = np.triu_indices(len(g), 1)
        keep = lv[i] != lv[j]
        i, j = i[keep], j[keep]
        if not len(i):
            continue
        f = {c: g[c].to_numpy(dtype=float) for c in ("uncached", "cached", "waited", "restarts", "idle_blocks", "evict_distance", "fits")}
        ev = np.where(f["evict_distance"] < 0, 1e5, f["evict_distance"])
        feats = [np.log1p(f["uncached"]), np.log1p(f["cached"]), np.log1p(np.maximum(f["waited"], 0)),
                 f["restarts"], (f["idle_blocks"] > 0).astype(float), np.log1p(ev), f["fits"]]
        rows.append(np.column_stack([lv[i] < lv[j]] + [v[i] - v[j] for v in feats]))
    m = np.vstack(rows)
    if len(m) > 300000:
        m = m[rng.choice(len(m), 300000, replace=False)]
    flip = rng.random(len(m)) < 0.5
    y = np.where(flip, 1 - m[:, 0], m[:, 0]).astype(int)
    X = np.where(flip[:, None], -m[:, 1:], m[:, 1:])
    sjf_acc = ((X[:, 0] < 0) == y)[X[:, 0] != 0].mean()
    fcfs_acc = ((X[:, 2] > 0) == y)[X[:, 2] != 0].mean()
    res = {}
    for label, idx in [("uncached", [0]), ("+waited", [0, 2]), ("+cache state", [0, 1, 2, 4, 5]),
                       ("+fits, restarts (all)", list(range(7)))]:
        res[label] = cross_val_score(LogisticRegression(max_iter=1000, fit_intercept=False), X[:, idx], y, cv=5).mean()
    sd = X.std(0)
    sd[sd == 0] = 1
    coef = LogisticRegression(max_iter=1000, fit_intercept=False).fit(X / sd, y).coef_[0]
    print(f"{SHORT[w]:8s} pairs {len(y)} | rule agreement: SJF {100 * sjf_acc:.1f}%, FCFS {100 * fcfs_acc:.1f}% | logistic: "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in res.items())
          + "\n         coefs: " + ", ".join(f"{n} {c:+.2f}" for n, c in zip(names, coef)))

print("\n==== 7. mean level (0-15, lower runs first) by uncached-prefill size and wait, decide calls")
for w in W:
    x = q[q.workload == w].copy()
    x["uncached tokens"] = pd.cut(x.uncached, [-1, 0, 256, 1024, 4096, 16384, 65536, 1e9],
                                  labels=["0", "1-256", "257-1k", "1k-4k", "4k-16k", "16k-64k", ">64k"])
    x["wb"] = pd.cut(x.waited, [-1, 1, 10, 60, 300, 1e9], labels=["<1s", "1-10s", "10-60s", "60-300s", ">300s"])
    t = x.groupby(["uncached tokens", "wb"], observed=False)["level"].mean().unstack()
    t.insert(0, "n", x.groupby("uncached tokens", observed=False).size())
    print(f"{SHORT[w]} (columns: time waited)")
    print(t.round(1).to_string())
