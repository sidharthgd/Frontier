"""p50 / p99 / p100 TTFT and e2e per eval workload for the baselines and every
non-abandoned PPO run's best checkpoint, from the CDF samples (6 seeds pooled)."""
import numpy as np

W = ["tracelab_c8", "tracelab_c10", "freeinference_c8", "freeinference_c10"]
base = {
    "FCFS": "fcfs",
    "age200": "sjf_prefill_tokens_age200",
    "rel30": "sjf_prefill_tokens_rel30",
    "plain SJF": "sjf_prefill_tokens",
}
runs = [
    ("imitation (age200 student)", "imitation/cdfs"),
    ("ppo_w0.01 @10", "ppo_w0.01/cdfs/best_iter0010"),
    ("ppo_w0.02 @10", "ppo_w0.02/cdfs/best_iter0010"),
    ("ppo_w0.04 @110", "ppo_w0.04/cdfs/best_iter0110"),
    ("ppo_w0.01_lr3e-5 @90", "ppo_w0.01_lr3e-5/cdfs/best_iter0090"),
    ("ppo_w0.02_lr3e-5 @80", "ppo_w0.02_lr3e-5/cdfs/best_iter0080"),
    ("ppo_imit_e2e_w2 @10", "ppo_imit_e2e_w2/cdfs/best_iter0010"),
    ("pure e2e w1 @20", "ppo_pure_e2e_w1/cdfs/best_iter0020"),
    ("pure e2e w1 seed1 @40", "ppo_pure_e2e_w1_seed1/cdfs/best_iter0040"),
    ("pure e2e w2 @30", "ppo_pure_e2e_w2/cdfs/best_iter0030"),
    ("pure e2e w6 @20", "ppo_pure_e2e_w6/cdfs/best_iter0020"),
]
rows = [(k, "cdfs_baselines/data/%s__" + v + ".npz") for k, v in base.items()]
rows += [(k, d + "/data/%s__policy_argmax.npz") for k, d in runs]

out = {}
for name, pat in rows:
    for w in W:
        x = np.load(pat % w)
        out[(name, w)] = {m: x[m] for m in ("ttft", "e2e")}

print("samples per workload (all policies should match):",
      {w: sorted({len(out[(n, w)]["ttft"]) for n, _ in rows}) for w in W})
for metric in ("ttft", "e2e"):
    print("\n==== %s (s): p50 / p99 / p100, 6 seeds pooled" % metric.upper())
    print("%-27s" % "" + "".join("%-24s" % w for w in W))
    for name, _ in rows:
        cells = []
        for w in W:
            v = out[(name, w)][metric]
            cells.append("%5.1f /%6.1f /%6.0f     " % (np.median(v), np.quantile(v, 0.99), v.max()))
        print("%-27s" % name + "".join(cells))

print("\n==== requests whose TTFT exceeds the FCFS maximum on that workload: count; top-3 TTFTs (s)")
for name, _ in rows:
    cells = []
    for w in W:
        t = np.sort(out[(name, w)]["ttft"])[::-1]
        fmax = out[("FCFS", w)]["ttft"].max()
        cells.append("%2d; %s" % ((t > fmax).sum(), ",".join("%.0f" % v for v in t[:3])))
    print("%-27s" % name + "".join("%-24s" % c for c in cells))
