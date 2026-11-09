"""
Headroom on agent sessions: scheduling rules on TraceLab sessions (closed
loop, prefix caching on) with the validated A100 + Llama-3.1-8B setup, paired
by seed against FCFS:

    python -m frontier.rl.agent_compare --output-dir rl_runs/agent_compare

Writes episodes.csv and summary.txt (% change vs FCFS, paired 95% CI).
"""

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

CALLS = "rl_runs/agent_traces/tracelab_calls.csv"
LOADS = (8, 12, 16, 24)  # concurrent sessions, starting within the first minute
POLICIES = (
    "fcfs",
    "sjf_prefill", "sjf_prefill_age400",
    "sjf_uncached", "sjf_uncached_age400", "sjf_uncached_age1600",
    "session_fewest",
    "sjf_total",  # oracle
    "oracle_cost",  # oracle
)
METRICS = (
    "mean_session_serving", "p99_session_serving", "mean_ttft", "p99_ttft",
    "mean_tbt", "cached_prefill_fraction", "num_preemptions",
)


def run_task(args):
    num_sessions, seed, policies, calls_per_session = args
    from frontier.rl.agent_trace import SessionWorkload
    from frontier.rl.agents import baseline_agent
    from frontier.rl.env import FrontierSchedulingEnv, a100_llama31_8b

    workload = SessionWorkload(
        CALLS, num_sessions=num_sessions, calls_per_session=calls_per_session,
        session_rate=num_sessions / 60.0,
    )
    env = FrontierSchedulingEnv(a100_llama31_8b(
        name=f"agents_{num_sessions}", sessions=workload,
        enable_prefix_caching=True, prediction_rounding_bits=5,
    ))
    rows = []
    for policy in policies:
        summary = env.run_episode(baseline_agent(policy, seed), seed).summary
        rows.append({"workload": f"sessions{num_sessions}", "seed": seed,
                     "policy": policy, **summary})
    print(f"done sessions={num_sessions} seed={seed}", flush=True)
    return rows


def summarize(table: pd.DataFrame) -> str:
    from scipy import stats

    lines = ["% change vs fcfs, mean over seeds ± paired 95% CI (negative = better);",
             "cached_prefill_fraction and num_preemptions are absolute means", ""]
    for workload, group in table.groupby("workload", sort=False):
        ref = group[group.policy == "fcfs"].set_index("seed")
        lines.append(f"== {workload} (n={len(ref)} seeds)")
        lines.append(f"{'policy':22s}" + "".join(f"{m[:16]:>18s}" for m in METRICS))
        for policy, rows in group.groupby("policy", sort=False):
            rows = rows.set_index("seed")
            cells = ""
            for m in METRICS:
                if m in ("cached_prefill_fraction", "num_preemptions"):
                    cells += f"{rows[m].mean():>16.2f}  "
                    continue
                ratio = 100 * (rows[m] / ref.loc[rows.index, m] - 1)
                half = (stats.t.ppf(0.975, len(ratio) - 1) * ratio.std(ddof=1)
                        / np.sqrt(len(ratio))) if len(ratio) > 1 else float("nan")
                cells += f"{ratio.mean():+9.1f} ±{half:5.1f}  "
            lines.append(f"{policy:22s}{cells}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loads", type=int, nargs="*", default=list(LOADS))
    parser.add_argument("--policies", nargs="*", default=list(POLICIES))
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--calls-per-session", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = [(n, s, args.policies, args.calls_per_session) for n in args.loads for s in args.seeds]
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        rows = [r for task_rows in pool.imap_unordered(run_task, tasks) for r in task_rows]
    order = {f"sessions{n}": i for i, n in enumerate(args.loads)}
    table = pd.DataFrame(rows).sort_values(["workload", "seed"], key=lambda c: c.map(order) if c.name == "workload" else c)
    table.to_csv(os.path.join(args.output_dir, "episodes.csv"), index=False)
    summary = summarize(table)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
