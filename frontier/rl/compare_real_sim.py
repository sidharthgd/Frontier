"""
Compare Frontier with a real vLLM run of the same closed-loop sessions
(frontier/rl/vllm_sessions.py), prefix caching on, FCFS:

    python -m frontier.rl.compare_real_sim --real real_sessions.csv \\
        --num-sessions 4 --calls-per-session 10 --seed 0 --num-blocks 52000 --max-model-len 800000

The workload arguments must match the ones the real run used. --num-blocks is
the KV block count in the real server's startup log.
"""

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    from frontier.rl.agent_trace import DEFAULT_SHARED_PREFIX, SessionWorkload
    from frontier.rl.agents import baseline_agent
    from frontier.rl.env import FrontierSchedulingEnv, h200_nemotron_1m

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--real", required=True)
    parser.add_argument("--calls", default="rl_runs/agent_traces/tracelab_calls.csv")
    parser.add_argument("--num-sessions", type=int, default=4)
    parser.add_argument("--calls-per-session", type=int, default=10)
    parser.add_argument("--session-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output", default=None, help="write the per-call table here")
    args = parser.parse_args()

    workload = SessionWorkload(
        args.calls, num_sessions=args.num_sessions, calls_per_session=args.calls_per_session,
        session_rate=args.session_rate,
        shared_prefix_tokens=DEFAULT_SHARED_PREFIX if args.shared_prefix else (),
    )
    env = FrontierSchedulingEnv(h200_nemotron_1m(
        name="validation", sessions=workload, enable_prefix_caching=True,
        num_blocks=args.num_blocks, max_model_len=args.max_model_len,
    ))
    episode = env.run_episode(baseline_agent("fcfs"), args.seed)

    by_session = {}
    for r in sorted(episode.requests, key=lambda r: r.id):
        by_session.setdefault(r.session_id, []).append(r)
    rows = []
    for session, calls in by_session.items():
        for k, r in enumerate(calls):
            if r.first_token_at is None:
                continue
            rows.append({"session": session, "call_index": k, "sim_ttft": r.first_token_at - r.arrived_at,
                         "sim_e2e": r.completed_at - r.arrived_at, "sim_cached": r.num_cached_prefill_tokens})
    sim = pd.DataFrame(rows)
    real = pd.read_csv(args.real)
    real = real[real.error.fillna("") == ""]
    both = real.merge(sim, on=["session", "call_index"])
    both["ttft_ratio"] = both.sim_ttft / both.ttft
    both["e2e_ratio"] = both.sim_e2e / both.e2e

    def corr(a, b):
        return float(np.corrcoef(np.log(a), np.log(b))[0, 1]) if len(a) > 2 else float("nan")

    print(f"{len(both)} calls compared ({len(real)} real, {len(sim)} simulated)")
    for name, real_col, sim_col, ratio in (("TTFT", "ttft", "sim_ttft", "ttft_ratio"), ("E2E", "e2e", "sim_e2e", "e2e_ratio")):
        err = (both[ratio] - 1).abs()
        print(f"{name}: real mean {both[real_col].mean():.2f}s  sim mean {both[sim_col].mean():.2f}s  "
              f"median |error| {100 * err.median():.0f}%  p90 {100 * err.quantile(0.9):.0f}%  "
              f"log-corr {corr(both[real_col], both[sim_col]):.3f}")
    if both.cached_tokens.notna().any():
        print(f"cached prompt tokens: real {both.cached_tokens.sum():.0f} ({100 * both.cached_tokens.sum() / both.prompt.sum():.1f}%)  "
              f"sim {both.sim_cached.sum():.0f} ({100 * both.sim_cached.sum() / both.prompt.sum():.1f}%)")
    if args.output:
        both.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
