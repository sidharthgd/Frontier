"""
Compare Frontier with a real vLLM run of a request trace (frontier/rl/vllm_replay.py):

    python -m frontier.rl.validate_requests --trace sweep_requests.csv --real real_sweep.csv \\
        --num-blocks 56345 --max-model-len 890000 --sequential

--sequential means the real run sent one request at a time (vllm_replay
--sequential), so the simulator is given the same requests spaced far apart
(its own timestamps are free). Otherwise the trace's arrival times are used.
"""

import argparse
import os
import tempfile

import numpy as np
import pandas as pd


def main() -> None:
    from frontier.rl.agents import baseline_agent
    from frontier.rl.env import FrontierSchedulingEnv, h200_nemotron_1m

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trace", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    trace = pd.read_csv(args.trace).sort_values("arrived_at").reset_index(drop=True)
    if args.sequential:
        trace["arrived_at"] = np.arange(len(trace)) * 1e5
    path = os.path.join(tempfile.mkdtemp(), "trace.csv")
    trace.to_csv(path, index=False)

    env = FrontierSchedulingEnv(h200_nemotron_1m(
        name="validation", arrivals="replay", replay_trace=path, num_blocks=args.num_blocks,
        max_model_len=args.max_model_len, max_tokens=args.max_model_len,
    ))
    episode = env.run_episode(baseline_agent("fcfs"), 0)
    done = sorted((r for r in episode.requests if r.first_token_at is not None), key=lambda r: r.arrived_at)
    sim = pd.DataFrame({
        "index": range(len(done)), "sim_ttft": [r.first_token_at - r.arrived_at for r in done],
        "sim_e2e": [r.completed_at - r.arrived_at for r in done],
        "sim_tbt": [np.diff(r.token_times).mean() if len(r.token_times) > 1 else np.nan for r in done],
    })
    real = pd.read_csv(args.real)
    real = real[real.error.fillna("") == ""]
    both = real.merge(sim, on="index")
    both["ttft_err"] = both.sim_ttft / both.ttft - 1
    both["e2e_err"] = both.sim_e2e / both.e2e - 1
    both["tbt_err"] = both.sim_tbt / both.mean_tbt - 1
    cols = ["num_prefill_tokens", "ttft", "sim_ttft", "ttft_err", "mean_tbt", "sim_tbt", "tbt_err", "e2e", "sim_e2e", "e2e_err"]
    print(both[cols].round(3).to_string(index=False))
    print(f"\nmedian |error|: TTFT {100 * both.ttft_err.abs().median():.1f}%  TBT {100 * both.tbt_err.abs().median():.1f}%  E2E {100 * both.e2e_err.abs().median():.1f}%")
    if args.output:
        both.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
