"""Build session traces with block hashes for prefix caching, and sample
training workloads from them (see frontier/trace_prep/README.md):

    python -m frontier.trace_prep freeinference release/<week>/traces/*.jsonl.zst -o OUT.csv
    python -m frontier.trace_prep tracelab syfi_coding_trace.jsonl.gz -o OUT.csv
    python -m frontier.trace_prep sharegpt ShareGPT_V3_unfiltered_cleaned_split.json -o OUT.csv
    python -m frontier.trace_prep sample OUT.csv -n 500 --qps 4 --seed 0 -o WORKLOAD.csv
    python -m frontier.trace_prep describe OUT.csv
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, help, max_tokens):
        command = commands.add_parser(name, help=help)
        command.add_argument("-o", "--output", required=True)
        command.add_argument(
            "--max-tokens", type=int, default=max_tokens,
            help=f"the model's context (default {max_tokens}; 0: no limit)",
        )
        command.add_argument(
            "--compact-at", type=int, default=120_000,
            help="compact a session's context when a call's prompt + output would pass "
            "this (default 120000; 0: drop calls over --max-tokens instead)",
        )
        command.add_argument("--compact-to", type=int, default=16_384,
                             help="tokens a compacted context restarts with (default 16384)")
        return command

    freeinference = add("freeinference", "FreeInference agentic trace (HF)", 131072)
    freeinference.add_argument("paths", nargs="+", help="<day>.jsonl.zst or .jsonl files")
    freeinference.add_argument("--harness", action="append", help="keep only these harnesses")

    tracelab = add("tracelab", "TraceLab coding-agent trace (GitHub release)", 131072)
    tracelab.add_argument("path", help="syfi_coding_trace.jsonl.gz")

    sharegpt = add("sharegpt", "ShareGPT, every turn, Llama-3 tokenizer", 4096)
    sharegpt.add_argument("path", help="ShareGPT_V3_unfiltered_cleaned_split.json")
    sharegpt.add_argument("--tokenizer", default=None)
    sharegpt.add_argument("--max-conversations", type=int, default=None)

    sample = commands.add_parser("sample", help="sample a training workload from a session trace")
    sample.add_argument("trace")
    sample.add_argument("-o", "--output", required=True)
    sample.add_argument("-n", "--num-requests", type=int, required=True)
    sample.add_argument("--qps", type=float, required=True)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--max-tokens", type=int, default=None)
    sample.add_argument("--session-time-scale", type=float, default=1.0)
    sample.add_argument("--turn-gap", type=float, default=30.0)
    sample.add_argument("--min-turn-gap", type=float, default=1.0)
    sample.add_argument("--phase", choices=["stationary", "start"], default="stationary")
    sample.add_argument("--open-loop", action="store_true",
                        help="no think_time: every call arrives at its sampled time")

    finalize = add("finalize", "add output ids (and compaction) to a session trace", 131072)
    finalize.add_argument("trace")

    describe = commands.add_parser("describe", help="summarize a session trace")
    describe.add_argument("trace")

    args = parser.parse_args()
    max_tokens = getattr(args, "max_tokens", None) or None
    compact_at = getattr(args, "compact_at", None) or None
    if compact_at is not None and max_tokens is not None:
        compact_at = min(compact_at, max_tokens)

    if args.command == "freeinference":
        from frontier.trace_prep.freeinference import convert

        stats = convert(args.paths, args.output, max_tokens, args.harness, compact_at, args.compact_to)
    elif args.command == "tracelab":
        from frontier.trace_prep.tracelab import convert

        stats = convert(args.path, args.output, max_tokens, compact_at, args.compact_to)
    elif args.command == "finalize":
        from frontier.trace_prep.common import finalize_trace

        stats = finalize_trace(args.trace, args.output, max_tokens, compact_at, args.compact_to)
    elif args.command == "sharegpt":
        from frontier.trace_prep.sharegpt import DEFAULT_TOKENIZER, convert

        stats = convert(
            args.path, args.output, args.tokenizer or DEFAULT_TOKENIZER,
            max_tokens, args.max_conversations,
        )
    elif args.command == "sample":
        from frontier.trace_prep.sessions import (
            load_session_trace,
            sample_session_workload,
            write_session_workload,
        )

        workload = sample_session_workload(
            load_session_trace(args.trace, max_tokens),
            args.num_requests, args.qps, args.seed,
            session_time_scale=args.session_time_scale,
            turn_gap=args.turn_gap, min_turn_gap=args.min_turn_gap, phase=args.phase,
            closed_loop=not args.open_loop,
        )
        write_session_workload(args.output, workload)
        stats = {"requests": len(workload), "sessions": workload["session_id"].nunique()}
    else:
        from frontier.trace_prep.common import describe as describe_trace

        print(describe_trace(args.trace))
        return

    print(dict(stats))
    if args.command != "sample":
        from frontier.trace_prep.common import describe as describe_trace

        print(describe_trace(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
