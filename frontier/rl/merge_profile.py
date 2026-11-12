"""
Merge the per-pass outputs of run_h200_profile.sh into one profile directory
that Frontier reads (data/profiling/compute/<device>/<model>/):

    python -m frontier.rl.merge_profile rl_runs/h200_profiling/prof \\
        data/profiling/compute/h200/llama3.1-8b-1m

- attention.csv: every pass's attention_combined.csv (standard rows plus, for
  the mixed passes, the true-mixed prefill+decode rows), unioned over columns
  and de-duplicated on the input columns (the mixed passes each re-measure the
  same small standard grid; the first measurement is kept).
- attention_kernel_only.csv, linear_op.csv, linear_op_kernel_only.csv: copied
  from the pass that produced them.
"""

import argparse
import glob
import os

import pandas as pd

# columns that identify a measured input; everything else is a measurement
_MEASURED_PREFIXES = ("time_stats.", "cpu_", "profiling_")


def _input_columns(frame: pd.DataFrame):
    return [c for c in frame.columns if not c.startswith(_MEASURED_PREFIXES)]


def merge_attention(src: str) -> pd.DataFrame:
    parts = []
    for path in sorted(glob.glob(os.path.join(src, "*", "compute", "*", "*", "attention_combined.csv"))):
        frame = pd.read_csv(path)
        frame["_pass"] = os.path.basename(path.split("/compute/")[0])
        parts.append(frame)
    merged = pd.concat(parts, ignore_index=True)
    keys = [c for c in _input_columns(merged) if c != "_pass"]
    before = len(merged)
    merged = merged.drop_duplicates(subset=keys, keep="first").drop(columns="_pass")
    print(f"attention.csv: {before} rows from {len(parts)} passes, {len(merged)} after de-duplication")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    merge_attention(args.src).to_csv(os.path.join(args.dst, "attention.csv"), index=False)
    for pass_name, name in (
        ("decode_kernel_only", "attention_kernel_only.csv"),
        ("linear_cuda_event", "linear_op.csv"),
        ("linear_kernel_only", "linear_op_kernel_only.csv"),
    ):
        found = glob.glob(os.path.join(args.src, pass_name, "compute", "*", "*", name))
        assert len(found) == 1, f"expected one {name} under {pass_name}, found {found}"
        frame = pd.read_csv(found[0])
        frame.to_csv(os.path.join(args.dst, name), index=False)
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
