#!/bin/bash
# Stop a PPO run started by run_chameleon_long.sh and resume it from its
# latest.pt (model, optimizer, iteration; frontier/rl/train.py resume_checkpoint),
# e.g. after a code fix or to raise its iteration cap:
#   bash scripts/rl/resume_chameleon_run.sh <run name> [--set iterations=400 ...]
# The old train.log is kept as train.<n>.log, and metrics lines past the
# checkpoint's iteration move to metrics.discarded.jsonl.
set -euo pipefail
cd "$(dirname "$0")/../.."
name=$1; shift
dir=rl_runs/chameleon/$name
session=${name//./_}
[ -f "$dir/latest.pt" ] || { echo "no $dir/latest.pt"; exit 1; }
tmux kill-session -t "$session" 2>/dev/null || true
# only the trainer's own python process: other command lines can mention the
# run too (a tmux server keeps the command of the session that started it)
pattern="^\.venv/bin/python -m frontier\.rl\.train .*--set output_dir=${dir//./\\.}( |$)"
for pid in $(pgrep -f "$pattern" || true); do kill "$pid" 2>/dev/null || true; done
sleep 10
iter=$(.venv/bin/python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['iteration'])" "$dir/latest.pt")
n=1; while [ -e "$dir/train.$n.log" ]; do n=$((n+1)); done
mv "$dir/train.log" "$dir/train.$n.log"
.venv/bin/python - "$dir/metrics.jsonl" "$iter" <<'PY'
import json, sys
path, iteration = sys.argv[1], int(sys.argv[2])
lines = open(path).read().splitlines()
keep = [l for l in lines if json.loads(l)["iteration"] <= iteration]
drop = [l for l in lines if json.loads(l)["iteration"] > iteration]
open(path, "w").write("".join(l + "\n" for l in keep))
with open(path.replace("metrics.jsonl", "metrics.discarded.jsonl"), "a") as f:
    f.write("".join(l + "\n" for l in drop))
print(f"resuming at iteration {iteration}, {len(drop)} later metrics lines set aside")
PY
# the original command plus the resume checkpoint and any new overrides
cmd="$(head -1 "$dir/command.txt" | sed 's/ > .*//') --set resume_checkpoint=$dir/latest.pt $*"
echo "$cmd > $dir/train.log 2>&1" >> "$dir/command.txt"
tmux new -d -s "$session" -c "$PWD" "ulimit -n 65536; export PYTHONPATH=$PWD WANDB_DISABLED=true OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1; $cmd > $dir/train.log 2>&1"
echo "resumed $name: $cmd"
