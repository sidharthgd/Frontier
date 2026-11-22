#!/bin/bash
# PPO runs on the Chameleon node (configs/rl/frontier_concurrency_long.yaml),
# side by side, each in its own tmux session. Run from the Frontier repo root
# after imitation (rl_runs/chameleon/imitation) has finished:
#   bash scripts/rl/run_chameleon_long.sh ppo <name> <init: imitation|fcfs|random> <ttft_hinge w> [--set ...]
#   bash scripts/rl/run_chameleon_long.sh cdfs <checkpoint> <output dir> <label>
#   bash scripts/rl/run_chameleon_long.sh baseline_cdfs
# "fcfs" init is pure PPO: no imitation checkpoint and no imitation anchor
# (ppo.imitation_coef=0), with the untrained model's middle-level probability
# raised to 0.99 so its sampled priorities start close to FCFS (one level for
# everything; ties go by arrival). "random" is pure PPO from the untrained
# model at the config's middle-level probability (0.25: nearly uniform levels).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD WANDB_DISABLED=true OMP_NUM_THREADS=8
# forkserver pools pass a few fds per worker start; 1024 runs out at ~60 workers
ulimit -n 65536
PY=.venv/bin/python
CONFIG=configs/rl/frontier_concurrency_long.yaml
ROOT=rl_runs/chameleon
IMITATION=$ROOT/imitation/latest.pt
SEEDS="1000000 1000001 1000002 1000003 1000004 1000005"
CDF_WORKLOADS=configs/rl/frontier_concurrency_cdf_workloads.yaml
BASELINES="fcfs sjf_prefill_tokens_age200 sjf_prefill_tokens_rel30 sjf_prefill_tokens"

ppo() {  # <name> <imitation|fcfs> <w> [extra overrides]
    local name=$1 init=$2 w=$3; shift 3
    local out=$ROOT/$name
    local sets=(--set output_dir=$out --set reward.weights.ttft_hinge=$w)
    if [ "$init" = imitation ]; then
        sets+=(--set init_checkpoint=$IMITATION)
    elif [ "$init" = random ]; then
        sets+=(--set ppo.imitation_coef=0)
    else
        sets+=(--set ppo.imitation_coef=0 --set model.init_middle_level_prob=0.99)
    fi
    mkdir -p $out
    local cmd="$PY -m frontier.rl.train --config $CONFIG ${sets[*]} $* > $out/train.log 2>&1"
    echo "$cmd" > $out/command.txt
    # a tmux session inherits the tmux server's environment and limits, not
    # this script's, so set them in the session's own command
    tmux new -d -s "$name" -c "$PWD" "ulimit -n 65536; export PYTHONPATH=$PWD WANDB_DISABLED=true OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1; $cmd"
    echo "started $name: $cmd"
}

cdfs() {  # <checkpoint> <output dir> <label>
    mkdir -p "$2/data"
    # baselines are simulated once (baseline_cdfs) and shared
    cp -n $ROOT/cdfs_baselines/data/*.npz "$2/data/" 2>/dev/null || true
    $PY -m frontier.rl.plot_cdfs --config $CONFIG --workloads $CDF_WORKLOADS \
        --checkpoint "$1" --output-dir "$2" --workers 4 --seeds $SEEDS \
        --policies $BASELINES policy_argmax --policy-label "$3" > "$2/run.log" 2>&1 \
        && echo "cdfs $2 done $(date)" || echo "cdfs $2 FAILED $(date)"
}

baseline_cdfs() {
    # baselines only; plot_cdfs still needs a --checkpoint path, unused here
    mkdir -p $ROOT/cdfs_baselines
    $PY -m frontier.rl.plot_cdfs --config $CONFIG --workloads $CDF_WORKLOADS \
        --checkpoint unused --output-dir $ROOT/cdfs_baselines --workers 12 \
        --seeds $SEEDS --policies $BASELINES \
        > $ROOT/cdfs_baselines/run.log 2>&1 \
        && echo "baseline cdfs done $(date)" || echo "baseline cdfs FAILED $(date)"
}

"$@"
