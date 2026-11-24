#!/bin/bash
# H200 1M-context experiments on the Vast box (rl_runs/h200/PLAN.md), each job in
# its own tmux session. Run from the Frontier repo root:
#   bash scripts/rl/run_h200.sh ppo <name> <ttft_hinge w> [--set ...]    pure PPO, near-random init
#   bash scripts/rl/run_h200.sh heuristics                                 FCFS/SJF/age200/rel30 + cpsjf, 18 seeds
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD WANDB_DISABLED=true OMP_NUM_THREADS=1
ulimit -n 65536
PY=.venv/bin/python
CONFIG=${CONFIG:-configs/rl/h200_concurrency.yaml}
ROOT=rl_runs/h200
SEEDS18="1000000 1000001 1000002 1000003 1000004 1000005 1000006 1000007 1000008 1000009 1000010 1000011 1000012 1000013 1000014 1000015 1000016 1000017"
ENV="ulimit -n 65536; export PYTHONPATH=$PWD WANDB_DISABLED=true OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1"

ppo() {  # <name> <w> [extra --set ...]: Cyrus's pure-PPO recipe (A100 headline run)
    local name=$1 w=$2; shift 2
    local out=$ROOT/$name
    mkdir -p $out
    local cmd="$PY -m frontier.rl.train --config $CONFIG --set output_dir=$out --set reward.weights.ttft_hinge=$w --set ppo.imitation_coef=0 --set reward.weights.e2e=1.0 --set reward.weights.e2e_normalized=0.0 --set group_size=8 --set num_workers=44 --set ppo.learning_rate=0.00005 --set ppo.target_kl=0.02 --set iterations=40 --set "eval_seeds=[1000000,1000001,1000002,1000003,1000004,1000005]" $* > $out/train.log 2>&1"
    echo "$cmd" > $out/command.txt
    tmux new -d -s "$name" -c "$PWD" "$ENV; $cmd"
    echo "started $name"
}

heuristics() {
    mkdir -p $ROOT/heuristics
    local a="$PY scripts/rl/concurrency_baseline_sweep.py --config $CONFIG --output-dir $ROOT/heuristics --baselines fcfs sjf_prefill_tokens sjf_prefill_tokens_age200 sjf_prefill_tokens_rel30 --eval-seeds $SEEDS18 --workers 12 > $ROOT/heuristics/run.log 2>&1"
    local b="$PY scripts/rl/ttft_tail_anatomy.py --config $CONFIG --output $ROOT/heuristics/cpsjf_requests.csv --seeds $SEEDS18 --policies fcfs cpsjf --workers 12 > $ROOT/heuristics/cpsjf.log 2>&1"
    echo "$a" > $ROOT/heuristics/command.txt; echo "$b" >> $ROOT/heuristics/command.txt
    tmux new -d -s heuristics -c "$PWD" "$ENV; $a"
    tmux new -d -s cpsjf -c "$PWD" "$ENV; $b"
    echo "started heuristics, cpsjf"
}

"$@"
