#!/bin/bash
# Unattended chain on the Vast box: wait for the capacity sweep, then start the
# heuristic sweep (incl. cache-protecting SJF) and three pure-PPO runs side by side.
cd "$(dirname "$0")/../.."
LOG=rl_runs/h200/chain.log
echo "waiting for capacity sweep $(date -u)" >> $LOG
while pgrep -f "concurrency_baseline_sweep.py --config configs/rl/h200_capacity" > /dev/null; do sleep 30; done
echo "capacity sweep done $(date -u)" >> $LOG
bash scripts/rl/run_h200.sh heuristics >> $LOG 2>&1
bash scripts/rl/run_h200.sh ppo ppo_pure_e2e_w1 1.0 >> $LOG 2>&1
bash scripts/rl/run_h200.sh ppo ppo_pure_e2e_w1_seed1 1.0 --set seed=1 >> $LOG 2>&1
bash scripts/rl/run_h200.sh ppo ppo_pure_e2e_sq0.01 0.0 --set reward.weights.starvation=0.01 >> $LOG 2>&1
echo "launched heuristics + 3 PPO runs $(date -u)" >> $LOG
