#!/bin/zsh
# Copy the Chameleon node's rl_runs/chameleon/ (logs, metrics, checkpoints,
# CDF plots and their sample data) to this laptop every 15 minutes, skipping
# the simulator scratch directories. Run from the Frontier repo root:
#   nohup zsh scripts/rl/sync_chameleon.sh > rl_runs/chameleon_sync.log 2>&1 &
# Stop it with: pkill -f sync_chameleon.sh
cd "$(dirname "$0")/../.."
HOST=${CHAMELEON_HOST:-cc@129.114.108.24}
INTERVAL=${SYNC_INTERVAL:-900}
mkdir -p rl_runs/chameleon
while true; do
    if rsync -az --timeout=120 \
        --exclude 'sim/' --exclude 'preload/' --exclude 'simulator_output/' --exclude 'references/' \
        "$HOST":~/Frontier/rl_runs/chameleon/ rl_runs/chameleon/; then
        echo "synced $(date)"
    else
        echo "sync FAILED $(date)"
    fi
    sleep $INTERVAL
done
