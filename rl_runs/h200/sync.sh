#!/bin/bash
# Pull rl_runs/h200 from the Vast box every 20 min (small files + checkpoints; not sim/ or preload/).
cd "$(dirname "$0")/../.."
while true; do
  ssh -o ConnectTimeout=20 -o BatchMode=yes -p 15047 root@79.161.30.134 \
    'cd /workspace/Frontier && tar czf - --exclude="*/sim" --exclude="*/preload" --exclude="*/references" rl_runs/h200 2>/dev/null' \
    2>/dev/null | tar xzf - 2>/dev/null && echo "synced $(date)" >> rl_runs/h200/sync.log
  sleep 1200
done
