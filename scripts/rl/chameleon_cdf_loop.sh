#!/bin/bash
# Every INTERVAL seconds (default 4 h), CDF plots for each PPO run's current
# best.pt under rl_runs/chameleon/<run>/cdfs/best_iter<N>, skipped when that
# iteration was already plotted (scripts/rl/run_chameleon_long.sh cdfs). Run
# from the Frontier repo root, after baseline_cdfs:
#   tmux new -d -s cdfs "bash scripts/rl/chameleon_cdf_loop.sh"
cd "$(dirname "$0")/../.."
INTERVAL=${INTERVAL:-14400}
while true; do
    for best in rl_runs/chameleon/ppo_*/best.pt; do
        [ -f "$best" ] || continue
        run=$(dirname "$best")
        iter=$(.venv/bin/python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['iteration'])" "$best")
        out=$run/cdfs/best_iter$(printf %04d "$iter")
        [ -d "$out" ] && continue
        cp "$best" "$run/cdfs_checkpoint.pt"  # best.pt may be replaced mid-run
        bash scripts/rl/run_chameleon_long.sh cdfs "$run/cdfs_checkpoint.pt" "$out" \
            "$(basename "$run") iter $iter (argmax)"
    done
    sleep "$INTERVAL"
done
