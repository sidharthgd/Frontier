# H200 + 1M-context redo of the Chameleon experiments (started 2026-11-24 00:30 UTC)

Box: Vast #62261089, `ssh -p 13056 root@185.17.198.197`, EPYC 7713 (128 allocated threads), ~500 GB RAM, 96 GB disk, $1.65/hr.
Code: branch `h200-1m-agents` (from Cyrus's `main`), repo at `/workspace/Frontier`, venv `/workspace/venv` (`.venv` symlink).
Not in git, copied by hand: `data/profiling/compute/h200/llama3.1-8b-1m/`, `data/processed_traces/tracelab_coding_1m.csv`.
Jobs run in tmux sessions on the box (`tmux ls`; a `keeper` session owns the server).

## Setup (done)
- Preset `h200_nemotron_1m` (frontier/rl/engine.py): H200 NVL + Nemotron-8B-UltraLong-1M, 56,345 KV blocks, max_model_len 890,000 (real vLLM settings, cache-on validated).
- TraceLab re-converted at the 1M limit (`tracelab_1m`): compaction at 880k instead of 120k; 588 compactions (was 28,579); prompt median 130k, p90 322k, p99 774k.
- H200 timing models built on the box (~40 min, cached in `cache/`). One episode takes ~3.5 min.
- Smoke result: at 8 / 10 agents FCFS is barely loaded (p99 TTFT 18 / 34 s, 0 / 2 preemptions of ~300 calls) -> loads must go up.
- FreeInference is NOT re-converted yet (raw trace not downloaded); TraceLab only for now.

## Step 1: capacity sweep (running) -> rl_runs/h200/capacity/
FCFS, plain (uncached) SJF, age200, rel30; TraceLab at 12/16/20/24/32/40 agents, synthetic at 18/30/45; 4 seeds.
Pick, from FCFS:
- LIGHT: highest TraceLab load with < 3% calls preempted (the analogue of A100 c8).
- PRESSURED: TraceLab load with ~10-20% preempted / falling hit rate (the analogue of A100 c10, 15.7% preempted).
- HEAVY: one level above PRESSURED, for the "does it hold across pressure" question.
- Synthetic training concurrency: the synthetic level whose FCFS preemption rate is closest to PRESSURED's; keep load_range [0.8, 1.25] around it.

## Step 2: heuristic sweep incl. cache-protecting SJF (18 seeds) -> rl_runs/h200/heuristics/
FCFS / SJF / age200 / rel30 (`concurrency_baseline_sweep.py`) and cpsjf (`ttft_tail_anatomy.py --policies cpsjf`) at LIGHT, PRESSURED, HEAVY.
Answers: does the A100 ordering of policies and the cpsjf mechanism hold with real context lengths, and how gains/tails scale with pressure. ~1-2 h.

## Step 3: PPO (pure, near-random init; Cyrus's best recipe) -> rl_runs/h200/ppo_*/
`configs/rl/h200_concurrency.yaml` with the chosen loads; reward e2e + 1 x ttft_hinge (tau = FCFS per-episode p95 TTFT), group 8, lr 5e-5, target_kl 0.02, imitation_coef 0, eval every 10 on 18 seeds, 60 iterations. Three runs side by side:
- ppo_pure_e2e_w1 (seed 0), ppo_pure_e2e_w1_seed1 (seed 1): the A100 headline, reproduced.
- ppo_pure_e2e_sq0.01: the squared tail term, for the tail end of the frontier.
Imitation+PPO is skipped (on the A100 it never left its teacher). ~8-10 h.

## Step 4: final 18-seed eval + head-to-head vs heuristics and cpsjf at all three loads.

## Rules
- Results synced to the laptop (`rl_runs/h200/`) regularly; checkpoints too.
- Keep Vast credit above ~$25 (about 15 h). A lapse restarts the box.
- Runs resume from `latest.pt` (`--set resume_checkpoint=...`).
