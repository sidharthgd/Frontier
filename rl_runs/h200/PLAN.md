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

## Corrections and changes (2026-11-24 ~07:00 UTC)
- **Bug found and fixed:** the synthetic training sessions (`synthetic:random`) still compacted at 120k (the generator's own default), so the first three PPO runs trained on ~128k-capped sessions while evaluating on 1M contexts. Fixed in `frontier/trace_prep/sessions.py` (sessions compact 10k below max_tokens when max_tokens > 131072); synthetic prompts p90 91k -> 211k, p99 120k -> 322k. Those runs were stopped at iteration 19-20 and moved to `abandoned_synth128k/`. The capacity sweep's synthetic rows (`capacity/`) are from the old generator and are superseded by `capacity_synth1m/`. TraceLab results (capacity, heuristics, cpsjf) are unaffected.
- **Moved boxes** to Vast #62308823 (`ssh -p 15047 root@79.161.30.134`): Threadripper PRO 7995WX, 192 threads, ~5.4 GHz, 258 GB limit, $1.46/hr. H200 timing models rebuilt there (box-to-box copy was blocked). Old box (#62261089) results pulled to the laptop; it can be destroyed.
- **PPO settings now:** 2 runs (ppo_pure_e2e_w1 seed 0 and seed 1), 44 workers each (memory: 258 GB limit), 40 iterations, evals every 10 on 6 seeds (1000000-1000005), full 18-seed comparison afterwards. The squared-tail run (sq0.01) waits until these finish.
- **Training concurrency:** picked from `capacity_synth1m/` (FCFS preemption closest to TraceLab c16's).

## 2026-11-24 07:30 UTC: PPO launched on the new box
- Synthetic capacity check (1M compaction, 6 seeds, FCFS): calls preempted 14% / 23% / 18% / 25% at 12 / 18 / 24 / 30 agents (final, 6 seeds; mean e2e 62 / 105 / 135 / 172 s; age200 cuts mean e2e 16% / 24% / 32% / 30%). Pressure is set mostly by each seed's sampled session sizes (seeds range 0% to 50%), not by the agent count. Training uses 18 agents (commit e9e7ed65).
- Launched 07:14 UTC: ppo_pure_e2e_w1 (seed 0) and ppo_pure_e2e_w1_seed1, 44 workers each, 100 iterations, eval every 10 on 6 seeds.
- Iteration 1: rollout 476 s + update 23 s = 8.3 min. Memory ~200 GB of 258 GB, stable.
- Estimate: ~13-14 h for 100 iterations (plus evals), so about iteration 50-55 by 15:00 UTC (10 AM EST), done ~22:00 UTC.

## 2026-11-24 14:30 UTC: first positive eval (run 0, iteration 20)
Eval seeds 1000000-05 (argmax), vs FCFS / age200 on the same seeds:
| load | PPO mean e2e | FCFS | age200 | PPO p99 TTFT | FCFS | age200 |
|---|---|---|---|---|---|---|
| c12 | 20.6 | 20.9 | 20.2 | 19.8 | 20.0 | 18.0 |
| c16 | 49.5 (-8.4%) | 54.0 | 47.4 | 76.7 | 60.5 | 86.5 |
| c20 | 108.6 (-5.2%) | 114.6 | 91.1 | 215.7 | 98.1 | 231.1 |
Iteration 10 had been -5% / -21% (c16 / c20) on return. Iterations average ~20-25 min (straggler episodes up to 2 h);
run 1 is ~5 iterations behind run 0.

## 2026-11-24 22:45 UTC: evals through iteration 40 (6 seeds, mean e2e / p99 TTFT, s)
FCFS on the same seeds: c12 20.9 / 20.0, c16 54.0 / 60.5, c20 114.6 / 98.1.
| run @iter | c12 | c16 | c20 |
|---|---|---|---|
| seed0 @10 | 20.5 / 18.6 | 57.7 / 72.8 | 138.2 / 143.9 |
| seed0 @20 | 20.6 / 19.8 | 49.5 / 76.7 | 108.6 / 215.7 |
| seed0 @30 | 20.8 / 16.8 | 51.0 / 69.5 | 123.8 / 154.1 |
| seed0 @40 | 20.6 / 18.8 | 50.3 / 104.4 | 127.7 / 222.3 |
| seed1 @10 | 22.2 / 17.2 | 68.1 / 89.8 | 152.1 / 163.9 |
| seed1 @20 | 20.3 / 19.6 | 60.4 / 198.5 | 129.3 / 199.2 |
| seed1 @30 | 21.1 / 18.5 | 65.6 / 106.7 | 160.7 / 215.5 |
Seed 0 peaked at 20 (beat FCFS at c16/c20), then regressed at c20. Seed 1 never beat FCFS at c16/c20.
