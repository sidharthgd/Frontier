# Long training run on Chameleon (2026-11-20 → lease end 2026-11-23 02:55 UTC)

Node: 2× EPYC 7763 (128 cores), 250 GB. Config `configs/rl/frontier_concurrency_long.yaml`
(same workloads, reward, model and teacher as `frontier_concurrency.yaml`; eval on 6 seeds).
Launcher `scripts/rl/run_chameleon_long.sh`. Everything lands in `rl_runs/chameleon/` and is
copied to the laptop every 15 min (`scripts/rl/sync_chameleon.sh`).

## Throughput facts that shape the plan

- Episodes take 80–125 s for synthetic (training) and about 150 s for eval workloads, roughly 3x a laptop's. Policy rollouts are slower than that.
- Each worker held about 7.5 GB, almost all of it predictor random forests. Workers now fork from
  one process that holds them (`share_predictors`): about 1.5 GB per worker plus 6.5 GB per run.
  With that, RAM allows about 4 runs × 28 workers.
- A PPO rollout needs its FCFS reference first (the hinge threshold is FCFS's p95 TTFT).
  `reference_cache_dir` + `prefetch_workers` cache references on disk (shared by all runs) and
  run the next iteration's during the current rollouts, so an iteration waits only on the slowest
  of its 24 rollouts plus the update. **Measured: about 6.3 min per iteration (about 340 s of rollouts plus about 36 s
  of update), so 250 iterations take about 26 h. The runs should finish around 2026-11-21 08:00 UTC,
  leaving about 43 h of lease.** Model inference is negligible (0.9 ms per decision); the wall time comes from the long-episode
  tail. More cores do not shorten an iteration. They only allow more runs at once.

## Step 0: baseline sweep (done, `baseline_sweep/`)

6 eval seeds per eval workload and 24 synthetic seeds, improvement over FCFS under
`e2e_normalized + w * ttft_hinge`:

| w | age200 eval mean (worst) | age200 synthetic | age100 eval / synth | plain SJF eval / synth |
|---|---|---|---|---|
| 0     | +0.032 (−0.034) | +0.110 | +0.033 / +0.122 | +0.069 / −0.161 |
| 0.01  | +0.009 (−0.073) | +0.056 | +0.004 / +0.035 | +0.032 / −0.549 |
| 0.02  | −0.011 (−0.107) | +0.005 | −0.021 / −0.046 | +0.001 / −0.913 |
| 0.03  | −0.027 (−0.134) | −0.043 | −0.043 / −0.125 | −0.025 / −1.261 |
| 0.05  | −0.056 (−0.175) | −0.137 | −0.082 / −0.274 | −0.070 / −1.921 |

- **Aging rate: keep 200.** age100 and age200 tie on the mean (eval +0.033 vs +0.032, synthetic +0.122 vs +0.110).
  age200 has the shorter tail (synthetic p99 TTFT 1.48x FCFS vs 1.84x; FreeInference c10 1.20x vs 1.33x), so it
  wins at every w > 0. age400/800 give up most of the mean gain. No significant change from the
  earlier sweeps.
- Plain SJF is the best on the *eval* mean at c8/c10 (+0.069). It doesn't starve at that load. It is
  terrible on the training workload (synthetic p99 TTFT 7x FCFS), though, so it is not a usable teacher.
- **The break-even ttft_hinge weight moved up.** age200 now breaks even with FCFS at w ≈ 0.015 (eval) and
  ≈ 0.022 (synthetic), not 0.006. At the current w = 0.01, copying the teacher already beats FCFS under
  the training reward, so the reward puts little pressure on the tail.
- The c8 workloads are close to FCFS-optimal: every policy is within ±3% on FreeInference c8.
  Most of the eval headroom is at c10.

## Step 1: imitation (done 05:30 UTC, `imitation/`)

DAgger with the age200 teacher: 96 teacher seeds, then 4 iterations × 64 mixture seeds (β 0.5 → 0.0625),
3 epochs per iteration, argmax evals at iterations 2 and 4. About 1–1.5 h.
All PPO runs start from **`imitation/latest.pt`** (the last iteration, trained on the most data).
`best.pt` is chosen under w = 0.01, so it is not reward-neutral. Imitation does not depend on the
reward, so this one checkpoint serves every fine-tune.

Result (argmax eval, 6 seeds × 4 workloads; teacher +0.011):

| checkpoint | vs FCFS | pair agreement | mean TTFT | p99 TTFT | TL c8 | TL c10 | FI c8 | FI c10 |
|---|---|---|---|---|---|---|---|---|
| iter 2 | +0.028 | 0.973 | 7.09 s | 69.6 s | +0.010 | +0.065 | −0.019 | +0.054 |
| iter 4 (`latest.pt`, PPO init) | +0.000 | 0.983 | 7.57 s | 79.9 s | +0.046 | −0.005 | −0.041 | +0.002 |

Iter 4 matches the teacher more closely, and the eval differences between the two are within the noise of
24 episodes. `checkpoints/iter_0002.pt` is kept in case it's useful as an alternative init.

## Steps 2–4: PPO, four runs side by side (started 05:30 UTC; 250 iterations, eval every 10)

Each run: 6 seeds × 4 rollouts per iteration (as in the laptop runs) on 24 workers + 3 prefetch
workers, about 50 GB.

| run | init | ttft_hinge w | why |
|---|---|---|---|
| `ppo_w0.01` | imitation | 0.01 | step 2: the current reward, just trained longer |
| `ppo_w0.02` | imitation | 0.02 | step 3: about the synthetic break-even, so the teacher's tail no longer pays off |
| `ppo_w0.04` | imitation | 0.04 | step 3: about 2x break-even, strong tail pressure; risk of drifting to FCFS |
| ~~`ppo_fcfs_w0.01`~~ | FCFS (pure PPO) | 0.01 | step 4: stopped, see below |
| `ppo_w0.01_lr3e-5` | imitation | 0.01 | `ppo_w0.01` with a 3x smaller step (lr 3e-5, target_kl 0.02) |

Pure PPO: no init checkpoint and `ppo.imitation_coef=0`. Its score head is zero-initialized, so the
argmax policy starts at exactly FCFS. The *sampled* policy does not: every waiting request is redrawn at every
decision, and a priority persists after admission and also sets preemption order, so a small chance of an
off-middle level per draw adds up to a lot of reordering. With the middle-level probability at 0.75, iteration-1
training scored −0.53 vs FCFS. At 0.95 (`abandoned_ppo_fcfs_w0.01_mid095/`, 20 iterations) training stayed at
−0.4 to −1.2 with no trend, entropy rose 0.23 → 0.32, KL stayed at about 0.005–0.01, and the argmax eval was exactly
FCFS (+0.000) at iterations 10 and 20. Exploration noise swamped any signal. At 0.99 (`abandoned_ppo_fcfs_w0.01_mid099/`, 10 iterations) training
still averaged −0.47 vs FCFS (−0.19 to −0.93), and argmax stayed exactly FCFS with KL ≈ 0.005.
**Finding (step 4): pure PPO from FCFS does not move off FCFS in this action space.** Every bit of
per-decision sampling noise costs far more than PPO's small policy changes win back, so the gradient stays
dominated by that noise. A pure-PPO run would need exploration that perturbs only at admission (or similar
structured exploration), not a different hinge weight. Stopped 2026-11-20 ~09:20 UTC so its slot could go to the
learning-rate variant below.

### Step-size variant (started 2026-11-20 ~09:20 UTC)

By iteration 30 the imitation runs had *declined* from their start, in training as well as eval
(`ppo_w0.01` 10-iteration train means −0.05 → −0.15 → −0.20; evals +0.019 / −0.028 / −0.049), with clip fraction
0.27–0.33, gradient norm about 8–9 against a clip of 0.5, and KL early stopping in 90% of the first 10 iterations.
Entropy and the value fit were fine. The suspected cause is steps too large for noisy 24-rollout advantages,
starting from a policy that already matches its teacher. `ppo_w0.01_lr3e-5`: same as `ppo_w0.01` but with
`ppo.learning_rate=3e-5` (was 1e-4) and `ppo.target_kl=0.02` (was 0.05). `ppo_w0.01` keeps running as the control. A higher w would make FCFS
a local optimum, as noted. Next candidate if it stalls: w = 0.003–0.005.

Note: the imitation runs keep `ppo.imitation_coef: 0.1` (an anchor toward age200, whose tail is
what we're trying to beat). This keeps them comparable with the earlier runs. If `ppo_w0.02`/`ppo_w0.04`
plateau with the tail still long, the next thing to try is the same w with the anchor at 0 or 0.03.

## Outputs and checkpoints

- `<run>/metrics.jsonl`, `train.log`, `latest.pt`, `best.pt` (argmax eval), `checkpoints/iter_NNNN.pt`
  every 10 iterations.
- CDFs (TTFT, TBT, e2e, batch size; 6 eval seeds pooled, argmax) against FCFS / age200 / plain SJF:
  `cdfs_baselines/` once, then `<run>/cdfs/best_iterNNNN/` every 4 h for each run whose best.pt changed
  (`scripts/rl/chameleon_cdf_loop.sh`, 4 workers, about 20 min each).
- To continue a run past 250 iterations, or after a crash: rerun its `command.txt` with
  `--set resume_checkpoint=<run>/latest.pt --set iterations=<N>`. This restores the model, optimizer, return
  normalization, iteration count and best eval. A killed run loses at most 10 iterations.

## Log

- **2026-11-20 ~09:40 UTC, memory leak.** Each env kept every reference episode it had ever loaded in memory
  (prefetched ones with observations), so workers grew by about 0.09 GB/h (about 10 GB/h across 108) and free memory
  fell to 16 GB. Fix: with a disk cache, memory keeps only the 8 most recent references, as slim copies
  (`frontier/rl/env.py`, tested). Each run resumed from its next checkpoint on the fixed code.
- **2026-11-20 10:11 UTC, all runs killed by the resume script.** It found the trainer with
  `pgrep -f output_dir=…`, which also matched the tmux server (a tmux server keeps the command line of the
  session that started it), so killing it ended every session. Lost: about 4 iterations of `ppo_w0.02` and about 5 of
  `ppo_w0.01_lr3e-5`; the other two were at their checkpoints. Fixed: the pattern now matches only the trainer's python
  process, and a `keeper` session (`sleep infinity`) now owns the tmux server. All four runs resumed at
  10:20 UTC (`resume_chameleon_run.sh`; old logs in `train.<n>.log`, metrics past the checkpoint in
  `metrics.discarded.jsonl`).

## Status at iteration 50 (argmax evals every 10 iterations, each scored under the run's own w)

| run | evals at iterations 10–50 | mean | p99 TTFT, last eval (TL c8 / TL c10 / FI c8 / FI c10) |
|---|---|---|---|
| FCFS (seed mean) | | | 43 / 78 / 49 / 80 |
| `ppo_w0.01` | +0.019 −0.028 −0.049 +0.012 +0.011 | −0.007 | 35 / 122 / 50 / 127 |
| `ppo_w0.02` | +0.007 −0.024 +0.001 −0.021 −0.009 | −0.009 | 37 / 105 / 47 / 128 |
| `ppo_w0.04` | −0.082 −0.041 −0.054 −0.088 −0.069 | −0.067 | 36 / 123 / 49 / 106 |
| `ppo_w0.01_lr3e-5` | −0.029 (iteration 10) | | 36 / 107 / 48 / 132 |

No run has made clear progress in 50 iterations. Eval-to-eval swings of about 0.05 are larger than any trend. The c10
p99 TTFTs stay at about 1.3–1.65x FCFS whatever the hinge weight. Even w = 0.04 does not pull the tail down.
- **2026-11-20 12:20 UTC: CDFs of each run's best.pt** (all early checkpoints, 6 seeds pooled), c10 p99 TTFT TL / FI:
  FCFS 93 / 98, age200 128 / 132, imitation 135 / 164, `ppo_w0.01`@10 **119 / 114**, `ppo_w0.02`@10 117 / 135,
  `ppo_w0.04`@20 134 / 122, `ppo_w0.01_lr3e-5`@10 126 / 155. Mean TTFT stays at the teacher's level (about 25% below FCFS).
  The early PPO steps do shorten the tail relative to imitation, but no later checkpoint has scored higher.
- **2026-11-20 12:45 UTC: stopped `ppo_w0.01`** (lr 1e-4) at iteration 70. Its evals went from +0.019 (10) to −0.073 / −0.063
  (60 / 70), while `ppo_w0.01_lr3e-5` went −0.029 → −0.006 → +0.018 (10–30). Its slot went to **`ppo_w0.02_lr3e-5`** (imitation init,
  w = 0.02, lr 3e-5, target_kl 0.02): the smaller step with the stronger tail weight.
- **2026-11-20 ~14:00 UTC: eval seeds 6 → 18** (1000000–1000017, 72 eval episodes; noise about 1.7x lower). After
  about 90 iterations no run had separated from the others: eval-to-eval swings of ±0.05 were as large as any difference.
  `ppo_w0.01_lr3e-5`'s early rise (−0.029 → +0.018) fell back to −0.015 at iteration 40, and `ppo_w0.02` (lr 1e-4) ranged
  −0.069 to +0.007. Each run resumes at its next checkpoint (`ppo_w0.02_lr3e-5` at 30, `ppo_w0.01_lr3e-5` at 60, `ppo_w0.02`
  and `ppo_w0.04` at 100) with `eval_seeds` overridden. best.pt is then chosen among 18-seed evals only; the earlier
  best checkpoints stay in `checkpoints/`. The first launch after a resume now also prefetches the eval FCFS references.
- **2026-11-20 ~16:00 UTC: memory ceiling and `ppo_w0.02` stopped.** Even with the reference leak fixed, each worker's
  private memory grows from about 1.45 to about 1.85 GB as its prediction cache fills, so four runs take about 230 GB and
  leave no room for evaluation jobs. With the user's OK, stopped `ppo_w0.02` (lr 1e-4) at iteration 100. Its small-step
  version `ppo_w0.02_lr3e-5` continues. Freed: about 57 GB.
- **18-seed checkpoint comparison** (`checkpoint_eval18/`; `scripts/rl/concurrency_baseline_sweep.py` with `ckpt:` entries,
  argmax, FCFS from the reference cache, rescored for any w): FCFS, age200, imitation, the early best checkpoints
  (`ppo_w0.01`@10, `ppo_w0.02`@10, `ppo_w0.04`@20) and the late ones of the stopped runs (`ppo_w0.01`@70, `ppo_w0.02`@100).
- **2026-11-20 16:12–18:45 UTC: thread oversubscription.** The checkpoint-comparison workers ran policy inference with
  torch's default thread count (all 128 cores), so 16 workers ran 2,048 threads and the load average reached about 1,700.
  Every trainer slowed down and the comparison finished no tasks. Fixed: `concurrency_baseline_sweep.py` workers call
  `torch.set_num_threads(1)` (as rollout workers do) and the script resumes from finished tasks. Restarted 18:45 UTC
  (the monitor now also reports a load above 300).

## Round 2 (2026-11-21 ~02:15 UTC)

**Comparison of 18-seed checkpoints** (`checkpoint_eval18/`): every learned policy lands within noise of age200. At c10,
mean TTFT is −20 to −24% vs FCFS, p99 TTFT +22 to +46%, mean e2e −5 to −9%. No PPO checkpoint shortened the c10 tail
below the teacher's, and there was no consistent early-vs-late trend.

**Wait-cap sweep** (`wait_cap_sweep/`, `sjf_prefill_tokens[_age200]_cap{10,20,40}`): at c10 the mean and tail trade
off almost one for one. A p99 TTFT near FCFS costs most of the mean-TTFT gain and all of the mean-e2e gain
(cap 40: −10% mean / +4% p99 TTFT on TL c10). On synthetic, cap 40 improves both (−11% / −8%).

**Tail anatomy** (`ttft_anatomy/`, top 2% TTFT, 6 seeds): FCFS's tail is preemption victims (65–78% restarted, often
large and mostly cached). Under age200 and RL, the c10 tail is **starved large uncached prompts** (median 74–102k
uncached tokens, 6–24% cached, queueing 118–145 s against 66–76 s under FCFS). At c8 they queue about as long as under
FCFS, but restarts halve, so the tail shrinks. Absolute aging (200 tok/s) needs about 375 s to lift a 75k uncached prompt above a fresh
small one. The observation already has wait time and uncached tokens, so the gap is credit assignment / reward, not features.

**Reward rescale for plain `e2e`**: `c_e2e` is about 60–80x `c_ttft_hinge`. Under `e2e + w * ttft_hinge`, age200 is +0.015 on
synthetic at w = 1 and −0.012 at w = 2 (break-even about 1.5). Above about 2, no heuristic in either sweep beats FCFS by more than
+0.003 on synthetic.

**Stopped** `ppo_w0.01_lr3e-5` (iteration 149), `ppo_w0.02_lr3e-5` (119), `ppo_w0.04` (188). Data and checkpoints kept.

**Running** (6 seeds × **8** rollouts = 48 workers + 3 prefetch each, lr 5e-5, target_kl 0.02, 18-seed eval every 10):

| run | start | reward | imitation anchor |
|---|---|---|---|
| `ppo_pure_e2e_w1` | untrained model, middle-level probability 0.25 (near random) | `e2e` + 1 × ttft_hinge | 0 |
| `ppo_imit_e2e_w2` | imitation (age200) `latest.pt` | `e2e` + 2 × ttft_hinge | 0 |

**Relative-aging sweep** (`relative_aging_sweep/`, `sjf_prefill_tokens_rel{10,30,100,300}`): rank by size / (1 + wait/τ),
so a request's effective size halves after τ seconds whatever its size. Tests whether size-aware aging serves the starved
class without giving up the mean. CDF loop paused until it finishes (memory).
- **2026-11-21 ~05:45 UTC: `ppo_pure_e2e_w1` crashed at iteration 28** in `group_advantages` (`IndexError`). A rollout with
  no transitions (a light synthetic draw where nothing ever waited) has no return to read, and with 8 rollouts per group
  that case finally came up. Fixed in `frontier/rl/ppo.py`: rollouts without decisions are left out of the
  leave-one-out baseline (zero baseline if no peer has any), with a test. Resumed from iteration 20 (7 iterations lost);
  `ppo_imit_e2e_w2` resumed onto the fix from its iteration-40 checkpoint (1 lost).
- **Results so far** (18-seed argmax eval; e2e + w × ttft_hinge): pure PPO +0.038 (iteration 10), **+0.046** (20), with c10 mean e2e
  −9 to −12% vs FCFS (age200 −7%, rel30 −3 to −4%) but c10 p99 TTFT +25 to +45%. Imitation + PPO +0.027 / +0.016 / +0.023 (10 / 20 / 30),
  level with its teacher (+0.023 under its reward).
- **Relative-aging sweep** (`relative_aging_sweep/`): rel30 (size / (1 + wait/30 s)) is the best TTFT heuristic. At c10 its mean TTFT
  is −16 to −18% with p99 TTFT +3 to +13% (age200 −16 to −21% / +11 to +21%), and on synthetic −22% / +5% (age200 −21% / +24%). The
  cost is a smaller mean-e2e gain (−3% vs −5%). It confirms the tail diagnosis: size-blind aging starves big uncached prompts.
- **2026-11-21 ~06:45 UTC: stopped `ppo_imit_e2e_w2`** at iteration 49 (evals +0.027 / +0.016 / +0.023 / iteration 40 in metrics; level
  with its teacher). Its slot went to **`ppo_pure_e2e_w2`**: pure PPO as `ppo_pure_e2e_w1` but `e2e` + **2** × ttft_hinge.
  Why 2 rather than about 1.4: at w = 1 pure PPO traded the TTFT tail for mean e2e (c10 p99 TTFT +25 to +45%). At w = 2 the best
  heuristic (rel30) is still above FCFS on synthetic (+0.009) while tail-heavy orderings fall below it (age200 −0.004, plain
  SJF −0.39), so tail-friendly orderings rank higher. At about w = 4 every heuristic is at or below FCFS (risk of collapsing to FCFS).
  It matches the stopped imitation run's reward (pure vs imitation at the same reward) and brackets w = 1.
- **2026-11-21 ~16:00 UTC: stopped `ppo_pure_e2e_w2`** at iteration 80 (evals +0.028 / +0.023 / +0.029 / +0.018 / +0.010 / +0.017 / +0.024 / +0.007;
  best at iteration 30). Averaged over evals, its c10 p99 TTFT (+39 / +47%) was no better than w = 1's (+40 / +38%), with about the same e2e.
  Doubling the hinge did not move the tail. Its slot went to **`ppo_pure_e2e_w6`** (pure PPO, `e2e` + 6 × ttft_hinge; same
  batch and lr) to test whether a much stronger tail term pulls PPO toward rel30-like tails, at the risk of collapsing to FCFS.
- `ppo_pure_e2e_w1` at iteration 100: evals +0.038 / +0.046 / +0.040 / +0.040 / +0.038 / +0.032 / +0.040 / +0.030 / +0.038 / +0.025, past its peak
  (best: iteration 20). Stopped at iteration 105 with the user's OK; replaced by **`ppo_pure_e2e_w1_seed1`** (identical but `seed=1`:
  different initial weights and training draws, same 18 eval seeds), to test whether the e2e result reproduces.
- **2026-11-22 ~02:20 UTC: retrospective and switch to a squared tail term** (`retrospective_quantiles.txt`). Pooled TTFT/e2e p50/p99/p100 from
  the CDF samples of every non-abandoned run's best checkpoint: pure PPO has the lowest c10 medians (TL TTFT p50 1.8–2.2 s, e2e p50
  30–31 s, against 33–35 s elsewhere), but on p99/p100 TTFT it is no better than the imitation runs. Every SJF-like policy starves
  about 1–1.5% of requests past FCFS's worst wait (17–28 of 1,800 at c10; rel30 16 / 7). Pure PPO's worst waits are the longest (336–730 s
  against FCFS 121–137, rel30 210–214). Plain SJF's p99 TTFT is about FCFS's, but its p100 is 2,023 s. The linear hinge does not stop a few
  long starvations.
  Stopped `ppo_pure_e2e_w1_seed1` (iteration 59; evals +0.038 / +0.029 / +0.040 / +0.041 / +0.018, so the e2e result reproduces) and `ppo_pure_e2e_w6`
  (iteration 63; −0.016 / +0.008 / +0.001 / +0.020 / +0.017 / +0.003). Started **`ppo_pure_e2e_sq0.01`** and **`ppo_pure_e2e_sq0.02`**: pure PPO, `e2e` +
  w × `starvation` (squared hinge Σ max(0, TTFT − τ)², τ = FCFS per-episode p95, q kept at 0.95 for consistency), linear hinge off.
  Rescoring the sweeps under this reward: at w = 0.01–0.02 plain SJF collapses (synthetic −4 to −9), age200 falls below FCFS on synthetic, and rel30
  is the best heuristic (+0.006 / +0.002 synthetic; +0.016 / +0.014 eval). Lease extended to about 2026-11-23 16:55 UTC.
- **2026-11-22 ~08:00 UTC: squared-hinge results so far.** (c10 TL / FI, 6-seed CDF pools.) sq0.02 @10: TTFT p100 170 / 157 s, 10 / 5 requests
  past FCFS's max (best worst case of any non-FCFS policy, rel30 210 / 214), but no e2e gain. sq0.02 @30: e2e −4 / −10%, p100 384 / 311 s.
  sq0.01 @20: e2e on a par with linear w = 1 (FI mean 55.3 s), p100 366 / 366 s. So every policy so far trades e2e for starvation; none beats rel30 on
  both. Stopped `ppo_pure_e2e_sq0.01` at iteration 34 (evals +0.001 / +0.023 / −0.069; best @20); replaced by **`ppo_pure_e2e_sq0.05`** (same, starvation weight 0.05;
  at that weight every heuristic is at or below FCFS on synthetic, so collapse to FCFS is possible).
- **2026-11-22 19:24–21:07 UTC: final 18-seed evaluation** (`final_eval18/`). All training stopped first (`ppo_pure_e2e_sq0.02` at iteration ~135,
  `ppo_pure_e2e_sq0.05` at ~85). 33 policies (4 heuristics plus every run's `best.pt` and `latest.pt`) × 4 eval workloads × 18 seeds,
  per call (`scripts/rl/ttft_tail_anatomy.py`, 100 workers). Tables: `final_eval18/tables.txt`, `headtohead.txt`. Write-up: **`RESULTS.md`**.
  Pure PPO e2e + 1×hinge is the fastest policy on e2e at c10: TL −13% median / −10% p99 / −12% mean vs FCFS, and −4 to −8% vs age200 / rel30
  (paired intervals exclude 0). It beats plain SJF on the mean and ties it on the median, with a worst TTFT of 447–484 s against SJF's 844–2,023 s.
  The mechanism is cache locality: hit rate 78% vs 75% (FCFS) and 76–77% (heuristics). At c8 everything is within ±2% of FCFS.
