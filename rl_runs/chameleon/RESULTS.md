# RL scheduling on closed-loop agent workloads: results

Final evaluation, 2026-11-22. 18 eval seeds × 4 eval workloads, 33 policies: 4 heuristics, plus `best.pt` **and**
`latest.pt` of every run. That's 5,400–5,472 scored calls per policy and workload.
- Data: `final_eval18/requests.csv.gz`, one row per scored call.
- Tables: `final_eval18/tables.txt` (from `final_eval_analysis.py`) and `final_eval18/headtohead.txt` (from `final_eval_headtohead.py`).
- Checkpoint iterations: `final_eval18/checkpoint_iterations.txt`.
- Chronological log of every run and decision: `PLAN.md`.

Bracketed ranges are 90% seed-bootstrap intervals: seeds resampled with replacement, and paired, so both
policies see the same seeds.

## Setup in one paragraph

Frontier simulator (vLLM v1 scheduler, H200 / Llama-2-7B, chunked prefill, prefix caching). The policy
assigns a priority level to every waiting call at each scheduling step. Workloads are closed-loop: N
agents, each running sessions back to back. A call's prompt is mostly the cached prefix of the agent's
previous call, and the agent thinks between calls. Each agent's scored work is a fixed quota: its first
30 calls after a 600 s warm-up at c10, 38 at c8.
- **Training:** synthetic random sessions (14–22 agents).
- **Evaluation:** real traces TraceLab (TL) and FreeInference (FI) at 8 and 10 agents (c8 / c10).
- **PPO:** group-relative advantages.
- **Reward:** `e2e` (per-call e2e, normalized by FCFS on the same seed) plus a TTFT tail term, either the linear
  `ttft_hinge` Σ max(0, TTFT − τ) or the squared `starvation` Σ max(0, TTFT − τ)², with τ = FCFS's per-episode
  p95 TTFT.
- **Heuristics:**
  - FCFS.
  - plain SJF on prefill tokens.
  - **age200**: SJF with absolute aging, 200 tokens/s of waiting. This was the imitation teacher.
  - **rel30**: SJF with relative aging, size / (1 + wait / 30 s).

Run names:
- *pure*: PPO from a near-random init with no imitation.
- *imit+PPO*: PPO from the age200 imitation student.
- *e2e+1h*: reward `e2e` + 1 × ttft_hinge.
- *0.02sq*: reward `e2e` + 0.02 × starvation.
- *@N*: checkpoint iteration.

## Headline

**At the heavier load (c10), pure PPO (`e2e` + 1 × ttft_hinge) has the lowest end-to-end latency of any
policy, learned or heuristic, and the result reproduces across two training seeds.**
- **vs FCFS on TraceLab c10:** −13% median e2e, −10% p99, −12% mean and −10% median time per agent.
- **vs the aging heuristics:** it beats age200 and rel30 on mean and median e2e with intervals that exclude zero.
- **vs plain SJF:** it matches plain SJF's median and beats its mean. Plain SJF starves calls for up to 2,023 s, while PPO's worst wait is 447–484 s.

For agentic workloads, e2e is the metric that matters: nobody watches every agent's tokens stream. What
counts is how long each step, and so the whole agent, takes.

**The cost is the extreme TTFT tail.**
- Median and mean TTFT are *better* than FCFS's: TL c10 TTFT p50 1.0 s against 1.5 s.
- About 0.5–0.8% of calls, the large mostly-uncached prompts, wait longer than FCFS's worst wait: 28–44 of 5,400.
- PPO's worst TTFT is 447–1,550 s, against FCFS's 139–149 s and rel30's 210–229 s.

**At light load (c8), FCFS is close to optimal.** Every policy is within about ±2% of it on median e2e.

### c10: per-call e2e (change vs FCFS) and per-agent completion time

| policy | TL e2e p50 | TL e2e p99 | TL e2e mean | TL agent p50 | FI e2e p50 | FI e2e p99 | FI e2e mean | FI agent p50 |
|---|---|---|---|---|---|---|---|---|
| FCFS (absolute) | 31.3 s | 633 s | 71.1 s | 2,488 s | 18.2 s | 508 s | 53.2 s | 1,884 s |
| plain SJF | −11% | −5% | −8% | −10% | −1% | −8% | −5% | −2% |
| age200 | −9% | −8% | −7% | −8% | −0% | −10% | −7% | −1% |
| rel30 | −7% | −5% | −4% | −7% | +1% | −1% | −3% | −2% |
| imitation (age200 student) @4 | −9% | −5% | −5% | −8% | −2% | −8% | −7% | −1% |
| imit+PPO e2e+2h @10 | −9% | −9% | −7% | −8% | −2% | −12% | −9% | −3% |
| **pure e2e+1h @20** | **−13%** | **−10%** | **−12%** | **−10%** | −1% | **−15%** | −9% | −3% |
| **pure e2e+1h, seed 1 @40** | **−13%** | −7% | −11% | −10% | −4% | −10% | −9% | **−5%** |
| pure e2e+2h @30 | −13% | −6% | −9% | −10% | −4% | −11% | **−11%** | −3% |
| pure e2e+6h @40 | −12% | −7% | −11% | −10% | −1% | −8% | −8% | −1% |
| pure e2e+0.01sq @20 | −12% | −6% | −10% | −9% | −4% | −14% | −10% | −3% |
| pure e2e+0.02sq @10 | −1% | −1% | +2% | −1% | −1% | −3% | −2% | −1% |
| pure e2e+0.02sq @40 | −7% | −5% | −5% | −5% | **−5%** | **−16%** | −10% | −3% |

Versus FCFS:
- TL c10, pure e2e+1h @20: p50 [−20, −8], p99 [−14, −6].
- TL c10, heuristics: p50 SJF [−18, −7], age200 [−15, −3], rel30 [−13, −2].
- FI c10: medians are within a few percent of FCFS for everyone (the trace has many short, fast calls). PPO's gain there is in the mean and p99.

### Head-to-head: PPO against each heuristic, paired by seed (c10)

Change in PPO's latency relative to the heuristic. Negative means PPO is faster.

| PPO checkpoint vs | TL e2e p50 | TL e2e mean | TL agent p50 | FI e2e p50 | FI e2e p99 | FI e2e mean |
|---|---|---|---|---|---|---|
| pure e2e+1h @20 vs plain SJF | −2.7% [−5, +1] | **−4.5% [−6, −3]** | −0.6% [−9, +1] | −0.1% [−4, +4] | **−7.8% [−18, −1]** | **−4.3% [−8, −1]** |
| pure e2e+1h @20 vs age200 | **−4.4% [−8, −2]** | **−5.5% [−7, −4]** | **−3.1% [−9, −1]** | −0.7% [−3, +3] | −5.5% [−12, +1] | **−2.8% [−5, −1]** |
| pure e2e+1h @20 vs rel30 | **−6.6% [−9, −4]** | **−8.0% [−10, −7]** | **−3.3% [−10, −1]** | −1.9% [−4, +2] | **−14.2% [−18, −6]** | **−6.4% [−9, −4]** |
| pure e2e+1h s1 @40 vs age200 | **−4.2% [−9, −2]** | **−5.1% [−7, −3]** | −2.1% [−8, +0] | **−3.8% [−7, −1]** | +0.0% [−6, +9] | −2.3% [−5, +1] |
| pure e2e+0.02sq @40 vs age200 | +1.8% [−2, +5] | +2.2% [+0, +4] | +2.3% [−1, +6] | **−4.7% [−8, −2]** | −6.3% [−12, +4] | **−3.7% [−6, −1]** |

The full grid, three PPO checkpoints against three heuristics, is in `final_eval18/headtohead.txt`.

What this supports:
- **Against the aging heuristics (age200, rel30),** pure PPO is faster on mean and median e2e at c10, with intervals clear of zero on TraceLab, and both training seeds agree.
- **Against plain SJF,** it is faster on mean e2e on both traces and roughly tied on median. Plain SJF isn't deployable:
  - its worst TTFT is 844 s (TL c10) and 2,023 s (FI c10);
  - on the synthetic training mix its p99 TTFT is 7× FCFS's.
- **Per-agent completion time** moves less than per-call latency, because each agent's span includes about 600–800 s of think time. Against FCFS it is −10% on TL c10 and −3 to −5% on FI c10. Against the heuristics, the gain is 2–4% and only clears zero against age200 and rel30 on TL.

### c10: TTFT, p50 / p99 / p100 (s)

| policy | TL c10 | FI c10 | calls past FCFS's worst TTFT (TL / FI, of 5,400) |
|---|---|---|---|
| FCFS | 1.5 / 93 / 149 | 0.2 / 90 / 139 | 0 / 0 |
| plain SJF | 1.3 / 117 / 844 | 0.2 / 88 / 2,023 | 35 / 33 |
| age200 | 1.5 / 110 / 395 | 0.2 / 101 / 385 | 27 / 37 |
| rel30 | 1.4 / 104 / 210 | 0.2 / 101 / 229 | 11 / 20 |
| pure e2e+1h @20 | **1.0** / 110 / 447 | 0.2 / 119 / 484 | 28 / 41 |
| pure e2e+1h s1 @40 | 1.1 / 131 / 676 | 0.2 / 122 / 1,550 | 42 / 44 |
| pure e2e+0.02sq @10 | 1.4 / 95 / 170 | 0.2 / 104 / 185 | 5 / 12 |
| pure e2e+0.02sq @40 | 1.6 / 126 / 529 | 0.2 / 116 / 316 | 34 / 40 |

The squared-hinge checkpoint at iteration 10 (pure e2e+0.02sq @10) has the best worst case of any non-FCFS
policy, but at FCFS-level e2e. That's the tradeoff in one row.

## Why the TTFT-vs-e2e tradeoff exists

The hypothesis is confirmed: **cache locality and fewer preemptions buy e2e, and starvation of the
cache-cold calls is the price.**

**1. Deferring large uncached prompts keeps other agents' prefixes resident.** A call whose prompt is
mostly cached (most of an agent's calls) is cheap. A large *uncached* prompt (a new session, or a call whose
prefix was evicted) needs a lot of prefill compute and a lot of fresh KV blocks. Admitting it evicts other
agents' cached prefixes and pushes running calls into preemption and recompute. The better a policy is at
deferring these calls, the higher its hit rate and the lower its e2e. **e2e follows the cache hit rate
almost monotonically**:

| c10 policy | TL hit rate | TL uncached prefill / episode | TL calls preempted | TL e2e mean | FI hit rate | FI calls preempted | FI e2e mean |
|---|---|---|---|---|---|---|---|
| FCFS | 74.8% | 4.27 M tok | 15.7% | 71.1 s | 78.9% | 8.4% | 53.2 s |
| pure e2e+0.02sq @10 (tail-capped) | 74.4% | 4.35 M | 17.6% | 72.5 s | 79.5% | 8.1% | 51.9 s |
| rel30 | 76.0% | 4.08 M | 13.5% | 68.2 s | 79.4% | 7.7% | 51.4 s |
| age200 | 76.3% | 4.03 M | 12.3% | 66.4 s | 80.1% | 6.7% | 49.5 s |
| plain SJF | 76.9% | 3.92 M | 11.2% | 65.7 s | 80.3% | 6.6% | 50.3 s |
| pure e2e+1h s1 @40 | 77.5% | 3.81 M | 9.6% | 63.0 s | 81.1% | 6.1% | 48.4 s |
| **pure e2e+1h @20** | **78.1%** | **3.72 M (−13%)** | **10.6%** | **62.8 s** | **81.3%** | 6.4% | **48.1 s** |

**PPO's advantage over the heuristics is better locality.** It recomputes 8% less prefill than age200 and 5%
less than plain SJF. It also has a third fewer preempted calls than FCFS.

**2. The closed loop compounds it.** A call that finishes sooner brings that agent's next call back sooner,
while its prefix is still resident, so the hit rate rises further. Serving the small, cached, fast calls
first also maximizes calls completed per second, the same argument that makes SJF optimal for the mean.

**3. The cost falls on exactly the deferred class.** The tail-anatomy study (`ttft_anatomy/`, 6 seeds)
found:
- **Under FCFS, the TTFT tail is preemption victims.** 65–78% of the top 2% were restarted, often large but mostly cached.
- **Under age200 and the learned policies, the c10 tail is starved large uncached prompts:**
  - median 74–102k uncached tokens, only 6–24% cached;
  - queueing 118–145 s, against 66–76 s under FCFS.
- Absolute aging at 200 tokens/s needs about 375 s to lift a 75k-token uncached prompt above a fresh small one.

**4. The starvation barely shows in e2e.** e2e p99 (about 600 s at c10) is set by long decodes. A few
hundred extra seconds of queueing on under 1% of calls barely moves the per-call median or p99, and it is
diluted further in an agent's total time. An e2e-driven optimizer therefore buys locality with starvation.
The tail terms in the reward trade along this frontier but don't escape it:
- **Linear hinge weights 1, 2 and 6:** TL c10 p100 447 / 730 / 511 s.
- **Squared hinge 0.02:** at iteration 10 it caps the worst case at 170 s, and the e2e gain disappears with it. By iteration 40 the e2e gain is partly back (FI c10 −5% median, −16% p99), and so is the starvation (316–529 s).

## c8 vs c10: what changes with load

| | c8 (TL / FI) | c10 (TL / FI) |
|---|---|---|
| FCFS median TTFT | 0.3 / 0.1 s | 1.5 / 0.2 s |
| FCFS cache hit rate | 86.9% / 86.9% | 74.8% / 78.9% |
| FCFS calls preempted | 2.6% / 2.0% | 15.7% / 8.4% |
| hit-rate spread across the 33 policies | 86.3–87.8% | 72–78% / 78–82% (diverged 0.05sq run: 66% TL) |
| best median-e2e change vs FCFS | −1.7% (TL), −0.8% (FI) | −14.5% (TL), −5.1% (FI) |

**At c8 there is little contention, so ordering barely matters.**
- KV memory mostly holds every active agent's prefix, and the hit rate is about 87% whatever the policy.
- Every policy's median e2e is within ±2% of FCFS, with intervals that include zero.
- On TL c8, pure PPO checkpoints hold all of the top 6 places by median e2e (−0.4 to −1.7%) and 7 of the top 8 by per-agent time (−2 to −4% per agent, against plain SJF's −2%), but the gaps are small.
- On FI c8, per-agent time is within ±0.5% of FCFS for every good policy. FCFS is effectively optimal there.
- The one sizable c8 effect is the p99: −3 to −8% for SJF-like policies, with wide intervals.

**At c10 the KV cache can no longer hold every agent's working set, and ordering decides which prefixes
survive.** That's where the 10–14% median-e2e gains come from, and also where starvation appears.

**What heavier load needs is *locality-aware admission*:** keep the agents whose prefixes are cached
running, and defer the cache-cold heavy prefills. Under light load, the best policy is one that does no harm.
Pure PPO does both:
- its `best.pt` checkpoints are within ±2% of FCFS on median e2e at c8 and 1–4% better on mean;
- it has the largest gains at c10.

Its tail gets worse with load in the same way as every SJF-like heuristic: 28–44 calls past FCFS's worst TTFT
at c10, against 4–12 at c8. A deployed version would want a load-dependent tail guard, for example relative
aging that switches on as KV occupancy rises. The observation already includes queue length and KV occupancy.

## best.pt vs latest.pt

You asked whether checkpoints past the reward-selected `best.pt` improved on median or p99 without the
reward showing it. **Partly on the median, never overall:**
- **Pure e2e+1h:**
  - the run-0 `latest.pt` (iteration 100) has the best TL c10 median of any policy: −14.5% vs −13.1% for `best.pt`;
  - but its TL c10 p99 is −8% against −10%, its FI c10 p99 is −7% against −15%, and its worst TTFT is 895 / 1,166 s against 447 / 484 s;
  - seed 1 behaves the same way: same median, worst TTFT 718 / 1,703 s.
- **Continued pure-PPO training keeps buying a little median at the price of the tail.** The tail term in the reward is what ranks `best.pt` above `latest.pt`, so the reward is tracking a real difference rather than missing one.
- **Squared-hinge runs degrade outright late in training.**
  - 0.02sq at iteration 130: TL c10 p99 +21%.
  - 0.05sq at iteration 80: TL c10 mean e2e +82%, p99 +122%.
  - A large squared penalty makes the returns heavy-tailed, and the policy eventually diverges.
- **Imitation-initialized runs:** `latest.pt` and `best.pt` are within noise of each other and of the age200 teacher, as throughout.

So `best.pt` is the right checkpoint to report. Its selection used the same 18 eval seeds, so there is mild
selection bias. The fact that `latest.pt` reproduces its median gains bounds that bias for the median, but not
for the tail.

## Paths we closed, and why

**Hard wait cap / deadline (`sjf_*_cap<s>`: calls waiting longer than the cap go first, by arrival).**
- Swept caps of 10, 20 and 40 s on plain SJF and age200 (`wait_cap_sweep/`).
- At c10 it trades mean for tail almost 1:1. Cap 40 on TL c10 brings p99 TTFT to within 4% of FCFS, but keeps only −10% mean TTFT and gives up *all* of the mean-e2e gain.
- The reason is structural. Under heavy load, many calls are past the cap, so the policy **degrades to FCFS** on exactly the calls, and in exactly the periods, where ordering matters. It gives back the locality gains above.
- It is a discontinuous rule, so it is also a hard imitation target, and its smooth version (rel30) gave a better tail for the same e2e.
- Since the goal is e2e, a mechanism that buys TTFT tail with e2e is the wrong direction.

**Imitation from rel30.**
1. **Imitation + PPO never left its teacher.**
   - Six imitation-initialized runs (hinge weights 0.01–0.04 and e2e + 2 × hinge; learning rates 1e-4 / 3e-5; 40–180 iterations) all stayed within noise of age200 on every metric, at both `best.pt` and `latest.pt`.
   - Only pure PPO from a near-random init found a new operating point.
   - Imitating rel30 would very likely just reproduce rel30, which we can already run as a heuristic.
2. **rel30 is at the wrong end of the frontier for the e2e story.** It has the best tail of the heuristics but the weakest e2e: at c10, mean e2e −4% / −3% vs FCFS and median −7% / +1%. Pure PPO beats it by 6–8% on mean e2e in the head-to-head.
3. **Timing.** rel30 came out of the relative-aging sweep after imitation had finished, when the compute was already committed to pure PPO, which was producing the result.

Fair caveat: we didn't test it. A rel30-initialized run with the e2e reward could land at a different point on
the frontier, with less e2e gain than pure PPO and a shorter tail. That's a reasonable follow-up. It isn't a
gap in the main claim, which compares against rel30 directly.

**Pure PPO from an FCFS-like init** (middle-level probability 0.95 / 0.99). It never moved off FCFS.
Priorities are redrawn at every decision, so a near-deterministic start produces no useful exploration.
Abandoned.

**Larger linear hinge weights (2, 6).** These didn't reduce p100. They gave the same median-e2e gain with a
similar tail.

**Squared hinge (0.01 / 0.02 / 0.05).**
- It traces the frontier from the tail end (0.02sq @10) toward the e2e end (0.01sq @20).
- 0.05 never beat FCFS, and late in training it diverged.

## Caveats

- **Simulator results:** Frontier, one GPU type and one model. Two real traces at two loads.
- **The 6-seed pools used during training overstated the gains.** At 6 seeds, pure PPO's TL c10 median e2e was −23% vs FCFS. At 18 seeds it is −13%. FCFS's own median moves from 39.4 s to 31.3 s between the two seed sets. Every number on this page is from the 18-seed evaluation.
- **Per-call vs per-agent.** Per-call e2e is the headline metric. Per-agent completion time, which also counts the knock-on delay a starved call causes to the rest of that agent's run, is reported alongside it.
  - It confirms the direction: pure e2e+1h @20 has the best TL c10 per-agent p50 and p90 (−10.5% / −11% vs FCFS). Its worst agent is −9%; pure e2e+2h @30 and e2e+6h @40 do better there, at −14% and −13%.
  - With only 180 agents per workload, and the gain diluted by think time, the per-agent gain is significant against FCFS but only marginal against the heuristics.
- **Model selection:** `best.pt` was chosen by the training reward on the same 18 eval seeds (see best.pt vs latest.pt above).
