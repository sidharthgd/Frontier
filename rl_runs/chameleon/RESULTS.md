# RL scheduling on closed-loop agent workloads: results

Final evaluation, 2026-11-22. 18 eval seeds × 4 eval workloads, 33 policies: 4 heuristics, plus `best.pt` **and**
`latest.pt` of every run. That's 5,400–5,472 scored calls per policy and workload.
- Data: `final_eval18/requests.csv.gz`, one row per scored call.
- Tables: `final_eval18/tables.txt` (from `final_eval_analysis.py`) and `final_eval18/headtohead.txt` (from `final_eval_headtohead.py`).
- Checkpoint iterations: `final_eval18/checkpoint_iterations.txt`.
- **Appendix:** TTFT and e2e p50 / p95 / p99 / p100 for every policy on all four workloads are in `final_eval18/appendix.md` (from `final_eval_appendix.py`; raw numbers in `final_eval18/quantiles.csv`).
- **Decision-level study of pure e2e+1h @20** (what its rankings do, and when they can be skipped): `decisions18/tables.txt`, from `scripts/rl/decision_anatomy.py`, `decision_analysis.py` and `head_agreement.py`. Per-episode latencies are in `decisions18/requests.csv.gz`; the per-call logs (`decisions.csv`, `queue.csv`, about 1 GB) stayed on the laptop.
- **Heuristics derived from it:** watermark SJF (`watermark18/`) and cache-protecting SJF (`cpsjf18/`), compared in `watermark18/tables.txt` (from `watermark_compare.py`).
- Chronological log of every run and decision: `PLAN.md`.

Bracketed ranges are 90% seed-bootstrap intervals: seeds resampled with replacement, and paired, so both
policies see the same seeds.

## Setup in one paragraph

Frontier simulator (vLLM v1 scheduler, A100 80GB / Llama-3.1-8B (`a100_llama31_8b` preset: 2,048-token steps, 256 sequences, 453k-token KV cache), chunked prefill, prefix caching). The policy
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

**At the heavier load (c10), pure PPO (`e2e` + 1 × ttft_hinge) has lower end-to-end latency than every
heuristic we started with, and the result reproduces across two training seeds.**
- **vs FCFS on TraceLab c10:** −13% median e2e, −8% p95, −10% p99, −12% mean and −10% median time per agent.
- **vs the aging heuristics:** it beats age200 and rel30 on mean and median e2e with intervals that exclude zero. On TL c10 it also beats them on e2e p95: −7.7% [−12, −2] vs age200 and −6.3% [−10, −1] vs rel30.
- **vs plain SJF:** it matches plain SJF's median and beats its mean (and its TL c10 p95, −4.4% [−9, −0]). Plain SJF starves calls for up to 2,023 s, while PPO's worst wait is 447–484 s.

**What PPO learned turns out to be a one-line rule, and that rule matches it** (section "What the policy
actually does").
- At the scheduler calls where ranking can change the batch, PPO's first-ranked request is, in 99.7–99.9% of
  calls, the waiting call with the most *idle* cached prefix blocks: KV that belongs to a waiting agent and
  that the next allocation would evict. Usually that call does not fit yet, so admission stops. PPO waits
  rather than let a cache-cold call evict that prefix.
- **Cache-protecting SJF** puts that call first and orders the rest by plain SJF. It needs no learning and no
  output-length oracle. On the same 18 seeds it is tied with PPO everywhere (TL c10 mean e2e −1.9% [−4, +1]
  vs PPO) and beats plain SJF at c10 (TL mean −6.3% [−8, −4], FI −5.2% [−8, −3]).
- So the RL contribution is *discovering* the mechanism. Once it is found, a heuristic can implement it.

For agentic workloads, e2e is the metric that matters: nobody watches every agent's tokens stream. What
counts is how long each step, and so the whole agent, takes.

**The cost is the extreme TTFT tail, and only the extreme tail.**
- Median, mean and p95 TTFT are *better* than FCFS's. TL c10 TTFT p50 is 1.0 s against 1.5 s, and p95 is 43.7 s against 61.1 s (−28%). Every reordering policy is 15–45% better than FCFS at p95 on c10.
- PPO's TTFT is no worse than FCFS's at any quantile up to about p98 at c10 and about p99.5 at c8. The loss is all above that.
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

### p95: does it change the analysis?

Not the conclusions, but it sharpens two of them (all quantiles for all policies are in `final_eval18/appendix.md`):

| c10, change vs FCFS [90%] | TL e2e p95 | FI e2e p95 | TL TTFT p95 | FI TTFT p95 |
|---|---|---|---|---|
| FCFS (absolute) | 253 s | 204 s | 61.1 s | 56.7 s |
| plain SJF | −3.4% [−6, +2] | −5.2% [−10, −0] | −37% | −45% |
| age200 | +0.0% [−5, +4] | −5.0% [−10, −1] | −30% | −45% |
| rel30 | −1.4% [−6, +2] | −3.3% [−7, +0] | −15% | −32% |
| **pure e2e+1h @20** | **−7.7% [−12, −1]** | **−6.9% [−14, −2]** | −28% | −39% |
| pure e2e+1h s1 @40 | −6.9% [−11, −2] | −3.1% [−12, +2] | −26% | −39% |

- **e2e p95 is where PPO separates from the heuristics most clearly on TL c10.** None of the three heuristics clears zero against FCFS there, and PPO beats each of them head to head (−4.4% vs plain SJF, −7.7% vs age200, −6.3% vs rel30; intervals exclude zero). On FI c10 the p95 gaps between PPO and the heuristics are 2–4%, with intervals that include zero.
- **TTFT p95 is not part of the tradeoff.** Reordering makes p95 TTFT much better, and the starvation only appears above p98. The TTFT cost is a property of the last 1–2% of calls.
- **At c8** the heuristics' and the `best.pt` checkpoints' e2e p95 is within −3.3% to +1.4% of FCFS, the same "ordering barely matters" picture as the median.

## Why the TTFT-vs-e2e tradeoff exists

The hypothesis is confirmed: **cache locality and fewer preemptions buy e2e, and starvation of the
cache-cold calls is the price.**

**1. Deferring large uncached prompts keeps other agents' prefixes resident.** A call whose prompt is
mostly cached (most of an agent's calls) is cheap. A large *uncached* prompt (a new session, or a call whose
prefix was evicted) needs a lot of prefill compute and a lot of fresh KV blocks. Admitting it evicts other
agents' cached prefixes and pushes running calls into preemption and recompute. The better a policy is at
deferring these calls, the higher its hit rate and the lower its e2e. (The decision-level study below pins
down *how* PPO does this: it holds admission rather than let a cache-cold call evict a waiting agent's idle
prefix.) **e2e follows the cache hit rate
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

## What the policy actually does, and how it beats plain SJF

Method: every scheduler call of pure e2e+1h @20 was logged on all 4 workloads × 18 seeds
(`scripts/rl/decision_anatomy.py`). That covers 1.07 M calls in the scored windows: the state before the
policy runs, every waiting request's level, uncached and cached tokens, idle cached blocks, wait and
restarts, and what admission then did. The instrumented run reproduces the final-eval latencies exactly.

**1. Most calls can't change anything.** vLLM calls the sort on every non-empty waiting queue, each
iteration (about every 50 ms of simulated time). Each call is classified before the policy runs:

| share of calls | blocked (no token budget / batch full) | nofit (no waiting request fits the free KV) | single (1 waiting, fits) | decide |
|---|---|---|---|---|
| TL c10 | 13% | 21% | 3% | 63% |
| FI c10 | 15% | 20% | 3% | 61% |
| TL c8 | 13% | 37% | 11% | 39% |
| FI c8 | 13% | 36% | 9% | 42% |

"Allfit" (every waiting request fits in any order) is 0.0% everywhere: the prompts are 20–80k tokens
against a 2,048-token step.

**2. Where order matters, PPO mostly uses it to *not admit*.** At 98.6–98.9% of decide calls, PPO's
first-ranked request does not fit, so admission stops and nothing is admitted. Every decide call has at least
one request that would fit, and plain SJF's first request would fit in 26–34% of them. Admitting calls (about 1.3%)
are the exception. So a pairwise model of PPO's ranking needs the "fits now" bit: requests that fit are ranked
*behind* (`decisions18/tables.txt` §6, logistic CV accuracy 97.6–99.6% with cache state and fit, against 74–88%
for uncached size alone).

**3. What it holds out for.** At those holds, the head of PPO's queue, compared with the smallest request that
would fit (TL c10; FI c10 is similar):

| | PPO's head | smallest fitting request |
|---|---|---|
| has an idle cached prefix | 100% | 11% |
| is the queue's most-cached request | 100% | |
| cached tokens (median) | 52,504 | 11,552 |
| uncached tokens (median) | 20,338 | 32,603 |
| hit fraction (median) | 0.73 | 0.22 |
| is a preempted restart | 42% | 7% |
| free KV at the hold (median) | 8.4% of the pool | (30% at calls that admit) |

The "free" KV at a hold is mostly the head's own idle prefix: about 2,800 blocks, roughly what is free. Any
admission would take those blocks from the free queue and evict a 50k-token prefix that the waiting agent
then has to recompute. PPO waits until enough memory frees for the cached call itself. It is protecting cached
prefixes, not reserving headroom.

**4. How often it is just SJF.** PPO's first-ranked request is plain SJF's (smallest uncached prefill) at 67–74%
of decide calls. Among the 1.3% that admit some but not all, the admitted set equals SJF's in 64–71%. When it
departs from SJF there, it admits a bigger request ahead of a smaller one (median 1.4–1.8× the uncached tokens)
that has more cached tokens (71–84% of such pairs), an idle cached prefix (68–84%), or is a restart (18–28%).
Mean levels also fall with wait time for small and medium prompts (mild aging). In the pairwise model, adding
wait time to uncached size gains 0–4 points of accuracy; cache state and the fit bit carry the rest. **Size is the main ordering key; idle cached prefixes override it.**

**5. The causal test: two heuristics built from these observations** (`watermark18/tables.txt`, 18 seeds,
paired against the final-eval runs):

| change vs FCFS: e2e mean / p50 / p95 / p99 | TL c10 | FI c10 | TL c8 | FI c8 | TTFT p100 TL / FI c10 | hit rate TL c10 |
|---|---|---|---|---|---|---|
| plain SJF | −7.5 / −10.7 / −3.4 / −5.3% | −5.4 / −1.0 / −5.2 / −8.0% | −2.8 / +1.3 / −1.8 / −6.2% | −1.8 / +1.3 / −1.3 / −0.7% | 844 / 2,023 s | 76.9% |
| pure e2e+1h @20 | −11.7 / −13.1 / −7.7 / −9.5% | −9.5 / −1.1 / −6.9 / −15.2% | −3.0 / −0.3 / −2.0 / −3.3% | −3.1 / +0.5 / −0.3 / −0.1% | 447 / 484 s | 78.1% |
| SJF + 10% KV watermark | +25.8 / −1.0 / +36.4 / +35.3% | +14.4 / −2.4 / +21.3 / +37.6% | +1.7 / +0.3 / +3.6 / +3.1% | +4.7 / +1.4 / +6.7 / +13.6% | 751 / 1,274 s | 72.5% |
| **cache-protecting SJF** | **−13.3 / −15.0 / −8.3 / −14.2%** | **−10.4 / −3.4 / −7.2 / −15.4%** | −3.4 / +0.3 / −3.7 / −3.3% | −2.3 / +1.0 / −1.4 / +0.0% | 366 / 1,416 s | 78.8% |

- **SJF + watermark** holds admission (puts the largest request first) whenever admitting SJF's first request would leave under 10% (or 20%) of KV free. It is much *worse* than FCFS: TL c10 mean +26% (20%: +38%), with a lower hit rate (72.5% / 69.9%) and, at 10%, more restarts. Holding memory for no particular request just idles capacity, so generic headroom isn't the mechanism.
- **Cache-protecting SJF** puts the waiting request with the most idle cached blocks first (if any has some) and orders the rest by plain SJF. Its first choice matches PPO's at 99.7–99.9% of decide calls (plain SJF matches at 67–74%). It is statistically tied with PPO on every workload (TL c10 vs PPO: mean −1.9% [−4, +1], p50 −2.1% [−5, +2], p95 −0.7% [−5, +6]; FI c10: −1.0% / −2.3% / −0.4%, all intervals include zero). At c10 it beats plain SJF (TL mean −6.3% [−8, −4], p50 −4.8% [−7, −0]; FI mean −5.2% [−8, −3]). At c8 it is within noise of plain SJF.
- **Its tail is no better than PPO's.** The worst TTFT is 366 s on TL c10 but 1,416 s on FI c10. It has no aging at all, so a deployable version would add rel30-style aging for the non-cached requests.

**So how does PPO beat plain SJF?** By being SJF, except that it never lets a cache-cold call evict a waiting
agent's idle prefix. It waits instead, and that is worth about 5–6% mean e2e at c10. The rule is simple, uses
only state vLLM already has (prefix-cache lookups and the free-block queue), and was found by the RL policy rather
than designed. Caveat: cache-protecting SJF was designed after looking at PPO's decisions on these same 18 seeds.
It has no tuned parameters, but a fresh seed set would be the clean test.

## Deployment cost: can the policy call be skipped?

**Yes, for the calls where no ranking can change the batch, and it can be checked before the policy runs.**
- **Blocked:** the remaining token budget is 0 or the batch is at max_num_seqs.
- **Nofit:** no waiting request's first chunk fits the free KV, counting the idle cached blocks its own hit would take off the free queue.

In both cases admission admits nothing whatever the order, and the RL hook already undoes a ranking that
admitted nothing. Skipping them (`skip_exact`, 72 episodes) reproduces **100% of call latencies
exactly**. The check itself costs 0.03 ms per call.

**Single-request calls are not free to skip, but nearly.** The lone request's priority is kept and later
picks preemption victims. Replacing the policy's priority for single (and allfit) calls with a fixed middle level, or an
SJF-rank level, changes mean e2e by −0.7% to +3.0% (TL c10 +1.1% / +3.0%) but TL c10 TTFT p100 by +63%. That is
also only another 3–11% of calls. These priorities are needed only at a later preemption, so the forward pass could
run off the critical path (after the batch launches) instead of being skipped.

**Decide calls (39–63%) cannot be skipped.** They are the 98.6% "hold" decisions above, and holding *is* the
policy.

**Cost and amortization** (Chameleon node CPU, one thread, PyTorch, unbatched):

| | TL c10 | FI c10 | TL c8 | FI c8 |
|---|---|---|---|---|
| features + forward, per call (mean / p99) | 1.29 / 1.58 ms | 1.30 / 1.62 ms | 1.31 / 1.85 ms | 1.30 / 1.82 ms |
| policy time per simulated second, every call | 5.2 ms (0.52%) | 4.9 ms | 1.7 ms | 2.0 ms |
| skipping blocked + nofit (exact) | 3.4 ms (−34%) | 3.2 ms (−35%) | 0.9 ms (−48%) | 1.1 ms (−47%) |
| decide calls only | 3.3 ms (−37%) | 3.0 ms (−38%) | 0.6 ms (−62%) | 0.8 ms (−58%) |

A call costs about 1.3 ms against a median of about 50 ms between scheduler iterations. On the critical path
that's about 2.6% of an iteration when it runs. Averaged over time it is 0.2–0.5% of wall time before skipping,
and 0.1–0.34% after the exact skip. The forward pass is 75% of the cost (0.97 ms for a 2-layer, d=64
transformer over queue plus running requests, CPU, batch 1), and features are 0.32 ms of Python. Since
PPO's decisions collapse to cache-protecting SJF, the cheapest deployment is the heuristic itself: it needs
no forward pass, only the prefix-cache lookups vLLM already does.

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
- **Cache-protecting SJF was found by reading PPO's decisions on the eval seeds.** It has no tuned parameters, and its 18-seed comparison is paired against runs from the same seeds. A fresh seed set (and the synthetic training mix) would be the clean confirmation, along with an aging guard for its tail.
- **Policy cost** is measured in the simulator's Python process on one CPU thread. A production scheduler would batch the forward pass or run it on the GPU's host thread pool, so the per-call numbers are an upper bound for this model size.

## Appendix

Full TTFT and e2e p50 / p95 / p99 / p100 tables for all 33 policies on TL / FI × c8 / c10, with the change vs FCFS
and p95 bootstrap intervals for the key policies: `final_eval18/appendix.md`.
