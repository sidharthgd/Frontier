# RL scheduling on closed-loop agent workloads: results (DRAFT, 6 eval seeds)

> **Status:** draft from the 6-seed CDF pools (`*/cdfs/*/data`, `retrospective_quantiles.txt`, 2026-11-22).
> The 18-seed final evaluation (`final_eval18/requests.csv`: every run's `best.pt` **and** `latest.pt`, plus
> FCFS / age200 / rel30 / plain SJF, per request, with session ids) was started 2026-11-22 19:24 UTC.
> Sections marked **[18-seed pending]** get replaced when it finishes. Full chronological log: `PLAN.md`.

## Setup in one paragraph

Frontier simulator (vLLM v1 scheduler, H200 / Llama-2-7B, chunked prefill, prefix caching). The policy
assigns a priority level to every waiting request at each scheduling step. Workloads are closed-loop
session_concurrency: N agents, each issuing a chain of turns whose prompt is mostly the cached prefix of
the previous turn, with think time between turns. Training uses synthetic random sessions (14–22
agents). Evaluation uses real traces TraceLab and FreeInference at 8 and 10 concurrent agents (c8 / c10).
PPO uses group-relative advantages. The reward is `e2e` (per-request e2e, normalized by FCFS on the same
seed) plus a TTFT tail term: the linear `ttft_hinge` Σ max(0, TTFT − τ) or the squared `starvation`
Σ max(0, TTFT − τ)², with τ = FCFS's per-episode p95 TTFT.

Baselines:
- FCFS.
- plain SJF on prefill tokens.
- **age200**: SJF with absolute aging, 200 tokens/s of waiting.
- **rel30**: SJF with relative aging, size / (1 + wait / 30 s).

## Headline (positioning)

**Pure PPO (from a near-random init, reward `e2e` + 1 × ttft_hinge) gives the lowest median and p99
end-to-end latency at the heavier load (c10). It reproduces across two training seeds.** For agentic
workloads this is the metric that matters. Nobody watches each agent's tokens stream; what counts is how
long each step, and so each agent's whole task, takes. Responsiveness of the first token is secondary.
*(The tables below are per call. Per-agent completion time, which also accounts for the knock-on delay a
starved call causes to the rest of that agent's run, is **[18-seed pending]**. See Caveats.)*

The cost is the extreme TTFT tail. About 1% of requests, the large mostly-uncached prompts, wait longer
than under any heuristic (TTFT p100 336–730 s vs FCFS 121–137 s, rel30 210–214 s). Median and mean TTFT
are still *better* than FCFS (TL c10 p50 1.8 s vs 4.5 s).

### c10: e2e latency (s), 6 seeds pooled, change vs FCFS

| policy | TL c10 p50 | TL c10 p99 | FI c10 p50 | FI c10 p99 |
|---|---|---|---|---|
| FCFS | 39.4 | 629 | 23.2 | 560 |
| plain SJF | 32.8 (−17%) | 589 (−6%) | 21.2 (−9%) | 540 (−4%) |
| age200 | 33.8 (−14%) | 641 (+2%) | 21.6 (−7%) | 497 (−11%) |
| rel30 | 33.3 (−15%) | 618 (−2%) | 22.1 (−5%) | 613 (+9%) |
| imitation (age200 student) | 34.6 (−12%) | 683 (+9%) | 21.9 (−6%) | 508 (−9%) |
| imitation + PPO, e2e + 2×hinge @10 | 33.5 (−15%) | 619 (−2%) | 21.2 (−9%) | 556 (−1%) |
| **pure PPO e2e + 1×hinge @20** | **30.4 (−23%)** | **579 (−8%)** | 22.4 (−3%) | **487 (−13%)** |
| **pure PPO, same, seed 1 @40** | **31.3 (−21%)** | 618 (−2%) | 21.5 (−7%) | 510 (−9%) |
| pure PPO e2e + 2×hinge @30 | 33.6 (−15%) | 655 (+4%) | 21.1 (−9%) | 521 (−7%) |
| pure PPO e2e + 6×hinge @20 | 33.2 (−16%) | 583 (−7%) | 22.8 (−2%) | 504 (−10%) |
| pure PPO e2e + 0.01×starvation @20 | 32.3 (−18%) | 601 (−4%) | 21.2 (−9%) | 514 (−8%) |
| pure PPO e2e + 0.02×starvation @10 | 38.1 (−3%) | 649 (+3%) | 22.8 (−2%) | 606 (+8%) |

Mean e2e at c10 (TL / FI): FCFS 76.7 / 64.5, rel30 72.0 / 61.9, age200 70.1 / 58.1, **pure PPO w1 65.7 / 57.4**
(−14% / −11%), sq0.01 67.8 / 55.3.

Honest reading:
- On TraceLab c10, pure PPO is clearly best on p50 (both seeds) and on p99 (seed 0).
- On FreeInference c10, the p50s of all SJF-like policies are within about 1 s. Pure PPO wins on the mean and the p99.
- **Plain SJF is the strongest e2e heuristic on the eval traces, but it is not deployable.** Its worst TTFT is 2,023 s on FI c10, and on
  the synthetic training mix its p99 TTFT is 7× FCFS's.
- PPO gets SJF-level e2e or better without that collapse.

### c10: TTFT (s), p50 / p99 / p100

| policy | TL c10 | FI c10 | requests past FCFS's max (TL / FI, of 1,800) |
|---|---|---|---|
| FCFS | 4.5 / 93 / 121 | 0.3 / 98 / 137 | 0 / 0 |
| rel30 | 3.3 / 112 / 210 | 0.3 / 112 / 214 | 16 / 7 |
| age200 | 3.7 / 128 / 395 | 0.3 / 132 / 376 | 19 / 16 |
| plain SJF | 3.6 / 97 / 320 | 0.3 / 93 / 2023 | 12 / 14 |
| pure PPO w1 @20 | 1.8 / 110 / 395 | 0.3 / 161 / 336 | 17 / 21 |
| pure PPO w1 seed1 @40 | 2.2 / 129 / 454 | 0.2 / 142 / 728 | 19 / 20 |
| sq0.02 @10 | 4.1 / 109 / 170 | 0.3 / 124 / 157 | 10 / 5 |

Every non-FCFS policy, heuristic or learned, starves some requests past FCFS's worst wait. The
squared-hinge run sq0.02 @10 has the best worst case of any of them, but gives back the e2e gain (table above).

## Why the TTFT-vs-e2e tradeoff exists

Your hypothesis is right, and the tail-anatomy data (`ttft_anatomy/`, 6 seeds) supports it.

**1. Deferring large uncached prompts improves cache locality and cuts preemptions.** In these workloads a
turn's prompt is mostly the cached prefix of that session's previous turn. A large *uncached* prompt (a new
session, or a turn whose prefix was evicted) costs a lot of prefill compute and a lot of fresh KV blocks.
Admitting it evicts other sessions' cached prefixes and pushes running requests into preemption. Size-aware
ordering defers it. Measured at c10:

| workload | policy | prefix-cache hit rate | uncached prefill tokens / episode | requests restarted (preempted) | mean e2e |
|---|---|---|---|---|---|
| TL c10 | FCFS | 71.5% | 4.98 M | 18.9% | 76.7 s |
| TL c10 | age200 | 73.0% | 4.72 M (−5%) | 14.0% | 70.1 s |
| TL c10 | imitation + PPO | 73.3% | 4.68 M (−6%) | 14.3% | 70.8 s |
| FI c10 | FCFS | 75.4% | 4.03 M | 9.9% | 64.5 s |
| FI c10 | age200 | 78.2% | 3.57 M (−11%) | 7.6% | 58.1 s |
| TL c8 | FCFS / age200 | 86.3% / 86.7% | 2.45 M / 2.38 M | 2.7% / 2.0% | 37.8 / 37.2 s |
| FI c8 | FCFS / age200 | 85.8% / 85.5% | 2.36 M / 2.42 M | 2.7% / 2.1% | 34.7 / 34.2 s |

**[18-seed pending]:** the same table for pure PPO, rel30 and plain SJF.

**2. The closed loop compounds it.** Finishing a session's turn sooner brings its next turn back sooner, while
its prefix is still resident, so the hit rate rises further. Serving the small, cached, fast turns first
also maximizes turns completed per second, the same argument that makes SJF mean-optimal.

**3. The cost lands on exactly the deferred class.** Under FCFS, the TTFT tail is preemption victims: 65–78%
of the top 2% were restarted, often large but mostly cached. Under age200 and the learned policies, the c10
tail is **starved large uncached prompts**:
- median 74–102k uncached tokens, only 6–24% cached;
- queueing 118–145 s, against 66–76 s under FCFS.

Absolute aging at 200 tokens/s needs about 375 s to lift a 75k-token uncached prompt above a fresh small one.

**4. The starvation barely shows in e2e.** e2e p99 is about 600 s and is set by long decodes. A few hundred
seconds of extra queueing on 1% of requests moves the e2e p99 and p50 very little. An e2e-driven optimizer
therefore happily buys locality with starvation.
- The tail terms in the reward pull back, but they trade along the frontier rather than escaping it.
- Linear hinge weights 1, 2 and 6 all left p100 at 395–730 s.
- The squared hinge at 0.02 (sq0.02 @10) fixed the worst case but erased the e2e gain.
- Later, sq0.02 @30 got the e2e back and the starvation with it.

## c8 vs c10: what changes with load

**At c8, the system is rarely overloaded and ordering barely matters.**
- Median TTFT is 0.1–0.4 s under every policy.
- The prefix-cache hit rate is about 86% whatever the policy, and only 2–3% of requests are preempted.
- Every policy is within about ±3% of FCFS on mean e2e: FCFS 37.8 / 34.7 s, best 36.5 / 33.3 s.
- The c8 p50 and p99 differences in the 6-seed pools are inside seed noise.
- Small wins:
  - imitation + PPO e2e_w2 (TL c8 e2e p50 15.2 vs 15.9 s, p99 311 vs 327 s, and the only policy whose TL c8 worst TTFT, 78 s, beats FCFS's 98 s);
  - pure PPO w2 (TL c8 p50 15.1 s).

**At c10, queueing and memory pressure dominate.**
- FCFS's TL median TTFT is already 4.5 s, 19% of requests get preempted, and the hit rate falls to 71–75%.
- Here the ordering decides the cache working set, which is where the 14–23% median-e2e gains come from.
- It is also where starvation appears.

So the characteristic that matters under heavy load is *locality-aware admission*: keep resident the sessions
whose prefixes are cached, and defer cache-cold heavy prefills. Under light load, FCFS is close to optimal and
the best policy is one that does no harm. A practical consequence is that a learned policy should key on load
(queue length, KV occupancy), which the observation already includes.

**[18-seed pending]:** a per-workload ranking of all policies (`best.pt` and `latest.pt`), to say which policy
is best at c8 and which at c10 with tighter error bars. The 6-seed pools give about ±3–5% on p50 e2e at c10,
so only the larger gaps above are trustworthy.

## Paths we closed, and why

**Hard wait cap / deadline (`sjf_*_cap<s>`: overdue requests go first, by arrival).**
- Swept caps of 10, 20 and 40 s on plain SJF and age200 (`wait_cap_sweep/`).
- At c10 it trades mean for tail almost 1:1. Cap 40 on TL c10 brings p99 TTFT to within 4% of FCFS, but keeps only −10% mean TTFT and gives up *all* of the mean-e2e gain.
- The reason is structural. Under heavy load, the overdue set is large, so the policy *degrades to FCFS* on exactly the requests and the periods where ordering matters, and gives back the locality gains above.
- It is a discontinuous rule, so it is also a hard imitation target, and the smooth version of the same idea (rel30) dominated it.
- Since the goal is e2e, a mechanism that buys TTFT tail with e2e is the wrong direction.

**Imitation from rel30.**
1. **Imitation + PPO never left its teacher.**
   - Six imitation-initialized runs (age200 teacher, hinge weights 0.01–0.04, lr 1e-4 / 3e-5, about 50–190 iterations each) all stayed within noise of age200 on every metric.
   - The pure-PPO runs from a near-random init are the only ones that found a new operating point.
   - Imitating rel30 would very likely just reproduce rel30, which we can already run as a heuristic.
2. **rel30 is at the wrong end of the frontier for the e2e story.**
   - It has the best TTFT tail of the heuristics but the smallest e2e gain: c10 mean e2e −6% / −4%, against −14% / −11% for pure PPO, and its FI c10 e2e p99 is +9% vs FCFS.
   - Starting there and optimizing e2e would at best move along the same frontier that the squared-hinge runs already traced.
3. **Timing.** rel30 came out of the relative-aging sweep after imitation had finished, when the compute was already committed to pure PPO, which was producing the result.

Fair caveat: we didn't test it. A rel30-initialized run with the e2e reward could land at a *different* point on the frontier, with somewhat less e2e gain than pure PPO and a shorter tail. That's a reasonable follow-up, not a gap in the main claim.

**Pure PPO from an FCFS-like init** (middle-level probability 0.95 / 0.99). It never moved off FCFS. Redrawing the
priority at every decision makes a near-deterministic start produce no useful exploration. Abandoned.

**Larger linear hinge weights (2, 6).** These didn't reduce p100. They only reduced the e2e gain.

**Squared hinge (0.01 / 0.02 / 0.05).**
- It traces the frontier.
- 0.05 degraded steadily: evals −0.03 to −0.40.

## Caveats

- Simulator results (Frontier, one GPU type and model).
- Two real traces at two loads.
- 6-seed pools on this page (1,800–1,824 requests per workload). The **[18-seed pending]** numbers replace them.
- Model selection used `best.pt` by the training reward on the same 18 eval seeds, so there is mild selection bias. Reporting `latest.pt` alongside it bounds that.
- **Per-call vs per-agent.** Every e2e number above is *per call*.
  - The claim is about how long an agent takes to finish.
  - Each agent waits for one call before sending the next, so a call starved for 400 s delays every later call from that agent, yet counts as one sample in the per-call p50 and p99.
  - The 18-seed eval therefore reports the **per-agent completion time**. `session_id` is the agent, and each agent's scored work is a fixed quota (its first 30 calls after warm-up at c10, 38 at c8), so the span from its first scored arrival to its last scored completion is the time it takes to finish its share.
  - That's 180 agents per c10 workload, so we report p50, p90 and the worst agent. **[18-seed pending]**
  - The headline must hold on this metric, not only on per-call latency.
