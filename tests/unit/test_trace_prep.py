"""Session traces with block hashes (frontier/trace_prep) and their replay."""

import dataclasses
import gzip
from collections import Counter
import json

import numpy as np
import pandas as pd
import pytest

from frontier.entities import Request
from frontier.request_generator.trace_replay_request_generator import _parse_block_hash_ids
from frontier.trace_prep import freeinference, sessions, tracelab
from frontier.trace_prep.common import BlockRuns, PlaceholderIds
from frontier.utils.block_hash_ids import decode_block_hash_ids, encode_block_hash_ids


def ids_of(text: str) -> list:
    return decode_block_hash_ids(text).tolist()


@pytest.mark.parametrize(
    "ids",
    [[], [7], [0, 1, 2, 3], [5, 6, 9, 10, 11, 2, 40], [2**50, 2**50 + 1, 3]],
)
def test_runs_round_trip(ids):
    text = encode_block_hash_ids(ids)
    assert ids_of(text) == ids
    if ids:
        assert _parse_block_hash_ids(text).tolist() == ids


def test_runs_are_compact_and_legacy_formats_parse():
    assert encode_block_hash_ids([0, 1, 2, 3, 9, 11, 12]) == "0-3|9|11-12"
    assert ids_of("1|2|5") == [1, 2, 5]
    assert ids_of("1,2,5") == [1, 2, 5]
    assert ids_of("[1, 2, 5]") == [1, 2, 5]
    assert ids_of("-3|4") == [-3, 4]
    with pytest.raises(ValueError):
        decode_block_hash_ids("9-3")


def test_request_stores_hashes_compactly_and_returns_ints():
    request = Request(arrived_at=0.0, num_prefill_tokens=40, num_decode_tokens=5,
                      block_hash_ids=decode_block_hash_ids("10-12|99"))
    hashes = request.prompt_block_hash_ids(16)  # 40 // 16 = 2 full blocks
    assert hashes == [10, 11] and all(type(h) is int for h in hashes)
    assert request.block_hash_ids == [10, 11, 12, 99]
    assert Request(0.0, 10, 1).prompt_block_hash_ids(16) == []


def test_block_runs_prefix_and_extend():
    runs = BlockRuns()
    runs.extend(0, 5)
    runs.extend(5, 3)  # adjacent: same run
    runs.extend(100, 2)
    assert runs.encode() == "0-7|100-101" and runs.num_blocks == 10
    assert runs.prefix(6).encode() == "0-5"
    assert runs.prefix(9).encode() == "0-7|100"
    placeholders = PlaceholderIds()
    first, second = placeholders.next(), placeholders.next()
    assert first != second and int(first) >= 1 << 50


def _tracelab_round(session, index, t, prompt, prefix, output, user="u", provider="claude"):
    return {
        "session_id": session, "round_index": index, "user": user, "provider": provider,
        "input_tokens_total": prompt, "prefix_tokens": prefix, "output_tokens": output,
        "timing_events": [{"event_type": "tool_result", "timestamp": f"2026-05-31T14:00:{t:02d}.000Z"}],
    }


def test_tracelab_synthesizes_shared_prefixes(tmp_path):
    rounds = [
        _tracelab_round("a", 0, 0, 1600, 0, 50),        # 100 prompt blocks, 103 with output
        _tracelab_round("a", 1, 5, 2000, 1590, 50),     # continues: shares prompt + output
        _tracelab_round("a", 2, 9, 2400, 2000, 50),     # continues
        _tracelab_round("a", 3, 14, 900, 800, 50),      # restart: shares its cache read, 50
        _tracelab_round("b", 0, 20, 1000, 640, 50),     # first round, cache read from user's session a
    ]
    source = tmp_path / "trace.jsonl.gz"
    with gzip.open(source, "wt") as f:
        for r in rounds:
            f.write(json.dumps(r) + "\n")
    output = tmp_path / "out.csv"
    stats = tracelab.convert(str(source), str(output), max_tokens=None)
    frame = pd.read_csv(output, dtype={"block_hash_ids": str}).sort_values("arrived_at")
    assert frame["ids_cover_output"].tolist() == [1] * 5
    ids = [ids_of(t) for t in frame["block_hash_ids"]]
    # (prompt + output) // 16 ids each
    assert [len(i) for i in ids] == [103, 128, 153, 59, 65]

    def shared(x, y):
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        return n

    assert shared(ids[0], ids[1]) == 103  # the previous prompt and its output
    assert shared(ids[1], ids[2]) == 128
    assert shared(ids[2], ids[3]) == 50
    assert shared(ids[3], ids[4]) == 40  # 640 // 16, from the user's latest round
    assert stats["first_round_parent_from_user"] == 1
    assert stats["continues_through_output"] == 2
    # the trunk keeps growing rounds a few runs long
    assert all(t.count("|") <= 2 for t in frame["block_hash_ids"])
    assert len(set(sum(ids[:3], []))) == 103 + 25 + 25  # new content gets new ids


def test_tracelab_sessions_share_their_providers_system_prompt(tmp_path):
    # 12 users' first rounds, no cache reads: the 10th percentile prompt
    # (here 1600 tokens = 100 blocks) is every claude context's system prompt
    rounds = [
        _tracelab_round(f"s{i}", 0, i, 1600 + 160 * i, 0, 50, user=f"u{i}") for i in range(12)
    ]
    source = tmp_path / "trace.jsonl.gz"
    with gzip.open(source, "wt") as f:
        for r in rounds:
            f.write(json.dumps(r) + "\n")
    output = tmp_path / "out.csv"
    tracelab.convert(str(source), str(output), max_tokens=None)
    ids = [ids_of(t) for t in pd.read_csv(output, dtype={"block_hash_ids": str})["block_hash_ids"]]
    system = tracelab.provider_prefix_blocks(tracelab.read_rounds(str(source), Counter()))["claude"]
    assert 100 <= system <= 120  # 10th percentile of 1600..3360 tokens
    # a prompt shorter than the system prompt holds part of it
    assert all(i[: min(system, len(i) - 3)] == ids[-1][: min(system, len(i) - 3)] for i in ids)
    longer = [i for i in ids if len(i) > system]
    assert len(longer) == 11 and len({i[system] for i in longer}) == 11  # then their own content


def test_freeinference_flattens_subagents_and_namespaces_weeks(tmp_path):
    def request(start, blocks, completion=10, calls=()):
        return {
            "start_ms": start, "block_ids": blocks,
            "provider_tokens": {"prompt": 0, "completion": completion},
            "role_tokens": {"system": 16 * len(blocks) + 3},
            "tool_calls": list(calls),
        }

    sub = {"session_id": "s2", "requests": [request(1500, [0, 1, 7])]}
    week1 = {"week": "2026-05-17_2026-05-23", "session_id": "s1", "requests": [
        request(1000, [0, 1, 2], calls=[{"name": "task", "subagent": sub}]),
        request(2000, [0, 1, 2, 3, 4]),
        request(3000, list(range(10)), completion=200),  # over max_tokens
    ]}
    week2 = {"week": "2026-05-24_2026-05-30", "session_id": "s3", "requests": [request(4000, [0, 1])]}
    source = tmp_path / "day.jsonl"
    source.write_text("\n".join(json.dumps(s) for s in (week1, week2)) + "\n")
    output = tmp_path / "out.csv"
    # without compaction, calls over max_tokens are dropped
    stats = freeinference.convert([str(source)], str(output), max_tokens=150, compact_at=None)
    assert stats["kept"] == 4 and stats["dropped_too_long"] == 1 and stats["sessions"] == 3
    frame = pd.read_csv(output, dtype={"block_hash_ids": str})
    assert list(frame["arrived_at"]) == [0.0, 0.5, 1.0, 3.0]
    assert list(frame["num_prefill_tokens"]) == [51, 51, 83, 35]  # 16 * blocks + 3
    total = frame["num_prefill_tokens"] + frame["num_decode_tokens"]
    assert total.floordiv(16).tolist() == [len(ids_of(t)) for t in frame["block_hash_ids"]]
    assert ids_of(frame["block_hash_ids"][1]) == [0, 1, 7]  # subagent, own session
    assert frame["session_id"].nunique() == 3
    assert ids_of(frame["block_hash_ids"][3])[0] == 1 << 36  # second week's namespace


def _session_trace(num_sessions=200, calls=20, gap=10.0, with_times=True):
    rows = []
    for s in range(num_sessions):
        for k in range(calls):
            row = {"session_id": s, "step": k, "num_prefill_tokens": 16 * (k + 1),
                   "num_decode_tokens": 8, "block_hash_ids": f"{s * 1000}-{s * 1000 + k}"}
            if with_times:
                row["arrived_at"] = 1e5 * s + gap * k
            rows.append(row)
    return pd.DataFrame(rows)


def test_sampler_keeps_session_gaps_and_rate(tmp_path):
    path = tmp_path / "trace.csv"
    _session_trace().to_csv(path, index=False)
    trace = sessions.load_session_trace(str(path))
    workload = sessions.sample_session_workload(trace, num_requests=300, qps=5.0, seed=1)
    assert len(workload) == 300
    assert workload["arrived_at"].is_monotonic_increasing and workload["arrived_at"].iloc[0] == 0
    span = workload["arrived_at"].iloc[-1]
    assert 300 / span == pytest.approx(5.0, rel=0.2)
    # each session's calls keep their 10 s gaps and their own hashes
    for _, calls in workload.groupby("session_id"):
        assert np.allclose(np.diff(calls["arrived_at"]), 10.0 * np.diff(calls["step"]))
    first_ids = workload["block_hash_ids"].str.split("-").str[0]
    assert (first_ids.groupby(workload["session_id"]).nunique() == 1).all()
    # stationary: sessions join mid-way, not only at step 0
    assert (workload.groupby("session_id")["step"].min() > 0).any()
    again = sessions.sample_session_workload(trace, num_requests=300, qps=5.0, seed=1)
    pd.testing.assert_frame_equal(workload, again)
    other = sessions.sample_session_workload(trace, num_requests=300, qps=5.0, seed=2)
    assert not workload["block_hash_ids"].equals(other["block_hash_ids"])


def test_sampler_weights_long_sessions_like_a_stationary_server(tmp_path):
    # half the sessions are 10x longer (in time and calls): in a stationary
    # server they send ~10x the calls of a window
    frame = pd.concat([
        _session_trace(400, calls=5, gap=10.0),
        _session_trace(400, calls=50, gap=10.0).assign(session_id=lambda f: f.session_id + 400),
    ])
    path = tmp_path / "trace.csv"
    frame.to_csv(path, index=False)
    trace = sessions.load_session_trace(str(path))
    share = np.mean([
        (sessions.sample_session_workload(trace, 200, 2.0, seed)["num_prefill_tokens"] > 16 * 5).mean()
        for seed in range(5)
    ])
    # calls past step 5 exist only in long sessions: 400 * 45 of the trace's
    # 400 * 55 calls. Drawing sessions uniformly instead would give ~0.63.
    assert share == pytest.approx(45 / 55, abs=0.06)


def test_sampler_gives_untimed_traces_think_time(tmp_path):
    path = tmp_path / "trace.csv"
    _session_trace(500, calls=4, with_times=False).to_csv(path, index=False)
    trace = sessions.load_session_trace(str(path))
    workload = sessions.sample_session_workload(trace, 200, 2.0, seed=0, turn_gap=20.0, min_turn_gap=3.0)
    gaps = np.concatenate([np.diff(c["arrived_at"]) for _, c in workload.groupby("session_id")])
    assert gaps.min() >= 3.0 - 1e-9 and 10 < gaps.mean() < 35


def test_sampler_filters_long_calls_and_reports_exhaustion(tmp_path):
    path = tmp_path / "trace.csv"
    _session_trace(5, calls=20).to_csv(path, index=False)
    trace = sessions.load_session_trace(str(path), max_tokens=16 * 10 + 8)
    assert trace.frame["num_prefill_tokens"].max() == 16 * 10
    with pytest.raises(ValueError, match="sessions give only"):
        sessions.sample_session_workload(trace, 1000, 100.0, seed=0)


def _cached_scheduler(cached_hashes):
    from types import SimpleNamespace

    from frontier.kv_cache.base_kv_cache_manager import KVCacheManager

    manager = KVCacheManager(block_size=16, num_gpu_blocks=64, enable_caching=True,
                             caching_hash_algo="builtin", num_preallocate_tokens=0)
    for i, block_hash in enumerate(cached_hashes):
        # what a finished prefill leaves: the hash bound to a cached block
        manager.block_pool.cached_block_hash_to_block[block_hash] = {i: manager.block_pool.blocks[i]}
    return SimpleNamespace(_kv_cache_manager=manager, block_size=16, num_free_blocks=64,
                           num_blocks=64, running_requests=[], max_tokens_in_batch=2048,
                           batch_size_cap=256)


def test_peek_cached_prefix_counts_hits_without_touching_stats():
    scheduler = _cached_scheduler(range(100, 140))  # 40 blocks cached
    manager = scheduler._kv_cache_manager
    request = Request(0.0, 50 * 16, 10, block_hash_ids=list(range(100, 150)))
    assert manager.peek_cached_prefix_tokens(request) == 40 * 16
    # the hit stops at the first uncached block
    gap = Request(0.0, 50 * 16, 10, block_hash_ids=[100, 101, 7] + list(range(103, 150)))
    assert manager.peek_cached_prefix_tokens(gap) == 2 * 16
    # vLLM computes the last prompt token: a full hit gives back a block
    full = Request(0.0, 40 * 16, 10, block_hash_ids=list(range(100, 140)))
    assert manager.peek_cached_prefix_tokens(full) == 39 * 16
    assert manager.prefix_cache_stats.queries == 0 and manager.prefix_cache_stats.hits == 0
    assert manager.peek_cached_prefix_tokens(Request(0.0, 800, 10)) == 0


def test_sjf_sizes_jobs_by_uncached_prompt_tokens():
    from frontier.rl.frontier_env import _OracleRequest
    from frontier.rl.history import job_size
    from frontier.rl.prefix_cache import cached_prefix_tokens

    scheduler = _cached_scheduler(range(100, 160))  # 60 blocks cached
    # mostly cached long call (60 of 63 blocks cached: ~3 blocks of work),
    # against a shorter call with nothing cached (20 blocks of work)
    cached_long = Request(0.0, 63 * 16, 50, block_hash_ids=list(range(100, 163)))
    uncached_short = Request(0.0, 20 * 16, 50, block_hash_ids=list(range(500, 520)))
    cached = [cached_prefix_tokens(scheduler, r) for r in (uncached_short, cached_long)]
    assert cached == [0, 60 * 16]
    oracle = [_OracleRequest(r, c) for r, c in zip((uncached_short, cached_long), cached)]
    # prompt and total SJF both put the cached long call first
    for key in ("prefill_tokens", "total_tokens"):
        sizes = [job_size(r, key, 2048) for r in oracle]
        assert sizes[1] < sizes[0], key
    assert [job_size(r, "prefill_tokens", 2048) for r in oracle] == [20 * 16, 3 * 16]


def test_output_ids_follow_into_the_next_prompt():
    from frontier.trace_prep.common import Call, FreshIds, add_output_ids

    fresh = FreshIds()
    # call 1's prompt is call 0's prompt + 2 blocks (its output) + 1 more
    calls = [
        Call({}, 48, 40, np.arange(3)),
        Call({}, 96, 20, np.array([0, 1, 2, 50, 51, 52])),
        Call({}, 32, 20, np.array([0, 9])),  # does not hold call 1's prompt
    ]
    add_output_ids(calls, fresh)
    assert calls[0].ids.tolist() == [0, 1, 2, 50, 51]  # (48 + 40) // 16 = 5
    assert calls[1].ids[:6].tolist() == [0, 1, 2, 50, 51, 52] and len(calls[1].ids) == 7
    assert calls[1].ids[6] >= 1 << 48  # fresh: no later prompt holds it
    assert calls[2].ids.tolist()[:2] == [0, 9] and len(calls[2].ids) == 3


def test_compaction_keeps_system_prompt_and_extends_the_compacted_context():
    from frontier.trace_prep.common import Call, FreshIds, compact_session

    fresh = FreshIds()

    def call(prompt_blocks, ids):
        # output of 32 tokens = 2 blocks; ids cover prompt + output
        return Call({}, 16 * prompt_blocks, 32, np.asarray(ids))

    chain = np.arange(1000)
    calls = [
        call(40, chain[:42]),     # fits (compact_at 1024 tokens = 64 blocks)
        call(60, chain[:62]),     # 60*16 + 32 = 992: fits
        call(80, chain[:82]),     # too long: compacts
        call(90, chain[:92]),     # extends the compacted context by its 8 new blocks
        call(20, np.r_[chain[:5], 5000 + np.arange(17)]),  # starts over: fits, runs as is
    ]
    stats = Counter()
    out = compact_session(calls, keep_blocks=8, compact_at=1024, compact_to=320, fresh=fresh, stats=stats)
    assert len(out) == 5 and stats["compactions"] == 1
    # compaction: 8 kept + 12 summary + 18 new (80 - the 62 shared with call 1)
    assert [c.prompt // 16 for c in out] == [40, 60, 38, 38 + 2 + 8, 20]
    compacted, following = out[2].ids, out[3].ids
    assert compacted[:8].tolist() == chain[:8].tolist()  # system prompt kept
    assert (compacted[8:] >= 1 << 48).all()  # summary, new tokens, output: fresh
    assert following[:40].tolist() == compacted.tolist()  # keeps prompt + output
    assert out[4].ids.tolist() == calls[4].ids.tolist()
    for c in out:
        assert len(c.ids) == (c.prompt + c.output) // 16 and c.prompt + c.output <= 1024


@pytest.mark.parametrize("preset", ["tracelab", "freeinference", "random"])
def test_synthetic_sessions_are_valid_session_traces(preset):
    from frontier.trace_prep import synthetic

    params = (
        synthetic.randomized_params(3) if preset == "random" else synthetic.PRESETS[preset]
    )
    params = dataclasses.replace(params, num_sessions=60)
    frame = synthetic.generate_session_frame(params, seed=1)
    again = synthetic.generate_session_frame(params, seed=1)
    pd.testing.assert_frame_equal(frame, again)  # seeded
    assert (frame.num_prefill_tokens + frame.num_decode_tokens <= params.compact_at).all()
    position = {}
    by_session = frame.sort_values(["session_id", "step"]).groupby("session_id")
    for _, calls in by_session:
        previous = None
        for row in calls.itertuples():
            ids = np.asarray(ids_of(row.block_hash_ids))
            assert len(ids) == (row.num_prefill_tokens + row.num_decode_tokens) // 16
            for j, i in enumerate(ids.tolist()):  # chained: an id has one position
                assert position.setdefault(i, j) == j
            if previous is not None:  # every call keeps at least the system prompt
                assert (ids[:16] == previous[:16]).all()
            previous = ids


def test_randomized_synthetic_workloads_differ_by_seed():
    from frontier.trace_prep import sessions, synthetic

    params = [synthetic.randomized_params(seed) for seed in range(20)]
    for name, (low, high) in synthetic.RANDOM_RANGES.items():
        values = [getattr(p, name) for p in params]
        assert min(values) >= low and max(values) <= high + 1, name
        assert len(set(values)) > 3, name
    a = sessions.sample_named_workload("synthetic:random", 100, 1.0, seed=0)
    b = sessions.sample_named_workload("synthetic:random", 100, 1.0, seed=1)
    c = sessions.sample_named_workload("synthetic:random", 100, 1.0, seed=0)
    assert len(a) == len(b) == 100 and a.equals(c) and not a.equals(b)
    assert a["think_time"].notna().any() and (a["ids_cover_output"] == 1).all()
