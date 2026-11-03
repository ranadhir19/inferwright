"""Smoke tests for 02-paged-kv.

Accounting invariants (used <= allocated <= total; free returns all slots),
block-id uniqueness, first-fit/coalescing behaviour, the external
fragmentation counter, and run-to-run determinism of the experiment
pipeline. Runnable with pytest or plain `python test_smoke.py`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")))

from llm_inference_engine.paged_kv import (  # noqa: E402
    BlockManager,
    ContiguousAllocator,
    gen_workload,
)


def test_paged_accounting_invariants():
    bm = BlockManager(total_slots=1024, block_size=16)
    assert bm.total_slots() == 1024
    assert bm.used_slots() == 0
    assert bm.free_slots() == 1024

    b0 = bm.alloc(0, 100)   # 7 blocks
    b1 = bm.alloc(1, 16)    # 1 block
    assert len(b0) == 7 and len(b1) == 1
    assert bm.used_slots() == 116
    assert bm.allocated_slots() == 8 * 16
    assert bm.used_slots() <= bm.allocated_slots() <= bm.total_slots()
    assert bm.free_slots() == 1024 - 8 * 16

    freed = bm.free(0)
    assert freed == 7 * 16
    assert bm.used_slots() == 16
    assert bm.free_slots() == 1024 - 16

    bm.free(1)
    assert bm.used_slots() == 0
    assert bm.free_slots() == bm.total_slots()


def test_paged_block_ids_unique_and_reused():
    bm = BlockManager(total_slots=256, block_size=16)
    seen = set()
    for i in range(8):
        blocks = bm.alloc(i, 20)
        assert blocks is not None
        assert not (set(blocks) & seen), "block id handed out twice"
        seen |= set(blocks)
    assert bm.alloc(99, 16) is None  # pool full
    bm.free(3)
    b = bm.alloc(99, 16)
    assert b is not None and set(b) <= seen  # recycled ids


def test_paged_fragmentation_stats():
    bm = BlockManager(total_slots=160, block_size=16)  # 10 blocks
    bm.alloc(0, 100)  # 7 blocks, 12 slots wasted
    f = bm.fragmentation()
    assert f["allocated_slots"] == 112
    assert f["used_slots"] == 100
    assert f["internal_waste_slots"] == 12
    assert abs(f["internal_waste_pct"] - 100 * 12 / 112) < 1e-9
    assert f["external_frag_events"] == 0  # fungible blocks: never external


def test_contiguous_first_fit_and_coalescing():
    ca = ContiguousAllocator(total_slots=64, max_len=16)
    assert ca.alloc(0, 10) == (0, 16)
    assert ca.alloc(1, 12) == (16, 16)
    assert ca.alloc(2, 5) == (32, 16)
    assert ca.free_slots() == 16
    assert ca.alloc(3, 16) == (48, 16)
    assert ca.alloc(4, 1) is None  # full
    assert ca.external_frag_events == 0  # truly full, not fragmented

    ca.free(1)  # frees (16,16); neighbours (0,16),(32,16) occupied
    assert ca.free_slots() == 16
    assert ca.alloc(4, 3) == (16, 16)  # first-fit reuses the hole
    ca.free(0)
    ca.free(2)
    ca.free(3)
    ca.free(4)
    assert ca.free_slots() == 64  # all coalesced back
    f = ca.fragmentation()
    assert f["used_slots"] == 0 and f["internal_waste_slots"] == 0


def test_contiguous_external_frag_counter():
    # White-box: free memory split into pieces smaller than max_len.
    ca = ContiguousAllocator(total_slots=12, max_len=6)
    ca._free = [(0, 3), (6, 3)]  # 6 free slots, no 6-run
    assert ca.alloc(0, 2) is None
    assert ca.external_frag_events == 1
    assert ca.free_slots() == 6


def test_contiguous_uniform_reservation_waste():
    ca = ContiguousAllocator(total_slots=2048, max_len=2048)
    ca.alloc(0, 128)
    f = ca.fragmentation()
    assert f["allocated_slots"] == 2048
    assert f["internal_waste_slots"] == 2048 - 128


def test_workload_reproducible_and_clipped():
    r1 = gen_workload(np.random.default_rng(11), n=500)
    r2 = gen_workload(np.random.default_rng(11), n=500)
    assert [(r.seq_id, r.arrival, r.length) for r in r1] == [
        (r.seq_id, r.arrival, r.length) for r in r2]
    assert all(1 <= r.length <= 2048 for r in r1)
    assert all(r.departure == r.arrival + r.length for r in r1)
    arrivals = [r.arrival for r in r1]
    assert arrivals == sorted(arrivals)
    assert all(0.0 <= a <= 600.0 for a in arrivals)


def test_experiment_pipeline_deterministic():
    exp_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
    sys.path.insert(0, exp_dir)
    import run_experiments as rexp

    res1, res2 = rexp.run_pipeline(), rexp.run_pipeline()
    assert rexp.headline(res1) == rexp.headline(res2)
    # Sanity on the actual physics: paged wastes far less than contiguous.
    assert res1["exp1_paged"]["internal_waste_pct"] < 10.0
    assert res1["exp1_contig"]["internal_waste_pct"] > 50.0
    assert (res1["exp1_paged"]["max_concurrent"]
            > res1["exp1_contig"]["max_concurrent"])


if __name__ == "__main__":
    test_paged_accounting_invariants()
    test_paged_block_ids_unique_and_reused()
    test_paged_fragmentation_stats()
    test_contiguous_first_fit_and_coalescing()
    test_contiguous_external_frag_counter()
    test_contiguous_uniform_reservation_waste()
    test_workload_reproducible_and_clipped()
    test_experiment_pipeline_deterministic()
    print("all smoke tests passed")
