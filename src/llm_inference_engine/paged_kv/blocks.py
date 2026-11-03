"""Paged and contiguous KV-cache allocators.

Both allocators manage a fixed pool of token slots (the analytical GPU
memory budget) and track the same accounting so experiments can compare
them directly:

- ``used_slots()``      — tokens actually occupied by live sequences.
- ``allocated_slots()`` — slots reserved (block-rounded for paged,
                          ``max_len``-rounded for contiguous).
- ``total_slots()``     — pool capacity.
- ``fragmentation()``   — internal waste and external-fragmentation events.

``alloc`` returns ``None`` when the request cannot be satisfied instead of
raising, so the simulation driver can count rejections. A failed allocation
where total free slots would have sufficed counts as one
*external fragmentation event*.
"""

from __future__ import annotations

import math


class BlockManager:
    """Paged KV-cache block manager (vLLM-style).

    The pool holds ``num_blocks = total_slots // block_size`` physical
    blocks. Allocating ``n_tokens`` for a sequence takes
    ``ceil(n_tokens / block_size)`` blocks off a free list; the returned
    block-id list is the sequence's block table. Blocks are fungible, so
    an allocation fails only when fewer than ``need`` blocks are free in
    total — external fragmentation is impossible by construction.
    """

    def __init__(self, total_slots: int, block_size: int) -> None:
        if total_slots <= 0:
            raise ValueError("total_slots must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.num_blocks = total_slots // block_size
        if self.num_blocks == 0:
            raise ValueError("total_slots smaller than one block")
        self._free: list[int] = list(range(self.num_blocks))
        self._seq_blocks: dict[int, list[int]] = {}
        self._seq_tokens: dict[int, int] = {}
        self.external_frag_events = 0

    # -- allocation -----------------------------------------------------
    def alloc(self, seq_id: int, n_tokens: int) -> list[int] | None:
        """Allocate blocks for ``seq_id``; ``None`` if the pool is full."""
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        if seq_id in self._seq_blocks:
            raise ValueError(f"seq_id {seq_id} already allocated")
        need = math.ceil(n_tokens / self.block_size)
        if len(self._free) < need:
            # Blocks are fungible: failure here means the pool is truly
            # full. Keep the honesty check anyway.
            if self.free_slots() >= need * self.block_size:
                self.external_frag_events += 1
            return None
        blocks = [self._free.pop() for _ in range(need)]
        self._seq_blocks[seq_id] = blocks
        self._seq_tokens[seq_id] = n_tokens
        return list(blocks)

    def free(self, seq_id: int) -> int:
        """Release ``seq_id``'s blocks; returns slots returned to the pool."""
        blocks = self._seq_blocks.pop(seq_id)
        del self._seq_tokens[seq_id]
        self._free.extend(blocks)
        return len(blocks) * self.block_size

    # -- accounting -----------------------------------------------------
    def used_slots(self) -> int:
        """Tokens actually occupied by live sequences."""
        return sum(self._seq_tokens.values())

    def allocated_slots(self) -> int:
        """Slots reserved, including unused block tails."""
        return sum(len(b) for b in self._seq_blocks.values()) * self.block_size

    def total_slots(self) -> int:
        return self.num_blocks * self.block_size

    def free_slots(self) -> int:
        return len(self._free) * self.block_size

    def live_sequences(self) -> int:
        return len(self._seq_blocks)

    def fragmentation(self) -> dict:
        allocated = self.allocated_slots()
        used = self.used_slots()
        waste = allocated - used
        return {
            "allocated_slots": allocated,
            "used_slots": used,
            "internal_waste_slots": waste,
            "internal_waste_pct": 100.0 * waste / allocated if allocated else 0.0,
            "external_frag_events": self.external_frag_events,
            "live_sequences": self.live_sequences(),
            "free_slots": self.free_slots(),
        }


class ContiguousAllocator:
    """Naive baseline: reserve ``max_len`` slots contiguously, first-fit.

    Each admitted sequence occupies a single contiguous run of ``max_len``
    slots regardless of its actual length. Free memory is tracked as a
    sorted list of ``(start, length)`` intervals; allocation scans for the
    first interval that fits and splits it, and ``free`` coalesces adjacent
    intervals. Allocation fails when no single interval fits even though
    total free memory suffices — that failure is recorded as one external
    fragmentation event.
    """

    def __init__(self, total_slots: int, max_len: int) -> None:
        if total_slots <= 0:
            raise ValueError("total_slots must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.max_len = max_len
        self._capacity = total_slots
        self._free: list[tuple[int, int]] = [(0, total_slots)]
        self._allocs: dict[int, tuple[int, int]] = {}  # seq_id -> (start, n_tokens)
        self.external_frag_events = 0

    # -- allocation -----------------------------------------------------
    def alloc(self, seq_id: int, n_tokens: int) -> tuple[int, int] | None:
        """Reserve ``max_len`` contiguous slots; ``None`` on failure."""
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        if seq_id in self._allocs:
            raise ValueError(f"seq_id {seq_id} already allocated")
        for i, (start, length) in enumerate(self._free):
            if length >= self.max_len:
                self._allocs[seq_id] = (start, n_tokens)
                if length == self.max_len:
                    del self._free[i]
                else:
                    self._free[i] = (start + self.max_len, length - self.max_len)
                return (start, self.max_len)
        # No single interval fits: external fragmentation if the total
        # free memory would have been enough.
        if self.free_slots() >= self.max_len:
            self.external_frag_events += 1
        return None

    def free(self, seq_id: int) -> int:
        """Release ``seq_id``'s reservation; returns slots returned."""
        start, _ = self._allocs.pop(seq_id)
        interval = (start, self.max_len)
        # Insert keeping the list sorted by start, then coalesce.
        idx = 0
        while idx < len(self._free) and self._free[idx][0] < start:
            idx += 1
        self._free.insert(idx, interval)
        if idx > 0:
            ps, pl = self._free[idx - 1]
            if ps + pl == start:
                self._free[idx - 1] = (ps, pl + self.max_len)
                del self._free[idx]
                idx -= 1
        if idx + 1 < len(self._free):
            ns, nl = self._free[idx + 1]
            cs, cl = self._free[idx]
            if cs + cl == ns:
                self._free[idx] = (cs, cl + nl)
                del self._free[idx + 1]
        return self.max_len

    # -- accounting -----------------------------------------------------
    def used_slots(self) -> int:
        return sum(n for _, n in self._allocs.values())

    def allocated_slots(self) -> int:
        return len(self._allocs) * self.max_len

    def total_slots(self) -> int:
        return self._capacity

    def free_slots(self) -> int:
        return sum(length for _, length in self._free)

    def live_sequences(self) -> int:
        return len(self._allocs)

    def fragmentation(self) -> dict:
        allocated = self.allocated_slots()
        used = self.used_slots()
        waste = allocated - used
        return {
            "allocated_slots": allocated,
            "used_slots": used,
            "internal_waste_slots": waste,
            "internal_waste_pct": 100.0 * waste / allocated if allocated else 0.0,
            "external_frag_events": self.external_frag_events,
            "live_sequences": self.live_sequences(),
            "free_slots": self.free_slots(),
        }
