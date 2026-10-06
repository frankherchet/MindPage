"""Turn MoE routing traces into page accesses and replay them through policies.

A trace holds, for every token, the top-k expert IDs each MoE layer selected.
Every ``(layer, expert)`` pair is one page ``L{layer}.E{expert}``; all experts
of a model have the same size. Tokens are ``prefill`` (the prompt) or
``decode`` (one generated token per step); expert paging is decided in decode.
"""

from __future__ import annotations

import heapq
from itertools import chain
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from mindpage.runtime.memory import WorkingSetManager
from mindpage.runtime.models import MemoryTier, Page, PageKind
from mindpage.runtime.policies import (
    FrequencyPolicy,
    LRUPolicy,
    PlacementPolicy,
    PriorityCostPolicy,
)
from mindpage.telemetry.recorder import TelemetryRecorder

from .workload import SimulationResult, WorkloadSimulator


def expert_page_id(layer: int, expert: int) -> str:
    return f"L{layer}.E{expert}"


@dataclass(frozen=True, slots=True)
class ExpertTrace:
    """Routed experts per token: ``experts[token][layer]`` is the top-k list."""

    experts: Sequence[Sequence[Sequence[int]]]
    prefill_tokens: int
    num_layers: int
    num_experts: int
    expert_bytes: int

    @property
    def decode_tokens(self) -> int:
        return len(self.experts) - self.prefill_tokens

    def page_ids(self) -> list[str]:
        """All expert pages of the model, layer-major."""
        return [
            expert_page_id(layer, expert)
            for layer in range(self.num_layers)
            for expert in range(self.num_experts)
        ]

    def accesses(self, phase: str = "decode") -> Iterator[str]:
        """Page accesses in execution order: token by token, layer by layer."""
        if phase == "decode":
            tokens = self.experts[self.prefill_tokens :]
        elif phase == "prefill":
            tokens = self.experts[: self.prefill_tokens]
        elif phase == "all":
            tokens = self.experts
        else:
            raise ValueError(f"unknown phase: {phase!r}")
        for token in tokens:
            for layer, chosen in enumerate(token):
                for expert in sorted(int(e) for e in chosen):
                    yield expert_page_id(layer, expert)


def simulate(
    accesses: Iterable[str],
    page_ids: Sequence[str],
    page_bytes: int,
    capacity_bytes: int,
    policy: PlacementPolicy,
) -> SimulationResult:
    """Replay through ``WorkloadSimulator``; every page starts in the CPU tier."""
    telemetry = TelemetryRecorder()
    manager = WorkingSetManager(
        {MemoryTier.GPU: capacity_bytes, MemoryTier.CPU: page_bytes * len(page_ids)},
        event_sink=telemetry,
    )
    for page_id in page_ids:
        manager.register(Page(page_id, PageKind.EXPERT, page_bytes), MemoryTier.CPU)
    return WorkloadSimulator(manager, policy, telemetry).run(accesses)


def replay_equal_size(
    accesses: Iterable[str],
    page_ids: Sequence[str],
    slots: int,
    policy: PlacementPolicy,
    warmup: Iterable[str] = (),
) -> tuple[int, int, int]:
    """Fast equivalent of :func:`simulate` for equal-size, priority-0 pages.

    ``WorkloadSimulator`` asks the policy to sort every GPU page on each miss,
    which is too slow for tens of thousands of experts and accesses. With equal
    page sizes each miss in a full cache evicts exactly one page, so the three
    built-in policies reduce to an ordered dict or a heap. ``PriorityCostPolicy``
    then orders exactly like ``FrequencyPolicy``. ``warmup`` accesses (e.g. the
    prefill) are replayed first but not counted. Returns
    ``(hits, misses, evictions)``; the tests check equality with ``simulate``.
    """
    if isinstance(policy, LRUPolicy):
        by_frequency = False
    elif isinstance(policy, FrequencyPolicy | PriorityCostPolicy):
        by_frequency = True
    else:
        raise TypeError(f"no fast replay for {type(policy).__name__}")

    # Mirror WorkingSetManager's clock: registration, then +1 per move/touch.
    clock = len(page_ids)
    last_access = {page_id: i + 1 for i, page_id in enumerate(page_ids)}
    count = dict.fromkeys(page_ids, 0)
    lru: OrderedDict[str, None] = OrderedDict()
    resident: set[str] = set()
    heap: list[tuple[int, int, str]] = []
    hits = misses = evictions = 0

    for page_id in chain(warmup, [None], accesses):
        if page_id is None:
            hits = misses = evictions = 0
            continue
        if page_id in resident:
            hits += 1
        else:
            misses += 1
            if slots <= 0:
                raise ValueError("capacity holds no page")
            if len(resident) >= slots:
                if by_frequency:
                    while True:
                        c, t, victim = heapq.heappop(heap)
                        if victim in resident and (c, t) == (count[victim], last_access[victim]):
                            break
                else:
                    victim, _ = lru.popitem(last=False)
                resident.discard(victim)
                evictions += 1
            resident.add(page_id)
            clock += 1  # move
        clock += 1  # touch
        last_access[page_id] = clock
        count[page_id] += 1
        if by_frequency:
            heapq.heappush(heap, (count[page_id], clock, page_id))
        else:
            lru[page_id] = None
            lru.move_to_end(page_id)
    return hits, misses, evictions


def reuse_distances(accesses: Iterable[str]) -> list[int | None]:
    """Accesses since the previous use of the same page; ``None`` on first use."""
    last: dict[str, int] = {}
    distances: list[int | None] = []
    for index, page_id in enumerate(accesses):
        previous = last.get(page_id)
        distances.append(None if previous is None else index - previous - 1)
        last[page_id] = index
    return distances
