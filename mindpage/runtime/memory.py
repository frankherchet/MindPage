from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .events import EventSink, EventType, RuntimeEvent
from .models import MemoryTier, Page, PageState


class CapacityError(RuntimeError):
    """Raised when a placement would exceed a tier's configured capacity."""


@dataclass(slots=True)
class _ResidentPage:
    page: Page
    tier: MemoryTier
    last_access: int
    access_count: int = 0


_TIER_RANK = {
    MemoryTier.GPU: 0,
    MemoryTier.CPU: 1,
    MemoryTier.STORAGE: 2,
}


class WorkingSetManager:
    """Deterministic, framework-independent residency bookkeeping.

    This implementation does not move tensors yet. It gives later backends one
    place to reason about logical residency, budgets, access history, and events.
    """

    def __init__(
        self,
        capacities: Mapping[MemoryTier, int],
        event_sink: EventSink | None = None,
    ) -> None:
        self._capacities = {tier: int(capacities.get(tier, 0)) for tier in MemoryTier}
        if any(value < 0 for value in self._capacities.values()):
            raise ValueError("capacities must be non-negative")
        self._pages: dict[str, _ResidentPage] = {}
        self._clock = 0
        self._event_sink = event_sink

    def capacity(self, tier: MemoryTier) -> int:
        return self._capacities[tier]

    def usage(self, tier: MemoryTier) -> int:
        return sum(
            resident.page.size_bytes
            for resident in self._pages.values()
            if resident.tier == tier
        )

    def available(self, tier: MemoryTier) -> int:
        return self.capacity(tier) - self.usage(tier)

    def register(self, page: Page, tier: MemoryTier) -> None:
        if page.page_id in self._pages:
            raise ValueError(f"page already registered: {page.page_id}")
        self._ensure_capacity(tier, page.size_bytes)
        self._clock += 1
        self._pages[page.page_id] = _ResidentPage(page, tier, self._clock)
        self._emit(EventType.REGISTER, page, target=tier)

    def tier_of(self, page_id: str) -> MemoryTier:
        return self._get(page_id).tier

    def state(self, page_id: str) -> PageState:
        resident = self._get(page_id)
        return PageState(
            page=resident.page,
            tier=resident.tier,
            last_access=resident.last_access,
            access_count=resident.access_count,
        )

    def touch(self, page_id: str) -> None:
        resident = self._get(page_id)
        self._clock += 1
        resident.last_access = self._clock
        resident.access_count += 1
        self._emit(EventType.ACCESS, resident.page, target=resident.tier)

    def move(self, page_id: str, target: MemoryTier) -> None:
        resident = self._get(page_id)
        source = resident.tier
        if source == target:
            self.touch(page_id)
            return
        self._ensure_capacity(target, resident.page.size_bytes)
        resident.tier = target
        self._clock += 1
        resident.last_access = self._clock

        if _TIER_RANK[target] < _TIER_RANK[source]:
            event_type = EventType.PROMOTE
        elif _TIER_RANK[target] > _TIER_RANK[source]:
            event_type = EventType.DEMOTE
        else:
            event_type = EventType.MOVE
        self._emit(event_type, resident.page, source=source, target=target)

    def pages_in(self, tier: MemoryTier) -> tuple[Page, ...]:
        return tuple(state.page for state in self.states_in(tier))

    def states_in(self, tier: MemoryTier) -> tuple[PageState, ...]:
        residents = sorted(
            (r for r in self._pages.values() if r.tier == tier),
            key=lambda r: (-r.page.priority, -r.last_access, r.page.page_id),
        )
        return tuple(
            PageState(r.page, r.tier, r.last_access, r.access_count) for r in residents
        )

    def snapshot(self) -> dict[str, MemoryTier]:
        return {page_id: resident.tier for page_id, resident in self._pages.items()}

    def _get(self, page_id: str) -> _ResidentPage:
        try:
            return self._pages[page_id]
        except KeyError as exc:
            raise KeyError(f"unknown page: {page_id}") from exc

    def _ensure_capacity(self, tier: MemoryTier, incoming_bytes: int) -> None:
        if incoming_bytes > self.available(tier):
            raise CapacityError(
                f"tier {tier.value!r} has {self.available(tier)} bytes available; "
                f"{incoming_bytes} bytes requested"
            )

    def _emit(
        self,
        event_type: EventType,
        page: Page,
        source: MemoryTier | None = None,
        target: MemoryTier | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        self._event_sink.record(
            RuntimeEvent(
                sequence=self._clock,
                event_type=event_type,
                page_id=page.page_id,
                page_kind=page.kind,
                size_bytes=page.size_bytes,
                source=source,
                target=target,
            )
        )
