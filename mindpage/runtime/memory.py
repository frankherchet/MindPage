from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import MemoryTier, Page


class CapacityError(RuntimeError):
    """Raised when a placement would exceed a tier's configured capacity."""


@dataclass(slots=True)
class _ResidentPage:
    page: Page
    tier: MemoryTier
    last_access: int


class WorkingSetManager:
    """Deterministic, framework-independent residency bookkeeping.

    This first implementation deliberately does not move tensors. It gives later
    runtime backends one place to reason about logical residency and budgets.
    """

    def __init__(self, capacities: Mapping[MemoryTier, int]) -> None:
        self._capacities = {tier: int(capacities.get(tier, 0)) for tier in MemoryTier}
        if any(value < 0 for value in self._capacities.values()):
            raise ValueError("capacities must be non-negative")
        self._pages: dict[str, _ResidentPage] = {}
        self._clock = 0

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

    def tier_of(self, page_id: str) -> MemoryTier:
        return self._get(page_id).tier

    def touch(self, page_id: str) -> None:
        resident = self._get(page_id)
        self._clock += 1
        resident.last_access = self._clock

    def move(self, page_id: str, target: MemoryTier) -> None:
        resident = self._get(page_id)
        if resident.tier == target:
            self.touch(page_id)
            return
        self._ensure_capacity(target, resident.page.size_bytes)
        resident.tier = target
        self.touch(page_id)

    def pages_in(self, tier: MemoryTier) -> tuple[Page, ...]:
        residents = sorted(
            (r for r in self._pages.values() if r.tier == tier),
            key=lambda r: (-r.page.priority, -r.last_access, r.page.page_id),
        )
        return tuple(r.page for r in residents)

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
