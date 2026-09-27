from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .models import PageState


class PlacementPolicy(ABC):
    """Select pages to evict from a constrained memory tier."""

    @abstractmethod
    def select_victims(
        self, candidates: Iterable[PageState], bytes_needed: int
    ) -> tuple[str, ...]:
        raise NotImplementedError

    @staticmethod
    def _take_until(states: Iterable[PageState], bytes_needed: int) -> tuple[str, ...]:
        if bytes_needed <= 0:
            return ()
        victims: list[str] = []
        freed = 0
        for state in states:
            if state.page.pinned:
                continue
            victims.append(state.page.page_id)
            freed += state.page.size_bytes
            if freed >= bytes_needed:
                return tuple(victims)
        return tuple(victims)


class LRUPolicy(PlacementPolicy):
    """Evict the least recently accessed unpinned pages first."""

    def select_victims(
        self, candidates: Iterable[PageState], bytes_needed: int
    ) -> tuple[str, ...]:
        ordered = sorted(candidates, key=lambda s: (s.last_access, s.page.page_id))
        return self._take_until(ordered, bytes_needed)


class FrequencyPolicy(PlacementPolicy):
    """Evict low-frequency pages first, then break ties by recency."""

    def select_victims(
        self, candidates: Iterable[PageState], bytes_needed: int
    ) -> tuple[str, ...]:
        ordered = sorted(
            candidates,
            key=lambda s: (s.access_count, s.last_access, s.page.page_id),
        )
        return self._take_until(ordered, bytes_needed)


class PriorityCostPolicy(PlacementPolicy):
    """Prefer evicting low-value pages that free useful capacity.

    Lower explicit priority and lower observed frequency are treated as lower
    value. For otherwise similar pages, larger pages are chosen first so the
    scheduler can satisfy a capacity request with fewer transfers.
    """

    def select_victims(
        self, candidates: Iterable[PageState], bytes_needed: int
    ) -> tuple[str, ...]:
        ordered = sorted(
            candidates,
            key=lambda s: (
                s.page.priority,
                s.access_count,
                -s.page.size_bytes,
                s.last_access,
                s.page.page_id,
            ),
        )
        return self._take_until(ordered, bytes_needed)
