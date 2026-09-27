from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mindpage.runtime.memory import CapacityError, WorkingSetManager
from mindpage.runtime.models import MemoryTier
from mindpage.runtime.policies import PlacementPolicy
from mindpage.telemetry.recorder import TelemetryRecorder


@dataclass(frozen=True, slots=True)
class SimulationResult:
    requests: int
    hits: int
    misses: int
    evictions: int
    hit_rate: float
    transfer_bytes: int


class WorkloadSimulator:
    """Replay logical page accesses against a placement policy."""

    def __init__(
        self,
        manager: WorkingSetManager,
        policy: PlacementPolicy,
        telemetry: TelemetryRecorder,
        hot_tier: MemoryTier = MemoryTier.GPU,
    ) -> None:
        self._manager = manager
        self._policy = policy
        self._telemetry = telemetry
        self._hot_tier = hot_tier

    def run(self, page_ids: Iterable[str]) -> SimulationResult:
        requests = hits = misses = evictions = 0

        for page_id in page_ids:
            requests += 1
            state = self._manager.state(page_id)
            if state.tier is self._hot_tier:
                hits += 1
                self._manager.touch(page_id)
                continue

            misses += 1
            bytes_needed = max(
                0,
                state.page.size_bytes - self._manager.available(self._hot_tier),
            )
            victims = self._policy.select_victims(
                self._manager.states_in(self._hot_tier), bytes_needed
            )

            for victim_id in victims:
                self._demote(victim_id)
                evictions += 1

            if state.page.size_bytes > self._manager.available(self._hot_tier):
                raise CapacityError(
                    f"policy {type(self._policy).__name__} could not free enough "
                    f"capacity for {page_id!r}"
                )

            self._manager.move(page_id, self._hot_tier)
            self._manager.touch(page_id)

        metrics = self._telemetry.metrics()
        return SimulationResult(
            requests=requests,
            hits=hits,
            misses=misses,
            evictions=evictions,
            hit_rate=(hits / requests) if requests else 0.0,
            transfer_bytes=metrics.transfer_bytes,
        )

    def _demote(self, page_id: str) -> None:
        state = self._manager.state(page_id)
        for tier in (MemoryTier.CPU, MemoryTier.STORAGE):
            if tier is self._hot_tier or tier is state.tier:
                continue
            if self._manager.available(tier) >= state.page.size_bytes:
                self._manager.move(page_id, tier)
                return
        raise CapacityError(f"no colder tier has capacity for {page_id!r}")
