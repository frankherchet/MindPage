from __future__ import annotations

from dataclasses import dataclass

from mindpage.runtime.events import EventType, RuntimeEvent


@dataclass(frozen=True, slots=True)
class TelemetryMetrics:
    events: int
    accesses: int
    promotions: int
    demotions: int
    transfers: int
    transfer_bytes: int


class TelemetryRecorder:
    """In-memory event recorder used by tests, simulations, and early backends."""

    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []

    def record(self, event: RuntimeEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)

    def metrics(self) -> TelemetryMetrics:
        promotions = sum(e.event_type is EventType.PROMOTE for e in self._events)
        demotions = sum(e.event_type is EventType.DEMOTE for e in self._events)
        transfer_events = tuple(
            e
            for e in self._events
            if e.event_type in {EventType.PROMOTE, EventType.DEMOTE, EventType.MOVE}
        )
        return TelemetryMetrics(
            events=len(self._events),
            accesses=sum(e.event_type is EventType.ACCESS for e in self._events),
            promotions=promotions,
            demotions=demotions,
            transfers=len(transfer_events),
            transfer_bytes=sum(e.size_bytes for e in transfer_events),
        )
