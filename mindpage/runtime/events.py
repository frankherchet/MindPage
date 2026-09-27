from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .models import MemoryTier, PageKind


class EventType(StrEnum):
    REGISTER = "register"
    ACCESS = "access"
    PROMOTE = "promote"
    DEMOTE = "demote"
    MOVE = "move"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    sequence: int
    event_type: EventType
    page_id: str
    page_kind: PageKind
    size_bytes: int
    source: MemoryTier | None = None
    target: MemoryTier | None = None


class EventSink(Protocol):
    def record(self, event: RuntimeEvent) -> None: ...
