from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class MemoryTier(StrEnum):
    """Logical placement tiers ordered from fastest to coldest storage."""

    GPU = "gpu"
    CPU = "cpu"
    STORAGE = "storage"


class PageKind(StrEnum):
    """Logical state classes managed by MindPage."""

    BASE = "base"
    EXPERT = "expert"
    ADAPTER = "adapter"
    KV = "kv"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class Page:
    """A logical unit of state that can be placed on a memory tier.

    A MindPage page is intentionally not tied to OS or CUDA page sizes. Backends
    may map one logical page to any physical representation that is appropriate.
    """

    page_id: str
    kind: PageKind
    size_bytes: int
    pinned: bool = False
    priority: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.page_id:
            raise ValueError("page_id must not be empty")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")


@dataclass(frozen=True, slots=True)
class PageState:
    """Observable runtime state for a logical page."""

    page: Page
    tier: MemoryTier
    last_access: int
    access_count: int
