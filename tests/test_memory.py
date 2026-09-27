import pytest

from mindpage import CapacityError, MemoryTier, Page, PageKind, WorkingSetManager


def manager() -> WorkingSetManager:
    return WorkingSetManager(
        {
            MemoryTier.GPU: 100,
            MemoryTier.CPU: 1_000,
            MemoryTier.STORAGE: 10_000,
        }
    )


def test_register_tracks_usage_and_tier() -> None:
    runtime = manager()
    page = Page("expert.1", PageKind.EXPERT, 40)

    runtime.register(page, MemoryTier.GPU)

    assert runtime.tier_of("expert.1") is MemoryTier.GPU
    assert runtime.usage(MemoryTier.GPU) == 40
    assert runtime.available(MemoryTier.GPU) == 60


def test_move_updates_both_tiers() -> None:
    runtime = manager()
    runtime.register(Page("adapter.project", PageKind.ADAPTER, 25), MemoryTier.CPU)

    runtime.move("adapter.project", MemoryTier.GPU)

    assert runtime.usage(MemoryTier.CPU) == 0
    assert runtime.usage(MemoryTier.GPU) == 25


def test_capacity_is_enforced() -> None:
    runtime = manager()
    runtime.register(Page("base", PageKind.BASE, 80), MemoryTier.GPU)

    with pytest.raises(CapacityError):
        runtime.register(Page("expert", PageKind.EXPERT, 30), MemoryTier.GPU)


def test_page_validation() -> None:
    with pytest.raises(ValueError):
        Page("", PageKind.KV, 1)

    with pytest.raises(ValueError):
        Page("kv.1", PageKind.KV, 0)


def test_pages_are_ordered_by_priority_then_recency() -> None:
    runtime = manager()
    runtime.register(Page("cold", PageKind.EXPERT, 10, priority=0.0), MemoryTier.GPU)
    runtime.register(Page("hot", PageKind.EXPERT, 10, priority=1.0), MemoryTier.GPU)
    runtime.register(Page("newer", PageKind.EXPERT, 10, priority=0.0), MemoryTier.GPU)

    assert [page.page_id for page in runtime.pages_in(MemoryTier.GPU)] == [
        "hot",
        "newer",
        "cold",
    ]


def test_touch_tracks_access_count() -> None:
    runtime = manager()
    runtime.register(Page("expert.1", PageKind.EXPERT, 10), MemoryTier.GPU)
    runtime.touch("expert.1")
    runtime.touch("expert.1")

    assert runtime.state("expert.1").access_count == 2
