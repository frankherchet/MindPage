from mindpage import (
    FrequencyPolicy,
    LRUPolicy,
    MemoryTier,
    Page,
    PageKind,
    PriorityCostPolicy,
    WorkingSetManager,
)


def runtime() -> WorkingSetManager:
    return WorkingSetManager(
        {MemoryTier.GPU: 100, MemoryTier.CPU: 1000, MemoryTier.STORAGE: 1000}
    )


def test_lru_selects_oldest_unpinned_page() -> None:
    manager = runtime()
    manager.register(Page("old", PageKind.EXPERT, 20), MemoryTier.GPU)
    manager.register(Page("pinned", PageKind.EXPERT, 20, pinned=True), MemoryTier.GPU)
    manager.register(Page("new", PageKind.EXPERT, 20), MemoryTier.GPU)
    manager.touch("new")

    assert LRUPolicy().select_victims(manager.states_in(MemoryTier.GPU), 20) == ("old",)


def test_frequency_prefers_less_used_page() -> None:
    manager = runtime()
    manager.register(Page("frequent", PageKind.EXPERT, 20), MemoryTier.GPU)
    manager.register(Page("rare", PageKind.EXPERT, 20), MemoryTier.GPU)
    manager.touch("frequent")
    manager.touch("frequent")
    manager.touch("rare")

    assert FrequencyPolicy().select_victims(manager.states_in(MemoryTier.GPU), 20) == ("rare",)


def test_priority_cost_prefers_low_priority_large_page() -> None:
    manager = runtime()
    manager.register(Page("small", PageKind.EXPERT, 10, priority=0), MemoryTier.GPU)
    manager.register(Page("large", PageKind.EXPERT, 30, priority=0), MemoryTier.GPU)
    manager.register(Page("important", PageKind.EXPERT, 40, priority=10), MemoryTier.GPU)

    assert PriorityCostPolicy().select_victims(
        manager.states_in(MemoryTier.GPU), 25
    ) == ("large",)
