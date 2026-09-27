from mindpage import (
    LRUPolicy,
    MemoryTier,
    Page,
    PageKind,
    TelemetryRecorder,
    WorkingSetManager,
    WorkloadSimulator,
)


def test_simulator_reports_hits_evictions_and_transfer_volume() -> None:
    telemetry = TelemetryRecorder()
    manager = WorkingSetManager(
        {MemoryTier.GPU: 20, MemoryTier.CPU: 100, MemoryTier.STORAGE: 100},
        event_sink=telemetry,
    )
    for page_id in ("a", "b", "c"):
        manager.register(Page(page_id, PageKind.EXPERT, 10), MemoryTier.CPU)

    result = WorkloadSimulator(manager, LRUPolicy(), telemetry).run(
        ["a", "b", "a", "c", "a"]
    )

    assert result.requests == 5
    assert result.hits == 2
    assert result.misses == 3
    assert result.evictions == 1
    assert result.hit_rate == 0.4
    assert result.transfer_bytes == 40
    assert manager.tier_of("a") is MemoryTier.GPU
    assert manager.tier_of("c") is MemoryTier.GPU
    assert manager.tier_of("b") is MemoryTier.CPU
