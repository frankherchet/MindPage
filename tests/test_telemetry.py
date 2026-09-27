from mindpage import MemoryTier, Page, PageKind, TelemetryRecorder, WorkingSetManager
from mindpage.runtime.events import EventType


def test_recorder_counts_access_and_transfer_bytes() -> None:
    telemetry = TelemetryRecorder()
    manager = WorkingSetManager(
        {MemoryTier.GPU: 100, MemoryTier.CPU: 100, MemoryTier.STORAGE: 100},
        event_sink=telemetry,
    )
    manager.register(Page("expert", PageKind.EXPERT, 25), MemoryTier.CPU)
    manager.move("expert", MemoryTier.GPU)
    manager.touch("expert")
    manager.move("expert", MemoryTier.CPU)

    metrics = telemetry.metrics()
    assert metrics.promotions == 1
    assert metrics.demotions == 1
    assert metrics.accesses == 1
    assert metrics.transfers == 2
    assert metrics.transfer_bytes == 50
    assert [event.event_type for event in telemetry.events] == [
        EventType.REGISTER,
        EventType.PROMOTE,
        EventType.ACCESS,
        EventType.DEMOTE,
    ]
