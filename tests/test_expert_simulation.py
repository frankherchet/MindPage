import random

import pytest

from mindpage import FrequencyPolicy, LRUPolicy, PriorityCostPolicy
from mindpage.simulation.experts import (
    ExpertTrace,
    expert_page_id,
    replay_equal_size,
    reuse_distances,
    simulate,
)


def _trace() -> ExpertTrace:
    # 2 prefill + 2 decode tokens, 2 layers, top-2 of 4 experts.
    return ExpertTrace(
        experts=[
            [[0, 1], [2, 3]],
            [[1, 2], [0, 3]],
            [[3, 1], [0, 1]],
            [[1, 3], [1, 0]],
        ],
        prefill_tokens=2,
        num_layers=2,
        num_experts=4,
        expert_bytes=10,
    )


def test_trace_to_page_accesses() -> None:
    trace = _trace()
    assert expert_page_id(3, 17) == "L3.E17"
    assert trace.decode_tokens == 2
    assert len(trace.page_ids()) == 8
    assert list(trace.accesses("decode")) == [
        "L0.E1", "L0.E3", "L1.E0", "L1.E1",
        "L0.E1", "L0.E3", "L1.E0", "L1.E1",
    ]
    assert len(list(trace.accesses("prefill"))) == 8
    assert len(list(trace.accesses("all"))) == 16
    with pytest.raises(ValueError):
        list(trace.accesses("other"))


def test_reuse_distances() -> None:
    assert reuse_distances(["a", "b", "a", "a", "c", "b"]) == [None, None, 1, 0, None, 3]


@pytest.mark.parametrize("policy", [LRUPolicy(), FrequencyPolicy(), PriorityCostPolicy()])
@pytest.mark.parametrize("slots", [1, 3, 7, 12])
def test_fast_replay_matches_workload_simulator(policy, slots: int) -> None:
    rng = random.Random(slots)
    page_ids = [expert_page_id(layer, e) for layer in range(3) for e in range(5)]
    # Skewed accesses so frequency and recency disagree.
    weights = [1 / (i + 1) for i in range(len(page_ids))]
    accesses = rng.choices(page_ids, weights=weights, k=400)

    expected = simulate(accesses, page_ids, 10, slots * 10 + 5, policy)
    hits, misses, evictions = replay_equal_size(accesses, page_ids, slots, policy)

    assert (hits, misses, evictions) == (expected.hits, expected.misses, expected.evictions)


def test_warmup_fills_cache_without_counting() -> None:
    page_ids = ["a", "b", "c"]
    assert replay_equal_size(["a", "b"], page_ids, 2, LRUPolicy()) == (0, 2, 0)
    assert replay_equal_size(["a", "b"], page_ids, 2, LRUPolicy(), warmup=["b", "a"]) == (2, 0, 0)
    assert replay_equal_size(["a"], page_ids, 2, LRUPolicy(), warmup=["a", "b", "c"]) == (0, 1, 1)
