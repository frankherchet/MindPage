from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from statistics import fmean
from typing import Iterable

from .models import TraceRun

ExpertKey = tuple[int, int]


def expert_counts(run: TraceRun) -> Counter[ExpertKey]:
    """Count routed selections by (layer, expert_id)."""

    return Counter((choice.layer, choice.expert_id) for choice in run.choices)


def top_expert_keys(run: TraceRun, limit: int = 32) -> tuple[ExpertKey, ...]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    counts = expert_counts(run)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(key for key, _ in ordered[:limit])


def jaccard(left: Iterable[ExpertKey], right: Iterable[ExpertKey]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def summarize_task_groups(
    runs: Iterable[TraceRun], *, top_n: int = 32
) -> dict[str, dict[str, object]]:
    """Summarize usage and within-group top-expert stability.

    Expert identity is layer-qualified. Expert 7 in layer 3 is therefore distinct
    from expert 7 in layer 4.
    """

    grouped: dict[str, list[TraceRun]] = defaultdict(list)
    for run in runs:
        grouped[run.task_group].append(run)

    summary: dict[str, dict[str, object]] = {}
    for task_group, group_runs in sorted(grouped.items()):
        aggregate: Counter[ExpertKey] = Counter()
        top_sets: list[tuple[ExpertKey, ...]] = []
        for run in group_runs:
            aggregate.update(expert_counts(run))
            top_sets.append(top_expert_keys(run, top_n))

        pair_scores = [jaccard(a, b) for a, b in combinations(top_sets, 2)]
        total = sum(aggregate.values())
        top_usage = [
            {
                "layer": layer,
                "expert_id": expert_id,
                "count": count,
                "share": (count / total) if total else 0.0,
            }
            for (layer, expert_id), count in sorted(
                aggregate.items(), key=lambda item: (-item[1], item[0])
            )[:top_n]
        ]
        summary[task_group] = {
            "runs": len(group_runs),
            "tokens": sum(run.token_count for run in group_runs),
            "selections": total,
            "mean_pairwise_jaccard": fmean(pair_scores) if pair_scores else None,
            "top_experts": top_usage,
        }
    return summary
