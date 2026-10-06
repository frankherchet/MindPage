"""Replay captured MoE expert traces through MindPage's placement policies.

Reads ``*.npz`` traces from ``moe_expert_capture.py``. Each task is replayed on
its own: every expert starts in the CPU tier, the GPU holds a fraction of all
expert bytes. Decode accesses are counted twice, once from a cold GPU cache and
once after replaying the prompt's prefill accesses uncounted.

Usage::

    python experiments/moe_expert_report.py traces/qwen3-coder --json report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mindpage import FrequencyPolicy, LRUPolicy, PriorityCostPolicy
from mindpage.simulation import ExpertTrace, replay_equal_size, reuse_distances
from mindpage.simulation.experts import simulate

POLICIES = {
    "LRU": LRUPolicy(),
    "Frequency": FrequencyPolicy(),
    "PriorityCost": PriorityCostPolicy(),
}
FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def load(path: Path) -> tuple[ExpertTrace, dict]:
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    trace = ExpertTrace(
        experts=data["experts"].tolist(),
        prefill_tokens=int(data["prefill_tokens"]),
        num_layers=meta["num_layers"],
        num_experts=meta["num_experts"],
        expert_bytes=meta["expert_bytes"],
    )
    return trace, meta


def analyse_task(path: Path) -> dict:
    """Replay one task for every policy, GPU share and start state."""
    trace, meta = load(path)
    page_ids = trace.page_ids()
    total = len(page_ids)
    decode = list(trace.accesses("decode"))
    prefill = list(trace.accesses("prefill"))
    counts = {}
    for name, policy in POLICIES.items():
        for fraction in FRACTIONS:
            slots = round(fraction * total)
            counts[name, fraction, "cold"] = replay_equal_size(decode, page_ids, slots, policy)[:2]
            counts[name, fraction, "warm"] = replay_equal_size(
                decode, page_ids, slots, policy, warmup=prefill
            )[:2]
    return {
        "meta": meta,
        "top_k": len(trace.experts[0][0]),
        "decode_tokens": trace.decode_tokens,
        "counts": counts,
        "distances": reuse_distances(decode),
        "distinct_fraction": len(set(decode)) / total,
    }


def percentile(sorted_values: list[int], q: float) -> int:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("traces", type=Path)
    cli.add_argument("--json", type=Path, help="write all numbers here")
    cli.add_argument(
        "--check", action="store_true",
        help="cross-check one task per policy against WorkloadSimulator (slow)",
    )
    cli.add_argument("--workers", type=int, default=4, help="tasks replayed in parallel")
    args = cli.parse_args()

    paths = sorted(args.traces.glob("*.npz"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [t for t in pool.map(analyse_task, paths) if t["decode_tokens"] > 0]
    if not tasks:
        raise SystemExit(f"no traces with decode tokens in {args.traces}")
    meta = tasks[0]["meta"]
    num_layers, num_experts = meta["num_layers"], meta["num_experts"]
    total_experts = num_layers * num_experts
    expert_bytes = meta["expert_bytes"]
    top_k = tasks[0]["top_k"]
    accesses_per_token = num_layers * top_k
    decode_tokens = sum(t["decode_tokens"] for t in tasks)
    per_task = [t["counts"] for t in tasks]

    rows = []
    for policy_name in POLICIES:
        for fraction in FRACTIONS:
            row = {"policy": policy_name, "fraction": fraction,
                   "slots": round(fraction * total_experts)}
            for start in ("cold", "warm"):
                hits = sum(c[policy_name, fraction, start][0] for c in per_task)
                misses = sum(c[policy_name, fraction, start][1] for c in per_task)
                row[f"hit_rate_{start}"] = hits / (hits + misses)
                row[f"bytes_per_token_{start}"] = misses * expert_bytes / decode_tokens
            warm = [
                h / (h + m) for h, m in (c[policy_name, fraction, "warm"] for c in per_task)
            ]
            row["hit_rate_warm_task_min"] = min(warm)
            row["hit_rate_warm_task_max"] = max(warm)
            rows.append(row)

    distances = sorted(d for t in tasks for d in t["distances"] if d is not None)
    first_uses = sum(d is None for t in tasks for d in t["distances"])
    reuse = {
        "accesses_per_token": accesses_per_token,
        "first_use_fraction": first_uses / (first_uses + len(distances)),
        "percentiles_accesses": {
            f"p{int(q * 100)}": percentile(distances, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
    }
    unique = [t["distinct_fraction"] for t in tasks]

    if args.check:
        trace, _ = load(paths[0])
        page_ids = trace.page_ids()
        decode = list(trace.accesses("decode"))
        for policy_name, policy in POLICIES.items():
            slots = round(0.2 * total_experts)
            fast = replay_equal_size(decode, page_ids, slots, policy)
            slow = simulate(decode, page_ids, expert_bytes, slots * expert_bytes, policy)
            assert fast == (slow.hits, slow.misses, slow.evictions), (policy_name, fast, slow)
            print(f"check {policy_name}: fast replay == WorkloadSimulator {fast}")

    full_miss = accesses_per_token * expert_bytes
    print(f"model: {meta['model_id']}")
    print(
        f"tasks: {len(tasks)}, decode tokens: {decode_tokens}, experts: "
        f"{num_layers} x {num_experts}, top-{top_k}, "
        f"{expert_bytes / 1e6:.1f} MB each"
    )
    print(
        f"distinct experts touched in decode per task: median "
        f"{statistics.median(unique):.0%} (min {min(unique):.0%}, max {max(unique):.0%})"
    )
    print(f"no cache: {full_miss / 1e6:.0f} MB per decode token\n")
    print("| policy | GPU share | hit rate cold | hit rate after prefill (task min-max) "
          "| MB/token cold | MB/token after prefill |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['policy']} | {r['fraction']:.0%} | {r['hit_rate_cold']:.1%} | "
            f"{r['hit_rate_warm']:.1%} ({r['hit_rate_warm_task_min']:.0%}-"
            f"{r['hit_rate_warm_task_max']:.0%}) | {r['bytes_per_token_cold'] / 1e6:.0f} | "
            f"{r['bytes_per_token_warm'] / 1e6:.0f} |"
        )
    p = reuse["percentiles_accesses"]
    print(
        f"\nreuse distance in decode (accesses between two uses; "
        f"{accesses_per_token} accesses per token): first use {reuse['first_use_fraction']:.1%}, "
        + ", ".join(f"{k} {v} ({v / accesses_per_token:.1f} tok)" for k, v in p.items())
    )
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "model_id": meta["model_id"],
                    "tasks": [t["meta"] for t in tasks],
                    "rows": rows,
                    "reuse": reuse,
                    "distinct_experts_fraction": unique,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
