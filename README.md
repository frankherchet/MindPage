# MindPage

> Virtual memory for LLM parameters, experts, adapters, and context.

MindPage is an experimental runtime and research project for reducing the amount of model state that must remain in GPU VRAM at any one time.

The central idea is to treat an LLM's active state as a **working set**. Frequently needed model components and context stay close to the GPU; less relevant state can live in CPU RAM or, eventually, SSD storage and be paged in when needed.

## Core hypothesis

For many real workloads, especially repository-level coding tasks, only a subset of all available model state is immediately useful. If that working set can be predicted and managed well, a system may trade some transfer overhead for substantially lower VRAM requirements without giving up too much quality or throughput.

MindPage separates four kinds of state:

1. **Base parameters** — the shared model backbone.
2. **Experts and adapters** — task- or domain-specific parameter modules such as MoE experts or LoRA adapters.
3. **Context state** — active and historical KV/context memory.
4. **External knowledge** — source code, symbols, tests, build metadata, and other repository state.

The long-term goal is a runtime that can place these across GPU VRAM, CPU RAM, and storage according to expected usefulness, transfer cost, and memory pressure.

## First use case: repository-level C++ coding

The initial demonstrator will focus on tasks such as:

> Implement a C++ feature in an existing repository and add or update its unit tests.

A first version does **not** require a new model architecture. It can combine:

- an open-weight code model,
- dynamically loaded LoRA adapters,
- structure-aware repository retrieval,
- compiler and test feedback,
- instrumentation for VRAM, RAM, latency, tokens/sec, and retrieval behavior.

This gives us a measurable baseline before attempting true MoE expert paging or KV paging.

## Design principles

- **Measure before optimizing.** Paging only matters if it beats simpler baselines in useful regimes.
- **Keep exact project facts external.** Fine-tuning should learn patterns and conventions; changing APIs and source facts should come from the live repository.
- **Separate policy from mechanism.** Routing decisions should be replaceable without rewriting storage and execution code.
- **Make every experiment reproducible.** Quality, latency, memory, and transfer volume are first-class metrics.
- **Prefer incremental proof.** Start with adapters and retrieval, then add expert paging, then context/KV paging.

## Repository layout

```text
mindpage/
  runtime/       # working-set abstractions and placement policy
  adapters/      # adapter loading/routing (planned)
  repo/          # repository indexing/retrieval (planned)
  benchmark/     # historical-task mining from git history
  telemetry/     # memory, transfer, and routing metrics (planned)

docs/
  architecture.md
  research.md
  roadmap.md

experiments/     # reproducible experiment scripts, see experiments/README.md
benchmarks/      # evaluation harnesses and datasets (planned)
tests/
```

## Research questions

MindPage starts with a few falsifiable questions:

1. Does project-specific adaptation improve repository-level coding beyond retrieval alone?
2. Are adapter or MoE-expert usage patterns stable enough to predict a useful GPU working set?
3. At what PCIe/NVLink bandwidth and task duration does paging become worthwhile?
4. Can context/KV paging and parameter paging share one scheduler without causing pathological stalls?
5. How much VRAM can be saved at equal task success rate and acceptable latency?

See [docs/research.md](docs/research.md) for the initial experimental plan.

## Status

**Phase 0 — project scaffold.**

The repository is being initialized around the working-set abstraction. No performance or quality claims are made yet.

## Quick start

Requires Python 3.11+.

```bash
python -m pip install -e '.[dev]'
pytest
```

## Roadmap

1. Working-set and memory-tier abstractions.
2. Telemetry and deterministic benchmark harness.
3. Base code-model + repository retrieval baseline.
4. Project LoRA baseline.
5. Adapter routing and hot-swap experiments.
6. MoE expert usage profiling and paging.
7. Context/KV paging experiments.
8. Unified placement scheduler.

See [docs/roadmap.md](docs/roadmap.md) for details.
