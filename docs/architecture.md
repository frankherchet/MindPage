# Architecture

## Goal

MindPage explores a runtime in which the active state of an LLM is treated as a working set rather than as a monolithic object that must remain entirely resident in GPU memory.

The architecture deliberately separates **what should be resident** from **how state is transferred and executed**.

## State classes

MindPage models four broad classes of state:

### Base parameters

Shared model weights needed by most requests. Dense models may require most of these on every token; MoE models create more opportunities for selective residency.

### Experts and adapters

Parameter subsets that may have task-dependent utility, including MoE experts and LoRA-style adapters.

### Context state

Runtime state derived from the active sequence, such as KV cache entries or other historical representations.

### External knowledge

Repository files, symbols, call relationships, build metadata, tests, compiler output, and other facts that should generally remain outside learned weights.

## Memory tiers

The initial runtime exposes three abstract tiers:

```text
GPU      fastest, smallest, execution working set
CPU      warm state, larger capacity, transfer cost to GPU
STORAGE  cold state, largest capacity, highest access latency
```

Hardware-specific backends will later map these abstractions to CUDA/HIP devices, pinned host memory, mmap, NVMe, or other transports.

## Components

```text
Task / request
      |
      v
+-------------------+
| Routing policy    |  predicts useful working set
+---------+---------+
          |
          v
+-------------------+
| Placement planner |  respects budgets and transfer costs
+---------+---------+
          |
          v
+-------------------+
| Runtime backends  |  load / evict / execute
+---------+---------+
          |
          +------------------+
          |                  |
          v                  v
    Telemetry           External knowledge
                        / repository index
```

### Routing policy

Produces usefulness estimates. The policy must be replaceable: static rules, observed frequency, LRU-style heuristics, learned predictors, or model-router signals should all be possible without changing storage code.

### Placement planner

Turns usefulness estimates into residency decisions under explicit GPU/CPU/storage budgets. It is responsible for deterministic decisions and should be testable without loading a model.

### Runtime backend

Performs actual transfers and execution. Backends are intentionally deferred until the placement API is stable.

### Telemetry

Records at least:

- resident bytes by tier,
- bytes transferred between tiers,
- transfer latency,
- page faults / cache misses,
- tokens per second,
- time to first token,
- task success / test success,
- router predictions and actual usage.

## Initial abstraction

The first code milestone defines a `Page` and `MemoryTier` independent of ML frameworks. A page is a logical unit of placement, not necessarily an operating-system page and not required to have a fixed byte size.

Examples:

```text
Page(kind=EXPERT,  id="layer.12.expert.7", size=...)
Page(kind=ADAPTER, id="project.cpp-style", size=...)
Page(kind=KV,      id="request.42.block.18", size=...)
```

The abstraction should stay coarse enough that different backends can choose sensible physical granularity.

## Coding demonstrator

The first application combines:

1. an open-weight code model,
2. live repository retrieval,
3. optional project-specific LoRA,
4. compiler/test feedback,
5. telemetry.

The repository index stores exact, changing project facts. Fine-tuning is reserved for stable patterns such as conventions and recurring transformations.

## Future unified scheduler

A later MindPage scheduler may jointly budget:

```text
GPU budget
  = base/shared weights
  + active experts
  + active adapters
  + active KV/context
  + temporary execution buffers
```

A joint scheduler is only justified if measurements show that coordinated placement beats independent caches.
