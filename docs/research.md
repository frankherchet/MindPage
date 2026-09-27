# Research plan

MindPage is an experimental project. Its claims should be phrased as hypotheses until measured.

## Primary hypothesis

For structured workloads such as repository-level coding, the useful model/context state at a given moment is smaller than the total available state often enough that dynamic placement can reduce peak VRAM while preserving an acceptable quality/latency envelope.

This hypothesis can fail. Transfer overhead, router mistakes, dense shared weights, or rapidly changing working sets may make paging slower or less useful than simpler approaches.

## Experiment 1: retrieval baseline

**Question:** How far can a base code model get with structure-aware repository retrieval and compiler/test feedback alone?

Compare:

- base model with manually supplied context,
- base model with automatic repository retrieval.

Record task success, compilation success, test success, tokens, latency, and context volume.

## Experiment 2: project adapter

**Question:** Does a project-specific LoRA learn stable project conventions beyond what retrieval provides?

Compare on held-out historical tasks:

- retrieval only,
- retrieval + project adapter.

Training data should be temporally separated from evaluation tasks where possible to reduce leakage from later commits.

The adapter should target stable behavior (style, common transformations, architecture patterns), not memorize changing APIs.

## Experiment 3: adapter routing

**Question:** Can multiple adapters be selected predictably enough to justify dynamic residency?

Potential adapters:

- generic C++ conventions,
- project conventions,
- unit-test patterns,
- subsystem-specific transformations.

Measure routing accuracy/usefulness, adapter load latency, VRAM savings, and end-to-end task success.

## Experiment 4: MoE usage profiles

**Question:** Do classes of coding tasks produce sufficiently stable expert-usage distributions to support expert prefetching/caching?

Collect expert activation traces for tasks grouped by language, build system, test framework, subsystem, and task type.

Evaluate whether a predictor based on task/repository features reduces expert cache misses compared with frequency and LRU baselines.

## Experiment 5: expert paging

**Question:** Under constrained VRAM, can selective expert residency improve usable model size or throughput relative to simpler CPU offload?

Key measurements:

- peak GPU bytes,
- CPU bytes,
- host-to-device/device-to-host bytes,
- stall time waiting for experts,
- tokens/sec,
- time to first token,
- quality/task success.

## Experiment 6: context/KV paging

**Question:** Can historical context be paged independently or jointly with parameter state without excessive random transfer overhead?

Start with existing context-offload mechanisms before implementing new kernels.

## Baselines

Every paging experiment should include at least:

1. all state resident when hardware permits,
2. static CPU offload,
3. simple LRU/frequency caching,
4. MindPage policy under test.

The purpose is not to prove paging is universally better. We want to identify regimes where it is useful and regimes where it is not.

## Initial coding benchmark

A useful project-level benchmark should consist of historical tasks for which a known-good patch and tests exist.

For each task capture:

- repository revision before the fix/feature,
- task description available at that time,
- accepted patch,
- relevant test suite,
- build/test commands.

Primary outcome: tests pass without regressions.

Secondary outcomes: number of agent iterations, generated tokens, wall-clock latency, retrieved context bytes/tokens, peak VRAM, transfer volume, and human-reviewable patch quality.

## Threats to validity

- Git history may leak solutions into retrieval or training data.
- Public code models may already contain public repository history.
- Expert IDs are not guaranteed to correspond to human-interpretable skills.
- A workload-specific cache can appear good while generalizing poorly.
- PCIe/NVLink/storage characteristics can dominate results.
- Quantization may change both routing behavior and quality.

These should be tracked explicitly in experiment reports.
