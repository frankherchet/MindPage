# Roadmap

The roadmap is organized around evidence, not feature count.

## Phase 0 — foundations

- [x] Define project scope and core hypothesis.
- [x] Define memory tiers and page kinds.
- [x] Add a framework-independent working-set manager.
- [x] Add telemetry event schema.
- [x] Add deterministic placement-policy simulation.
- [x] Add CI for unit tests.
- [ ] Add lint/format checks.

**Exit criterion:** placement logic is deterministic, tested, and independent of a specific ML runtime.

## Phase 1 — repository coding baseline

- [ ] Choose initial open-weight code model and inference runtime.
- [ ] Add C++ repository index (symbols, definitions, references, includes).
- [ ] Add task-driven retrieval.
- [ ] Add compile/test execution adapter.
- [ ] Build a small historical-task benchmark.
- [ ] Record quality, latency, token, and memory baselines.

**Exit criterion:** reproducible end-to-end coding tasks with no project fine-tuning.

## Phase 2 — project adaptation

- [ ] Build training examples from historical changes without evaluation leakage.
- [ ] Train a small project LoRA.
- [ ] Compare retrieval-only vs retrieval + LoRA.
- [ ] Measure hot-swap cost.

**Exit criterion:** evidence for or against keeping project behavior in an adapter.

## Phase 3 — routed adapters

- [ ] Add multiple adapters.
- [ ] Implement static routing baseline.
- [ ] Add learned/task-feature routing experiment.
- [ ] Measure residency and transfer behavior.

**Exit criterion:** determine whether adapter routing is worth its complexity.

## Phase 4 — MoE expert profiling

- [x] Add a framework-independent router trace schema and task-group analysis.
- [x] Add a Hugging Face Transformers MoE tracing backend.
- [x] Run the first real MoE model trace and archive the environment/configuration.
- [x] Capture per-layer expert activation traces for the benchmark task groups.
- [ ] Analyze task-conditioned usage stability and between-group separation.
- [x] Implement LRU/frequency cache simulation before real transfers.

**Exit criterion:** show a cache simulation driven by real router traces with meaningful hit-rate/VRAM trade-offs before building paging infrastructure.

## Phase 5 — expert paging

- [ ] Implement CPU-resident expert store.
- [ ] Add GPU expert cache.
- [ ] Add asynchronous prefetch where supported.
- [ ] Compare with static offload.

**Exit criterion:** identify hardware/workload regimes where dynamic expert paging improves the Pareto frontier.

## Phase 6 — context paging

- [ ] Integrate an existing KV/context offload backend.
- [ ] Expose context blocks through the same placement model.
- [ ] Evaluate independent vs joint scheduling.

## Phase 7 — unified MindPage runtime

- [ ] Joint scheduler for experts, adapters, and context.
- [ ] Cost model incorporating bandwidth and latency.
- [ ] Predictive prefetch.
- [ ] Stable benchmark suite and reproducible experiment reports.

## Non-goals for early phases

- Training a foundation model from scratch.
- Inventing custom CUDA kernels before measurements justify them.
- Treating expert IDs as named semantic skills without evidence.
- Storing exact, rapidly changing repository facts in model weights.
