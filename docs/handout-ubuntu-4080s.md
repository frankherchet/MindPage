# Handout: work on the Ubuntu RTX 4080 Super machine

You are Claude on the home machine: RTX 4080 Super with 16 GB VRAM and 32 GB RAM (details in [`.memento/available-hardware.md`](../.memento/available-hardware.md)). Kolibri-1 (~78 GB in FP8) does not fit here; it runs only on the work A4000 via [`experiments/kolibri1_lora_smoke.py`](../experiments/kolibri1_lora_smoke.py). This machine does the groundwork that every later Kolibri experiment depends on.

There are two independent tasks. Ask the user which one to start with. Task A runs on the CPU and Task B on the GPU, so both can run at the same time.

| task | needs | answers |
|---|---|---|
| A: validate mined tasks | CPU, cmake, C++ compiler | Which historical tasks are usable benchmark items? |
| B: MoE expert traces | GPU | Is expert usage on C++ work stable enough for a GPU working set to pay off? (research question 2) |

## Setup

1. Get the code. The task miner lives in PR [frankherchet/MindPage#3](https://github.com/frankherchet/MindPage/pull/3). If it is merged, work from `main`; otherwise branch from `claude/cpp-task-mining`.
2. `python -m pip install -e '.[dev]'` (Python 3.11+), then `pytest`.
3. `git clone https://github.com/fmtlib/fmt ~/src/fmt`. Use a full clone: a `--filter=blob:none` clone makes the miner fetch every blob one by one and run for minutes instead of under a second.
4. `python -m mindpage.benchmark ~/src/fmt --since 2023-01-01 --cutoff 2026-01-01 > fmt-tasks.jsonl`

**Done when:** `pytest` is green and `fmt-tasks.jsonl` has 333 lines, give or take new commits on fmt.

## Task A: validate mined tasks

A mined task is only a benchmark item if its tests prove the patch: the commit's tests **fail** without the source change and **pass** with it (fail-to-pass, as in SWE-bench). The record format and mining rules are in [`experiments/README.md`](../experiments/README.md#historical-c-tasks).

Steps per task:

1. Check out `parent` in a separate git worktree of fmt.
2. Apply the test change: `git checkout <commit> -- <test_files>`.
3. Configure, build and run the tests with cmake and `ctest`. Record which tests fail. A test target that no longer compiles also counts as failing, but record it as its own status.
4. Apply the source change: `git checkout <commit> -- <source_files>`. Build and test again.
5. Classify the task as `valid` (at least one test goes from failing to passing and nothing regresses), `no_fail_before`, `fails_after` or `build_error`. Store the names of the tests that flip.

Notes:

- Start with the 64 `eval` tasks; `train` tasks only need validation later, for training data.
- fmt builds are incremental across neighbouring commits: reuse one build directory per worktree and use `ccache`.
- Put the logic in `mindpage/benchmark/validate.py` with a CLI next to the miner. Write it to read the miner's JSONL and output JSONL with an added status, and add one unit test that runs on a tiny cmake project with no network access.

**Done when:** every `eval` task has a status, the counts per status are reported to the user, and [`experiments/README.md`](../experiments/README.md) records the counts and the exact commands.

## Task B: MoE expert traces into the cache simulator

Kolibri-1 is architecturally Qwen3-MoE, so a smaller Qwen3-MoE is a good stand-in. Expert IDs do not transfer between models; reuse patterns, locality and hit rate versus cache size do.

| model | experts | expert size | fits here |
|---|---|---|---|
| `allenai/OLMoE-1B-7B-0125-Instruct` | 16 layers × 64, top-8 | ~13 MB (BF16) | fully, in BF16 (~14 GB) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 48 layers × 128, top-8 | ~9.4 MB (BF16) | only quantized or partly offloaded (~61 GB in BF16) |
| Kolibri-1, for comparison | 50 layers × 384, top-6 | ~3.9 MB (FP8) | no |

Bring the pipeline up on OLMoE, then produce the real results on Qwen3-Coder-30B-A3B. Verify that the chosen quantization actually covers the MoE expert weights in the installed `transformers` version; recent versions store experts as fused 3D tensors, which some quantizers skip.

Steps:

1. **Prompts:** for each `eval` task from Task A (or from the miner if A is not done), build a prompt from the commit message and the source files at `parent`, cut to fit the context window.
2. **Capture:** run greedy generation (~256 new tokens) with `transformers`. Per layer and token, record the top-k expert IDs, taken from `output_router_logits=True` or a forward hook on each layer's router. Top-k of the logits equals top-k of the softmax. Label every token as `prefill` or `decode`: expert paging is decided in `decode`, while `prefill` touches nearly every expert.
3. **Convert:** turn traces into page accesses with page ID `L{layer}.E{expert}` and page size = expert bytes. Put the converter in `mindpage/simulation/` with a unit test; the capture script goes in `experiments/` because it needs a GPU.
4. **Simulate:** replay `decode` accesses through `WorkloadSimulator` with `LRUPolicy`, `FrequencyPolicy` and `PriorityCostPolicy`. Sweep the GPU capacity from 5 % to 50 % of all expert bytes and keep every expert resident in the CPU tier.
5. **Report:** for each policy and capacity, give the hit rate and transfer bytes per decode token, plus the reuse distance distribution (accesses between two uses of the same expert).

**Done when:** the table from step 5 exists for at least 20 tasks on Qwen3-Coder-30B-A3B, and [`experiments/README.md`](../experiments/README.md) has a section with the question, method, commands, results, and what they imply for Kolibri-1 (384 smaller experts, top-6).

## Conventions

- One branch and one PR per task, with experiment documentation in [`experiments/README.md`](../experiments/README.md).
- Scripts that need a GPU live in `experiments/` and stay out of CI; logic that runs on the CPU lives in `mindpage/` with a unit test.
- Record decisions, gotchas and unfinished work in `.memento/` (memento skill).
