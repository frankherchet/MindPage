# Experiments

Reproducible experiment scripts. Each script is self-contained and documents its own requirements; none of them run in CI.

## Kolibri-1 LoRA smoke test

[`kolibri1_lora_smoke.py`](kolibri1_lora_smoke.py)

**Question:** Can vLLM serve [Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) with a LoRA adapter on a 16 GB GPU, with the routed experts offloaded to CPU RAM? This gates fine-tuning Kolibri-1 for C++: an adapter that cannot be served is not worth training.

**Model facts** (from `config.json` and the [`aleph-alpha-inference`](https://github.com/Aleph-Alpha/aleph-alpha-inference) vLLM plugin):

- 78 B parameters total, ~3.5 B active per token, FP8 checkpoint (~78 GB).
- 50 layers, 384 routed experts + 1 ungated shared expert per layer, top-6 routing.
- One routed expert is ~3.9 MB in FP8. Routed experts are ~97 % of all weights, so the always-needed part (attention, embeddings, shared experts) is only ~3–4 GB.
- Architecturally Qwen3-MoE plus four changes: sigmoid routing with selection on `logits + bias`, an ungated shared expert, extra norms after attention and MoE, and a 4:1 sliding-window/full-attention layer pattern.
- No `transformers` modeling code is published; the vLLM plugin is the only implementation.

**Method:** The script writes two dummy PEFT adapters (rank 16) on `q/k/v/o_proj`, and optionally on the shared expert MLP:

| adapter | `lora_B` | expected greedy output |
|---|---|---|
| zero | all zeros | identical to the base model |
| random | small random values | different from the base model |

It also prints the base model's decode throughput under CPU offload. That number is the first measurement for the "static CPU offload" baseline in [`docs/research.md`](../docs/research.md).

**Requirements:**

- 16 GB GPU and ≥ 128 GB RAM (the experts alone take ~75 GB).
- vLLM 0.29 and the Kolibri plugin: `pip install aleph-alpha-inference`.

**Run:**

```bash
python experiments/kolibri1_lora_smoke.py --cpu-offload-gb 70
python experiments/kolibri1_lora_smoke.py --cpu-offload-gb 70 --shared-experts
```

Exit code 0 means both checks passed.

**Known risks:**

- **FP8 on Ampere** (e.g. RTX A4000): no native FP8 tensor cores, so vLLM needs a weight-only FP8 kernel for the block-scaled MoE weights. It is unverified whether vLLM 0.29 provides one. There is no fallback on a 128 GB machine, because the BF16 checkpoint needs ~156 GB.
- **Offload speed:** it is unknown whether vLLM streams all offloaded weights each step (seconds per token) or reads only the selected experts (tens of milliseconds per token). The measured throughput decides whether a full retrieval baseline (Experiment 1) is practical on this hardware.
- **Shared-expert LoRA** may be rejected by vLLM even if attention LoRA works. Run without `--shared-experts` first.

**Next steps if it passes:** Kolibri port to `transformers` with a logit parity check against vLLM, then QLoRA on attention and the shared expert with the router and routed experts frozen. Training needs ~40 GB of VRAM and therefore a cloud GPU.

## Historical C++ tasks

Benchmark tasks are mined with `mindpage.benchmark` (no GPU needed):

```bash
python -m mindpage.benchmark /path/to/repo --since 2023-01-01 --cutoff 2026-01-01 > tasks.jsonl
```

A task is a non-merge commit that changes both C++ sources and test files and at most `--max-source-lines` source lines (default 200). Each JSONL record holds the commit, its parent, the date, the commit message as task description, the source and test files, and a `train`/`eval` split by `--cutoff`. Patches are reproduced with `git diff <parent> <commit> -- <files>`.

On [fmtlib/fmt](https://github.com/fmtlib/fmt) (commits since 2023, cutoff 2026-01-01) this yields 333 tasks, 269 `train` and 64 `eval`.

Contamination caveat: public repositories are very likely in the model's training data. Only commits after the model's release (Kolibri-1: 2026-10-03) are reliably unseen.

### Fail-to-pass validation

A mined task is only a benchmark item if its tests prove the patch. `mindpage.benchmark.validate` checks out `parent` in a git worktree, applies the commit's `test_files`, builds with cmake and runs `ctest`, then applies the `source_files` and runs again. It writes the input records with these fields added:

| field | meaning |
|---|---|
| `status` | `valid`: at least one test goes from failing to passing and none from passing to failing. `no_fail_before`: every test already passes before the source change. `fails_after`: no test flips, or one regresses. `build_error`: the project does not configure, or a test target does not compile, after the source change. |
| `fail_to_pass` | tests that flip from failing to passing |
| `build_fail_before` | tests whose target does not compile before the source change (counted as failing) |
| `pass_to_fail`, `failing_after` | regressions and every test that still fails after the change |

Test executables are deleted before each build, so a target that does not compile is recorded as `build_fail` instead of running a stale binary. Each worker reuses one worktree and build directory for a contiguous run of tasks in history order, and builds go through `ccache`.

**Run** (fmt, 64 `eval` tasks, 4 workers × 4 jobs on 16 cores, 44 minutes, about 2.5 minutes per task and worker):

```bash
python -m mindpage.benchmark ~/src/fmt --since 2023-01-01 --cutoff 2026-01-01 > fmt-tasks.jsonl
python -m mindpage.benchmark.validate ~/src/fmt fmt-tasks.jsonl --split eval \
    --workers 4 --jobs 4 --work-dir /tmp/validate-work \
    --cmake-arg=-DFMT_PEDANTIC=ON --cmake-arg=-DCMAKE_CXX_STANDARD=23 \
    > fmt-eval-validated.jsonl
```

Toolchain: GCC 15.2, CMake from Ubuntu 26.04, Make generator. `FMT_PEDANTIC=ON` registers `compile-error-test`, `nolocale-test`, `noexception-test` and the other tests fmt only builds in its own CI; `CMAKE_CXX_STANDARD=23` enables the C++20/23 parts of `std-test` and friends that GCC's default C++17 compiles out. Without both, tasks that only touch those tests would wrongly count as `no_fail_before`.

**Results** (fmt at 2026-10-04, 64 `eval` tasks):

| status | tasks |
|---|---|
| `valid` | 37 |
| `no_fail_before` | 26 |
| `fails_after` | 0 |
| `build_error` | 1 |

- 15 of the 37 valid tasks flip only because the new tests do not compile without the source change (new API); the other 22 have a test that compiles and fails at runtime. Most tasks flip a single test; `format-test` (10) and `std-test` (6) flip most often.
- No valid task has a test that still fails afterwards, so there is no environment-specific noise in this setup.
- About half of the 26 `no_fail_before` tasks are refactorings and cleanups ("Simplify copy", "Move std::byte formatter to std.h"). The rest include real fixes whose bug a plain build does not expose, such as an out-of-bounds read (`8a7aea04`) and an out-of-range float-to-int conversion (`de4c6c50`); these may flip under `-fsanitize=address,undefined`, which is not tried yet. Two tasks need C++26 reflection (`e27cc20b`, `e589a16e`), which GCC 15 does not provide.
- The `build_error` task (`44f2c7a`, "Make base.h a compatibility header") also changes the top-level `CMakeLists.txt`, which the miner neither counts as source nor as test file. The task cannot be reproduced from its file lists, so it is correctly excluded.
- Not covered: `module-test` needs C++ modules and the Ninja generator and is not built here; the 3 tasks touching it were judged on the remaining tests.
