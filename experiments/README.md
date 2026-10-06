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

## MoE expert traces in the cache simulator

[`moe_expert_capture.py`](moe_expert_capture.py) (GPU) and [`moe_expert_report.py`](moe_expert_report.py) (CPU)

**Question:** Is expert usage during C++ work stable enough that a GPU-resident working set of experts pays off (research question 2)? Concretely: which hit rate and how many host-to-GPU bytes per decode token does a GPU expert cache of a given size achieve?

**Models:** Kolibri-1 is architecturally Qwen3-MoE but does not fit this machine, so a smaller Qwen3-MoE stands in. Expert IDs do not transfer between models; reuse patterns, locality and hit rate versus cache size do.

| model | experts | expert size (BF16) | accesses per decode token |
|---|---|---|---|
| `allenai/OLMoE-1B-7B-0125-Instruct` (pipeline bring-up) | 16 layers × 64, top-8 | 12.6 MB | 128 |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | 48 layers × 128, top-8 | 9.4 MB | 384 |
| Kolibri-1, for comparison | 50 layers × 384, top-6 | ~3.9 MB (FP8) | 300 |

Qwen3-Coder runs from Qwen's official FP8 checkpoint (block-wise e4m3, ~31 GB) with ~11 GB of weights on the GPU and the rest offloaded to CPU RAM. In transformers 5.18 the fused 3D expert tensors load as `FP8Experts` with `float8_e4m3fn` weights, so the quantization does cover the experts; the router (`mlp.gate`) stays in BF16. bitsandbytes 4-bit was ruled out because transformers 5.18 does not quantize the fused expert tensors with it. FP8 needs `pip install kernels==0.17.0`.

**Method:**

1. **Prompts:** the `valid` eval tasks from the fail-to-pass validation above. Each prompt is the commit message plus the task's source files at `parent`; files that do not fit 3072 tokens are cut to a window centred on the first changed line.
2. **Capture:** greedy generation of 256 new tokens with `transformers`. A forward hook on each layer's `*TopKRouter` records the expert indices the layer actually computes. Every token is labelled `prefill` or `decode`.
3. **Convert:** each `(layer, expert)` is a page `L{layer}.E{expert}` of the expert's BF16 size (`mindpage.simulation.ExpertTrace`). Accesses run token by token, layer by layer.
4. **Simulate:** decode accesses of each task are replayed with `LRUPolicy`, `FrequencyPolicy` and `PriorityCostPolicy`; every expert stays resident in the CPU tier, the GPU holds 5–50 % of all expert bytes. Each task starts once from an empty GPU cache ("cold") and once after replaying its prefill uncounted ("after prefill"), which is what a real runtime would see. `WorkloadSimulator` re-sorts every GPU page on each miss, which is too slow for 6144 experts and ~100k accesses per task, so the report uses `replay_equal_size`, an exact equivalent for equal-size pages; unit tests and `--check` compare both.
5. **Report:** hit rate and host-to-GPU bytes per decode token per policy and cache size, plus the reuse distance distribution (accesses between two uses of the same expert). Evicting an expert costs no transfer, because expert weights are read-only.

With equal page sizes and no priorities, `PriorityCostPolicy` sorts by exactly the same key as `FrequencyPolicy` (access count, then recency), so the two always give identical results.

**Run:**

```bash
python experiments/moe_expert_capture.py --model allenai/OLMoE-1B-7B-0125-Instruct \
    --tasks fmt-eval-validated.jsonl --status valid --repo ~/src/fmt --out traces/olmoe
python experiments/moe_expert_capture.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --tasks fmt-eval-validated.jsonl --status valid --repo ~/src/fmt --out traces/qwen3-coder
python experiments/moe_expert_report.py traces/qwen3-coder --json qwen3-coder-report.json --check
```

The `.npz` traces are not committed (~1 MB per task); they are reproducible with the commands above.

**Environment:** RTX 4080 Super (16 GB), 32 GB RAM, torch 2.14.1 + CUDA 13.0, transformers 5.18.0, accelerate, kernels 0.17.0, NVIDIA driver 580.178. Qwen3-Coder with `--max-gpu-memory 11GiB`: higher budgets run out of GPU memory in the FP8 expert kernel during the 3k-token prefill.

**Results:** 37 `valid` eval tasks per model, median prompt 3034 tokens, 255 decode tokens each (9435 in total). GPU share is the fraction of all expert bytes the GPU cache holds; "after prefill" starts decode with the cache state the prefill left behind.

Qwen3-Coder-30B-A3B (6144 experts, 384 accesses per decode token, 3.6 GB per token without cache):

| GPU share | LRU hit rate | LRU MB/token | Frequency hit rate (cold) | Frequency hit rate (after prefill) | Frequency MB/token (after prefill) |
|---|---|---|---|---|---|
| 5 % | 0.0 % | 3624 | 29.1 % | 12.6 % | 3168 |
| 10 % | 48.0 % | 1883 | 45.6 % | 20.2 % | 2893 |
| 15 % | 61.6 % | 1393 | 57.4 % | 27.5 % | 2627 |
| 20 % | 70.6 % | 1066 | 66.2 % | 34.7 % | 2367 |
| 25 % | 77.6 % | 810 | 73.0 % | 41.8 % | 2110 |
| 30 % | 83.1 % | 614 | 78.4 % | 48.8 % | 1855 |
| 40 % | 90.4 % | 347 | 86.2 % | 62.6 % | 1355 |
| 50 % | 94.4 % | 203 | 91.0 % | 75.6 % | 886 |

LRU columns are after prefill; LRU from a cold cache is within 2 points. Per-task LRU hit rates after prefill spread by about ±10 points (e.g. 62–83 % at 20 %). `PriorityCostPolicy` equals `FrequencyPolicy` in every cell, as expected.

OLMoE (1024 experts, 128 accesses per token) shows the same shape at lower hit rates: LRU after prefill 0 % at 5–10 %, 56 % at 20 %, 85 % at 50 %; Frequency 28 % at 5 %, 57 % at 20 %, 82 % at 50 %.

Reuse distance in decode (Qwen3-Coder): median 2 tokens, p75 7, p90 17, p99 76 tokens; 4.8 % of decode accesses are an expert's first use in that task. Within 255 decode tokens a task touches a median 78 % of all experts (54–85 %); OLMoE 93 %.

**Findings:**

1. **There is no small working set.** A task uses most experts within a few hundred tokens, so the hit rate grows smoothly with cache size instead of saturating early. At 25 % of expert bytes on the GPU, LRU still misses 22 % of accesses.
2. **But there is strong short-term reuse.** Half of all reuses happen within 2 tokens. A recency cache therefore helps a lot: transfers drop from 3.6 GB to 0.8 GB per token at 25 % and to 0.2 GB at 50 %.
3. **LRU collapses below one token's footprint.** When the cache holds fewer experts than one decode token uses (layers × top-k: 384 for Qwen3, 128 for OLMoE), LRU always evicts the expert needed next and hits 0 %. Above that threshold LRU beats Frequency at every size on Qwen3-Coder; on OLMoE the two stay within 3 points, with Frequency slightly ahead.
4. **Prefill routing does not predict decode routing.** The top 25 % experts by prefill use overlap only 41 % (28–57 %) with the top 25 % by decode use, and cover 40 % of decode accesses, against 75 % for the same-size set chosen from decode itself. Frequency counts carried over from the prefill therefore hurt on Qwen3-Coder: 20 % instead of 46 % hit rate at 10 % GPU share. On OLMoE the loss is at most 2.5 points, so the size of the effect depends on the model. A runtime should reset or decay frequency statistics at the prefill-to-decode boundary, and prefetching decode experts from prefill statistics is unlikely to pay off.
5. **Static offload is far off.** The accelerate CPU offload used for capturing (11 GB weights on the GPU, the rest streamed every step) needed 669 s per task, i.e. on the order of seconds per decode token. At 25 GB/s PCIe, LRU at 25 % would move 0.8 GB per token, about 32 ms.

**What this implies for Kolibri-1** (50 layers × 384 experts, top-6, ~3.9 MB per expert in FP8, ~75 GB of experts):

- One token uses 300 experts = 1.6 % of all experts, so the LRU collapse threshold is far below any realistic cache size.
- On a 16 GB GPU with ~3–4 GB for the always-needed weights, about 11 GB (~15 % of expert bytes) remain for an expert cache. If Kolibri's hit-rate curve resembles Qwen3-Coder's at the same share (~60 % at 15 %), that is ~120 experts or ~470 MB per decode token, roughly 20 ms at 25 GB/s. Streaming all offloaded weights costs seconds per token. Expert-level paging with an LRU cache is therefore worth building and is the next step towards the Phase 5 CPU expert store.
- Unverified: Kolibri's finer-grained experts (384 per layer, top-6) and its sigmoid routing with bias may spread usage more or less than Qwen3-MoE. The first Kolibri routing trace on the A4000 should rerun `moe_expert_report.py` before any paging code is tuned.

**Limits:** one repository (fmt), greedy decoding, 255 decode tokens per task, prompts cut to 3072 tokens, one cold cache per task (no reuse across tasks), FP8 weights for Qwen3-Coder (the router stays BF16, but FP8 expert outputs can shift later routing slightly), transfer cost only, no prefetching or overlap with compute modelled.
