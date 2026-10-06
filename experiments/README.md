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
