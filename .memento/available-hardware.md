---
title: Available GPU hardware
type: context
created: 2026-10-05
updated: 2026-10-05
---

Two machines are available to run experiments:

- Home, Ubuntu: RTX 4080 Super, 16 GB VRAM, 32 GB RAM.
- Work: RTX A4000 (Ampere, no FP8 tensor cores), 16 GB VRAM, 128 GB RAM.

Kolibri-1 (FP8, ~78 GB) cannot run at home, since 16 GB VRAM + 32 GB RAM is not enough. On the A4000 it only runs with vLLM `--cpu-offload-gb` (~70 GB in RAM). Training Kolibri (QLoRA ~40 GB) fits neither machine and needs a cloud GPU.

**Why:** The fine-tuning plan for Kolibri-1 depends on this.
**How to apply:** Run Kolibri experiments (`experiments/kolibri1_lora_smoke.py`) on the A4000. At home, use only model-independent work (task mining) or small models.
