---
title: Open tasks for the Ubuntu 4080S machine
type: todo
created: 2026-10-05
updated: 2026-10-06
---

Both handout tasks (`docs/handout-ubuntu-4080s.md`) and the `train` validation are done; results are in `experiments/README.md`. Data that is not committed lives in `~/mindpage-data/`: mined and validated task JSONL (`fmt-train-final.jsonl` is the merged train result), MoE traces (`olmoe/`, `qwen3-coder/`) and their reports.

Still open: try `-fsanitize=address,undefined` on the `no_fail_before` tasks (26 eval, 100 train); several are real fixes (out-of-bounds read, out-of-range conversion) that a plain build does not expose.

**Why:** Kolibri-1 does not fit on this machine; this is the groundwork for the Kolibri experiments.
**How to apply:** Delete this memo once the sanitizer run is done or dropped.
