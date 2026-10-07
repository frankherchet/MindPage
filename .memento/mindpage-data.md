---
title: Uncommitted experiment data in ~/mindpage-data
type: context
created: 2026-10-07
updated: 2026-10-07
---

Data that `experiments/README.md` marks as "not committed" lives in `~/mindpage-data/` on the Ubuntu 4080S machine: mined and validated fmt task JSONL (`fmt-eval-validated.jsonl`, `fmt-train-final.jsonl` for train, `fmt-nfb-sanitized.jsonl` for the ASan/UBSan re-run of the `no_fail_before` tasks), MoE traces (`olmoe/`, `qwen3-coder/`) and their reports.

All work planned for this machine (handout tasks, train validation, sanitizer pass) is done as of 2026-10-07. The remaining open work is Kolibri-1, which needs the A4000 or a cloud GPU.

**Why:** These files are inputs for later experiments and cannot be found from the repo.
**How to apply:** Look here before re-mining or re-validating tasks.
