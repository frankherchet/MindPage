---
title: Open tasks for the Ubuntu 4080S machine
type: todo
created: 2026-10-05
updated: 2026-10-05
---

Task A (fail-to-pass validation) is done on branch `claude/validate-fmt-tasks`: 37 of 64 eval tasks are valid; counts and commands are in `experiments/README.md`. Validating the 269 `train` tasks is still open (same command with `--split train`, about 3 hours).

Two tasks were waiting for the home machine (RTX 4080 Super, 32 GB RAM): validating the mined fmt tasks (fail-to-pass) and capturing MoE expert traces into the cache simulator. Steps and completion criteria are in `docs/handout-ubuntu-4080s.md`.

**Why:** Kolibri-1 does not fit on this machine. These tasks prepare the benchmark and the expert-paging evidence that Kolibri experiments need.
**How to apply:** On the Ubuntu machine, read the handout first. Delete this memo once both tasks are done and documented in `experiments/README.md`.
