---
title: Open tasks for the Ubuntu 4080S machine
type: todo
created: 2026-10-05
updated: 2026-10-06
---

Both handout tasks (`docs/handout-ubuntu-4080s.md`) are done, each on its own branch with results in `experiments/README.md`:

- Task A, fail-to-pass validation: branch `claude/validate-fmt-tasks`. 37 of 64 fmt eval tasks are valid.
- Task B, MoE expert traces: branch `claude/moe-expert-traces` (includes PR #1). 37 tasks traced on Qwen3-Coder-30B-A3B-FP8 and OLMoE; traces and reports are untracked in `traces/` of that worktree (`~/git/MindPage-moe-traces`).

Still open: validate the 269 `train` tasks (same command with `--split train`, about 3 hours), and try `-fsanitize=address,undefined` on the 26 `no_fail_before` eval tasks.

**Why:** Kolibri-1 does not fit on this machine; this is the groundwork for the Kolibri experiments.
**How to apply:** Delete this memo once the train validation is done.
