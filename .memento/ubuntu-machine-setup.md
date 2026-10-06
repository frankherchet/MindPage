---
title: Ubuntu 4080S machine setup gotchas
type: context
created: 2026-10-05
updated: 2026-10-05
---

- NVIDIA driver: after a kernel update the prebuilt module package can lag behind (`linux-modules-nvidia-580-open-generic-hwe-26.04` was kept back), so `nvidia-smi` finds no driver. Fix: `sudo apt install linux-modules-nvidia-580-open-generic-hwe-26.04 && sudo modprobe nvidia`. Secure Boot is on; the Canonical-signed prebuilt modules load fine, DKMS is not involved.
- Python: only `python3` (3.14) exists. The project venv is `.venv/` in the main checkout (torch 2.14 + CUDA 13, transformers 5.18). It is an editable install of the main checkout; in other worktrees run with `PYTHONPATH=.`.
- FP8 models in transformers need `kernels==0.17.*` (downloads the finegrained-fp8 kernel from the Hub on first use).
- `huggingface_hub` keeps blobs in a shared store under `~/.cache/huggingface/hub/blobs/`; `du` on a model directory shows only symlinks, use `du -shL .../snapshots/<rev>/`.
- Loading a 30 GB checkpoint takes ~12 minutes here (the disk reads at ~40 MB/s under load).

**Why:** Each of these cost time on 2026-10-05.
**How to apply:** Check these first when GPU or model loading misbehaves on this machine.
