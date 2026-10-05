"""Capture per-token MoE expert choices during greedy generation (needs a GPU).

For each mined C++ task the prompt is the commit message plus the task's source
files at ``parent``, cut to ``--max-prompt-tokens``. A forward hook on every
layer's top-k router records the expert indices the model actually computes,
for the prompt (prefill) and for each generated token (decode).

Output per task: ``<out>/<commit>.npz`` with ``experts`` (tokens x layers x k,
uint16), ``prefill_tokens`` and a JSON ``meta`` string; read it with
``mindpage.simulation.ExpertTrace`` via ``experiments/moe_expert_report.py``.

Usage::

    python experiments/moe_expert_capture.py --model allenai/OLMoE-1B-7B-0125-Instruct \\
        --tasks fmt-tasks.jsonl --repo ~/src/fmt --out traces/olmoe --limit 3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are an expert C++ developer working on the fmt library."


def first_changed_line(repo: Path, task: dict, path: str) -> int:
    """First line of ``path`` at ``parent`` that the commit changes (1-based)."""
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "-U0", task["parent"], task["commit"], "--", path],
        capture_output=True, text=True,
    ).stdout
    match = re.search(r"^@@ -(\d+)", diff, re.MULTILINE)
    return int(match.group(1)) if match else 1


def build_prompt(tokenizer, task: dict, repo: Path, max_tokens: int) -> str:
    """Commit message plus source files at ``parent``, files cut to fit."""
    header = (
        f"Implement the following change in the fmt repository.\n\n"
        f"## Change\n{task['description']}\n\n## Files to modify\n"
    )
    footer = "\nWrite the modified code."
    budget = max_tokens - len(tokenizer(SYSTEM + header + footer).input_ids) - 64
    files = []
    for path in task["source_files"]:
        show = subprocess.run(
            ["git", "-C", str(repo), "show", f"{task['parent']}:{path}"],
            capture_output=True, text=True,
        )
        if show.returncode == 0:
            files.append((path, show.stdout))
    per_file = max(0, budget // max(1, len(files)))
    blocks = []
    for path, text in files:
        # Large headers do not fit; keep a window centred on the first change.
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > per_file:
            line = first_changed_line(repo, task, path)
            prefix = "".join(text.splitlines(keepends=True)[: max(0, line - 1)])
            center = len(tokenizer(prefix, add_special_tokens=False).input_ids)
            start = min(max(0, center - per_file // 2), len(ids) - per_file)
            ids = ids[start : start + per_file]
        blocks.append(f"### {path}\n```cpp\n{tokenizer.decode(ids)}\n```\n")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": header + "".join(blocks) + footer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class RouterRecorder:
    """Forward hooks on ``*TopKRouter`` modules; record selected expert indices."""

    def __init__(self, model) -> None:
        self.layers: dict[int, list[torch.Tensor]] = {}
        self.handles = []
        for name, module in model.named_modules():
            if type(module).__name__.endswith("TopKRouter"):
                layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
                self.handles.append(module.register_forward_hook(self._hook(layer)))
        if not self.handles:
            raise RuntimeError("no *TopKRouter modules found; unsupported architecture")

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            # (router_logits, router_scores, router_indices); indices: tokens x k
            self.layers.setdefault(layer, []).append(output[2].detach().to("cpu", torch.int16))

        return hook

    def reset(self) -> None:
        self.layers = {}

    def experts(self) -> np.ndarray:
        per_layer = [torch.cat(self.layers[i]) for i in sorted(self.layers)]
        return torch.stack(per_layer, dim=1).numpy().astype(np.uint16)


def expert_quantization(model) -> str:
    """Describe how the expert weights are stored, to catch silently skipped quantization."""
    for name, module in model.named_modules():
        if type(module).__name__.endswith("Experts"):
            params = {n: (str(p.dtype), tuple(p.shape)) for n, p in module.named_parameters()}
            return f"{type(module).__name__} {params}"
    return "no *Experts module found"


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--model", required=True)
    cli.add_argument("--tasks", required=True, type=Path, help="miner or validator JSONL")
    cli.add_argument("--repo", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--split", default="eval")
    cli.add_argument("--status", help="only tasks with this validation status, e.g. valid")
    cli.add_argument("--limit", type=int)
    cli.add_argument("--max-prompt-tokens", type=int, default=3072)
    cli.add_argument("--max-new-tokens", type=int, default=256)
    cli.add_argument("--max-gpu-memory", default="14GiB")
    cli.add_argument("--max-cpu-memory", default="26GiB")
    args = cli.parse_args()

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    tasks = [t for t in tasks if t["split"] == args.split]
    if args.status:
        tasks = [t for t in tasks if t.get("status") == args.status]
    tasks.sort(key=lambda t: t["date"])
    tasks = tasks[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
        max_memory={0: args.max_gpu_memory, "cpu": args.max_cpu_memory},
    )
    model.eval()
    config = model.config
    num_experts = config.num_experts
    # Reference size: one expert's gate, up and down projections in BF16.
    expert_bytes = 3 * config.hidden_size * config.moe_intermediate_size * 2 if hasattr(
        config, "moe_intermediate_size"
    ) else 3 * config.hidden_size * config.intermediate_size * 2
    placement = sorted({str(d) for d in getattr(model, "hf_device_map", {}).values()})
    quantization = expert_quantization(model)
    print(f"experts: {quantization}\nplacement: {placement}", flush=True)
    recorder = RouterRecorder(model)

    for task in tasks:
        path = args.out / f"{task['commit']}.npz"
        if path.exists():
            continue
        prompt = build_prompt(tokenizer, task, args.repo, args.max_prompt_tokens)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prefill = inputs.input_ids.shape[1]
        recorder.reset()
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens
            )
        seconds = time.monotonic() - start
        experts = recorder.experts()
        decode = experts.shape[0] - prefill
        meta = {
            "model_id": args.model,
            "commit": task["commit"],
            "num_layers": experts.shape[1],
            "num_experts": num_experts,
            "top_k": experts.shape[2],
            "expert_bytes": expert_bytes,
            "prefill_tokens": prefill,
            "decode_tokens": decode,
            "generated_tokens": output.shape[1] - prefill,
            "seconds": round(seconds, 1),
            "experts_storage": quantization,
            "placement": placement,
        }
        np.savez_compressed(path, experts=experts, prefill_tokens=prefill, meta=json.dumps(meta))
        print(
            f"{task['commit'][:10]} prefill={prefill} decode={decode} {seconds:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
