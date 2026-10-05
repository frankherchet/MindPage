"""Smoke test: can vLLM serve Kolibri-1 with a LoRA adapter under CPU offload?

Builds two dummy PEFT adapters and compares greedy outputs against the base
model:

- zero adapter (lora_B = 0): output must equal the base output,
- random adapter: output must differ from the base output.

Also prints decode throughput, the first data point for the "static CPU
offload" baseline in docs/research.md.

Target: 16 GB GPU + 128 GB RAM. Needs vLLM 0.29 + aleph-alpha-inference
(``pip install aleph-alpha-inference``)::

    python experiments/kolibri1_lora_smoke.py --cpu-offload-gb 70
    python experiments/kolibri1_lora_smoke.py --cpu-offload-gb 70 --shared-experts
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL = "Aleph-Alpha/Kolibri-1"
PROMPTS = [
    "// C++17: return the index of the first element greater than x in a sorted vector\n",
    "Explain the difference between std::unique_ptr and std::shared_ptr in two sentences.",
]


def module_shapes(config: dict, shared_experts: bool) -> dict[str, tuple[int, int]]:
    """Map HF module suffix to (in_features, out_features)."""
    hidden = config["hidden_size"]
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    shapes = {
        "self_attn.q_proj": (hidden, q),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (q, hidden),
    }
    if shared_experts:
        inter = config["shared_expert_intermediate_size"]
        shapes |= {
            "mlp.shared_experts.gate_proj": (hidden, inter),
            "mlp.shared_experts.up_proj": (hidden, inter),
            "mlp.shared_experts.down_proj": (inter, hidden),
        }
    return shapes


def write_adapter(
    path: Path, config: dict, rank: int, shared_experts: bool, zero: bool
) -> None:
    shapes = module_shapes(config, shared_experts)
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for layer in range(config["num_hidden_layers"]):
        for module, (fan_in, fan_out) in shapes.items():
            key = f"base_model.model.model.layers.{layer}.{module}"
            tensors[f"{key}.lora_A.weight"] = (
                torch.randn(rank, fan_in, generator=generator) / fan_in**0.5
            ).bfloat16()
            b = torch.zeros(fan_out, rank)
            if not zero:
                b = torch.randn(fan_out, rank, generator=generator) * 0.02
            tensors[f"{key}.lora_B.weight"] = b.bfloat16()

    path.mkdir(parents=True)
    save_file(tensors, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": MODEL,
                "r": rank,
                "lora_alpha": 2 * rank,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": sorted({m.split(".")[-1] for m in shapes}),
            }
        )
    )


def generate(llm: LLM, lora: LoRARequest | None) -> tuple[list[str], float]:
    params = SamplingParams(temperature=0.0, max_tokens=64)
    start = time.perf_counter()
    outputs = llm.generate(PROMPTS, params, lora_request=lora)
    elapsed = time.perf_counter() - start
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return [o.outputs[0].text for o in outputs], tokens / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cpu-offload-gb", type=float, default=70)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--shared-experts", action="store_true",
                        help="also target the shared expert MLP")
    ns = parser.parse_args()

    from huggingface_hub import hf_hub_download

    config = json.loads(Path(hf_hub_download(ns.model, "config.json")).read_text())

    with tempfile.TemporaryDirectory() as tmp:
        zero_path, random_path = Path(tmp, "zero"), Path(tmp, "random")
        write_adapter(zero_path, config, ns.rank, ns.shared_experts, zero=True)
        write_adapter(random_path, config, ns.rank, ns.shared_experts, zero=False)

        llm = LLM(
            model=ns.model,
            cpu_offload_gb=ns.cpu_offload_gb,
            max_model_len=ns.max_model_len,
            enable_lora=True,
            max_lora_rank=ns.rank,
            max_loras=1,
            enforce_eager=True,
        )
        base, base_tps = generate(llm, None)
        zero, _ = generate(llm, LoRARequest("zero", 1, str(zero_path)))
        rand, _ = generate(llm, LoRARequest("random", 2, str(random_path)))

    print(f"decode throughput (base, cpu_offload_gb={ns.cpu_offload_gb}): "
          f"{base_tps:.2f} tok/s")
    for prompt, b, z, r in zip(PROMPTS, base, zero, rand):
        print(f"\n--- {prompt.strip()[:60]}\nbase:   {b!r}\nzero:   {z!r}\nrandom: {r!r}")

    zero_ok = zero == base
    random_ok = rand != base
    print(f"\nzero adapter == base:   {'PASS' if zero_ok else 'FAIL'}")
    print(f"random adapter != base: {'PASS' if random_ok else 'FAIL'}")
    raise SystemExit(0 if zero_ok and random_ok else 1)


if __name__ == "__main__":
    main()
