# MoE routing trace experiment

This experiment tests a prerequisite for expert paging: **do related tasks reuse a sufficiently stable set of layer-qualified experts?**

MindPage does not assume that an expert has a human-readable semantic role. We only measure whether task groups produce repeatable routing patterns that a cache or prefetcher could exploit.

## What is traced

The first backend uses Hugging Face Transformers models that expose `router_logits` when called with `output_router_logits=True`. It performs a prompt-only forward pass and records the selected top-k experts for every token and MoE layer.

Expert identity is always `(layer, expert_id)`. Expert `7` in two different layers is not treated as the same expert.

## Install

```bash
pip install -e '.[dev,moe]'
```

## Run

```bash
mindpage-moe-trace \
  --model mistralai/Mixtral-8x7B-v0.1 \
  --input experiments/prompts/moe_tasks.jsonl \
  --output traces.jsonl \
  --summary summary.json \
  --max-length 512
```

The model is intentionally a CLI argument. Start with any causal MoE architecture supported by your installed Transformers version and hardware. Large checkpoints may require quantization/offload; that is a separate variable and should be recorded with the experiment.

## Outputs

`traces.jsonl` stores the raw task group, prompt tokens, and per-token expert choices. `summary.json` reports aggregate usage and mean pairwise Jaccard overlap of each run's most-used `(layer, expert)` keys.

A high within-group overlap is not enough to justify expert paging by itself. The next analysis should compare:

- within-group overlap vs between-group overlap,
- LRU/frequency cache hit rate using the real trace order,
- how early a task-level predictor can identify useful experts,
- bytes transferred and stall time under realistic expert sizes.

## Current limits

- Prompt forward pass only; generation-time routing is not yet traced.
- Router logits must be exposed by the model output.
- Scores are stored for diagnostics, but the initial analysis counts selections rather than score mass.
- This experiment measures routing behavior only. No expert weights are moved yet.
