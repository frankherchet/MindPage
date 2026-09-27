from __future__ import annotations

import argparse
import json
from pathlib import Path

from mindpage.integrations import TransformersMoETracer
from mindpage.tracing import read_prompts, summarize_task_groups, write_traces


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Trace MoE router decisions by task group")
    cli.add_argument("--model", required=True, help="Hugging Face model id or local path")
    cli.add_argument("--input", required=True, help="JSONL prompts: task_group + prompt")
    cli.add_argument("--output", required=True, help="JSONL trace output")
    cli.add_argument("--summary", required=True, help="JSON summary output")
    cli.add_argument("--max-length", type=int, default=512)
    cli.add_argument("--top-k", type=int, default=None)
    cli.add_argument("--top-n-summary", type=int, default=32)
    cli.add_argument("--device-map", default="auto")
    cli.add_argument("--dtype", default="auto")
    cli.add_argument("--trust-remote-code", action="store_true")
    return cli


def main() -> None:
    args = parser().parse_args()
    prompts = read_prompts(args.input)
    tracer = TransformersMoETracer.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    runs = [
        tracer.trace_prompt(
            record.prompt,
            task_group=record.task_group,
            max_length=args.max_length,
            top_k=args.top_k,
        )
        for record in prompts
    ]
    write_traces(args.output, runs)
    Path(args.summary).write_text(
        json.dumps(
            summarize_task_groups(runs, top_n=args.top_n_summary),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
