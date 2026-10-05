from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import TraceRun


@dataclass(frozen=True, slots=True)
class PromptRecord:
    task_group: str
    prompt: str


def read_prompts(path: str | Path) -> tuple[PromptRecord, ...]:
    records: list[PromptRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                records.append(PromptRecord(str(data["task_group"]), str(data["prompt"])))
            except KeyError as exc:
                raise ValueError(f"line {line_number}: missing {exc.args[0]!r}") from exc
    return tuple(records)


def write_traces(path: str | Path, runs: Iterable[TraceRun]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")


def read_traces(path: str | Path) -> tuple[TraceRun, ...]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return tuple(
            TraceRun.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        )
