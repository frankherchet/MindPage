"""Mine historical coding tasks from a C++ repository's git history.

A task is a non-merge commit that changes both C++ source and test files. The
model sees the repository at the parent revision plus the commit message and
must produce the source change; the commit's test change is applied afterwards
to judge it. Patches are not stored: ``git diff <parent> <commit> -- <files>``
reproduces them.

Usage::

    python -m mindpage.benchmark /path/to/repo --cutoff 2025-01-01 > tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

CPP_SUFFIXES = frozenset(
    {".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".ipp", ".inl"}
)


@dataclass(frozen=True, slots=True)
class CodingTask:
    commit: str
    parent: str
    date: str
    description: str
    source_files: tuple[str, ...]
    test_files: tuple[str, ...]
    source_lines_changed: int
    split: str


def is_test_path(path: str) -> bool:
    return any("test" in part.lower() for part in PurePosixPath(path).parts)


def mine_tasks(
    repo: str,
    *,
    cutoff: str | None = None,
    max_source_lines: int = 200,
    since: str | None = None,
) -> Iterator[CodingTask]:
    """Yield tasks oldest first.

    Commits dated before ``cutoff`` (ISO date) are marked ``train``, the rest
    ``eval``, so adapters never see changes that come after an evaluation task.
    """
    args = [
        "git", "-C", repo, "log", "--no-merges", "--no-renames", "--reverse",
        "--numstat", "--format=%x1e%H%x1f%P%x1f%aI%x1f%B%x1d",
    ]
    if since:
        args.append(f"--since={since}")
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout

    for record in out.split("\x1e")[1:]:
        header, numstat = record.split("\x1d", 1)
        commit, parents, date, message = header.split("\x1f", 3)

        source: list[str] = []
        tests: list[str] = []
        source_lines = 0
        for line in numstat.strip().splitlines():
            added, deleted, path = line.split("\t", 2)
            if is_test_path(path):
                tests.append(path)
            elif PurePosixPath(path).suffix.lower() in CPP_SUFFIXES:
                source.append(path)
                if added != "-":  # binary files report "-"
                    source_lines += int(added) + int(deleted)

        if not source or not tests or source_lines > max_source_lines:
            continue
        yield CodingTask(
            commit=commit,
            parent=parents.split()[0],
            date=date,
            description=message.strip(),
            source_files=tuple(source),
            test_files=tuple(tests),
            source_lines_changed=source_lines,
            split="train" if cutoff and date[:10] < cutoff else "eval",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("--cutoff", help="ISO date; earlier commits become 'train'")
    parser.add_argument("--since", help="ignore commits before this date")
    parser.add_argument("--max-source-lines", type=int, default=200)
    ns = parser.parse_args(argv)

    for task in mine_tasks(
        ns.repo, cutoff=ns.cutoff, max_source_lines=ns.max_source_lines, since=ns.since
    ):
        print(json.dumps(asdict(task)), file=sys.stdout)


if __name__ == "__main__":
    main()
