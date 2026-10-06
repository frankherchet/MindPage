"""Validate mined tasks: the commit's tests must fail without its source change.

For each task the repository is checked out at ``parent`` and the commit's test
files are applied. The tests are built and run (cmake + ctest), then the
commit's source files are applied and the tests run again. A task is ``valid``
when at least one test goes from failing to passing and no test goes from
passing to failing (fail-to-pass, as in SWE-bench).

Statuses:

- ``valid``: at least one fail-to-pass test, no pass-to-fail test.
- ``no_fail_before``: every test passes before the source change.
- ``fails_after``: no test flips to passing, or some test regresses.
- ``build_error``: the commit itself does not configure or a test target in it
  does not compile.

A test whose target does not compile before the source change counts as failing
and is listed in ``build_fail_before``. Test executables are deleted before
every build so that a failed build never runs a stale binary.

Usage::

    python -m mindpage.benchmark.validate /path/to/repo tasks.jsonl \\
        --split eval --workers 4 --work-dir /tmp/validate > validated.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

PASS = "pass"
FAIL = "fail"
BUILD_FAIL = "build_fail"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True, text=True
    )


def apply_files(worktree: Path, commit: str, files: list[str]) -> None:
    """Bring ``files`` to their state in ``commit``, deleting files it removed."""
    for path in files:
        if _git(worktree, "cat-file", "-e", f"{commit}:{path}", check=False).returncode == 0:
            _git(worktree, "checkout", commit, "--", path)
        elif (worktree / path).exists():
            _git(worktree, "rm", "-q", "--", path)


class TestRunner:
    """Configure, build and run a cmake project's ctest tests in one build dir."""

    def __init__(
        self,
        source: Path,
        build: Path,
        *,
        jobs: int,
        timeout: int,
        cmake_args: list[str],
    ) -> None:
        self.source = source
        self.build = build
        self.jobs = jobs
        self.timeout = timeout
        self.cmake_args = list(cmake_args)
        self.ninja = shutil.which("ninja") is not None
        if self.ninja:
            self.cmake_args += ["-G", "Ninja"]
        if shutil.which("ccache"):
            self.cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]
        # Worktrees live at different paths; relative paths let ccache share hits.
        self.env = {**os.environ, "CCACHE_BASEDIR": str(source.parent)}

    def _run(self, args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args, capture_output=True, text=True, env=self.env, timeout=timeout
        )

    def _tests(self) -> dict[str, Path | None]:
        """Map each test to its command; ``None`` if ctest cannot find it."""
        out = self._run(["ctest", "--test-dir", str(self.build), "--show-only=json-v1"])
        return {
            test["name"]: Path(test["command"][0]) if test.get("command") else None
            for test in json.loads(out.stdout).get("tests", [])
        }

    def run(self) -> dict[str, str] | None:
        """Return ``{test: status}``, or ``None`` if configuring fails."""
        self.build.mkdir(parents=True, exist_ok=True)
        configure = self._run(
            ["cmake", "-S", str(self.source), "-B", str(self.build), *self.cmake_args]
        )
        if configure.returncode != 0:
            return None
        build = self.build.resolve()
        for exe in self._tests().values():
            # Only executables built here are ours to delete.
            if exe is not None and build in exe.resolve().parents:
                exe.unlink(missing_ok=True)

        keep_going = ["-k", "0"] if self.ninja else ["-k"]
        self._run(
            ["cmake", "--build", str(self.build), "-j", str(self.jobs), "--", *keep_going]
        )

        failed_log = self.build / "Testing" / "Temporary" / "LastTestsFailed.log"
        failed_log.unlink(missing_ok=True)
        self._run(
            [
                "ctest", "--test-dir", str(self.build), "-j", str(self.jobs),
                "--timeout", str(self.timeout),
            ]
        )
        failed: set[str] = set()
        if failed_log.exists():
            for line in failed_log.read_text().splitlines():
                failed.add(line.split(":", 1)[1] if ":" in line else line)

        # A test whose executable was not rebuilt has no command any more.
        return {
            name: BUILD_FAIL if exe is None else FAIL if name in failed else PASS
            for name, exe in self._tests().items()
        }


def classify(before: dict[str, str] | None, after: dict[str, str] | None) -> dict:
    """Turn before/after test statuses into a task status and flipped tests.

    ``None`` means the project did not configure. Before the source change that
    counts as every test failing to build.
    """
    after_ok = after is not None and BUILD_FAIL not in after.values()
    after = after or {}
    if before is None:
        before = {t: BUILD_FAIL for t in after}
    fail_to_pass = sorted(t for t, s in before.items() if s != PASS and after.get(t) == PASS)
    pass_to_fail = sorted(
        t for t, s in before.items() if s == PASS and after.get(t, PASS) != PASS
    )
    if not after_ok:
        status = "build_error"
    elif all(s == PASS for s in before.values()):
        status = "no_fail_before"
    elif not fail_to_pass or pass_to_fail:
        status = "fails_after"
    else:
        status = "valid"
    return {
        "status": status,
        "fail_to_pass": fail_to_pass,
        "pass_to_fail": pass_to_fail,
        "build_fail_before": sorted(t for t, s in before.items() if s == BUILD_FAIL),
        "failing_after": sorted(t for t, s in after.items() if s != PASS),
    }


def validate_task(task: dict, worktree: Path, runner: TestRunner) -> dict:
    """Check out ``task`` before and after its source change and classify it."""
    start = time.monotonic()
    _git(worktree, "checkout", "-q", "--force", "--detach", task["parent"])
    _git(worktree, "clean", "-fdxq")
    apply_files(worktree, task["commit"], list(task["test_files"]))
    before = runner.run()
    apply_files(worktree, task["commit"], list(task["source_files"]))
    after = runner.run()
    return {
        **task,
        **classify(before, after),
        "validate_seconds": round(time.monotonic() - start, 1),
    }


def validate_tasks(
    repo: Path,
    tasks: list[dict],
    work_dir: Path,
    *,
    workers: int = 1,
    jobs: int | None = None,
    timeout: int = 300,
    cmake_args: list[str] | None = None,
    on_result=None,
) -> list[dict]:
    """Validate ``tasks`` with one git worktree and build dir per worker.

    Each worker gets a contiguous run of tasks in history order, so its builds
    stay incremental. Results are returned in completion order.
    """
    repo = repo.resolve()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(tasks, key=lambda t: t["date"])
    workers = max(1, min(workers, len(tasks)))
    jobs = jobs or max(1, (os.cpu_count() or 1) // workers)
    size = -(-len(tasks) // workers) if tasks else 0
    chunks = [tasks[i : i + size] for i in range(0, len(tasks), size)]

    results: list[dict] = []
    lock = Lock()

    def work(index: int, chunk: list[dict]) -> None:
        worktree = work_dir / f"worktree-{index}"
        if not worktree.exists():
            _git(repo, "worktree", "add", "-q", "--detach", str(worktree), chunk[0]["parent"])
        runner = TestRunner(
            worktree,
            work_dir / f"build-{index}",
            jobs=jobs,
            timeout=timeout,
            cmake_args=cmake_args or [],
        )
        for task in chunk:
            result = validate_task(task, worktree, runner)
            with lock:
                results.append(result)
                if on_result:
                    on_result(result)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(work, i, c) for i, c in enumerate(chunks)]:
            future.result()
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", type=Path)
    parser.add_argument("tasks", type=Path, help="JSONL from python -m mindpage.benchmark")
    parser.add_argument("--split", help="only validate tasks of this split")
    parser.add_argument("--work-dir", type=Path, default=Path("validate-work"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, help="build/test jobs per worker")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per test")
    parser.add_argument(
        "--skip-done", type=Path, help="JSONL of earlier results; skip their commits"
    )
    parser.add_argument(
        "--cmake-arg", action="append", default=[], help="extra cmake configure argument"
    )
    ns = parser.parse_args(argv)

    tasks = [json.loads(line) for line in ns.tasks.read_text().splitlines() if line.strip()]
    if ns.split:
        tasks = [t for t in tasks if t["split"] == ns.split]
    if ns.skip_done and ns.skip_done.exists():
        done = {
            json.loads(line)["commit"]
            for line in ns.skip_done.read_text().splitlines()
            if line.strip()
        }
        tasks = [t for t in tasks if t["commit"] not in done]

    def emit(result: dict) -> None:
        print(json.dumps(result), flush=True)
        print(f"{result['commit'][:10]} {result['status']}", file=sys.stderr, flush=True)

    validate_tasks(
        ns.repo,
        tasks,
        ns.work_dir,
        workers=ns.workers,
        jobs=ns.jobs,
        timeout=ns.timeout,
        cmake_args=ns.cmake_arg,
        on_result=emit,
    )


if __name__ == "__main__":
    main()
