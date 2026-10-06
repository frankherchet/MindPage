import os
import subprocess
from pathlib import Path

from mindpage.benchmark import mine_tasks
from mindpage.benchmark.tasks import is_test_path


def _commit(repo: Path, date: str, message: str, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", repo, "-c", "commit.gpgsign=false", "commit", "-qm", message],
        check=True,
        env=env,
    )


def test_is_test_path() -> None:
    assert is_test_path("test/format-test.cc")
    assert is_test_path("src/foo_unittest.cpp")
    assert not is_test_path("include/fmt/format.h")


def test_mines_commits_with_source_and_test_changes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    _commit(tmp_path, "2024-01-01T00:00:00", "init", {"src/a.cc": "1\n", "README": "x"})
    _commit(
        tmp_path,
        "2024-06-01T00:00:00",
        "Fix overflow\n\nDetails.",
        {"src/a.cc": "1\n2\n", "test/a_test.cc": "t\n"},
    )
    _commit(tmp_path, "2024-07-01T00:00:00", "docs only", {"README": "y"})
    _commit(
        tmp_path,
        "2025-02-01T00:00:00",
        "Add feature",
        {"src/a.cc": "1\n2\n3\n", "tests/b_test.cpp": "t\n"},
    )
    _commit(
        tmp_path,
        "2025-03-01T00:00:00",
        "Too big",
        {"src/a.cc": "x\n" * 50, "test/a_test.cc": "u\n"},
    )

    tasks = list(mine_tasks(str(tmp_path), cutoff="2025-01-01", max_source_lines=20))

    assert [t.description for t in tasks] == ["Fix overflow\n\nDetails.", "Add feature"]
    assert [t.split for t in tasks] == ["train", "eval"]
    assert tasks[0].source_files == ("src/a.cc",)
    assert tasks[0].test_files == ("test/a_test.cc",)
    assert tasks[0].source_lines_changed == 1
    assert len(tasks[0].parent) == 40
