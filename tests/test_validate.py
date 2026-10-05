import shutil
import subprocess
from pathlib import Path

import pytest

from mindpage.benchmark.validate import classify, validate_tasks

from test_benchmark import _commit

CMAKE = """\
cmake_minimum_required(VERSION 3.16)
project(tiny CXX)
enable_testing()
add_library(lib src/lib.cc)
target_include_directories(lib PUBLIC src)
foreach(name old-test new-test)
  if(EXISTS ${CMAKE_SOURCE_DIR}/test/${name}.cc)
    add_executable(${name} test/${name}.cc)
    target_link_libraries(${name} lib)
    add_test(NAME ${name} COMMAND ${name})
  endif()
endforeach()
"""


def test_classify() -> None:
    assert classify({"a": "fail"}, {"a": "pass"})["status"] == "valid"
    assert classify({"a": "pass"}, {"a": "pass"})["status"] == "no_fail_before"
    assert classify({"a": "fail"}, {"a": "fail"})["status"] == "fails_after"
    assert classify({"a": "fail", "b": "pass"}, {"a": "pass", "b": "fail"})["status"] == (
        "fails_after"
    )
    assert classify({"a": "fail"}, {"a": "build_fail"})["status"] == "build_error"
    assert classify({"a": "fail"}, None)["status"] == "build_error"
    result = classify(None, {"a": "pass"})
    assert result["status"] == "valid"
    assert result["build_fail_before"] == ["a"]


@pytest.mark.skipif(
    not (shutil.which("cmake") and shutil.which("ctest") and shutil.which("c++")),
    reason="needs cmake and a C++ compiler",
)
def test_validates_tiny_cmake_project(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", repo], check=True)
    _commit(
        repo,
        "2025-01-01T00:00:00",
        "init",
        {
            "CMakeLists.txt": CMAKE,
            "src/lib.h": "int answer();\n",
            "src/lib.cc": '#include "lib.h"\nint answer() { return 41; }\n',
            "test/old-test.cc": '#include "lib.h"\nint main() { return answer() > 0 ? 0 : 1; }\n',
        },
    )
    # Fixes answer() and adds twice(); old-test checks the fix, new-test the
    # new function, so it does not compile without the source change.
    _commit(
        repo,
        "2025-02-01T00:00:00",
        "Fix answer, add twice",
        {
            "src/lib.h": "int answer();\nint twice(int);\n",
            "src/lib.cc": (
                '#include "lib.h"\nint answer() { return 42; }\n'
                "int twice(int x) { return 2 * x; }\n"
            ),
            "test/old-test.cc": '#include "lib.h"\nint main() { return answer() == 42 ? 0 : 1; }\n',
            "test/new-test.cc": '#include "lib.h"\nint main() { return twice(2) == 4 ? 0 : 1; }\n',
        },
    )
    # Changes source and tests, but the test already passes before.
    _commit(
        repo,
        "2025-03-01T00:00:00",
        "Refactor",
        {
            "src/lib.cc": (
                '#include "lib.h"\nint answer() { return 42; }\n'
                "int twice(int x) { return x + x; }\n"
            ),
            "test/new-test.cc": '#include "lib.h"\nint main() { return twice(3) == 6 ? 0 : 1; }\n',
        },
    )
    log = subprocess.run(
        ["git", "-C", repo, "log", "--reverse", "--format=%H %P %aI"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()[1:]
    tasks = []
    for line in log:
        commit, parent, date = line.split()
        tasks.append(
            {
                "commit": commit,
                "parent": parent,
                "date": date,
                "source_files": ["src/lib.h", "src/lib.cc"],
                "test_files": ["test/old-test.cc", "test/new-test.cc"],
                "split": "eval",
            }
        )

    results = validate_tasks(repo, tasks, tmp_path / "work", workers=1, jobs=2, timeout=30)
    by_commit = {r["commit"]: r for r in results}

    fix = by_commit[tasks[0]["commit"]]
    assert fix["status"] == "valid"
    assert fix["fail_to_pass"] == ["new-test", "old-test"]
    assert fix["build_fail_before"] == ["new-test"]
    assert fix["pass_to_fail"] == []

    refactor = by_commit[tasks[1]["commit"]]
    assert refactor["status"] == "no_fail_before"
