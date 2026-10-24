"""Тесты песочницы (sandbox.py). Без LLM и сети."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from sandbox import ExecResult, run_code

try:
    import resource
except ImportError:
    resource = None

HAS_RESOURCE = resource is not None and hasattr(os, "setsid")


@pytest.fixture(autouse=True)
def _hang_guard():
    """Страховка от зависшего теста (SIGALRM есть на macOS/Linux)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("тест завис — похоже на баг в sandbox")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


def test_simple_code_runs(tmp_path: Path):
    res = run_code('print("hello sandbox")', cwd=tmp_path, timeout=30)
    assert res.error is None
    assert res.ok
    assert "hello sandbox" in res.stdout
    assert res.returncode == 0
    assert not res.timed_out


def test_result_file_is_parsed(tmp_path: Path):
    code = (
        "import os, json\n"
        "json.dump({'cv_score': 0.75, 'notes': ['a', 'b']}, "
        "open(os.environ['SANDBOX_RESULT'], 'w'))\n"
    )
    res = run_code(code, cwd=tmp_path, timeout=30)
    assert res.ok
    assert res.result == {"cv_score": 0.75, "notes": ["a", "b"]}
    # result-файл не должен попасть в артефакты
    assert not any("sandbox_result" in name for name in res.artifacts)


def test_timeout_kills_infinite_loop(tmp_path: Path):
    t0 = time.monotonic()
    res = run_code("while True:\n    pass", cwd=tmp_path, timeout=1)
    elapsed = time.monotonic() - t0
    assert res.timed_out
    assert not res.ok
    assert elapsed < 10, f"убийство заняло {elapsed:.1f} сек"
    assert res.duration_sec >= 1


def test_child_processes_are_killed(tmp_path: Path):
    marker = "sandbox_child_marker_9f8e7d"
    code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)  # {marker}'])\n"
        "time.sleep(120)\n"
    )
    res = run_code(code, cwd=tmp_path, timeout=1)
    assert res.timed_out
    time.sleep(0.5)  # дать killpg отработать
    import psutil

    leftover = [
        p.info["pid"]
        for p in psutil.process_iter(["cmdline"])
        if p.info["cmdline"] and any(marker in part for part in p.info["cmdline"])
    ]
    assert leftover == [], f"выжившие дочерние процессы: {leftover}"


def test_env_has_no_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-secret-for-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy")
    code = (
        "import os\n"
        "print('KEY_PRESENT=' + str('OPENAI_API_KEY' in os.environ))\n"
        "print('AWS_PRESENT=' + str('AWS_SECRET_ACCESS_KEY' in os.environ))\n"
        "print('PATH_PRESENT=' + str('PATH' in os.environ))\n"
    )
    res = run_code(code, cwd=tmp_path, timeout=30)
    assert res.ok
    assert "KEY_PRESENT=False" in res.stdout
    assert "AWS_PRESENT=False" in res.stdout
    assert "PATH_PRESENT=True" in res.stdout


def test_nonzero_returncode(tmp_path: Path):
    res = run_code("import sys\nsys.exit(3)", cwd=tmp_path, timeout=30)
    assert not res.ok
    assert res.returncode == 3
    assert not res.timed_out


def test_syntax_error_is_reported(tmp_path: Path):
    res = run_code("def broken(:", cwd=tmp_path, timeout=30)
    assert not res.ok
    assert res.returncode != 0
    assert "SyntaxError" in res.stderr


def test_stdout_is_truncated(tmp_path: Path):
    res = run_code("print('x' * 50000)", cwd=tmp_path, timeout=30,
                   max_output_chars=2_000)
    assert res.ok
    assert len(res.stdout) <= 2_500
    assert "усечено" in res.stdout


def test_artifacts_detected(tmp_path: Path):
    code = (
        "import pandas as pd\n"
        "pd.DataFrame({'a': [1, 2]}).to_parquet('features.parquet')\n"
        "open('notes.txt', 'w').write('hi')\n"
    )
    res = run_code(code, cwd=tmp_path, timeout=60)
    assert res.ok
    assert res.artifacts.get("features.parquet", 0) > 0
    assert res.artifacts.get("notes.txt", 0) == 2


@pytest.mark.skipif(not HAS_RESOURCE, reason="RLIMIT_AS доступен только на POSIX")
def test_memory_limit(tmp_path: Path):
    # RLIMIT_AS нельзя опустить ниже текущего VM процесса (особенность macOS),
    # поэтому лимит подбираем от VM родителя. Адресное пространство считается
    # без обращения к страницам — аллокации больших bytearray падают быстро.
    import psutil

    vms_mb = psutil.Process().memory_info().vms // 1024 // 1024
    limit_mb = max(512, vms_mb + 1024)
    chunks_gb = (limit_mb // 1024) + 2  # гарантированно больше лимита
    code = (
        "chunks = []\n"
        "try:\n"
        f"    for _ in range({chunks_gb}):\n"
        "        chunks.append(bytearray(1024 ** 3))\n"
        "except MemoryError:\n"
        "    print('MEMORY_ERROR_CAUGHT')\n"
        "print('CHUNKS', len(chunks))\n"
    )
    res = run_code(code, cwd=tmp_path, timeout=120, memory_limit_mb=limit_mb)
    assert res.error is None
    assert "MEMORY_ERROR_CAUGHT" in res.stdout or res.returncode != 0, (
        f"лимит памяти не сработал: {res.stdout!r} rc={res.returncode}"
    )


def test_workdir_is_created(tmp_path: Path):
    target = tmp_path / "nested" / "work"
    res = run_code("import os; print(os.getcwd())", cwd=target, timeout=30)
    assert res.ok
    assert str(target) in res.stdout


def test_broken_json_result_is_error(tmp_path: Path):
    code = (
        "import os\n"
        "open(os.environ['SANDBOX_RESULT'], 'w').write('{not json')\n"
    )
    res = run_code(code, cwd=tmp_path, timeout=30)
    assert not res.ok
    assert res.error is not None
    assert "не распарсился" in res.error
