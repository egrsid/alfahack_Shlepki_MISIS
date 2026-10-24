"""Песочница для запуска кода, сгенерированного LLM.

Контракт (см. CLAUDE.md): run_code(code, cwd, timeout, memory_limit) -> ExecResult.

Гарантии:
- жёсткий таймаут: по истечении весь выводимый процесс убивается, включая
  дерево дочерних процессов (psutil, recursive) и всю группу (killpg);
- чистое окружение: белый список переменных, всё, что похоже на секрет
  (KEY/TOKEN/SECRET/PASS/...), вырезано принудительно — LLM-код не видит
  OPENAI_API_KEY и другие чувствительные значения;
- лимит вывода: stdout/stderr усекаются до max_output_chars;
- результат передаётся через файл: путь кладём в env SANDBOX_RESULT,
  код пишет туда JSON, sandbox парсит его в ExecResult.result.

Ограничения: сеть и pip изнутри процесса НЕ отсекаются (на платформе это
не страшно — интернета нет; полная изоляция — задача Docker, см. этап
feature/docker-check). Отсеявшийся через double-fork процесс теоретически
может уцелеть — на практике setsid + killpg закрывают сценарии LLM-кода.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

try:  # POSIX-only; на платформе (Linux) доступен всегда
    import resource
except ImportError:  # pragma: no cover - Windows dev-машины
    resource = None

DEFAULT_TIMEOUT = 300          # секунд на выполнение
DEFAULT_MAX_OUTPUT = 12_000    # символов stdout/stderr суммарно
KILL_GRACE = 3.0               # секунд между мягким и жёстким убийством
RESULT_ENV = "SANDBOX_RESULT"
RESULT_DEFAULT_NAME = "sandbox_result.json"
_MAX_ARTIFACTS = 100           # не больше записей в artifacts

# Белый список переменных окружения. Всё остальное (включая OPENAI_API_KEY,
# AWS_*, прокси и т.п.) внутрь не попадает.
_ENV_ALLOW = re.compile(
    r"^(PATH|HOME|USER|LOGNAME|HOSTNAME|TMPDIR|TMP|TEMP|SHELL|"
    r"LANG|LC_ALL|LC_CTYPE|"
    r"SYSTEMROOT|SYSTEMDRIVE|COMSPEC|WINDIR|NUMBER_OF_PROCESSORS|"
    r"OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|"
    r"PYTHONIOENCODING)$"
)
# Двойная страховка: даже внутри белого списка режем то, что пахнет секретом.
_ENV_DENY = re.compile(r"(KEY|TOKEN|SECRET|PASS|CREDENTIAL|AUTH)", re.IGNORECASE)


@dataclass
class ExecResult:
    """Результат запуска кода в песочнице."""

    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    duration_sec: float = 0.0
    result: Any = None                       # распарсенный JSON из SANDBOX_RESULT
    artifacts: dict[str, int] = field(default_factory=dict)  # имя файла -> байт
    workdir: str = ""                        # реальный каталог выполнения
    error: str | None = None                 # ошибка самой песочницы (не кода)

    @property
    def ok(self) -> bool:
        """Код отработал успешно (без таймаута, rc=0, песочница не падала)."""
        return self.error is None and not self.timed_out and self.returncode == 0


def _build_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Окружение по белому списку, без секретов."""
    env = {
        name: value
        for name, value in os.environ.items()
        if _ENV_ALLOW.match(name) and not _ENV_DENY.search(name)
    }
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n[... усечено {len(text) - limit} символов ...]\n" + text[-half:]


def _child_setup(memory_limit_mb: int | None) -> None:  # вызывается в дочернем процессе
    if hasattr(os, "setsid"):
        os.setsid()  # своя группа процессов — убиваем всей группой
    if memory_limit_mb and resource is not None:
        # best-effort: основной механизм — обёртка через `ulimit` (см. run_code),
        # здесь дублируем для надёжности на Linux. Не должен ронять дочерний
        # процесс, если лимит меньше текущего VM родителя (бывает на macOS).
        try:
            limit = int(memory_limit_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):
            pass


def _build_cmd(code: str, memory_limit_mb: int | None) -> list[str]:
    """Команда запуска. Лимит памяти ставим через `ulimit` в оболочке:
    шелл крошечный, поэтому лимит применяется всегда, в отличие от
    setrlimit до exec в процессе с большим VM."""
    if memory_limit_mb and os.name == "posix":
        kb = int(memory_limit_mb) * 1024
        return [
            "/bin/sh", "-c",
            f"ulimit -v {kb} 2>/dev/null || ulimit -d {kb} 2>/dev/null; exec \"$@\"",
            "sh", sys.executable, "-I", "-u", "-c", code,
        ]
    return [sys.executable, "-I", "-u", "-c", code]


def _kill_group(pid: int) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _kill_tree(proc: subprocess.Popen) -> None:
    """Мягкое завершение дерева, затем SIGKILL группе и выжившим."""
    try:
        root = psutil.Process(proc.pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        procs = []
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=KILL_GRACE)
    _kill_group(proc.pid)  # добиваем группу (дети, появившиеся после снапшота)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=KILL_GRACE)


def _collect(proc: subprocess.Popen, timeout: float | None) -> tuple[str, str, bool]:
    """communicate с гарантией, что процесс не останется висеть."""
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        return "", "", True


def _snapshot(workdir: Path) -> set[str]:
    """Относительные пути существующих файлов (для вычисления артефактов)."""
    found: set[str] = set()
    for i, path in enumerate(workdir.rglob("*")):
        if i >= _MAX_ARTIFACTS * 10:
            break
        if path.is_file():
            found.add(str(path.relative_to(workdir)))
    return found


def run_code(
    code: str,
    cwd: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    memory_limit_mb: int | None = None,
    result_name: str = RESULT_DEFAULT_NAME,
    max_output_chars: int = DEFAULT_MAX_OUTPUT,
) -> ExecResult:
    """Запустить Python-код в изолированном процессе и вернуть ExecResult.

    Параметры:
        code:            строка с Python-кодом.
        cwd:             рабочий каталог (подкаталог задачи или /tmp).
                         Если None — создаётся временный каталог; sandbox его
                         не удаляет, путь возвращается в ExecResult.workdir.
        timeout:         жёсткий лимит секунд, после него дерево процессов
                         принудительно убивается.
        memory_limit_mb: если задан, дочернему процессу выставляется
                         RLIMIT_AS (best-effort, на Linux платформы надёжно).
        result_name:     имя файла результата в cwd; путь к нему кладётся
                         в env SANDBOX_RESULT, ожидается JSON.
        max_output_chars: суммарный лимит stdout/stderr.
    """
    start = time.monotonic()
    if cwd is None:
        workdir = Path(tempfile.mkdtemp(prefix="sandbox_"))
    else:
        workdir = Path(cwd)
        workdir.mkdir(parents=True, exist_ok=True)

    result_path = workdir / result_name
    if result_path.exists():  # не читать результат прошлого запуска
        result_path.unlink()
    before = _snapshot(workdir)

    result = ExecResult(workdir=str(workdir))
    env = _build_env({RESULT_ENV: str(result_path)})

    try:
        proc = subprocess.Popen(
            _build_cmd(code, memory_limit_mb),
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=(lambda: _child_setup(memory_limit_mb))
            if hasattr(os, "setsid") or memory_limit_mb
            else None,
        )
    except (OSError, ValueError) as exc:
        result.error = f"не удалось запустить процесс: {exc}"
        result.duration_sec = time.monotonic() - start
        return result

    stdout, stderr, timed_out = _collect(proc, timeout)
    if timed_out:
        _kill_tree(proc)
        stdout, stderr, still = _collect(proc, KILL_GRACE)
        if still:  # практически недостижимо, но процесс не должен пережить нас
            try:
                proc.kill()
            except OSError:
                pass
            stdout, stderr = stdout or "", stderr or ""

    if result_path.exists():
        try:
            result.result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            result.error = f"результат не распарсился: {exc}"
        try:
            result_path.unlink()
        except OSError:
            pass

    after = _snapshot(workdir)
    for name in sorted(after - before - {result_name})[:_MAX_ARTIFACTS]:
        try:
            result.artifacts[name] = (workdir / name).stat().st_size
        except OSError:
            result.artifacts[name] = -1

    result.stdout = _truncate(stdout or "", max_output_chars)
    result.stderr = _truncate(stderr or "", max_output_chars)
    result.returncode = proc.returncode
    result.timed_out = timed_out
    result.duration_sec = time.monotonic() - start
    return result
