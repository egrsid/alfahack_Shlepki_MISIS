"""Сборка плоского ZIP для платформы с проверками (вне ZIP). Зона P1, сделано с согласия P1.

    python tools/package.py --base <папка с файлами текущего архива> [--p3 <корень репо>] [--override файл.py]...
                            --out solution.zip [--smoke]

Файлы берутся из --base (распакованный текущий архив команды); --p3 подменяет ровно модули P3 (fallback.py, metrics.py,
preprocess.py, submission.py) версиями из репозитория; --override — любые явно названные файлы. Целые папки поверх
базы не накладываются: так в архив не попадут чужие или устаревшие файлы (например, solution.py v0 из main).
В архив идут только корневые *.py и requirements.txt — так их копирует платформа.

Проверки (любая ошибка → код возврата 1, архив не пишется):
- есть solution.py и requirements.txt; requirements.txt совпадает с выданным организаторами (--requirements);
- каждый .py компилируется; все импорты разрешаются: модуль либо в архиве, либо установлен (stdlib/requirements);
- нет ключей (sk-..., OPENAI_API_KEY=...) и .env; размер < 100 МБ;
- --smoke: в чистой временной папке `python3 solution.py task.md` на маленькой задаче с LLM_DISABLED=1
  и проверкой submission.csv (колонки, число строк, id, нет пропусков).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 100 * 1024 * 1024
P3_MODULES = ("fallback.py", "metrics.py", "preprocess.py", "submission.py")
SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|OPENAI_API_KEY\s*=\s*['\"][^'\"]+['\"]")


def collect(base: Path, overrides: list[Path]) -> dict[str, Path]:
    """Имя файла в корне архива → откуда его взять: всё из base, затем явно названные файлы overrides."""
    files: dict[str, Path] = {}
    for path in sorted(base.iterdir()):
        if path.is_file() and (path.suffix == ".py" or path.name == "requirements.txt"):
            files[path.name] = path
    for path in overrides:
        if not path.is_file():
            raise FileNotFoundError(f"--override: нет файла {path}")
        files[path.name] = path
    return files


def _optional_guard(handlers: list[ast.ExceptHandler]) -> bool:
    """try/except ImportError (или шире) — импорт необязательный: без модуля код работает дальше."""
    for handler in handlers:
        names = [handler.type] if not isinstance(handler.type, ast.Tuple) else list(handler.type.elts)
        if handler.type is None or any(isinstance(n, ast.Name) and n.id in ("ImportError", "ModuleNotFoundError",
                                                                            "Exception") for n in names):
            return True
    return False


def local_imports(path: Path) -> set[str]:
    """Модули, без которых файл упадёт: все импорты, кроме обёрнутых в try/except ImportError."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    optional: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and _optional_guard(node.handlers):
            optional.update(id(child) for stmt in node.body for child in ast.walk(stmt))
    names = set()
    for node in ast.walk(tree):
        if id(node) in optional:
            continue
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def check(files: dict[str, Path], requirements: Path) -> list[str]:
    problems = []
    for required in ("solution.py", "requirements.txt"):
        if required not in files:
            problems.append(f"нет {required}")
    if "requirements.txt" in files and requirements.is_file():
        if files["requirements.txt"].read_text(encoding="utf-8").split() != requirements.read_text(encoding="utf-8").split():
            problems.append(f"requirements.txt отличается от {requirements} (платформа ставит только выданный список)")
    modules = {name[:-3] for name in files if name.endswith(".py")}
    for name, path in files.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        if SECRET.search(text):
            problems.append(f"{name}: похоже на ключ API — убрать")
        if not name.endswith(".py"):
            continue
        try:
            compile(text, name, "exec")
        except SyntaxError as error:
            problems.append(f"{name}: SyntaxError {error.msg} (строка {error.lineno})")
            continue
        for module in sorted(local_imports(path) - modules):
            if importlib.util.find_spec(module) is None:
                problems.append(f"{name}: import {module} — нет ни в архиве, ни среди установленных пакетов")
    total = sum(path.stat().st_size for path in files.values())
    if total > MAX_BYTES:
        problems.append(f"размер {total / 2**20:.1f} МБ > 100 МБ")
    return problems


def make_task(root: Path, seed: int = 0) -> Path:
    """Маленькая регрессия в формате платформы: категория, пропуски, дата — для дымового прогона."""
    rng = np.random.default_rng(seed)
    n_train, n_test = 400, 120
    n = n_train + n_test
    df = pd.DataFrame({"id": np.arange(n) + 1, "x": rng.normal(size=n), "city": rng.choice(["a", "b", "c"], n),
                       "day": (pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"))
                       .strftime("%Y-%m-%d")})
    df.loc[rng.random(n) < 0.1, "x"] = np.nan
    y = 3 * df.x.fillna(0) + df.city.map({"a": 1.0, "b": 2.0, "c": 3.0}) + rng.normal(scale=0.2, size=n)
    root.mkdir(parents=True, exist_ok=True)
    df.iloc[:n_train].assign(target=y[:n_train].to_numpy()).to_csv(root / "train.csv", index=False)
    df.iloc[n_train:].to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": df.id[n_train:], "target": 0.0}).to_csv(root / "sample_submission.csv", index=False)
    (root / "task.md").write_text("Предсказать target. Метрика: RMSE.", encoding="utf-8")
    return root


def smoke(archive: Path, timeout: float) -> list[str]:
    """Распаковать архив в чистую папку с задачей и запустить решение так, как это делает платформа."""
    with tempfile.TemporaryDirectory() as tmp:
        task = make_task(Path(tmp) / "task")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(task)
        env = {"PATH": os.environ.get("PATH", os.defpath), "LLM_DISABLED": "1", "PYTHONIOENCODING": "utf-8",
               "FALLBACK_THREADS": "2"}
        try:
            run = subprocess.run([sys.executable, "solution.py", "task.md"], cwd=task, env=env, capture_output=True,
                                 text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return [f"дымовой прогон не уложился в {timeout:.0f} с"]
        if run.returncode != 0:
            return [f"solution.py завершился с кодом {run.returncode}: {run.stderr.strip()[-800:]}"]
        out = task / "submission.csv"
        if not out.is_file():
            return ["submission.csv не создан"]
        sub, sample = pd.read_csv(out), pd.read_csv(task / "sample_submission.csv")
        problems = []
        if list(sub.columns) != list(sample.columns):
            problems.append(f"колонки {list(sub.columns)} != {list(sample.columns)}")
        if len(sub) != len(sample) or set(sub["id"]) != set(sample["id"]):
            problems.append("строки/id не совпадают с sample_submission.csv")
        if sub.isna().any().any():
            problems.append("в submission.csv есть пропуски")
        return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, required=True, help="папка с файлами текущего архива")
    parser.add_argument("--p3", type=Path, help="корень репозитория: подменить модули P3 (" + ", ".join(P3_MODULES) + ")")
    parser.add_argument("--override", type=Path, action="append", default=[], help="файл, который подменяет одноимённый")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements.txt",
                        help="выданный организаторами requirements.txt")
    parser.add_argument("--smoke", action="store_true", help="дымовой прогон solution.py без LLM")
    parser.add_argument("--smoke-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    overrides = [args.p3 / name for name in P3_MODULES] if args.p3 else []
    files = collect(args.base, overrides + args.override)
    print("в архив:", ", ".join(f"{name} ← {path.parent.name or path.parent}" for name, path in sorted(files.items())))
    problems = check(files, args.requirements)
    if problems:
        print("ОШИБКИ:\n- " + "\n- ".join(problems))
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in sorted(files.items()):
            zf.write(path, arcname=name)  # строго в корень архива, без папок
    if args.smoke:
        problems = smoke(tmp, args.smoke_timeout)
        if problems:
            tmp.unlink()
            print("ДЫМОВОЙ ПРОГОН НЕ ПРОШЁЛ:\n- " + "\n- ".join(problems))
            return 1
        print("дымовой прогон: ok")
    os.replace(tmp, args.out)
    print(f"готово: {args.out} ({args.out.stat().st_size / 1024:.0f} КБ, файлов {len(files)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
