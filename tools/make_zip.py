"""Сборка сабмита на платформу: плоский ZIP (все файлы в корне архива, без папок). Вне ZIP.

    .venv/bin/python tools/make_zip.py                 # → dist/solution_<коммит>.zip
    .venv/bin/python tools/make_zip.py --smoke         # + дымовой прогон solution.py в чистой папке (как платформа)
    .venv/bin/python tools/make_zip.py --out dist/x.zip

Берутся ровно файлы FILES из корня репозитория (в корне только модули решения; вспомогательное — в tools/).
Проверки перед упаковкой: все файлы на месте, нет ключей и абсолютных путей к своей машине, размер < 100 МБ, и
проверки tools/package.py (P3): каждый файл компилируется, все обязательные импорты — в архиве или установлены,
requirements.txt не изменён. tools/package.py — для сборки из уже распакованного архива с подменой модулей.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# точка входа, модули команды (P2 llm_client, P3 fallback/preprocess/metrics/submission, P5 sandbox), агент (P4)
# и модули ролей P1/P2/P5, написанные P4 (tuning, ensemble, task_understanding, guards, automl)
FILES = ("solution.py", "requirements.txt",
         "fallback.py", "preprocess.py", "metrics.py", "submission.py",
         "llm_client.py", "sandbox.py",
         "agent.py", "agent_eval.py", "prompts_agent.py", "memory.py", "error_analysis.py", "postprocess.py",
         "tuning.py", "ensemble.py", "final.py", "task_understanding.py", "guards.py", "automl.py")
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{16,}|/Users/|/home/[a-z]")
MAX_BYTES = 100 * 1024 * 1024


def check(root: Path = ROOT) -> list[str]:
    problems = []
    for name in FILES:
        path = root / name
        if not path.is_file():
            problems.append(f"{name}: нет файла")
        elif SECRET.search(path.read_text(encoding="utf-8")):
            problems.append(f"{name}: похоже на ключ или абсолютный путь")
    if problems:
        return problems
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import package  # tools/package.py (P3): компиляция, импорты, requirements, ключи
    return package.check({name: root / name for name in FILES}, root / "requirements.txt")


def build(out: Path, root: Path = ROOT) -> Path:
    problems = check(root)
    if problems:
        raise SystemExit("сборка остановлена:\n  " + "\n  ".join(problems))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / name, arcname=name)  # arcname без папок: всё в корне архива
    if out.stat().st_size > MAX_BYTES:
        raise SystemExit(f"{out}: {out.stat().st_size} байт > 100 МБ")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true", help="дымовой прогон solution.py без LLM в чистой папке")
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() or "local"
    out = build(args.out or ROOT / "dist" / f"solution_{commit}.zip")
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    print(f"{out}: {out.stat().st_size // 1024} КБ, файлов {len(names)}: {', '.join(names)}")
    if args.smoke:
        import package
        problems = package.smoke(out, timeout=900.0)
        print("дымовой прогон:", "ok" if not problems else "ОШИБКИ:\n- " + "\n- ".join(problems))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
