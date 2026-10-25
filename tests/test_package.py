import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import package  # noqa: E402

SOLUTION = '''import sys
from pathlib import Path

import fallback
import submission


def main():
    spec = fallback.guess_spec(Path(sys.argv[1]).resolve())
    fallback.run(spec, config=fallback.FallbackConfig(time_budget=60, models=("lightgbm",)))
    return 0 if not submission.check_file(spec) else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def _base(tmp_path: Path, solution: str = SOLUTION) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    (base / "solution.py").write_text(solution, encoding="utf-8")
    shutil.copy(ROOT / "requirements.txt", base / "requirements.txt")
    return base


def test_builds_flat_zip_and_smoke_passes(tmp_path):
    out = tmp_path / "solution.zip"
    base = _base(tmp_path)
    (base / "baseline.py").write_text("raise SystemExit('stale file from the base archive')\n", encoding="utf-8")
    code = package.main(["--base", str(base), "--p3", str(ROOT), "--out", str(out), "--smoke",
                         "--smoke-timeout", "300"])
    assert code == 0 and out.is_file()
    names = zipfile.ZipFile(out).namelist()
    assert "solution.py" in names and "fallback.py" in names and "requirements.txt" in names
    assert zipfile.ZipFile(out).read("solution.py").decode() == SOLUTION  # решение команды не подменено v0 из репо
    assert "check_llm.py" not in names  # из репозитория берутся только модули P3
    assert all("/" not in name for name in names)  # строго корень, без папок


def test_missing_module_secret_and_requirements_are_caught(tmp_path):
    base = _base(tmp_path, SOLUTION.replace("import submission", "import submission\nimport not_packed_module"))
    (base / "leak.py").write_text('KEY = "sk-' + "a" * 30 + '"\n', encoding="utf-8")
    (base / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    out = tmp_path / "bad.zip"
    assert package.main(["--base", str(base), "--p3", str(ROOT), "--out", str(out)]) == 1
    assert not out.exists()
    problems = package.check(package.collect(base, [ROOT / m for m in package.P3_MODULES]), ROOT / "requirements.txt")
    text = " ".join(problems)
    assert "not_packed_module" in text and "ключ" in text and "requirements.txt" in text


def test_guarded_imports_are_optional(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("try:\n    from schemas import X\nexcept ImportError:\n    X = None\n"
                    "def f():\n    import agent\n", encoding="utf-8")
    assert package.local_imports(path) == {"agent"}  # schemas — необязательный, agent — нужен (ленивый импорт)
