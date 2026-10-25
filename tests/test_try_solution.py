"""tools/try_solution.py: весь конвейер solution.solve с MockLLM — агент стартует от рецепта fallback."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import try_solution  # noqa: E402
from mocks import make_regression_task  # noqa: E402


def test_dry_run_reports_recipe_start_and_agent_experiment(tmp_path):
    spec = make_regression_task(tmp_path / "src")
    code = try_solution.main([str(spec.task_dir / "task.md"), "--dry-run", "--minutes", "15",
                              "--out", str(tmp_path / "try")])
    report = json.loads((tmp_path / "try" / "src" / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and report["submission_problems"] == []
    assert report["recipe_mode"] and report["fallback"]
    assert report["experiments"] and report["experiments"][0]["kind"] == "encoding"
    assert "task_analyst" in report["llm_usage"]["roles"] and "coder" in report["llm_usage"]["roles"]
    assert not (spec.task_dir / ".agent_work").exists()  # исходная папка задачи не тронута
