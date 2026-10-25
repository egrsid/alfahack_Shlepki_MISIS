"""solution.py: fallback → агент на тех же фолдах → запись только при честном выигрыше; раскладка ZIP."""

import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import agent
import fallback
import solution
from mocks import GOOD_CODE, MockLLM, fenced, make_multiclass_task, make_regression_task, plan, useless_code
from test_agent_loop import FAST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_zip  # noqa: E402

CFG = agent.AgentConfig(**FAST, auto_tune=False)  # правило записи проверяется без настоящего подбора (tuning.py)


def deadline(research: float = 600) -> solution.Deadline:
    return solution.Deadline(time.monotonic(), 3600, research)


def test_fallback_only_without_llm(tmp_path):
    spec = make_regression_task(tmp_path / "reg")
    assert solution.solve(spec.task_dir / "task.md", llm=None, deadline=deadline()) == []
    sub = pd.read_csv(spec.task_dir / "submission.csv")
    assert list(sub.columns) == ["id", "target"] and len(sub) == 150 and sub["target"].std() > 0
    assert not (spec.task_dir / ".agent_work").exists()  # агент не запускался


def test_agent_writes_submission_only_when_clearly_better(tmp_path, caplog, monkeypatch):
    # fallback — один LightGBM: признак x1*x2 (главный сигнал синтетики) честно лучше; CatBoost fallback выучил бы его сам
    run = fallback.run
    monkeypatch.setattr(fallback, "run", lambda spec, deadline=None: run(
        spec, deadline, fallback.FallbackConfig(models=("lightgbm",), try_log_target=False)))
    spec = make_regression_task(tmp_path / "reg")
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    with caplog.at_level(logging.INFO):
        assert solution.solve(spec.task_dir / "task.md", llm=llm, deadline=deadline(), agent_config=CFG) == []
    assert "submission.csv ← агент" in caplog.text
    caplog.clear()

    spec = make_regression_task(tmp_path / "reg2")
    llm = MockLLM({"planner": [plan("Scaled x3")], "coder": [fenced(useless_code("x3s"))]})
    with caplog.at_level(logging.INFO):
        assert solution.solve(spec.task_dir / "task.md", llm=llm, deadline=deadline(), agent_config=CFG) == []
    assert "submission.csv ← агент" not in caplog.text


def test_agent_failure_keeps_fallback_submission(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls")
    llm = MockLLM({"planner": [RuntimeError("boom")] * 5})
    assert solution.solve(spec.task_dir / "task.md", llm=llm, deadline=deadline(), agent_config=CFG) == []
    assert set(pd.read_csv(spec.task_dir / "submission.csv")["kind"]) <= {"high", "low", "mid"}


def test_no_research_window_skips_agent(tmp_path):
    spec = make_regression_task(tmp_path / "reg")
    llm = MockLLM()
    assert solution.solve(spec.task_dir / "task.md", llm=llm, deadline=deadline(research=0)) == []
    assert [role for role, _, _ in llm.calls] == ["task_analyst"]  # только понимание задачи, агент не запускался


class FakeClient:
    prompt_tokens, completion_tokens = 120, 30

    def __init__(self, error=None):
        self.error, self.kw = error, None

    def chat(self, messages, **kw):
        self.kw = kw
        if self.error is not None:
            raise self.error
        return "ok"


class GatewayError(Exception):
    status_code = 400


def test_agent_llm_adapter():
    client = FakeClient()
    llm = solution.AgentLLM(client)
    assert llm.chat([], max_tokens=5, extra_body={"x": 1}) == "ok" and client.kw == {"max_tokens": 5,
                                                                                      "extra_body": {"x": 1}}
    assert (llm.prompt_tokens, llm.completion_tokens) == (120, 30) and not hasattr(llm, "chat_json")
    wrapped = agent.LLMUnavailable("bad request")
    wrapped.__cause__ = GatewayError("reasoning is not supported")
    with pytest.raises(GatewayError):  # 400 — исходной ошибкой: агент повторит запрос без extra_body
        solution.AgentLLM(FakeClient(wrapped)).chat([])
    with pytest.raises(agent.LLMUnavailable):  # бюджет/сеть — как есть: агент остановится
        solution.AgentLLM(FakeClient(agent.LLMUnavailable("Budget has been exceeded"))).chat([])


def test_make_llm_respects_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert solution.make_llm(tmp_path) is None
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.setenv("LLM_DISABLED", "1")
    assert solution.make_llm(tmp_path) is None


def test_zip_is_flat_and_runs_in_clean_folder(tmp_path):
    assert make_zip.check() == []
    archive = make_zip.build(tmp_path / "solution.zip")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert sorted(names) == sorted(make_zip.FILES) and all("/" not in n for n in names)
        z.extractall(tmp_path / "platform")
    task = make_multiclass_task(tmp_path / "platform" / "task")
    for name in make_zip.FILES:  # как на платформе: модули рядом с task.md, cwd — каталог задачи
        shutil.copy(tmp_path / "platform" / name, task.task_dir / name)
    env = {"PATH": os.defpath, "LLM_DISABLED": "1", "HOME": str(tmp_path), "SOLUTION_AUTOML_SECONDS": "60"}
    result = subprocess.run([sys.executable, "-s", "solution.py", "task.md"], cwd=task.task_dir, env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    sub = pd.read_csv(task.task_dir / "submission.csv")
    assert list(sub.columns) == ["id", "kind"] and len(sub) == 150


def test_automl_stage_writes_only_when_clearly_better(tmp_path, caplog, monkeypatch):
    import automl

    spec = make_multiclass_task(tmp_path / "cls")
    monkeypatch.setenv("SOLUTION_AUTOML_SECONDS", "90")
    with caplog.at_level(logging.INFO):
        assert solution.solve(spec.task_dir / "task.md", llm=None, deadline=deadline()) == []
    assert "automl: AutoGluon multiclass" in caplog.text and ("AutoML: " in caplog.text)
    assert not (spec.task_dir / ".agent_work" / "automl").exists()

    caplog.clear()
    spec = make_regression_task(tmp_path / "reg")
    monkeypatch.setattr(automl, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.INFO):
        assert solution.solve(spec.task_dir / "task.md", llm=None, deadline=deadline()) == []
    assert "AutoML упал" in caplog.text  # сбой AutoML не трогает сабмит fallback


def test_automl_budget_leaves_window_for_agent(monkeypatch):
    monkeypatch.setenv("SOLUTION_AUTOML_SECONDS", "2400")
    assert solution.automl_seconds(deadline(research=1200), with_agent=True) == pytest.approx(300, abs=2)
    assert solution.automl_seconds(deadline(research=1200), with_agent=False) == pytest.approx(1200, abs=2)
    assert solution.automl_seconds(deadline(research=10_000), with_agent=True) == 2400
    monkeypatch.setenv("SOLUTION_AUTOML_SECONDS", "0")
    assert solution.automl_seconds(deadline(), with_agent=False) == 0


def test_environment_is_logged(caplog):
    with caplog.at_level(logging.INFO):
        found = solution.log_environment()
    assert found["lightgbm"] and "окружение: Python" in caplog.text and set(found) == set(solution.OPTIONAL_PACKAGES)
