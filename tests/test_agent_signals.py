"""Сигналы планировщику (agent_eval.recipe_signals) и фильтр бесполезных новых колонок (agent.useless_new_columns)."""

import numpy as np
import pandas as pd
import pytest

import agent
import agent_eval
import fallback
import prompts_agent as P
from mocks import GOOD_CODE, MockLLM, SimpleDeadline, feature_code, fenced, make_regression_task, plan
from test_agent_loop import FAST, experiments

ONE = fallback.FallbackConfig(models=("lightgbm",), try_log_target=False, bag_seeds=1)
LOG_CODE = feature_code('df["log_x3"] = np.log1p(df["x3"] - df["x3"].min())\ndf["x3_sq"] = (df["x3"] + 100) ** 2')
WIDE_CODE = feature_code("\n".join(f'df["x1_x2_{i}"] = df["x1"] * df["x2"] + {i} * df["x3"]' for i in range(10)))


@pytest.fixture
def task(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    return spec, fallback.run(spec, None, ONE)


def test_useless_new_columns_drops_monotone_transforms_and_copies():
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({"a": rng.normal(size=300), "b": rng.uniform(1, 5, 300), "c": rng.choice(["x", "y"], 300)})
    out = raw.assign(log_b=np.log(raw["b"]), scaled_a=raw["a"] * 3 + 1, ab=raw["a"] * raw["b"],
                     ab_copy=raw["a"] * raw["b"] * 2, city=raw["c"] + "_")
    drop, notes, problem = agent.useless_new_columns(out, raw, max_new=8)
    assert problem == "" and drop == ["log_b", "scaled_a", "ab_copy"]
    assert "monotone transform of b" in notes[0] and "duplicates ab" in notes[2]
    _, _, problem = agent.useless_new_columns(out, raw, max_new=3)
    assert problem.startswith("too many new columns (5 > 3)")


def test_signals_describe_baseline_importance_and_drift(task):
    spec, best = task
    text = agent_eval.recipe_signals(agent_eval.RecipeEvaluator(best, threads=2), threads=2, rounds=50)
    assert text.startswith("- Baseline models (blend weights): lightgbm 1.00")
    assert "Feature importance (gain share): x" in text and "drift" in text


def _research(spec, best, coder, **cfg):
    evaluate = agent_eval.RecipeEvaluator(best, threads=2)
    llm = MockLLM({"planner": [plan("Hypothesis")], "coder": [fenced(code) for code in coder]})
    agent.research(spec, best, llm, SimpleDeadline(1200), evaluate=evaluate,
                   config=agent.AgentConfig(**{**FAST, "max_iters": 1, **cfg}))
    return evaluate, llm


def test_only_useless_columns_are_rejected_without_evaluation_or_repair(task):
    spec, best = task
    evaluate, llm = _research(spec, best, [LOG_CODE] * 2)
    row = experiments(spec)[0]
    assert row["status"] == "rejected" and row["error"].startswith(agent.USELESS_REJECT)
    assert "monotone transform of x3" in row["error"] and evaluate.calls == 1
    assert [role for role, _, _ in llm.calls].count("coder") == 1
    planner = next(m for role, m, _ in llm.calls if role == "planner")
    assert "## Baseline facts" in planner[1]["content"] and "Feature importance" in planner[1]["content"]


def test_too_many_new_columns_asks_coder_to_fix(task):
    spec, best = task
    evaluate, llm = _research(spec, best, [WIDE_CODE, GOOD_CODE], screen=False)
    coder = [m for role, m, _ in llm.calls if role == "coder"]
    assert len(coder) == 2 and "too many new columns (10 > 8)" in coder[1][1]["content"]  # ремонт с причиной
    assert experiments(spec)[0]["status"] == "accepted"


def test_plan_prompt_with_signals_fits_budget():
    profile = "\n".join(f"col{i} | num | 1.0 | 1000 | 0..100 | заметка о колонке номер {i}" for i in range(80))
    messages = P.build_plan_messages(task_line="type: regression", task_text="Задача. " * 400, profile=profile,
                                     base_code="x" * 3000, cv_text="CV rmse = 41", memory="m" * 1500, n=6,
                                     recipe=True, signals="- " + "s" * 2000, errors="e" * 800)
    assert "## Baseline facts" in messages[1]["content"] and P.messages_tokens(messages) <= 4000
