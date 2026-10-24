import json
import time

import numpy as np
import pytest

import agent
from mocks import (DROP_ROWS_CODE, EXCEPTION_CODE, GOOD_CODE, INF_CODE, NO_FUNCTION_CODE, SLOW_CODE, MockLLM,
                   SimpleDeadline, StubEvaluator, feature_code, fenced, make_best, make_multiclass_task,
                   make_regression_task, plan, useless_code)

# авто-стратегии (выбор модели, постобработка) проверяются в своих тестах: здесь цикл ведёт себя как раньше
FAST = dict(min_start_seconds=10, llm_reserve_seconds=0, auto_model_search=False, postprocess=False)
GOOD_ON_TOP = feature_code('df["x1_x2"] = df["x1"] * df["x2"]\ndf["x3_scaled"] = df["x3"] * 2.0 + 1.0')


@pytest.fixture
def reg_task(tmp_path):
    return make_regression_task(tmp_path / "reg")


def run_research(spec, llm, *, best=None, evaluator=None, deadline=None, **cfg):
    evaluator = evaluator or StubEvaluator(n_splits=3, n_estimators=60)
    best = best if best is not None else make_best(spec, evaluator)
    result = agent.research(spec, best, llm, deadline or SimpleDeadline(600),
                            config=agent.AgentConfig(**{**FAST, **cfg}), evaluate=evaluator)
    return result, best, evaluator


def experiments(spec):
    path = spec.task_dir / ".agent_work" / "experiments.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def coder_prompts(llm):
    return [messages[-1]["content"] for role, messages, _ in llm.calls if role == "coder"]


# ---------------------------------------------------------------- preflight

def test_llm_none_returns_empty(reg_task):
    evaluator = StubEvaluator(n_splits=3, n_estimators=50)
    assert agent.research(reg_task, None, None, SimpleDeadline(600), evaluate=evaluator) == []
    assert evaluator.calls == 0


def test_llm_disabled_env_skips_real_client(reg_task, monkeypatch):
    class RealClient(MockLLM):
        offline = False

    monkeypatch.setenv("LLM_DISABLED", "1")
    llm = RealClient()
    assert agent.research(reg_task, None, llm, SimpleDeadline(600)) == []
    assert llm.calls == []


def test_deadline_exhausted_before_start(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    result, _, evaluator = run_research(reg_task, llm, deadline=SimpleDeadline(1))
    assert result == [] and llm.calls == [] and evaluator.calls == 1  # только fallback-кандидат из теста


def test_no_evaluator_and_local_off_returns_empty(reg_task):
    config = agent.AgentConfig(**FAST, local_evaluator=False)
    assert agent.research(reg_task, None, MockLLM(), SimpleDeadline(600), config=config) == []


def test_stub_evaluator_is_deterministic(reg_task, tmp_path):
    evaluator = StubEvaluator(n_splits=3, n_estimators=50)
    first, second = make_best(reg_task, evaluator), make_best(reg_task, evaluator)
    assert first.cv_score == second.cv_score and len(first.meta["fold_scores"]) == 3
    assert first.oof.shape == (400,) and np.isfinite(first.test_pred).all()

    cls = make_best(make_multiclass_task(tmp_path / "cls"), evaluator)
    assert 0 < cls.cv_score <= 1 and cls.oof.shape == (450, 3)


# ---------------------------------------------------------------- happy path

def test_happy_path_accumulates_and_returns_better(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3")],
                   "coder": [fenced(GOOD_CODE), fenced(GOOD_ON_TOP)]})
    result, best, _ = run_research(reg_task, llm)

    assert len(result) == 1
    cand = result[0]
    assert cand.cv_score < best.cv_score and cand.name.startswith("agent_01")
    assert cand.meta["hypothesis"] == "Interaction x1*x2" and cand.meta["iter"] == 1
    assert 'df["x1_x2"]' in cand.meta["code"] and cand.meta["model"] == "lightgbm"
    assert cand.oof.shape == (400,) and cand.test_pred.shape == (150,)
    assert 'df["x1_x2"]' in coder_prompts(llm)[1]  # база второго эксперимента — код принятого
    assert [r["status"] for r in experiments(reg_task)] == ["accepted", "rejected"]
    assert llm.calls[0][1][-1]["content"].endswith("/no_think")
    assert not (reg_task.task_dir / "submission.csv").exists()
    assert not list((reg_task.task_dir / ".agent_work" / "input").glob("*.parquet"))


def test_multiclass_greater_is_better(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls")
    llm = MockLLM({"planner": [plan("Product a*b")],
                   "coder": [fenced(feature_code('df["ab"] = df["a"] * df["b"]'))]})
    result, best, _ = run_research(spec, llm)
    assert len(result) == 1 and result[0].cv_score > best.cv_score and result[0].oof.shape == (450, 3)


def test_best_with_code_is_used_as_base(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    (agent_best,), _, evaluator = run_research(reg_task, llm)

    llm2 = MockLLM({"planner": [plan("Scaled x3")], "coder": [fenced(GOOD_ON_TOP)]})
    calls_before = evaluator.calls
    result, _, _ = run_research(reg_task, llm2, best=agent_best, evaluator=evaluator)
    assert result == []  # без улучшения над best
    assert evaluator.calls == calls_before + 1  # эталон не пересчитывался: только сам эксперимент
    assert 'df["x1_x2"]' in coder_prompts(llm2)[0]


# ---------------------------------------------------------------- ремонт и формат

def test_format_fix_then_repair_by_traceback(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2")],
                   "coder": ["I would add an interaction feature.", fenced(EXCEPTION_CODE), fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm)
    assert len(result) == 1 and llm.count("coder") == 3
    assert "KeyError" in coder_prompts(llm)[2] and "no_such_column" in coder_prompts(llm)[2]


def test_garbage_plan_does_not_crash(reg_task):
    llm = MockLLM({"planner": ["total garbage", "more garbage {", "still nothing"]})
    result, _, _ = run_research(reg_task, llm)
    assert result == [] and llm.count("coder") == 0 and llm.count("planner") == 4


def test_code_without_function_and_exceptions_three_failures_stop(reg_task):
    llm = MockLLM({"planner": [plan("h one", "h two", "h three", "h four", "h five")],
                   "coder": [fenced(NO_FUNCTION_CODE), fenced(NO_FUNCTION_CODE),
                             fenced(EXCEPTION_CODE), fenced(EXCEPTION_CODE),
                             "no code here", "still no code", fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm)
    rows = experiments(reg_task)
    assert result == [] and llm.count("coder") == 6
    assert [r["status"] for r in rows] == ["failed"] * 3
    assert "build_features" in rows[0]["error"] and "KeyError" in rows[1]["error"]
    assert "code block" in rows[2]["error"]


def test_timeout_rows_and_inf_are_failures(reg_task):
    llm = MockLLM({"planner": [plan("slow one", "drop rows", "inf ratio")],
                   "coder": [fenced(SLOW_CODE), fenced(SLOW_CODE), fenced(DROP_ROWS_CODE), fenced(DROP_ROWS_CODE),
                             fenced(INF_CODE), fenced(INF_CODE)]})
    result, _, _ = run_research(reg_task, llm, features_timeout=2, max_consecutive_failures=5)
    errors = [r["error"] for r in experiments(reg_task)]
    assert result == [] and len(errors) == 3
    assert "timeout" in errors[0] and "rows" in errors[1] and "inf" in errors[2]


def test_duplicate_code_is_not_rerun(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Same interaction again")],
                   "coder": [fenced(GOOD_CODE), fenced(GOOD_CODE)]})
    result, _, evaluator = run_research(reg_task, llm)
    rows = experiments(reg_task)
    assert len(result) == 1 and rows[1]["status"] == "rejected" and "same code as experiment #1" in rows[1]["error"]
    assert evaluator.calls == 3  # fallback из теста + эталон + первый эксперимент


def test_ignored_params_are_reported_to_memory(reg_task):
    code = GOOD_CODE.replace("PARAMS = {}", 'PARAMS = {"is_unbalance": True, "num_leaves": 15}')
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(code)]})
    result, _, _ = run_research(reg_task, llm)
    row = experiments(reg_task)[0]
    assert len(result) == 1 and result[0].meta["params"] == {"num_leaves": 15}
    assert row["status"] == "accepted" and "is_unbalance" in row["error"] and "num_leaves" not in row["error"]


# ---------------------------------------------------------------- память экспериментов

def test_experiment_records_diff_columns_and_base(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3")],
                   "coder": [fenced(GOOD_CODE), fenced(GOOD_ON_TOP)]})
    run_research(reg_task, llm)
    rows = experiments(reg_task)
    assert rows[0]["columns"] == ["x1", "x2"] and 'df["x1_x2"] = df["x1"] * df["x2"]' in rows[0]["diff"]
    assert rows[0]["base_hash"] == agent.code_hash(agent.BASE_CODE)
    assert rows[1]["base_hash"] == agent.code_hash(GOOD_CODE)  # кумулятивно: база — принятый #1
    assert rows[1]["columns"] == ["x3"] and "x1_x2" not in rows[1]["diff"]


def test_columns_of_spec_groups_count_as_touched(reg_task):
    code = feature_code('for col in spec.get("cat_cols", []):\n    df[col + "_n"] = df[col].map(df[col].value_counts())')
    llm = MockLLM({"planner": [plan("Count encoding of categoricals", kind="encoding")], "coder": [fenced(code)]})
    run_research(reg_task, llm)
    assert experiments(reg_task)[0]["columns"] == ["city"]


def test_rephrased_rejected_hypothesis_is_not_rerun(reg_task):
    llm = MockLLM({"planner": [plan("Count encoding of city", kind="encoding"),
                               plan("Frequency encoding for city", "Interaction x1*x2", kind="encoding")],
                   "coder": [fenced(useless_code("city_count")), fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm)
    rows = experiments(reg_task)
    assert [r["hypothesis"] for r in rows] == ["Count encoding of city", "Interaction x1*x2"]
    assert rows[0]["status"] == "rejected" and len(result) == 1 and llm.count("coder") == 2


def test_rephrased_duplicates_within_one_plan_are_dropped(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "x1 times x2 interaction", "Scaled x3")],
                   "coder": [fenced(GOOD_CODE), fenced(GOOD_ON_TOP)]})
    run_research(reg_task, llm)
    assert [r["hypothesis"] for r in experiments(reg_task)] == ["Interaction x1*x2", "Scaled x3"]


def planner_prompts(llm):
    return [messages[-1]["content"] for role, messages, _ in llm.calls if role == "planner"]


def test_replan_prompt_shows_coverage(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    run_research(reg_task, llm)
    first, replan = planner_prompts(llm)[0], planner_prompts(llm)[-1]
    assert "No experiments yet." in first
    assert "Kinds not tried yet: encoding, missing, outliers, target_transform" in replan
    assert "Columns no experiment has used yet: x3, city" in replan
    assert "[accepted] features | Interaction x1*x2" in replan


def test_coder_sees_diff_of_accepted_steps(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3")],
                   "coder": [fenced(GOOD_CODE), fenced(GOOD_ON_TOP)]})
    run_research(reg_task, llm)
    second = coder_prompts(llm)[1]
    assert "Accepted steps (already in the current best):" in second
    assert '    + df["x1_x2"] = df["x1"] * df["x2"]' in second


# ---------------------------------------------------------------- анализ худших объектов

def test_planner_gets_error_analysis_and_it_is_refreshed_after_accept(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    run_research(reg_task, llm)
    first, replan = planner_prompts(llm)[0], planner_prompts(llm)[-1]
    assert "## Where the current model errs\nError analysis of the current best model (out-of-fold):" in first
    section = lambda text: text.split("## Where the current model errs")[1].split("## Memory")[0]
    assert section(first) != section(replan)  # после принятия x1*x2 ошибки другие


def test_error_analysis_can_be_disabled(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    run_research(reg_task, llm, errors_chars=0)
    assert all("Where the current model errs" not in p for p in planner_prompts(llm))


def test_broken_oof_does_not_break_research(reg_task):
    class NoOof(StubEvaluator):
        def __call__(self, *args, **kw):
            cand = super().__call__(*args, **kw)
            cand.oof = None
            return cand

    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm, evaluator=NoOof(n_splits=3, n_estimators=60))
    assert len(result) == 1 and all("Where the current model errs" not in p for p in planner_prompts(llm))


# ---------------------------------------------------------------- рассуждающая модель (qwen3.8)

class BadRequest(Exception):
    status_code = 400


def test_empty_answer_is_retried_with_more_tokens(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": ["", fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm, code_max_tokens=1000)
    assert len(result) == 1
    assert [kw["max_tokens"] for role, _, kw in llm.calls if role == "coder"] == [1000, 2500]


def test_reasoning_hint_is_dropped_after_bad_request(reg_task):
    llm = MockLLM({"planner": [BadRequest("unknown field: reasoning"), plan("Interaction x1*x2")],
                   "coder": [fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm)
    kws = [kw for _, _, kw in llm.calls]
    assert len(result) == 1
    assert kws[0]["extra_body"] == {"reasoning": {"enabled": False}}
    assert all("extra_body" not in kw for kw in kws[1:])


def test_gateway_html_error_is_one_line_failure(reg_task):
    page = "<html><head><title>504 Gateway Time-out</title></head><body><center>nginx/1.28.3</center></body></html>"
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3")],
                   "coder": [RuntimeError(page), fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm)
    first = experiments(reg_task)[0]
    assert first["status"] == "failed" and first["error"] == "internal error: RuntimeError: 504 Gateway Time-out"
    assert len(result) == 1


def test_budget_error_with_status_400_still_stops(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3")],
                   "coder": [fenced(GOOD_CODE), BadRequest("Budget has been exceeded")]})
    result, _, _ = run_research(reg_task, llm)
    assert len(result) == 1 and llm.count("coder") == 2 and llm.count("planner") == 1


# ---------------------------------------------------------------- стоп-критерии

def test_llm_unavailable_mid_run_returns_accumulated(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3", "Third")],
                   "coder": [fenced(GOOD_CODE), agent.LLMUnavailable("Budget has been exceeded")]})
    result, _, _ = run_research(reg_task, llm)
    assert len(result) == 1 and llm.count("coder") == 2 and llm.count("planner") == 1


def test_budget_error_text_from_unknown_exception_stops(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3")],
                   "coder": [fenced(GOOD_CODE), RuntimeError("Error 429: Budget has been exceeded!")]})
    result, _, _ = run_research(reg_task, llm)
    assert len(result) == 1 and llm.count("coder") == 2 and llm.count("planner") == 1


def test_transient_llm_error_is_one_failure(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Interaction again")],
                   "coder": [TimeoutError("read timeout"), fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm)
    assert len(result) == 1 and [r["status"] for r in experiments(reg_task)] == ["failed", "accepted"]


def test_plateau_stops(reg_task):
    llm = MockLLM({"planner": [plan("u one", "u two", "u three", "u four")],
                   "coder": [fenced(useless_code(f"u{i}")) for i in range(4)]})
    result, _, _ = run_research(reg_task, llm, max_no_improve=2)
    assert result == [] and llm.count("coder") == 2
    assert [r["status"] for r in experiments(reg_task)] == ["rejected", "rejected"]


def test_max_iters(reg_task):
    llm = MockLLM({"planner": [plan("u one", "u two", "u three")],
                   "coder": [fenced(useless_code(f"u{i}")) for i in range(3)]})
    run_research(reg_task, llm, max_iters=1)
    assert llm.count("coder") == 1


def test_token_budget_stops(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3")],
                   "coder": [fenced(GOOD_CODE), fenced(GOOD_ON_TOP)]}, tokens=(1000, 200))
    result, _, _ = run_research(reg_task, llm, max_tokens=2000)
    assert len(result) == 1 and llm.count("coder") == 1


def test_deadline_runs_out_mid_run(reg_task):
    deadline = SimpleDeadline(600)

    def expire_then_answer(messages):
        deadline.end = time.monotonic()
        return fenced(GOOD_ON_TOP)

    llm = MockLLM({"planner": [plan("Interaction x1*x2", "Scaled x3", "Third")],
                   "coder": [fenced(GOOD_CODE), expire_then_answer, fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm, deadline=deadline)
    assert len(result) == 1 and llm.count("coder") == 2 and llm.count("planner") == 1


# ---------------------------------------------------------------- сбои харнесса и защита

def test_evaluator_always_failing_returns_empty(reg_task):
    def broken(*args, **kwargs):
        raise RuntimeError("evaluator is broken")

    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    best = make_best(reg_task, StubEvaluator(n_splits=3, n_estimators=60))
    result = agent.research(reg_task, best, llm, SimpleDeadline(600), config=agent.AgentConfig(**FAST),
                            evaluate=broken)
    assert result == [] and llm.calls == []


def test_evaluator_error_on_experiment_is_failure(reg_task):
    stub = StubEvaluator(n_splits=3, n_estimators=60)
    best = make_best(reg_task, stub)

    def flaky(spec, build, params, deadline):
        if stub.calls >= 2:
            raise ValueError("bad params")
        return stub(spec, build, params, deadline)

    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE), fenced(GOOD_CODE)]})
    result = agent.research(reg_task, best, llm, SimpleDeadline(600), config=agent.AgentConfig(**FAST),
                            evaluate=flaky)
    rows = experiments(reg_task)
    assert result == [] and rows[0]["status"] == "failed" and "evaluation failed" in rows[0]["error"]


def test_sandbox_crash_is_contained(reg_task):
    def broken_sandbox(code, cwd, timeout):
        raise OSError("sandbox is down")

    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    evaluator = StubEvaluator(n_splits=3, n_estimators=60)
    result = agent.research(reg_task, make_best(reg_task, evaluator), llm, SimpleDeadline(600),
                            config=agent.AgentConfig(**FAST), evaluate=evaluator, run_code=broken_sandbox)
    assert result == [] and llm.calls == []


def test_profile_failure_falls_back_to_mini_profile(reg_task):
    def broken_profile(train, test, spec):
        raise RuntimeError("eda failed")

    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    evaluator = StubEvaluator(n_splits=3, n_estimators=60)
    result = agent.research(reg_task, make_best(reg_task, evaluator), llm, SimpleDeadline(600),
                            config=agent.AgentConfig(**FAST), evaluate=evaluator, profile=broken_profile)
    assert len(result) == 1 and "x1 | num" in llm.calls[0][1][1]["content"]


def test_result_filtered_by_input_best(reg_task):
    perfect = agent.Candidate("fallback", 0.0, 0.0, np.zeros(400), np.zeros(150), {})
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    result, _, _ = run_research(reg_task, llm, best=perfect)
    assert result == [] and experiments(reg_task)[0]["status"] == "accepted"


def test_suspicious_jump_rejected_by_guards(reg_task):
    seen = []

    def guards(spec, cand, baseline):
        seen.append((cand.cv_score, baseline.cv_score))
        return ["leak suspected"]

    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    evaluator = StubEvaluator(n_splits=3, n_estimators=60)
    result = agent.research(reg_task, make_best(reg_task, evaluator), llm, SimpleDeadline(600),
                            config=agent.AgentConfig(**FAST), evaluate=evaluator, guards=guards)
    rows = experiments(reg_task)
    assert result == [] and len(seen) == 1 and rows[0]["status"] == "rejected" and "guards" in rows[0]["error"]


def test_critic_runs_before_replan(reg_task):
    llm = MockLLM({"planner": [plan("u one", "u two"), plan()],
                   "coder": [fenced(useless_code("u1")), fenced(useless_code("u2"))],
                   "critic": ["Interactions between x1 and x2 were never tried."]})
    run_research(reg_task, llm)
    plans = [messages[-1]["content"] for role, messages, _ in llm.calls if role == "planner"]
    assert llm.count("critic") == 1 and "Interactions between x1 and x2" in plans[1]
