import logging
import time

import numpy as np
import pandas as pd

import ensemble
import fallback
import final
import solution
from mocks import make_multiclass_task, make_regression_task
from p3_fixtures import classification_task

ONE = fallback.FallbackConfig(models=("lightgbm",), try_log_target=False, bag_seeds=1, write=False)


def soon(seconds: float = 120.0) -> float:
    return time.monotonic() + seconds


def test_regression_members_share_recipe_folds(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    best = fallback.run(spec, None, ONE)
    zoo = final.train(spec, best, until=soon(), threads=2, n_random=2)
    names = [c.name for c in zoo]
    assert "final_lgbm_slow" in names and any(n.startswith("final_lgbm_r") for n in names)
    for cand in zoo:
        assert cand.oof.shape == best.oof.shape and cand.test_pred.shape == best.test_pred.shape
        assert np.isfinite(cand.oof).all() and np.isfinite(cand.test_pred).all()
        assert np.array_equal(cand.meta["folds"], best.meta["folds"])
    mixed = ensemble.blend(spec, [fallback.honest_view(best), *fallback.pool(best), *zoo])
    if mixed is not None:
        assert set(mixed.meta["blend"]) <= {best.name, *[c.name for c in fallback.pool(best)], *names}


def test_regression_with_negative_target_has_no_log_members(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=400)  # y = 5·x1·x2 + … — есть отрицательные
    best = fallback.run(spec, None, ONE)
    ctx = final.context(best)
    plan = final.members(ctx, final.reference(ctx)) + final.random_members(ctx, final.reference(ctx))
    assert all(m.transform != "log1p" for m in plan)


def test_imbalanced_macro_f1_members_use_class_weights(tmp_path):
    spec, _ = classification_task(tmp_path / "cls", n_classes=3)  # дисбаланс 4:1
    best = fallback.run(spec, None, ONE)
    ctx = final.context(best)
    plan = final.members(ctx, final.reference(ctx))
    assert {m.name for m in plan} == {"cat_sqrtbal", "xgb_classw", "lgbm_bal_reg", "lgbm_bal_slow"}
    zoo = final.train(spec, best, until=soon(), threads=2, n_random=0)
    assert zoo, "хотя бы одна модель зоопарка обучилась"
    for cand in zoo:
        assert cand.oof.shape == best.oof.shape and cand.meta["classes"] == best.meta["classes"]
        assert np.allclose(cand.oof.sum(axis=1), 1.0, atol=1e-6)


def test_no_time_or_no_recipe_gives_nothing(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls")
    best = fallback.run(spec, None, ONE)
    assert final.train(spec, best, until=time.monotonic()) == []
    bare = fallback.Candidate("const", 0.0, 0.0, best.oof, best.test_pred, {})
    assert final.train(spec, bare, until=soon()) == []


def test_failed_member_is_skipped(tmp_path, monkeypatch, caplog):
    spec = make_regression_task(tmp_path / "reg", n_train=400)
    best = fallback.run(spec, None, ONE)
    real = fallback.cv_model

    def flaky(data, name, model, *args, **kw):
        if model == "xgboost":
            raise MemoryError("boom")
        return real(data, name, model, *args, **kw)

    monkeypatch.setattr(fallback, "cv_model", flaky)
    with caplog.at_level(logging.WARNING, logger="final"):
        zoo = final.train(spec, best, until=soon(), threads=2, n_random=0)
    assert zoo and all(c.meta["model"] != "xgboost" for c in zoo)
    assert "упал: MemoryError" in caplog.text


def test_bag_averages_seeds_of_weighted_members(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=400)
    best = fallback.run(spec, None, ONE)
    zoo = final.train(spec, best, until=soon(), threads=2, n_random=0)
    first, rest = zoo[0], zoo[1:]
    weights = {first.name: 0.6, **{c.name: 0.01 for c in rest}}
    bagged = final.bag(spec, best, zoo, weights, until=soon(), threads=2)
    assert [b.name for b in bagged] == [f"{first.name}_bag3"]
    assert bagged[0].meta["bagged_from"] == first.name and bagged[0].meta["seeds"] == 3
    assert not np.allclose(bagged[0].oof, first.oof)  # другие сиды — другие деревья
    assert final.bag(spec, best, zoo, weights, until=time.monotonic(), threads=2) == []


def test_solve_runs_zoo_and_keeps_valid_submission(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SOLUTION_FINAL_SECONDS", "120")
    run = fallback.run
    monkeypatch.setattr(fallback, "run", lambda spec, deadline=None: run(
        spec, deadline, fallback.FallbackConfig(models=("lightgbm",), try_log_target=False, bag_seeds=1)))
    spec = make_regression_task(tmp_path / "reg")
    deadline = solution.Deadline(time.monotonic(), 3600, 600)
    with caplog.at_level(logging.INFO):
        assert solution.solve(spec.task_dir / "task.md", llm=None, deadline=deadline) == []
    assert "зоопарк A:" in caplog.text and "зоопарк B:" in caplog.text and "final: lgbm_slow" in caplog.text
    assert "final: lgbm_b0" in caplog.text  # этап B: свои случайные варианты
    sub = pd.read_csv(spec.task_dir / "submission.csv")
    assert len(sub) == 150 and np.isfinite(sub["target"]).all()


def test_layout_of_three_hours(monkeypatch):
    monkeypatch.delenv("SOLUTION_FINAL_SECONDS", raising=False)  # conftest выключает зоопарк — здесь умолчания
    d = solution.Deadline(time.monotonic(), solution.TOTAL_SECONDS, solution.RESEARCH_END)
    assert solution.RESEARCH_END == 7500 and solution.zoo_a_end(d, with_agent=True) == 4200  # T+2:05, T+1:10
    assert solution.zoo_a_end(d, with_agent=False) == solution.RESEARCH_END
    agent_d = solution.agent_deadline(d)
    assert agent_d.research_end == d.research_end and agent_d.total == 8700  # доводка агента — до T+2:25
    assert 9500 < solution.zoo_seconds(d, d.total) <= d.total - solution.FINAL_RESERVE  # зоопарк B — до T+2:40
    short = solution.Deadline(time.monotonic(), 3600, 1800)  # локальный прогон с коротким окном
    assert solution.zoo_a_end(short, with_agent=True) == 1800 - solution.AGENT_MIN_SECONDS
    monkeypatch.setenv("SOLUTION_FINAL_SECONDS", "0")
    assert solution.zoo_seconds(d, 4200) == 0
