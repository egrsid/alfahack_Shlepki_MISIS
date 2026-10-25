import types

import numpy as np
import pytest

import agent_eval
import ensemble
import fallback
import solution
from mocks import identity_features, make_multiclass_task, make_regression_task

ONE = fallback.FallbackConfig(models=("lightgbm",), try_log_target=False)


def agent_like(spec, best, model):
    meta = best.meta
    evaluate = agent_eval.LocalEvaluator(folds=meta["folds"], rows=meta.get("train_rows"), early_stopping=50,
                                         threads=2)
    return evaluate(spec, identity_features, {"model": model}, None)


def test_blend_of_fallback_and_agent_on_same_folds(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    best = fallback.run(spec, None, ONE)
    other = agent_like(spec, best, "xgboost")
    mixed = ensemble.blend(spec, [best, other])
    if mixed is not None:  # бленд может и не понадобиться — тогда None
        assert len(mixed.meta["fold_scores"]) == len(best.meta["fold_scores"])
        assert set(mixed.meta["blend"]) <= {best.name, other.name} and len(mixed.meta["blend"]) == 2
        assert mixed.test_pred.shape == best.test_pred.shape
    broken = types.SimpleNamespace(name="short", oof=best.oof[:-1], test_pred=best.test_pred, meta={})
    assert ensemble.blend(spec, [best, broken]) is None  # другие строки — не смешиваем


def test_class_columns_are_aligned_by_name(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls")
    best = fallback.run(spec, None, ONE)
    classes = list(best.meta["classes"])
    perm = classes[::-1]
    index = [classes.index(c) for c in perm]
    shuffled = types.SimpleNamespace(name="perm", oof=best.oof[:, index], test_pred=best.test_pred[:, index],
                                     meta={"classes": perm})
    oof, test_pred = ensemble._aligned(shuffled, best)
    assert np.allclose(oof, best.oof) and np.allclose(test_pred, best.test_pred)
    alien = types.SimpleNamespace(name="alien", oof=best.oof, test_pred=best.test_pred,
                                  meta={"classes": ["x", "y", "z"]})
    assert ensemble._aligned(alien, best) is None


def cand(name, scores):
    return types.SimpleNamespace(name=name, cv_score=float(np.mean(scores)), meta={"fold_scores_eff": scores})


def test_pick_takes_best_clearly_better_option(monkeypatch):
    spec = types.SimpleNamespace(metric="rmse")
    best = cand("fallback", [1.0, 1.1, 0.9, 1.0])
    agent_c = cand("agent", [0.95, 1.04, 0.86, 0.95])
    monkeypatch.setattr(ensemble, "blend", lambda spec, cands: cand("ensemble", [0.9, 1.0, 0.8, 0.9]))
    assert solution.pick(spec, best, [agent_c]).name == "ensemble"
    monkeypatch.setattr(ensemble, "blend", lambda spec, cands: None)
    assert solution.pick(spec, best, [agent_c]).name == "agent"
    noisy = cand("noisy", [0.8, 1.3, 0.7, 1.2])
    assert solution.pick(spec, best, [noisy]) is None
    assert solution.pick(spec, best, []) is None


def test_pick_survives_broken_ensemble(monkeypatch):
    def boom(spec, cands):
        raise RuntimeError("boom")
    monkeypatch.setattr(ensemble, "blend", boom)
    spec = types.SimpleNamespace(metric="macro_f1")
    best = cand("fallback", [0.5, 0.52, 0.48, 0.5])
    assert solution.pick(spec, best, [cand("agent", [0.6, 0.62, 0.58, 0.6])]).name == "agent"


def test_fallback_context_and_pool(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    best = fallback.run(spec, None)
    ctx = best.meta["context"]
    assert ctx.members and abs(sum(ctx.members.values()) - 1.0) < 1e-6
    assert set(ctx.members) <= {r.name for r in ctx.runs}
    members = fallback.pool(best)
    assert len(members) == len(ctx.runs) >= 2
    assert all(m.oof.shape == best.oof.shape and m.test_pred.shape == best.test_pred.shape for m in members)


def test_pool_blend_is_scored_honestly(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    y = ensemble._truth(spec, types.SimpleNamespace(meta={"train_rows": None, "classes": None}))
    rng = np.random.default_rng(0)
    folds = np.arange(len(y)) % 5
    meta = {"folds": folds, "train_rows": None, "classes": None, "metric": "rmse"}
    cands = [types.SimpleNamespace(name=f"m{i}", oof=y + rng.normal(0, 1, len(y)), test_pred=np.zeros(150),
                                   meta=dict(meta)) for i in range(4)]
    mixed = ensemble.blend(spec, cands)
    in_sample = [ensemble.metrics.score("rmse", y[folds == f], mixed.oof[folds == f]) for f in range(5)]
    assert mixed.meta["honest_oof"] and len(mixed.meta["blend"]) >= 2
    assert np.mean(mixed.meta["fold_scores"]) >= np.mean(in_sample) - 1e-9  # чужие веса не лучше подобранных на себе
    assert np.mean(mixed.meta["fold_scores"]) < min(ensemble.metrics.score("rmse", y, c.oof) for c in cands)
