import numpy as np
import pandas as pd

import ensemble
import fallback
from p3_fixtures import Spec, write_task

TWO = fallback.FallbackConfig(models=("lightgbm", "randomforest"), try_log_target=False, bag_seeds=1, write=False)


def period_task(root, n_per_year: int = 250, seed: int = 0) -> Spec:
    """Годы 1–4 в train, год 5 в test; штат «late» есть только в году 4, штат «new» — только в test."""
    rng = np.random.default_rng(seed)
    years = np.repeat([1, 2, 3, 4, 5], n_per_year)
    state = rng.choice(["a", "b", "c"], len(years)).astype(object)
    state[(years == 4) & (rng.random(len(years)) < 0.5)] = "late"
    state[(years == 5) & (rng.random(len(years)) < 0.3)] = "new"
    x = rng.normal(size=len(years))
    effect = pd.Series(state).map({"a": 0.0, "b": 1.0, "c": -1.0, "late": 4.0, "new": 4.0}).to_numpy()
    y = 3 * x + effect + 0.2 * years + rng.normal(scale=0.3, size=len(years))
    frame = pd.DataFrame({"id": np.arange(len(years)), "year": years, "state": state, "x": x})
    train, test = frame[years < 5].assign(target=y[years < 5]), frame[years == 5].reset_index(drop=True)
    write_task(root, train, test, pd.DataFrame({"id": test.id, "target": 0.0}), "Predict target. Metric: RMSE.")
    return Spec(root, "id", "target", "regression", "rmse", False)


def test_period_task_blends_on_recent_rows_and_handles_new_levels(tmp_path):
    spec = period_task(tmp_path / "reg")
    best = fallback.run(spec, None, TWO)
    data = best.meta["context"].data
    assert data.cv_scheme["kind"] == "period"

    recent = ensemble.recent_rows(best.meta, len(data.y))
    assert recent is not None and set(np.asarray(data.period_order)[recent]) == {3, 4}
    stress, risky = ensemble.new_level_rows(best.meta, np.asarray(best.meta["folds"]))
    test = pd.read_csv(spec.task_dir / "test.csv")
    assert risky.sum() == (test.state == "new").sum() > 0
    train = pd.read_csv(spec.task_dir / "train.csv")
    assert stress.sum() >= (train.state == "late").sum() >= ensemble.STRESS_MIN_ROWS

    cands = [fallback.honest_view(best), *fallback.pool(best)]
    mixed = ensemble.blend(spec, cands)
    plain = ensemble.blend(spec, cands, recent=False, new_levels=False)
    if mixed is None:  # одна модель лучше всех — бленда нет, проверять нечего
        return
    assert mixed.meta["blend_rows"] == "recent periods"
    assert mixed.meta["new_levels"]["test_rows"] == int(risky.sum())
    same_weights = ensemble.blend(spec, cands, new_levels=False)
    assert np.allclose(mixed.test_pred[~risky], same_weights.test_pred[~risky])  # меняются только новые уровни
    assert np.allclose(mixed.oof, same_weights.oof) and mixed.meta["fold_scores"] == same_weights.meta["fold_scores"]
    if plain is not None:
        assert plain.meta["blend_rows"] == "all" and "new_levels" not in plain.meta


def test_random_cv_uses_all_rows_and_no_stress(tmp_path):
    from mocks import make_regression_task

    spec = make_regression_task(tmp_path / "reg", n_train=400)
    best = fallback.run(spec, None, TWO)
    n = len(best.meta["context"].data.y)
    assert ensemble.recent_rows(best.meta, n) is None
    levels = ensemble.new_level_rows(best.meta, np.asarray(best.meta["folds"]))
    assert levels is not None and not levels[1].any()  # в test нет новых городов
    assert ensemble.recent_rows({}, n) is None and ensemble.new_level_rows({}, np.zeros(n, dtype=int)) is None


def test_pick_compares_period_tasks_on_recent_folds(tmp_path):
    import solution

    spec = period_task(tmp_path / "reg")
    best = fallback.run(spec, None, TWO)
    folds = ensemble.recent_folds(best.meta, len(best.oof))
    data = best.meta["context"].data
    order, fold_of = np.asarray(data.period_order), np.asarray(best.meta["folds"])
    assert folds is not None and {int(np.median(order[fold_of == f])) for f in folds} == {3, 4}

    scores = list(best.meta["fold_scores_eff"])
    view = solution.on_folds(best, folds)
    assert view.meta["fold_scores_eff"] == [scores[i] for i in folds] and best.meta["fold_scores_eff"] == scores
    assert solution.on_folds(best, None) is best

    # претендент хуже на ранних фолдах, но лучше на поздних — по схеме «по периодам» записывается
    better_late = fallback.Candidate("late", 0.0, 0.0, best.oof, best.test_pred,
                                     {**best.meta, "honest": None, "fold_scores": None,
                                      "fold_scores_eff": [s * (0.9 if i in folds else 1.5) for i, s in enumerate(scores)],
                                      "cv_score_post": None})
    better_late.meta["cv_score_post"] = float(np.mean(better_late.meta["fold_scores_eff"]))
    assert solution.pick(spec, best, [better_late], best, "test") is better_late


def test_period_blend_is_written_unless_clearly_worse_than_fallback(tmp_path):
    import solution

    spec = period_task(tmp_path / "reg")
    best = fallback.run(spec, None, TWO)
    folds = ensemble.recent_folds(best.meta, len(best.oof))
    scores = list(fallback.honest_view(best).meta["fold_scores_eff"])

    def blend_like(factors):
        eff = [s * factors.get(i, 1.0) for i, s in enumerate(scores)]
        return fallback.Candidate("ensemble", float(np.mean(eff)), 0.0, best.oof, best.test_pred,
                                  {**best.meta, "honest": None, "blend": {"a": 0.5, "b": 0.5}, "honest_oof": True,
                                   "fold_scores": eff, "fold_scores_eff": eff, "cv_score_post": float(np.mean(eff))})

    noisy = blend_like({folds[0]: 1.03, folds[1]: 0.99})   # чуть хуже в среднем
    assert solution.pick(spec, best, [noisy], best, "test") is noisy
    steady = blend_like({i: 1.06 for i in folds})            # стабильно хуже на 6% — в допуске (как на копии public)
    assert solution.pick(spec, best, [steady], best, "test") is steady
    worse = blend_like({i: 1.3 for i in folds})             # хуже на 30% — сломанный бленд
    assert solution.pick(spec, best, [worse], best, "test") is None
    assert solution.within_tolerance("roc_auc", 0.785, 0.80, 0.1) and not solution.within_tolerance("roc_auc", 0.77, 0.80, 0.1)
    single = blend_like({folds[0]: 1.03, folds[1]: 0.99})
    single.meta.pop("blend")                                 # не бленд — только строгое правило
    assert solution.pick(spec, best, [single], best, "test") is None


def test_new_level_rows_patch_any_written_candidate(tmp_path):
    import solution

    spec = period_task(tmp_path / "reg")
    best = fallback.run(spec, None, TWO)
    mixed = ensemble.blend(spec, [fallback.honest_view(best), *fallback.pool(best)])
    if mixed is None or "new_levels" not in mixed.meta:
        return
    patched = solution.with_new_levels(best, mixed)
    index = np.asarray(mixed.meta["new_levels"]["test_index"])
    changed = np.flatnonzero(~np.isclose(patched.test_pred, best.test_pred))
    assert set(changed) <= set(index.tolist()) and len(changed) > 0
    assert np.allclose(patched.test_pred[index], mixed.test_pred[index])
    assert patched.name.endswith("+new_levels") and "test_index" not in patched.meta["new_levels"]
    assert solution.with_new_levels(patched, mixed) is None and solution.with_new_levels(mixed, mixed) is None
    assert solution.with_new_levels(best, None) is None


def test_blend_within_tolerance_is_written_for_random_cv(tmp_path):
    import solution
    from mocks import make_regression_task

    spec = make_regression_task(tmp_path / "reg", n_train=400)
    best = fallback.run(spec, None, TWO)
    assert ensemble.recent_folds(best.meta, len(best.oof)) is None
    scores = list(fallback.honest_view(best).meta["fold_scores_eff"])

    def blend_like(factor):
        eff = [s * factor for s in scores]
        return fallback.Candidate("ensemble", float(np.mean(eff)), 0.0, best.oof, best.test_pred,
                                  {**best.meta, "honest": None, "blend": {"a": 0.5, "b": 0.5}, "honest_oof": True,
                                   "fold_scores": eff, "fold_scores_eff": eff, "cv_score_post": float(np.mean(eff))})

    close = blend_like(1.04)
    assert solution.pick(spec, best, [close], best, "test") is close      # в допуске 5% — бленд
    assert solution.pick(spec, best, [blend_like(1.08)], best, "test") is None
