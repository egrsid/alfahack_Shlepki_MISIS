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
