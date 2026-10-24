import numpy as np
import pandas as pd
from p3_fixtures import Spec

import preprocess as P


def _frames(n_train=300, n_test=100, seed=0):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame({
        "num": rng.normal(size=n),
        "numstr": [f"{v:.1f}".replace(".", ",") for v in rng.uniform(0, 100, n)],
        "date": (pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")).strftime("%Y-%m-%d"),
        "city": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
        "grade": rng.integers(1, 4, n),
        "comment": [" ".join(rng.choice(["good", "bad", "fast", "slow", "price", "delivery", "item"], rng.integers(3, 9)))
                    + f" order {i}" for i in range(n)],
        "user_id": np.arange(n) + 7,
        "token": [f"tok-{i:08x}{seed}" for i in range(n)],
        "const": 3,
        "num_copy": None,
        "only_train": rng.normal(size=n),
        "flag": rng.random(n) < 0.5,
    })
    df["num_copy"] = df["num"]
    df.loc[n_train:, "only_train"] = np.nan
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def test_kinds_drops_and_shapes():
    train, test = _frames()
    prep = P.prepare(train, test)
    report = prep.report
    assert report["kinds"]["num"] == "num" and report["kinds"]["numstr"] == "numstr"
    assert report["kinds"]["date"] == "date" and report["kinds"]["comment"] == "text"
    assert report["kinds"]["city"] == "cat" and report["kinds"]["flag"] == "num"
    assert report["dropped"]["const"] == "constant"
    assert report["dropped"]["user_id"] == "id_like"
    assert report["dropped"]["token"] == "unique_strings"
    assert report["dropped"]["only_train"] == "empty_in_test"
    assert report["dropped"]["num_copy"] == "duplicate_of:num"
    assert len(prep.X) == len(train) and len(prep.X_test) == len(test)
    assert list(prep.X.columns) == list(prep.X_test.columns)
    assert {"date__year", "date__month", "date__dow", "date__ts"} <= set(prep.X.columns)
    assert {"comment__len", "comment__words"} <= set(prep.X.columns)
    assert prep.X["numstr"].dtype.kind == "f" and prep.X["numstr"].notna().all()
    assert "date" not in prep.X.columns and "comment" not in prep.X.columns


def test_categories_rare_and_consistent():
    train, test = _frames(n_train=2000, n_test=500)
    train.loc[[0, 1], "city"] = "rare1"  # 2 вхождения < min_count=5
    test.loc[0, "city"] = "never_seen"
    prep = P.prepare(train, test)
    assert prep.cat_cols == ["city"]
    cats = prep.X["city"].cat.categories
    assert P.RARE in cats and "rare1" not in cats
    assert prep.X_test["city"].dtype == prep.X["city"].dtype  # одинаковые категории в train и test
    assert prep.X_test.loc[0, "city"] == P.RARE  # уровень, которого нет в train, — как редкий


def test_spec_hints_override_detection():
    train, test = _frames()
    spec = Spec(task_dir=None, id_col="id", target="y", task_type="regression", metric="rmse",
                cat_cols=["grade"], drop_cols=["numstr"], text_cols=[])
    prep = P.prepare(train, test, spec)
    assert prep.report["kinds"]["grade"] == "cat" and "grade" in prep.cat_cols
    assert prep.report["dropped"]["numstr"] == "spec.drop_cols"


def test_unsafe_column_names_are_renamed():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a[1]": rng.normal(size=50), 'b:"x"': rng.normal(size=50), "площадь, м2": rng.normal(size=50)})
    prep = P.prepare(frame.iloc[:40], frame.iloc[40:])
    assert set(prep.X.columns) == {"a_1", "b__x", "площадь__м2"}
    assert prep.report["renamed"]["a_1"] == "a[1]"


def test_model_frames():
    train, test = _frames(n_train=2000, n_test=500)
    prep = P.prepare(train, test, config=P.PrepConfig(max_cat_levels=2))
    assert prep.high_card == ["city"]
    lgb_frame, lgb_cats = P.model_frame(prep, prep.X, "lightgbm")
    assert lgb_frame["city"].dtype.kind == "f" and lgb_cats == []  # высокая кардинальность → коды по частоте
    cb_frame, cb_cats = P.model_frame(prep, prep.X, "catboost")
    assert cb_cats == ["city"] and cb_frame["city"].map(type).eq(str).all()
    assert prep.X["city"].dtype.name == "category"  # исходный кадр не меняется


def test_dates_with_time_and_missing_values():
    stamps = pd.Series(["2021-03-01 10:30:00", "2021-03-02 23:15:00", None, "2021-03-04 00:00:00"] * 10)
    frame = pd.DataFrame({"when": stamps, "x": np.arange(40)})
    prep = P.prepare(frame.iloc[:30], frame.iloc[30:])
    assert "when__hour" in prep.X.columns
    assert prep.X["when__year"].isna().sum() == stamps.iloc[:30].isna().sum()
