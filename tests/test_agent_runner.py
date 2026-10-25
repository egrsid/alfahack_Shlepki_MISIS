import time

import numpy as np
import pandas as pd
import pytest

import agent
from mocks import (DROP_ROWS_CODE, EXCEPTION_CODE, GOOD_CODE, INF_CODE, SLOW_CODE, feature_code,
                   make_regression_task)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("task")
    spec = make_regression_task(root)
    data = agent.load_task_data(spec)
    workdir = root / ".agent_work"
    return spec, data, workdir, agent.write_inputs(data, workdir)


def run(env, code, name, timeout=60.0, **cfg):
    _, data, workdir, inputs = env
    return agent.run_features(code, workdir / name, inputs, data, agent.AgentConfig(**cfg),
                              agent._local_run_code, timeout)


def test_load_task_data_hides_target_and_id(env):
    _, data, _, inputs = env
    assert list(data.train_x.columns) == ["x1", "x2", "x3", "city"] == list(data.test_x.columns)
    assert len(data.y) == 400 and data.spec_dict["target"] == "target" and data.spec_dict["cat_cols"] == ["city"]
    assert 0 < data.nan_share < 0.05
    assert list(agent.load_frame(str(inputs[0])).columns) == ["x1", "x2", "x3", "city"]


def test_load_task_data_finds_files_in_subdir(tmp_path):
    spec = make_regression_task(tmp_path / "regression" / "participants")
    spec.task_dir = tmp_path
    assert len(agent.load_task_data(spec).train_x) == 400


def test_mini_profile(env):
    _, data, _, _ = env
    profile = agent._mini_profile(data)
    cols = {c["name"]: c for c in profile["columns"]}
    assert profile["n_train"] == 400 and profile["n_test"] == 150
    assert cols["city"]["dtype"] == "cat" and sorted(cols["city"]["top"]) == ["a", "b", "c"]
    assert cols["x3"]["dtype"] == "num" and cols["x3"]["missing_pct"] > 0
    assert set(profile["target"]) >= {"name", "mean", "std", "skew"}


def test_good_code_builds_features(env):
    out = run(env, GOOD_CODE, "good")
    assert out.error == "", out.error
    assert list(out.x_train.columns) == ["x1", "x2", "x3", "city", "x1_x2"] == list(out.x_test.columns)
    assert isinstance(out.x_train["city"].dtype, pd.CategoricalDtype)
    assert out.x_train["city"].dtype == out.x_test["city"].dtype
    assert np.allclose(out.x_train["x1_x2"], out.x_train["x1"] * out.x_train["x2"])
    assert not (env[2] / "good" / "feat_train.frame").exists()  # артефакты подчищены


def test_base_code_is_identity_plus_prep(env):
    assert agent.validate_code(agent.BASE_CODE) == []
    out = run(env, agent.BASE_CODE, "base")
    assert out.error == "" and list(out.x_train.columns) == ["x1", "x2", "x3", "city"]


def test_exception_in_code_returns_traceback(env):
    out = run(env, EXCEPTION_CODE, "exc")
    assert out.x_train is None and "KeyError" in out.error and "no_such_column" in out.error


def test_infinite_loop_is_killed_by_timeout(env):
    start = time.monotonic()
    out = run(env, SLOW_CODE, "slow", timeout=3)
    assert out.timed_out and "timeout" in out.error and time.monotonic() - start < 20


@pytest.mark.parametrize("code, fragment", [
    (DROP_ROWS_CODE, "rows must not be added or dropped"),
    (INF_CODE, "inf values"),
    (feature_code('df["row"] = np.arange(len(df))'), "row index"),
    (feature_code("pass").replace("return X_train, X_test", "return X_train.sample(frac=1, random_state=0), X_test"),
     "reordered"),
    (feature_code("pass").replace("return X_train, X_test", 'return X_train.assign(extra=1.0), X_test'),
     "columns differ"),
    (feature_code("pass").replace("return X_train, X_test", "return [X_train, X_test]"), "tuple"),
    (feature_code("pass").replace("return X_train, X_test", "return X_train.values, X_test"), "DataFrame"),
    (feature_code("for i in range(30):\n    df[f'nan_{i}'] = np.nan"), "NaN avalanche"),
])
def test_bad_outputs_fail_with_reason(env, code, fragment):
    out = run(env, code, "bad")
    assert out.x_train is None and fragment in out.error, out.error


def test_sandbox_env_has_no_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    result = agent._local_run_code("import os\nprint(sorted(os.environ))\n", tmp_path, 30)
    assert result.returncode == 0 and "OPENAI" not in result.stdout and "sk-secret" not in result.stdout


def _data(train_x, test_x, **spec):
    spec_dict = {"target": "y", "id_col": "id", **spec}
    return agent.TaskData(train_x, test_x, None, spec_dict, nan_share=0.0)


def test_validate_features_types():
    n = 120
    train = pd.DataFrame({"s": ["a", "b"] * (n // 2), "flag": [True, False] * (n // 2),
                          "when": pd.date_range("2020-01-01", periods=n, freq="7D"),
                          "const": 1.0, "v": np.linspace(0, 1, n) ** 2})
    test = pd.DataFrame({"s": ["c", "a"] * 10, "flag": [True] * 20,
                         "when": pd.date_range("2021-01-01", periods=20, freq="D"), "const": 1.0,
                         "v": np.ones(20)})
    x_train, x_test, problems, notes = agent.validate_features(train, test, _data(train, test), agent.AgentConfig())
    assert problems == []
    assert list(x_train.columns) == ["s", "flag", "when", "v"]  # константа выброшена
    assert list(x_train["s"].cat.categories) == ["a", "b", "c"] and x_test["s"].dtype == x_train["s"].dtype
    assert x_train["flag"].dtype == "float64" and x_test["when"].dtype == "float64"
    assert any("constant" in n for n in notes)


def test_validate_features_rejects_target_too_many_columns_and_mismatch():
    train = pd.DataFrame({"y": [1.0, 2.0], "a": [1.0, 2.0]})
    problems = agent.validate_features(train, train, _data(train, train), agent.AgentConfig())[2]
    assert any("target/id" in p for p in problems)
    wide = pd.DataFrame(np.ones((2, 5)), columns=list("abcde"))
    problems = agent.validate_features(wide, wide, _data(wide, wide), agent.AgentConfig(max_feature_cols=3))[2]
    assert any("feature columns" in p for p in problems)
    a, b = pd.DataFrame({"m": [1.0, 2.0]}), pd.DataFrame({"m": ["x", "y"]})
    problems = agent.validate_features(a, b, _data(a, b), agent.AgentConfig())[2]
    assert any("mismatched" in p for p in problems)


def test_build_features_adapter_matches_subset_rows():
    x_train, x_test = pd.DataFrame({"f": np.arange(10.0)}), pd.DataFrame({"f": np.arange(3.0)})
    build = agent.make_build_features(x_train, x_test)
    full_train, full_test = build(pd.DataFrame(index=range(10)), pd.DataFrame(index=range(3)), {})
    assert full_train.equals(x_train) and full_test.equals(x_test)
    part, _ = build(pd.DataFrame(index=[1, 5, 7]), None, {})
    assert part["f"].tolist() == [1.0, 5.0, 7.0]
    with pytest.raises(ValueError):
        build(pd.DataFrame(index=[100, 200]), None, {})
