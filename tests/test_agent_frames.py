"""Обмен таблицами харнесс ↔ песочница без pyarrow/fastparquet (на платформе их не оказалось) и без pickle."""

import sys

import numpy as np
import pandas as pd
import pytest

import agent
from mocks import GOOD_CODE, MockLLM, StubEvaluator, fenced, make_best, make_regression_task, plan
from test_agent_loop import FAST, experiments


def test_frame_roundtrip_keeps_values_and_types(tmp_path):
    frame = pd.DataFrame({
        "i": [1, 2, 3], "f": [0.5, np.nan, 2.0], "b": [True, False, True],
        "nullable_int": pd.array([1, None, 3], dtype="Int64"), "nullable_bool": pd.array([True, None, False]),
        "s": ["a", None, "в"], "cat": pd.Categorical(["x", "y", "x"]), "mixed": [1, "two", None],
        "d": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
        "tz": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 00:00", "2024-01-03 00:00"]).tz_localize("Europe/Moscow"),
        7: [1.0, 2.0, 3.0],
    })
    agent.save_frame(frame, str(tmp_path / "t.frame"))
    back = agent.load_frame(str(tmp_path / "t.frame"))
    assert list(back.columns) == [str(c) for c in frame.columns] and len(back) == 3
    assert back["i"].dtype == np.int64 and back["b"].dtype == bool and np.isnan(back["f"][1])
    assert np.isnan(back["nullable_int"][1]) and back["nullable_int"][2] == 3.0
    assert back["s"].tolist() == ["a", None, "в"] and back["cat"].tolist() == ["x", "y", "x"]
    assert back["mixed"].tolist() == ["1", "two", None]
    assert pd.isna(back["d"][1]) and back["d"][2] == pd.Timestamp("2024-03-01")
    assert back["tz"][0] == pd.Timestamp("2024-01-01 07:00")  # UTC без зоны
    assert not list((tmp_path / "t.frame").glob("*.pkl"))


def test_frame_with_rows_but_no_columns(tmp_path):
    agent.save_frame(pd.DataFrame(index=range(4)), str(tmp_path / "e.frame"))
    assert agent.load_frame(str(tmp_path / "e.frame")).shape == (4, 0)


def test_no_parquet_anywhere_in_the_exchange():
    assert "parquet" not in agent.RUNNER and "parquet" not in agent.FRAME_IO
    assert "read_pickle" not in agent.FRAME_IO and "allow_pickle=False" in agent.FRAME_IO


def test_agent_works_without_pyarrow(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")

    for name in ("pyarrow", "fastparquet"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", missing)
    monkeypatch.setattr(pd, "read_parquet", missing)
    spec = make_regression_task(tmp_path / "reg")
    evaluator = StubEvaluator(n_splits=3, n_estimators=60)
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    result = agent.research(spec, make_best(spec, evaluator), llm, None, config=agent.AgentConfig(**FAST),
                            evaluate=evaluator)
    assert len(result) == 1 and experiments(spec)[0]["status"] == "accepted"
