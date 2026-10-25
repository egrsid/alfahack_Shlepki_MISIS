import json

import numpy as np
import pandas as pd
import pytest

import task_understanding as tu
from mocks import MockLLM


def make_task(root, metric_line="Метрика: RMSE.", with_dictionary=True):
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame({"id": np.arange(n), "x": rng.normal(size=n),
                          "city": rng.choice(["msk", "spb"], n),
                          "created": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
                          "review": [f"очень хороший дом номер {i} рядом парк" for i in range(n)],
                          "leak_flag": rng.integers(0, 2, n)})
    frame["y"] = frame["x"] * 3 + rng.normal(size=n)
    root.mkdir(parents=True)
    frame.iloc[:200].to_csv(root / "train.csv", index=False)
    frame.iloc[200:].drop(columns="y").to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": frame["id"].iloc[200:], "y": 0.0}).to_csv(root / "sample_submission.csv", index=False)
    if with_dictionary:
        pd.DataFrame({"feature": ["x", "city"], "unit": ["м", None],
                      "description_ru": ["площадь", "город"]}).to_csv(root / "data_dictionary.csv", index=False)
    (root / "task.md").write_text(f"# Цена\nПредскажите `y`. {metric_line}\nКолонка leak_flag — утечка.\n",
                                  encoding="utf-8")
    return root / "task.md"


def answer(**fields):
    base = {"metric": "rmse", "task_type": "regression", "date_cols": [], "text_cols": [], "cat_cols": [],
            "drop_cols": []}
    return MockLLM({"task_analyst": [json.dumps({**base, **fields})]})


def test_without_llm_gives_heuristics_and_dictionary_notes(tmp_path):
    spec = tu.understand_task(make_task(tmp_path / "t"), None)
    assert spec.target == "y" and spec.metric == "rmse" and spec.source == "heuristic"
    assert spec.feature_notes["x"] == "площадь [м]" and spec.feature_notes["city"] == "город"


def test_valid_answer_is_applied(tmp_path):
    llm = answer(date_cols=["created"], text_cols=["review"], cat_cols=["city"], drop_cols=["leak_flag"])
    spec = tu.understand_task(make_task(tmp_path / "t"), llm)
    assert spec.date_cols == ["created"] and spec.text_cols == ["review"] and spec.cat_cols == ["city"]
    assert spec.drop_cols == ["leak_flag"] and spec.source == "llm"
    assert llm.calls[0][2]["max_tokens"] == tu.MAX_TOKENS


def test_bad_fields_are_rejected(tmp_path):
    llm = answer(metric="macro_f1", date_cols=["x", "nope"], text_cols=["city"], cat_cols=["x"],
                 drop_cols=["x", "city"])  # x/city не упомянуты в task.md как утечки
    spec = tu.understand_task(make_task(tmp_path / "t", metric_line=""), llm)
    assert spec.metric == "rmse"  # метрика классификации к регрессии не подходит
    assert spec.date_cols == [] and spec.text_cols == [] and spec.cat_cols == [] and spec.drop_cols == []


def test_metric_from_llm_only_when_text_has_none(tmp_path):
    spec = tu.understand_task(make_task(tmp_path / "a", metric_line="Качество оценивается по MAE."), answer(metric="rmse"))
    assert spec.metric == "mae"  # из текста — главнее
    spec = tu.understand_task(make_task(tmp_path / "b", metric_line=""), answer(metric="Mean Absolute Error"))
    assert spec.metric == "mae" and spec.greater_is_better is False


@pytest.mark.parametrize("reply", ["not json at all", RuntimeError("Budget has been exceeded"), '["list"]'])
def test_llm_failure_falls_back_to_heuristics(tmp_path, reply):
    llm = MockLLM({"task_analyst": [reply]})
    spec = tu.understand_task(make_task(tmp_path / "t", with_dictionary=False), llm)
    assert spec.target == "y" and spec.metric == "rmse" and spec.source == "heuristic" and spec.date_cols == []


class Gateway400(Exception):
    status_code = 400


def test_retry_without_reasoning_hint_on_400(tmp_path):
    llm = MockLLM({"task_analyst": [Gateway400("reasoning not supported"),
                                    json.dumps({"date_cols": ["created"]})]})
    spec = tu.understand_task(make_task(tmp_path / "t"), llm)
    assert spec.date_cols == ["created"] and "extra_body" not in llm.calls[1][2]
