import pytest

import agent
from mocks import GOOD_CODE, NO_FUNCTION_CODE

PLAN = '{"hypotheses": [{"title": "log area", "kind": "features"}]}'


@pytest.mark.parametrize("text", [
    PLAN,
    f"<think>maybe {{\"a\": 1}} ... let me think</think>\n{PLAN}",
    f"Sure! Here is the plan:\n```json\n{PLAN}\n```\nHope it helps.",
    f"```\n{PLAN}\n```",
    'Plan below.\n{"hypotheses": [{"title": "log area", "kind": "features",},],}',               # висячие запятые
    '{\n  // main idea\n  "hypotheses": [{"title": "log area", /* x */ "kind": "features"}]  # end\n}',
    "{'hypotheses': [{'title': 'log area', 'kind': 'features'}]}",                                # питоновский литерал
    "Step [1]: thinking... answer: " + PLAN,                                                       # мусорный список раньше
    "<think>unclosed reasoning " + PLAN,
])
def test_extract_json_dirty_answers(text):
    assert agent.extract_json(text) == {"hypotheses": [{"title": "log area", "kind": "features"}]}


def test_extract_json_list_and_literals():
    assert agent.extract_json('[{"a": true, "b": null}]') == [{"a": True, "b": None}]
    assert agent.extract_json("{'a': True, 'b': None}") == {"a": True, "b": None}


@pytest.mark.parametrize("text", ["", None, "no json here", "<think>{\"a\": 1}</think> nothing", "[1, 2, 3]",
                                  '{"a": 1'])
def test_extract_json_garbage_raises(text):
    with pytest.raises(ValueError):
        agent.extract_json(text)


CODE = "import pandas as pd\n\ndef build_features(train, test, spec):\n    return train, test\n"


@pytest.mark.parametrize("text", [
    f"```python\n{CODE}```",
    f"<think>I could write ```python\nx = 1\n``` but no</think>\nHere:\n```python\n{CODE}```\nExplanation.",
    f"```py\n{CODE}```",
    f"```\n{CODE}```",
    f"Some text\n{CODE}",                                     # без ограждений
    f"```python\n{CODE}",                                     # незакрытый блок (обрезан max_tokens)
    f"```python\nx = 1\n```\nfinal:\n```python\n{CODE}```",  # берём блок с build_features
    f"```json\n{{\"a\": 1}}\n```\n```python\n{CODE}```",
])
def test_extract_code_variants(text):
    assert agent.extract_code(text) == CODE


@pytest.mark.parametrize("text", [None, "", "I cannot do that.", "```json\n{\"a\": 1}\n```",
                                  "<think>```python\ndef build_features(a, b, c):\n    pass\n```</think>"])
def test_extract_code_none(text):
    assert agent.extract_code(text) is None


def test_validate_good_code():
    assert agent.validate_code(GOOD_CODE) == []
    assert agent.validate_code('"""doc"""\nfrom __future__ import annotations\nimport numpy as np\n'
                               "from sklearn.decomposition import PCA\n\n"
                               "def build_features(train, test, spec):\n    return train, test\n\nPARAMS = {}\n") == []


@pytest.mark.parametrize("code, fragment", [
    ("def build_features(train, test, spec)\n    return train, test\n", "SyntaxError"),
    (NO_FUNCTION_CODE, "no top-level function"),
    ("def build_features(train, test):\n    return train, test\n", "three arguments"),
    ("import os\n" + CODE, "import of `os`"),
    ("import subprocess\n" + CODE, "import of `subprocess`"),
    ("from socket import socket\n" + CODE, "import of `socket`"),
    ("import requests\n" + CODE, "import of `requests`"),
    ("import openai\n" + CODE, "import of `openai`"),
    ("import shutil\n" + CODE, "import of `shutil`"),
    ("from . import x\n" + CODE, "relative imports"),
    (CODE + "\ndef f():\n    return open('train.csv').read()\n", "`open`"),
    (CODE + "\ndef f():\n    return eval('1')\n", "`eval`"),
    (CODE + "\ndef f():\n    exec('x = 1')\n", "`exec`"),
    (CODE + "\ndef f():\n    return __import__('os')\n", "`__import__`"),
    (CODE + "\ndef f():\n    return getattr(pd, 'read_csv')\n", "`getattr`"),
    (CODE + "\ndef f():\n    return pd.read_csv('../../train.csv')\n", "`.read_csv`"),
    (CODE + "\ndef f(df):\n    df.to_csv('submission.csv')\n", "`.to_csv`"),
    (CODE + "\ndef f(df):\n    return df.__class__.__subclasses__()\n", "dunder attribute"),
    (CODE + "\nfor i in range(3):\n    pass\n", "top-level `For`"),
    (CODE + "\nprint('hi')\n", "top-level `Expr`"),
])
def test_validate_bad_code(code, fragment):
    problems = agent.validate_code(code)
    assert problems and any(fragment in p for p in problems), problems


def test_extract_constants():
    consts, problems = agent.extract_constants(
        'MODEL = "catboost"\nPARAMS: dict = {"depth": 6}\nTARGET_TRANSFORM = None\nOTHER = 1\n')
    assert consts == {"MODEL": "catboost", "PARAMS": {"depth": 6}, "TARGET_TRANSFORM": None} and problems == []
    consts, problems = agent.extract_constants("PARAMS = dict(depth=6)\n")
    assert consts == {} and "literal" in problems[0]


def test_normalize_model_config_whitelist_and_bounds():
    model, params, transform, notes = agent.normalize_model_config(
        {"MODEL": "LightGBM", "PARAMS": {"num_leaves": 63.0, "learning_rate": 5, "objective": "huber",
                                         "class_weight": "balanced", "subsample": "0.8"},
         "TARGET_TRANSFORM": "log1p"}, "regression")
    assert model == "lightgbm" and transform == "log1p"
    assert params == {"num_leaves": 63, "learning_rate": 0.5, "class_weight": "balanced"}
    assert any("objective" in n for n in notes) and any("clipped" in n for n in notes)
    assert any("subsample" in n for n in notes)


def test_normalize_model_config_rejects_unknown_model_and_log_for_classification():
    model, params, transform, notes = agent.normalize_model_config(
        {"MODEL": "random_forest", "PARAMS": [1], "TARGET_TRANSFORM": "log1p"}, "multiclass")
    assert (model, params, transform) == ("lightgbm", {}, None) and len(notes) == 3
    assert agent.normalize_model_config({}, "binary")[:3] == ("lightgbm", {}, None)
