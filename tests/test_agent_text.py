import pytest

import agent
from mocks import (EXCEPTION_CODE, MockLLM, SimpleDeadline, StubEvaluator, identity_features, make_regression_task,
                   make_text_task)
from test_agent_loop import FAST


@pytest.fixture(scope="module")
def text_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("text")
    spec = make_text_task(root)
    data = agent.load_task_data(spec)
    workdir = root / ".agent_work"
    return spec, data, workdir, agent.write_inputs(data, workdir)


def test_free_text_column_is_detected_and_removed_from_categories(text_env, tmp_path):
    _, data, _, _ = text_env
    assert data.spec_dict["text_cols"] == ["review"] and data.spec_dict["cat_cols"] == ["city"]
    assert agent.load_task_data(make_regression_task(tmp_path / "reg")).spec_dict["text_cols"] == []


def test_text_base_code_adds_tfidf_svd_features(text_env):
    spec, data, workdir, inputs = text_env
    assert agent.base_code_for(data.spec_dict) is agent.BASE_CODE_TEXT
    assert agent.base_code_for({"text_cols": []}) is agent.BASE_CODE
    assert agent.validate_code(agent.BASE_CODE_TEXT) == []
    out = agent.run_features(agent.BASE_CODE_TEXT, workdir / "text", inputs, data, agent.AgentConfig(),
                             agent._local_run_code, 120.0)
    assert out.error == "", out.error
    svd = [c for c in out.x_train.columns if c.startswith("review_svd_")]
    assert len(svd) == 20 and "review" not in out.x_train.columns and "review_len" in out.x_train.columns
    assert list(out.x_train.columns) == list(out.x_test.columns) and len(out.x_test) == len(data.test_x)

    evaluator = StubEvaluator(n_splits=3, n_estimators=60)
    with_text = evaluator(spec, agent.make_build_features(out.x_train, out.x_test), {}, None)
    raw = evaluator(spec, identity_features, {}, None)  # отзыв как категория: почти все значения уникальны
    assert with_text.cv_score > raw.cv_score + 0.1


def test_research_starts_from_text_base_and_falls_back_without_tfidf(tmp_path, monkeypatch):
    spec = make_text_task(tmp_path / "t1")
    agent.research(spec, None, MockLLM(), SimpleDeadline(600), config=agent.AgentConfig(**FAST),
                   evaluate=StubEvaluator(n_splits=3, n_estimators=60))
    assert "TfidfVectorizer" in (spec.task_dir / ".agent_work" / "exp_00_base" / "features.py").read_text()

    monkeypatch.setattr(agent, "BASE_CODE_TEXT", EXCEPTION_CODE)  # TF-IDF-модуль падает
    spec = make_text_task(tmp_path / "t2")
    agent.research(spec, None, MockLLM(), SimpleDeadline(600), config=agent.AgentConfig(**FAST),
                   evaluate=StubEvaluator(n_splits=3, n_estimators=60))
    assert (spec.task_dir / ".agent_work" / "exp_00_base" / "features.py").read_text() == agent.BASE_CODE
