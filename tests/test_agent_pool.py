"""Пул агента для финального бленда и объединение «почти принятых» гипотез (режим рецепта)."""

import types
from collections import deque

import numpy as np

import agent
import ensemble
import solution
from memory import Experiment, ExperimentMemory
from mocks import make_regression_task


def cand(score, name="c", **meta):
    return agent.Candidate(name, float(score), 0.0, np.zeros(3), np.zeros(2),
                           {"fold_scores": [float(score)] * 5, **meta})


def research_stub(**cfg):
    loop = object.__new__(agent._Research)
    loop.cfg, loop.recipe, loop.greater, loop.hidden = agent.AgentConfig(**cfg), True, False, None
    loop.pool, loop.near_misses, loop.queue = [], [], deque()
    loop.memory, loop.accepted, loop.best = ExperimentMemory(), [], None
    return loop


def test_pool_keeps_best_candidates_only_for_the_blend():
    loop = research_stub(pool_size=2)
    for n, score in enumerate([10.3, 10.1, 10.2], 1):  # RMSE: лучшие — 10.1 и 10.2
        loop._add_to_pool(cand(score, recipe_raw={"oof": np.zeros(3)}), n)
    assert [c.name for c in loop.pool] == ["agent_pool_02", "agent_pool_03"]
    assert all(c.meta["pool_only"] and "recipe_raw" not in c.meta for c in loop.pool)
    loop.accepted = [cand(9.0, "agent_04_features")]
    assert [c.name for c in loop.result()] == ["agent_04_features", "agent_pool_02", "agent_pool_03"]


def test_near_misses_are_combined_into_one_hypothesis(caplog):
    loop = research_stub()
    for i, title in ((1, "Ratio a/b"), (2, "Group mean of c")):
        loop.memory.add(Experiment(id=i, hypothesis=title, kind="features", status="rejected", gain=0.1 * i,
                                   diff=f"+    df['{title[:5]}'] = 1"))
    loop.near_misses = [1]
    loop._queue_combination()
    assert not loop.queue  # одного мало
    loop.near_misses.append(2)
    with caplog.at_level("INFO"):
        loop._queue_combination()
    hypothesis = loop.queue[0]
    assert hypothesis.title == f"{agent.COMBINE_PREFIX} #1 Ratio a/b + #2 Group mean of c"
    assert "df['Ratio']" in hypothesis.rationale and "df['Group']" in hypothesis.rationale
    assert loop.near_misses == [] and "queued combination of near misses #1, #2" in caplog.text


def test_pool_members_go_only_into_the_blend(tmp_path, monkeypatch):
    spec = make_regression_task(tmp_path / "reg")
    best = types.SimpleNamespace(name="fallback", meta={})
    strong = cand(1.0, "agent_pool_01", pool_only=True)
    seen = {}

    def fake_blend(spec_, cands, **kw):
        seen["names"] = [c.name for c in cands]
        return None
    monkeypatch.setattr(ensemble, "blend", fake_blend)
    current = cand(10.0, "fallback")
    assert solution.improve(spec, best, current, [], "агент", members=[strong]) is current  # отдельно не пишется
    assert "agent_pool_01" in seen["names"]
