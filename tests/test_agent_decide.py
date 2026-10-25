import math

import numpy as np
import pytest

import agent

CFG = agent.AgentConfig()


def cand(score, std=0.0, folds=None):
    meta = {"fold_scores": folds} if folds is not None else {}
    return agent.Candidate("c", score, std, np.zeros(1), np.zeros(1), meta)


@pytest.mark.parametrize("greater, new, accept", [
    (False, 9.0, True),    # RMSE ниже — лучше
    (False, 11.0, False),
    (True, 0.80, True),    # F1 выше — лучше
    (True, 0.70, False),
])
def test_direction(greater, new, accept):
    current = cand(10.0 if not greater else 0.75, std=0.01)
    decision = agent.decide(cand(new), current, greater, CFG)
    assert decision.accept is accept and (decision.gain > 0) is accept


@pytest.mark.parametrize("greater", [True, False])
def test_equal_score_is_rejected(greater):
    decision = agent.decide(cand(0.5), cand(0.5, std=0.0), greater, CFG)
    assert not decision.accept and decision.gain == 0


def test_gain_within_noise_is_rejected():
    current = cand(10.0, std=1.0)  # SE = 1/sqrt(5) ≈ 0.447
    assert not agent.decide(cand(9.7), current, False, CFG).accept
    assert agent.decide(cand(9.5), current, False, CFG).accept


def test_min_relative_gain_without_noise():
    current = cand(100.0, std=0.0)  # порог = 5e-4 · 100 = 0.05
    assert not agent.decide(cand(99.97), current, False, CFG).accept
    assert agent.decide(cand(99.9), current, False, CFG).accept


def test_paired_folds_beat_noisy_std():
    # фолды сильно различаются между собой, но новый кандидат стабильно лучше на каждом
    base = [8.0, 12.0, 9.0, 11.0, 10.0]
    current = cand(np.mean(base), std=np.std(base), folds=base)
    better = [x - 0.2 for x in base]
    paired = agent.decide(cand(np.mean(better), folds=better), current, False, CFG)
    unpaired = agent.decide(cand(np.mean(better)), cand(np.mean(base), std=np.std(base)), False, CFG)
    assert paired.accept and "paired" in paired.reason
    assert not unpaired.accept


def test_paired_folds_noisy_difference_is_rejected():
    base = [10.0, 10.0, 10.0, 10.0, 10.0]
    mixed = [9.0, 11.0, 9.2, 10.8, 9.9]  # в среднем лучше на 0.02, но разброс разниц большой
    decision = agent.decide(cand(np.mean(mixed), folds=mixed), cand(10.0, folds=base), False, CFG)
    assert not decision.accept


def test_non_finite_scores():
    assert not agent.decide(cand(math.nan), cand(1.0), True, CFG).accept
    assert agent.decide(cand(1.0), cand(math.nan), True, CFG).accept
    assert agent.decide(cand(1.0, folds=[1.0, math.nan]), cand(0.5, std=math.nan), True, CFG).accept


def test_suspicious_jump_is_flagged():
    current = cand(10.0, std=0.1)
    assert agent.decide(cand(5.0), current, False, CFG).suspicious
    assert not agent.decide(cand(9.8), current, False, CFG).suspicious
