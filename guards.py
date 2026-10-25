"""Проверки кандидата перед записью (роль P5; написано P4 — владелец может переписать).

    check_candidate(spec, cand, baseline) -> list[str]   # пусто — проблем не найдено

Вызывают агент (на «чудесном» скачке CV — agent.py, Research._judge) и solution.py (перед записью кандидата
агента или бленда). baseline — кандидат, которому верим (fallback или текущий лучший агента): у него те же строки
test, поэтому сравниваем с ним, а не с ответами (их нет). Проверки:
- форма и конечность oof/test_pred;
- регрессия: test-прогноз не уходит от baseline (корреляция, сдвиг среднего, выбросы за его диапазон);
- классификация: доли предсказанных классов на test близки к baseline; уверенность на OOF не выше, чем на test,
  заметно сильнее, чем у baseline (признак утечки таргета в признаки: на train модель «знает» ответ, на test — нет).
"""

from __future__ import annotations

from typing import Any

import numpy as np

MIN_CORR = 0.5           # регрессия: корреляция test-прогнозов с baseline
MAX_SHIFT_STD = 0.5      # регрессия: |сдвиг среднего| ≤ этого × std baseline
MAX_OUTSIDE = 0.01       # регрессия: доля прогнозов за диапазоном baseline (расширенным на половину размаха)
MAX_CLASS_TV = 0.2       # классификация: total variation долей классов на test
MAX_CONFIDENCE_GAP = 0.1  # классификация: (уверенность OOF − test) кандидата минус то же у baseline


def _confidence_gap(cand: Any) -> float:
    oof, test = np.asarray(cand.oof, dtype=float), np.asarray(cand.test_pred, dtype=float)
    return float(oof.max(axis=1).mean() - test.max(axis=1).mean())


def check_candidate(spec: Any, cand: Any, baseline: Any = None) -> list[str]:
    problems: list[str] = []
    test = np.asarray(cand.test_pred, dtype=float)
    has_oof = getattr(cand, "oof", None) is not None  # evaluator может не отдать OOF — это не порок прогноза
    if not np.isfinite(test).all() or (has_oof and not np.isfinite(np.asarray(cand.oof, dtype=float)).all()):
        problems.append("oof or test_pred has NaN/inf")
    if baseline is None:
        return problems
    base = np.asarray(baseline.test_pred, dtype=float)
    if test.shape != base.shape:
        return problems + [f"test_pred shape {test.shape} differs from baseline {base.shape}"]
    if problems:
        return problems

    if getattr(spec, "task_type", "") == "regression" or test.ndim == 1:
        std = float(base.std())
        if std > 0 and float(test.std()) > 0:
            corr = float(np.corrcoef(test, base)[0, 1])
            if corr < MIN_CORR:
                problems.append(f"test predictions correlate {corr:.2f} with baseline (< {MIN_CORR})")
            shift = abs(float(test.mean() - base.mean()))
            if shift > MAX_SHIFT_STD * std:
                problems.append(f"test mean shifted by {shift:.4g} (> {MAX_SHIFT_STD}·std of baseline {std:.4g})")
        span = float(base.max() - base.min())
        outside = float(np.mean((test < base.min() - 0.5 * span) | (test > base.max() + 0.5 * span)))
        if outside > MAX_OUTSIDE:
            problems.append(f"{outside:.1%} of test predictions are far outside the baseline range")
        return problems

    k = test.shape[1]
    ours = np.bincount(test.argmax(axis=1), minlength=k) / len(test)
    theirs = np.bincount(base.argmax(axis=1), minlength=k) / len(base)
    tv = 0.5 * float(np.abs(ours - theirs).sum())
    if tv > MAX_CLASS_TV:
        problems.append(f"predicted class shares on test differ from baseline by TV={tv:.2f} (> {MAX_CLASS_TV})")
    if has_oof and getattr(baseline, "oof", None) is not None and np.ndim(baseline.oof) == 2:
        gap = _confidence_gap(cand) - _confidence_gap(baseline)
        if gap > MAX_CONFIDENCE_GAP:
            problems.append(f"OOF is {gap:.2f} more confident than test vs baseline — looks like target leakage")
    return problems
