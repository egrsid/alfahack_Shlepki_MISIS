"""Заглушки модулей команды для тестов агента (вне ZIP).

MockLLM       — вместо llm_client (P2): ответы по сценарию для каждой роли промпта.
StubEvaluator — вместо evaluator.evaluate (P1): LocalEvaluator агента (K-fold, фиксированный сид).
StubTuner     — вместо tuning.tune (P1): заданный ответ или маленький перебор сетки тем же evaluator.
SimpleDeadline — вместо budget.Deadline (P1).
SpecStub      — поля TaskSpec из schemas.py (P1), пока его нет в main.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent import Candidate
from agent_eval import LocalEvaluator


@dataclass
class SpecStub:
    task_dir: Path
    id_col: str
    target: str
    task_type: str
    metric: str
    greater_is_better: bool
    feature_notes: dict[str, str] = field(default_factory=dict)
    drop_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    date_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    raw_task_text: str = ""
    source: str = "heuristic"


class SimpleDeadline:
    def __init__(self, seconds: float) -> None:
        self.end = time.monotonic() + seconds

    def left(self) -> float:
        return max(0.0, self.end - time.monotonic())

    def stage(self, name: str) -> float:
        return self.left()


class MockLLM:
    """Отдаёт ответы по сценарию: script = {"planner": [...], "coder": [...], "critic": [...]}.

    Роль берётся из первой строки системного промпта («Role: coder.»). Элемент сценария — строка,
    dict (для chat_json), исключение (бросается) или callable(messages). Когда сценарий роли
    исчерпан, отдаётся нейтральный ответ из DEFAULTS.
    """

    offline = True  # не сетевая модель: LLM_DISABLED=1 её не отключает
    DEFAULTS = {"planner": '{"hypotheses": []}', "coder": "Sorry, no code.", "critic": "Nothing to add."}

    def __init__(self, script: dict[str, list] | None = None, tokens: tuple[int, int] = (1000, 200)) -> None:
        self.script = {role: list(items) for role, items in (script or {}).items()}
        self.tokens = tokens
        self.calls: list[tuple[str, list[dict], dict]] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def role(messages: list[dict]) -> str:
        found = re.search(r"Role:\s*(\w+)", messages[0]["content"]) if messages else None
        return found.group(1).lower() if found else "unknown"

    def _next(self, messages: list[dict], kw: dict) -> Any:
        role = self.role(messages)
        self.calls.append((role, messages, kw))
        self.usage["prompt_tokens"] += self.tokens[0]
        self.usage["completion_tokens"] += self.tokens[1]
        queue = self.script.get(role, [])
        item = queue.pop(0) if queue else self.DEFAULTS.get(role, "")
        if isinstance(item, BaseException):
            raise item
        return item(messages) if callable(item) else item

    def chat(self, messages: list[dict], **kw: Any) -> str:
        item = self._next(messages, kw)
        return item if isinstance(item, str) else json.dumps(item)

    def chat_json(self, messages: list[dict], schema: dict, **kw: Any) -> dict:
        item = self._next(messages, kw)
        if isinstance(item, str):
            try:
                return json.loads(item)
            except json.JSONDecodeError as error:  # так же ведёт себя клиент P2 после своих попыток починки
                raise ValueError(f"invalid JSON: {error}") from error
        return item

    def count(self, role: str) -> int:
        return sum(1 for r, _, _ in self.calls if r == role)


# ---------------------------------------------------------------- evaluator

class StubEvaluator(LocalEvaluator):
    """Заглушка evaluator.evaluate(spec, build_features, params, deadline) -> Candidate (P1).

    Это LocalEvaluator агента (agent_eval.py) с 4 потоками по умолчанию: тесты не зависят от числа ядер.
    params = {"model": "lightgbm|catboost|xgboost", "params": {...}, "target_transform": None|"log1p"}.
    """

    def __init__(self, n_splits: int = 5, seed: int = 42, n_estimators: int = 300,
                 learning_rate: float = 0.05, threads: int = 4) -> None:
        super().__init__(n_splits=n_splits, seed=seed, n_estimators=n_estimators, learning_rate=learning_rate,
                         threads=threads)


class StubTuner:
    """Заглушка tuning.tune(spec, build_features, cfg, deadline) -> dict (P1).

    result — что вернуть (dict) или исключение (бросается). С evaluator и grid — маленький перебор:
    каждый набор из grid поверх cfg["params"] оценивается тем же evaluator, возвращается лучший
    (для dry-run и живых прогонов, пока у P1 нет настоящего подбора на Optuna).
    """

    def __init__(self, result: Any = None, *, evaluator: Any = None, grid: list[dict] | tuple = (),
                 greater_is_better: bool = False) -> None:
        self.result, self.evaluator, self.grid, self.greater = result, evaluator, list(grid), greater_is_better
        self.calls: list[tuple[dict, float | None]] = []

    def __call__(self, spec: Any, build_features: Any, cfg: dict, deadline: Any = None) -> dict:
        self.calls.append((cfg, deadline.left() if deadline is not None else None))
        if isinstance(self.result, BaseException):
            raise self.result
        if self.evaluator is None:
            return dict(self.result or {})
        best, best_score = dict(cfg.get("params") or {}), None
        for params in self.grid:
            if deadline is not None and deadline.left() < 5:
                break
            trial = {**(cfg.get("params") or {}), **params}
            score = self.evaluator(spec, build_features, {**cfg, "params": trial}, deadline).cv_score
            if best_score is None or (score > best_score if self.greater else score < best_score):
                best, best_score = trial, score
        return best


LIGHTGBM_GRID = [{"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63},
                 {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50},
                 {"n_estimators": 400, "learning_rate": 0.08, "num_leaves": 127, "colsample_bytree": 0.7}]


def identity_features(train: pd.DataFrame, test: pd.DataFrame, spec: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train, test


def make_best(spec: SpecStub, evaluator: StubEvaluator) -> Candidate:
    """Аналог fallback-кандидата P3: модель на исходных признаках, без кода агента."""
    cand = evaluator(spec, identity_features, {"model": "lightgbm"}, None)
    cand.name = "fallback"
    return cand


# ---------------------------------------------------------------- синтетические задачи

def make_regression_task(root: Path, n_train: int = 400, n_test: int = 150, seed: int = 0) -> SpecStub:
    """y = 5·x1·x2 + 0.5·x3 + эффект города + шум: признак x1*x2 даёт крупный выигрыш CV."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame({"id": np.arange(n) + 1000, "x1": rng.uniform(-1, 1, n), "x2": rng.uniform(-1, 1, n),
                       "x3": rng.normal(size=n), "city": rng.choice(["a", "b", "c"], n)})
    y = (5 * df.x1 * df.x2 + 0.5 * df.x3 + df.city.map({"a": 0.0, "b": 0.5, "c": -0.5})
         + rng.normal(scale=0.1, size=n))
    df.loc[rng.random(n) < 0.05, "x3"] = np.nan
    root.mkdir(parents=True, exist_ok=True)
    df.iloc[:n_train].assign(target=y[:n_train]).to_csv(root / "train.csv", index=False)
    df.iloc[n_train:].to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": df.id.iloc[n_train:], "target": 0.0}).to_csv(root / "sample_submission.csv", index=False)
    text = "# Synthetic regression\nPredict `target`. Metric: RMSE (lower is better)."
    (root / "task.md").write_text(text, encoding="utf-8")
    return SpecStub(task_dir=root, id_col="id", target="target", task_type="regression", metric="rmse",
                    greater_is_better=False, cat_cols=["city"], raw_task_text=text,
                    feature_notes={"x1": "первый фактор", "x2": "второй фактор", "x3": "шумовой признак",
                                   "city": "код города"})


def make_multiclass_task(root: Path, n_train: int = 450, n_test: int = 150, seed: int = 1) -> SpecStub:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame({"id": np.arange(n), "a": rng.normal(size=n), "b": rng.normal(size=n),
                       "proto": rng.choice(["tcp", "udp", "icmp"], n)})
    score = df.a * df.b + rng.normal(scale=0.05, size=n)
    label = np.where(score > 0.4, "high", np.where(score < -0.4, "low", "mid"))
    root.mkdir(parents=True, exist_ok=True)
    df.iloc[:n_train].assign(kind=label[:n_train]).to_csv(root / "train.csv", index=False)
    df.iloc[n_train:].to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": df.id.iloc[n_train:], "kind": "mid"}).to_csv(root / "sample_submission.csv", index=False)
    text = "# Synthetic multiclass\nPredict `kind`. Metric: Macro F1 (higher is better)."
    (root / "task.md").write_text(text, encoding="utf-8")
    return SpecStub(task_dir=root, id_col="id", target="kind", task_type="multiclass", metric="macro_f1",
                    greater_is_better=True, cat_cols=["proto"], raw_task_text=text)


def make_text_task(root: Path, n_train: int = 600, n_test: int = 200, seed: int = 2) -> SpecStub:
    """Бинарная: класс задают слова в отзыве (85% отзывов с «правильным» словом). Колонка review в spec отмечена
    категорией, как это делают эвристики по dtype: агент сам должен распознать в ней текст."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    label = rng.choice(["neg", "pos"], n)
    filler = np.array("the product delivery box price item color size was very quite really and it".split())
    words = {"pos": np.array(["great", "excellent", "love", "perfect", "wonderful"]),
             "neg": np.array(["awful", "terrible", "hate", "broken", "poor"])}
    reviews = []
    for kind in label:
        tone = kind if rng.random() < 0.85 else ("neg" if kind == "pos" else "pos")
        parts = list(rng.choice(filler, rng.integers(6, 12))) + list(rng.choice(words[tone], rng.integers(1, 3)))
        reviews.append(" ".join(rng.permutation(parts)))
    df = pd.DataFrame({"id": np.arange(n), "x1": rng.normal(size=n), "city": rng.choice(["a", "b"], n),
                       "review": reviews})
    root.mkdir(parents=True, exist_ok=True)
    df.iloc[:n_train].assign(sentiment=label[:n_train]).to_csv(root / "train.csv", index=False)
    df.iloc[n_train:].to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": df.id.iloc[n_train:], "sentiment": "pos"}).to_csv(root / "sample_submission.csv", index=False)
    text = "# Synthetic text classification\nPredict `sentiment`. Metric: Macro F1 (higher is better)."
    (root / "task.md").write_text(text, encoding="utf-8")
    return SpecStub(task_dir=root, id_col="id", target="sentiment", task_type="binary", metric="macro_f1",
                    greater_is_better=True, cat_cols=["city", "review"], raw_task_text=text)


# ---------------------------------------------------------------- ответы LLM для сценариев

def fenced(code: str) -> str:
    return f"Here is the code:\n```python\n{code}```\nDone."


def plan(*titles: str, kind: str = "features") -> dict:
    return {"analysis": "x1 and x2 interact; x3 is noise.",
            "hypotheses": [{"id": f"h{i}", "title": title, "kind": kind, "rationale": "test",
                            "expected_gain": "high", "cost": "low"} for i, title in enumerate(titles, 1)]}


def feature_code(body: str, header: str = "") -> str:
    """Модуль по шаблону агента; body — строки внутри цикла по (X_train, X_test) с переменной df."""
    lines = "\n".join("        " + line for line in body.strip().splitlines())
    return (f"import numpy as np\nimport pandas as pd\n{header}\n\n"
            "def build_features(train, test, spec):\n"
            "    X_train, X_test = train.copy(), test.copy()\n"
            "    for df in (X_train, X_test):\n"
            f"{lines}\n"
            "    return X_train, X_test\n\n\n"
            'MODEL = "lightgbm"\nPARAMS = {}\nTARGET_TRANSFORM = None\n')


GOOD_CODE = feature_code('df["x1_x2"] = df["x1"] * df["x2"]')
EXCEPTION_CODE = feature_code('df["oops"] = df["no_such_column"] * 2')
SLOW_CODE = feature_code("while True:\n    pass")
INF_CODE = feature_code('df["ratio"] = df["x1"] / (df["x2"] * 0.0)')
NO_FUNCTION_CODE = "import pandas as pd\n\n\ndef make_features(train, test):\n    return train, test\n"
DROP_ROWS_CODE = ("import pandas as pd\n\n\ndef build_features(train, test, spec):\n"
                  "    return train.iloc[:-5].copy(), test.copy()\n")


def useless_code(name: str) -> str:
    """Монотонное преобразование одного признака: деревьям ничего не даёт."""
    return feature_code(f'df["{name}"] = df["x3"] * 2.0 + 1.0')
