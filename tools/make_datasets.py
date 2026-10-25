"""Тестовые задачи для проверки универсальности (роль P5, написано P4 — владелец может переписать). Вне ZIP.

    .venv/bin/python tools/make_datasets.py                    # → .agent_work/datasets/<имя>/
    .venv/bin/python tools/make_datasets.py --out путь --only bin_auc,reg_no_id

Каждая задача — папка в формате платформы: `task/` (task.md, train.csv, test.csv, sample_submission.csv,
иногда data_dictionary.csv — иногда во вложенной папке, как в public) и рядом `answers.csv` (таргет test) с
`meta.json` (метрика). `solution.py` видит только `task/`. Данные — синтетика sklearn и встроенные в sklearn
датасеты (без интернета). Прогон — `tools/run_all.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".agent_work" / "datasets"


@dataclass
class Task:
    """Готовая задача: полные данные (с таргетом), разбиение и то, как её описать."""
    frame: pd.DataFrame
    target: str | list[str]
    metric: str             # имя метрики в metrics.py — для оценки ответа
    metric_text: str        # как метрика названа в task.md
    title: str
    id_col: str | None = "id"
    lang: str = "ru"
    nested: str | None = None        # файлы в подпапке, как в public («multiclass/participants»)
    dictionary: bool = True
    shuffle_sample: bool = False     # порядок id в образце не как в test
    proba: bool = False              # в образце вероятности (AUC, logloss)
    test_share: float = 0.3
    split: Callable[[pd.DataFrame, np.random.Generator], np.ndarray] | None = None  # маска test
    notes: dict[str, str] = field(default_factory=dict)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _words(rng: np.random.Generator, vocab: list[str], n: int) -> str:
    return " ".join(rng.choice(vocab, size=n))


def _add_noise_columns(frame: pd.DataFrame, rng: np.random.Generator, *, missing: float = 0.0,
                       categories: int = 0, high_card: int = 0, date: bool = False) -> pd.DataFrame:
    n = len(frame)
    frame = frame.copy()
    if categories:
        frame["region"] = rng.choice([f"R{i}" for i in range(categories)], size=n)
    if high_card:
        frame["shop_code"] = rng.choice([f"S{i:05d}" for i in range(high_card)], size=n)
    if date:
        start = np.datetime64("2021-01-01")
        frame["created_at"] = (start + rng.integers(0, 900, size=n).astype("timedelta64[D]")).astype(str)
    if missing:
        numeric = [c for c in frame.columns if c.startswith("f")]
        for col in numeric[: max(1, len(numeric) // 2)]:
            frame.loc[rng.random(n) < missing, col] = np.nan
    return frame


# ---------------------------------------------------------------- задачи

def bin_auc() -> Task:
    """Бинарная, ROC-AUC (в образце вероятности), встроенный датасет, нестандартное имя id."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    frame = data.frame.rename(columns=lambda c: c.replace(" ", "_"))
    frame.insert(0, "patient_id", np.arange(10_000, 10_000 + len(frame)))
    return Task(frame, "target", "roc_auc", "ROC-AUC", "Диагностика опухоли", id_col="patient_id", proba=True)


def bin_f1_str() -> Task:
    """Бинарная, F1, таргет-строка yes/no, дисбаланс 9:1, пропуски, категории, вложенные пути."""
    from sklearn.datasets import make_classification
    x, y = make_classification(6000, 15, n_informative=6, weights=[0.9], flip_y=0.02, random_state=1)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame = _add_noise_columns(frame, _rng(1), missing=0.1, categories=7)
    frame["churn"] = np.where(y == 1, "yes", "no")
    frame.insert(0, "id", np.arange(len(frame)))
    return Task(frame, "churn", "f1", "F1-score (положительный класс — `yes`)", "Отток клиентов",
                nested="binary/participants")


def multi_imbalance() -> Task:
    """Мультикласс 6 классов со строковыми метками, macro F1, дисбаланс 100:1, высокая кардинальность, дата."""
    from sklearn.datasets import make_classification
    x, y = make_classification(12_000, 20, n_informative=10, n_classes=6, n_clusters_per_class=1,
                               weights=[0.55, 0.25, 0.1, 0.06, 0.035, 0.005], random_state=2)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame = _add_noise_columns(frame, _rng(2), missing=0.05, high_card=800, date=True)
    frame["segment"] = np.array(["alpha", "beta", "gamma", "delta", "eps", "zeta"])[y]
    frame.insert(0, "id", np.arange(len(frame)))
    return Task(frame, "segment", "macro_f1", "Macro F1", "Сегмент клиента")


def small_wine() -> Task:
    """Мало строк (178), 3 класса с целыми метками, accuracy, без data_dictionary."""
    from sklearn.datasets import load_wine
    frame = load_wine(as_frame=True).frame
    frame.insert(0, "id", np.arange(len(frame)))
    return Task(frame, "target", "accuracy", "Accuracy", "Сорт вина", dictionary=False)


def digits_weighted() -> Task:
    """10 классов, 64 признака, weighted F1, английское описание."""
    from sklearn.datasets import load_digits
    data = load_digits(as_frame=True)
    frame = data.frame
    frame.insert(0, "id", np.arange(len(frame)))
    return Task(frame, "target", "weighted_f1", "weighted F1-score", "Handwritten digits", lang="en")


def reg_mae_en() -> Task:
    """Регрессия, MAE, мало строк (442), английское описание, перемешанный образец."""
    from sklearn.datasets import load_diabetes
    frame = load_diabetes(as_frame=True, scaled=False).frame
    frame.insert(0, "id", np.arange(len(frame)))
    return Task(frame, "target", "mae", "Mean Absolute Error (MAE)", "Diabetes progression", lang="en",
                shuffle_sample=True)


def reg_rmsle_skewed() -> Task:
    """Регрессия, RMSLE, скошенный положительный таргет, дата, текст, пропуски."""
    rng = _rng(3)
    n = 8000
    frame = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(8)})
    frame["rooms"] = rng.integers(1, 6, size=n)
    vocab_good, vocab_bad = ["ремонт", "вид", "парк", "метро", "новый"], ["шум", "трасса", "старый", "сырость"]
    filler = ["квартира", "дом", "этаж", "окна", "кухня", "балкон", "рядом", "школа"]
    good = rng.random(n) < 0.5
    frame["description"] = [
        _words(rng, filler, 6) + " " + _words(rng, vocab_good if g else vocab_bad, 3) for g in good]
    log_price = 12 + 0.4 * frame["f0"] + 0.25 * frame["rooms"] + 0.5 * good + 0.2 * frame["f1"] ** 2 \
        + rng.normal(scale=0.3, size=n)
    frame["price"] = np.exp(log_price).round(0)
    frame = _add_noise_columns(frame, rng, missing=0.08, date=True, categories=12)
    frame.insert(0, "listing_id", np.arange(500_000, 500_000 + n))
    return Task(frame, "price", "rmsle", "RMSLE", "Цена квартиры", id_col="listing_id")


def reg_no_id() -> Task:
    """Регрессия без id-колонки: образец — одна колонка таргета, порядок строк как в test; R²."""
    from sklearn.datasets import make_regression
    x, y = make_regression(5000, 12, n_informative=8, noise=15, random_state=4)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    return Task(frame, "y", "r2", "коэффициент детерминации R²", "Регрессия без идентификатора", id_col=None)


def reg_mape_period() -> Task:
    """Регрессия, MAPE, test — новый год (как в public-регрессии: фолды по периодам)."""
    rng = _rng(5)
    n = 9000
    year = rng.integers(1, 7, size=n)
    frame = pd.DataFrame({"year_factor": year, "store": rng.choice([f"st{i}" for i in range(40)], size=n)})
    for i in range(6):
        frame[f"f{i}"] = rng.normal(size=n)
    base = 100 + 8 * frame["f0"] + 5 * frame["f1"] * frame["f2"] + 3 * year
    frame["sales"] = (base + rng.normal(scale=6, size=n)).clip(lower=5)
    frame.insert(0, "id", np.arange(n))
    return Task(frame, "sales", "mape", "MAPE", "Продажи магазина",
                split=lambda f, r: (f["year_factor"] == 6).to_numpy())


def wide_logloss() -> Task:
    """Широкая бинарная (300 признаков), logloss — нужны вероятности."""
    from sklearn.datasets import make_classification
    x, y = make_classification(3000, 300, n_informative=20, n_redundant=30, random_state=6)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["label"] = y
    frame.insert(0, "id", np.arange(len(frame)))
    return Task(frame, "label", "logloss", "Log Loss", "Широкая бинарная задача", proba=True, dictionary=False)


def text_cls() -> Task:
    """Классификация по свободному тексту (3 класса), accuracy."""
    rng = _rng(7)
    n = 6000
    topics = {"sport": ["матч", "гол", "тренер", "команда", "счёт"],
              "finance": ["курс", "банк", "кредит", "ставка", "акции"],
              "tech": ["смартфон", "процессор", "сеть", "код", "сервер"]}
    filler = [w for w in string.ascii_lowercase[:12]] + ["сегодня", "новость", "эксперт", "заявил", "итог"]
    labels = rng.choice(list(topics), size=n, p=[0.5, 0.3, 0.2])
    texts = []
    for label in labels:
        own = rng.random() < 0.85
        vocab = topics[label] if own else topics[rng.choice(list(topics))]
        texts.append(_words(rng, filler, int(rng.integers(5, 15))) + " " + _words(rng, vocab, int(rng.integers(1, 4))))
    frame = pd.DataFrame({"id": np.arange(n), "text": texts, "length_hint": rng.integers(0, 3, size=n),
                          "topic": labels})
    return Task(frame, "topic", "accuracy", "Accuracy", "Тематика новости")


def multi_target() -> Task:
    """Два таргета регрессии в образце (мульти-таргет)."""
    from sklearn.datasets import make_regression
    x, y = make_regression(4000, 10, n_targets=2, noise=10, random_state=8)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["t1"], frame["t2"] = y[:, 0], y[:, 1]
    frame.insert(0, "id", np.arange(len(frame)))
    return Task(frame, ["t1", "t2"], "rmse", "RMSE (среднее по двум таргетам)", "Два таргета")


TASKS: dict[str, Callable[[], Task]] = {f.__name__: f for f in (
    bin_auc, bin_f1_str, multi_imbalance, small_wine, digits_weighted, reg_mae_en, reg_rmsle_skewed, reg_no_id,
    reg_mape_period, wide_logloss, text_cls, multi_target)}


# ---------------------------------------------------------------- запись

def _task_md(task: Task, name: str, prefix: str, columns: list[str], n_train: int, n_test: int) -> str:
    targets = task.target if isinstance(task.target, list) else [task.target]
    files = ["train.csv", "test.csv", "sample_submission.csv"] + (["data_dictionary.csv"] if task.dictionary else [])
    out_cols = ([task.id_col] if task.id_col else []) + targets
    if task.lang == "en":
        lines = [f"# {task.title}", "", "## Task", "",
                 f"Predict `{', '.join(targets)}` for every row of `test.csv`.", "", "## Data", "",
                 "Paths are relative to this file:", ""]
        lines += [f"- `{prefix}{f}`" for f in files]
        lines += ["", f"train: {n_train} rows, test: {n_test} rows.", "", "## Evaluation", "",
                  f"The primary metric is **{task.metric_text}**.", "", "## Submission", "",
                  f"`submission.csv` next to this file with columns `{','.join(out_cols)}`."]
    else:
        lines = [f"# {task.title}", "", "## Задача", "",
                 f"Предскажите `{', '.join(targets)}` для каждой строки `test.csv`.", "", "## Данные", "",
                 "Пути указаны относительно каталога этого Markdown-файла:", ""]
        lines += [f"- `{prefix}{f}`" for f in files]
        lines += ["", f"В train {n_train} строк, в test {n_test}. Признаки: {', '.join(columns[:12])}"
                  + (" и другие." if len(columns) > 12 else "."), "", "## Метрика", "",
                  f"Основная метрика — **{task.metric_text}**.", "", "## Формат результата", "",
                  f"Файл `submission.csv` рядом с этим файлом, столбцы `{','.join(out_cols)}`."]
        if task.proba:
            lines.append("Для каждой строки — вероятность положительного класса.")
    return "\n".join(lines) + "\n"


def write_task(name: str, task: Task, out: Path, seed: int = 0) -> Path:
    rng = _rng(seed)
    frame = task.frame.reset_index(drop=True)
    targets = task.target if isinstance(task.target, list) else [task.target]
    is_test = task.split(frame, rng) if task.split else rng.random(len(frame)) < task.test_share
    train, test = frame[~is_test].reset_index(drop=True), frame[is_test].reset_index(drop=True)

    root = out / name
    if root.exists():
        shutil.rmtree(root)
    task_dir = root / "task"
    data_dir = task_dir / task.nested if task.nested else task_dir
    data_dir.mkdir(parents=True)
    prefix = f"{task.nested}/" if task.nested else ""

    train.to_csv(data_dir / "train.csv", index=False)
    test.drop(columns=targets).to_csv(data_dir / "test.csv", index=False)
    sample = pd.DataFrame({task.id_col: test[task.id_col]}) if task.id_col else pd.DataFrame(index=test.index)
    for col in targets:
        if task.proba:
            sample[col] = 0.5
        elif pd.api.types.is_numeric_dtype(train[col]) and train[col].nunique() > 20:
            sample[col] = float(train[col].mean())
        else:
            sample[col] = train[col].mode().iloc[0]
    if task.shuffle_sample:
        sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample.to_csv(data_dir / "sample_submission.csv", index=False)
    if task.dictionary:
        rows = [{"feature": c, "role": "target" if c in targets else ("id" if c == task.id_col else "feature"),
                 "dtype": "numeric" if pd.api.types.is_numeric_dtype(frame[c]) else "category",
                 "description_ru": task.notes.get(c, f"Поле {c}")} for c in frame.columns]
        pd.DataFrame(rows).to_csv(data_dir / "data_dictionary.csv", index=False)
    features = [c for c in frame.columns if c not in targets and c != task.id_col]
    (task_dir / "task.md").write_text(_task_md(task, name, prefix, features, len(train), len(test)), encoding="utf-8")

    answers = test[([task.id_col] if task.id_col else []) + targets]
    answers.to_csv(root / "answers.csv", index=False)
    (root / "meta.json").write_text(json.dumps({"metric": task.metric, "target": task.target, "id_col": task.id_col,
                                                "proba": task.proba}, ensure_ascii=False), encoding="utf-8")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", default="", help="имена через запятую")
    args = parser.parse_args(argv)
    names = [n for n in args.only.split(",") if n] or list(TASKS)
    for name in names:
        root = write_task(name, TASKS[name](), args.out)
        print(f"{name}: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
