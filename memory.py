"""Память экспериментов агента.

Журнал: что пробовали, что приняли/отклонили и почему, что падало, что изменилось в коде.
Отдаёт сжатую сводку для промпта (принятые шаги не выкидываются никогда), ловит перефразированные
дубли гипотез, показывает «что ещё не пробовали», пишет и читает JSONL.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

STATUSES = ("accepted", "rejected", "failed")


def code_hash(code: str) -> str:
    """Хеш кода без учёта пустых строк и хвостовых пробелов — для поиска повторов."""
    lines = [line.rstrip() for line in code.strip().splitlines() if line.strip()]
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def normalize_title(title: str) -> str:
    return re.sub(r"[\W_]+", " ", title.lower()).strip()


def _code_lines(code: str) -> list[str]:
    return [line.rstrip() for line in (code or "").splitlines() if line.strip()]


def _prefer_added(lines: list[str], limit: int) -> list[str]:
    """Не больше limit строк: сначала добавленные (+), потом удалённые; исходный порядок сохраняется."""
    order = sorted(range(len(lines)), key=lambda i: (not lines[i].startswith("+"), i))[:limit]
    return [lines[i] for i in sorted(order)]


def code_diff(old: str, new: str, max_lines: int = 15) -> str:
    """Только изменённые строки (+/-) без контекста; при обрезке важнее добавленные. max_lines=0 — без обрезки."""
    changed = [line for line in difflib.unified_diff(_code_lines(old), _code_lines(new), lineterm="", n=0)
               if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    if max_lines and len(changed) > max_lines:
        changed = _prefer_added(changed, max_lines) + [f"... (+{len(changed) - max_lines} more changed lines)"]
    return "\n".join(changed)


# ---------------------------------------------------------------- похожие гипотезы

_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "and", "with", "on", "in", "by", "from", "as", "into", "at", "per", "via",
    "add", "adding", "added", "create", "creating", "new", "use", "using", "apply", "applying", "compute",
    "computing", "make", "build", "try", "switch", "feature", "features", "column", "columns", "col", "cols",
    "variable", "variables", "value", "values", "based", "all", "each", "simple", "basic", "extra", "additional",
    "existing", "raw", "transform", "transformation", "transformed", "model", "outlier", "outliers", "extreme",
}
_SYNONYMS = {
    "frequency": "count", "freq": "count",
    "ratio": "div", "divide": "div", "divided": "div", "division": "div", "proportion": "div", "share": "div",
    "fraction": "div",
    "interaction": "product", "multiply": "product", "multiplication": "product", "cross": "product",
    "times": "product",
    "log1p": "log", "logarithm": "log", "logarithmic": "log", "ln": "log",
    "root": "sqrt",
    "nan": "missing", "null": "missing", "na": "missing", "missingness": "missing", "isnull": "missing",
    "isna": "missing",
    "imputation": "impute", "imputing": "impute", "fill": "impute", "fillna": "impute", "filling": "impute",
    "encode": "encoding", "encoded": "encoding", "encoder": "encoding",
    "aggregate": "agg", "aggregation": "agg", "groupby": "agg", "group": "agg", "grouped": "agg", "grouping": "agg",
    "difference": "diff", "subtract": "diff", "subtraction": "diff", "minus": "diff", "delta": "diff",
    "total": "sum", "average": "mean", "avg": "mean",
    "clipping": "clip", "winsorize": "clip", "winsorizing": "clip", "winsorization": "clip", "cap": "clip",
    "capping": "clip", "truncate": "clip",
    "remove": "drop", "removing": "drop", "dropping": "drop", "exclude": "drop", "delete": "drop",
    "indicator": "flag", "binary": "flag", "dummy": "flag",
    "infrequent": "rare", "temperature": "temp",
    "lgbm": "lightgbm", "lgb": "lightgbm", "xgb": "xgboost", "cb": "catboost",
    "raise": "increase", "higher": "increase", "larger": "increase", "more": "increase",
    "lower": "decrease", "reduce": "decrease", "smaller": "decrease", "less": "decrease",
    "tuning": "tune", "optimize": "tune",
}


_OPPOSITES = (("increase", "decrease"),)


def _canonical(word: str) -> str:
    word = _SYNONYMS.get(word, word)
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = _SYNONYMS.get(word[:-1], word[:-1])
    return word


def title_tokens(title: str) -> set[str]:
    """Смысловые слова названия: без стоп-слов, синонимы и формы приведены к одному слову."""
    words = (_canonical(w) for w in normalize_title(title).split())
    return {w for w in words if w and w not in _STOPWORDS}


def mentioned_columns(text: str, known: Iterable[str]) -> set[str]:
    """Известные колонки, названные в тексте целым словом (floor_area или «floor area»)."""
    text = (text or "").lower()
    found = set()
    for name in known:
        for variant in {name.lower(), name.lower().replace("_", " ")}:
            if variant and re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", text):
                found.add(name)
                break
    return found


def _overlap(a: set[str], b: set[str]) -> float:
    """Среднее Жаккара и коэффициента перекрытия: «то же + уточнение» тоже похоже."""
    if not a or not b:
        return 0.0
    common = len(a & b)
    return (common / len(a | b) + common / min(len(a), len(b))) / 2


def similar_titles(a: str, b: str) -> float:
    return _overlap(title_tokens(a), title_tokens(b))


def same_idea(title_a: str, kind_a: str, title_b: str, kind_b: str, columns: Iterable[str] = (),
              threshold: float = 0.6) -> bool:
    """Одна и та же гипотеза другими словами: тот же kind и (при названных колонках — те же колонки)
    похожие действия над ними."""
    if kind_a != kind_b:
        return False
    if normalize_title(title_a) == normalize_title(title_b):
        return True
    tokens_a, tokens_b = title_tokens(title_a), title_tokens(title_b)
    if any((x in tokens_a and y in tokens_b) or (y in tokens_a and x in tokens_b) for x, y in _OPPOSITES):
        return False  # «увеличить» и «уменьшить» одно и то же — разные гипотезы
    columns = list(columns)
    cols_a, cols_b = mentioned_columns(title_a, columns), mentioned_columns(title_b, columns)
    if cols_a and cols_b:
        if cols_a != cols_b:
            return False
        names = title_tokens(" ".join(cols_a))
        ops_a, ops_b = tokens_a - names, tokens_b - names
        return (not ops_a and not ops_b) or _overlap(ops_a, ops_b) >= threshold
    return _overlap(tokens_a, tokens_b) >= threshold


_STRING_LITERAL = re.compile(r"""(['"])((?:\\.|(?!\1).)*)\1""")


def columns_in(code: str, known: Iterable[str]) -> list[str]:
    """Известные колонки, упомянутые в коде строковыми литералами (df["col"]), в порядке появления."""
    known = set(known)
    found = (match.group(2) for match in _STRING_LITERAL.finditer(code or ""))
    return list(dict.fromkeys(name for name in found if name in known))


@dataclass
class Experiment:
    id: int
    hypothesis: str
    kind: str
    status: str  # accepted | rejected | failed
    cv_score: float | None = None
    cv_std: float | None = None
    gain: float | None = None
    code_hash: str = ""
    error: str = ""
    duration_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    ts: float = field(default_factory=time.time)
    columns: list[str] = field(default_factory=list)  # исходные колонки, которых касается изменение
    diff: str = ""                                    # сжатый дифф кода относительно базы
    base_hash: str = ""                               # code_hash базы, от которой сделан эксперимент


_FIELDS = {f.name for f in fields(Experiment)}


def _num(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.5g}"


def _one_line(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


_FAILURE_CAUSES = re.compile(r"timeout|static check|NaN avalanche|inf values|row count|columns differ|"
                             r"look like a row index|evaluation failed|no time left|no \S+ code block|"
                             r"same code|[A-Z]\w*(?:Error|Exception)")


def _failure_cause(error: str) -> str:
    """Короткая причина падения для итога по видам: тип ошибки или вид проверки харнесса."""
    found = _FAILURE_CAUSES.search(error or "")
    return found.group(0) if found else _one_line(error or "unknown", 30)


def _diff_preview(diff: str, max_lines: int = 3) -> list[str]:
    lines = [line for line in (diff or "").splitlines() if line.strip() and not line.startswith("... (+")]
    preview = [_one_line(line, 100) for line in _prefer_added(lines, max_lines)]
    return preview + ([f"... (+{len(lines) - max_lines} lines)"] if len(lines) > max_lines else [])


def _format(exp: Experiment) -> str:
    head = f"#{exp.id} [{exp.status}] {exp.kind} | {_one_line(exp.hypothesis, 80)}"
    if exp.status == "failed":
        return f"{head} | error: {_one_line(exp.error, 140)}"
    gain = "" if exp.gain is None else f" (gain {exp.gain:+.4g})"
    reason = f" | {_one_line(exp.error, 160)}" if exp.error else ""
    return f"{head} | cv={_num(exp.cv_score)}{gain}{reason}"


class ExperimentMemory:
    def __init__(self, columns: Iterable[str] = ()) -> None:
        self.experiments: list[Experiment] = []
        self.notes: list[str] = []
        self.columns: list[str] = [str(c) for c in columns]  # исходные колонки задачи — для дублей и покрытия

    def next_id(self) -> int:
        return len(self.experiments) + 1

    def add(self, exp: Experiment) -> Experiment:
        if exp.status not in STATUSES:
            raise ValueError(f"unknown status: {exp.status}")
        self.experiments.append(exp)
        return exp

    def add_note(self, text: str, keep: int = 3) -> None:
        """Заметка критика; хранятся только последние keep."""
        text = _one_line(text, 600)
        if text:
            self.notes.append(text)
            del self.notes[:-keep]

    def tried_titles(self) -> list[str]:
        return [exp.hypothesis for exp in self.experiments]

    def tried_kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for exp in self.experiments:
            counts[exp.kind] = counts.get(exp.kind, 0) + 1
        return counts

    def find_code(self, digest: str) -> Experiment | None:
        return next((exp for exp in self.experiments if exp.code_hash == digest), None)

    def same_idea(self, title_a: str, kind_a: str, title_b: str, kind_b: str, threshold: float = 0.6) -> bool:
        return same_idea(title_a, kind_a, title_b, kind_b, self.columns, threshold)

    def find_similar(self, title: str, kind: str, threshold: float = 0.6) -> Experiment | None:
        """Уже проверенная гипотеза с той же идеей. Упавшую можно повторить один раз
        (могла упасть из-за кода, а не идеи): дублем она считается со второго падения."""
        matches = [e for e in self.experiments if self.same_idea(title, kind, e.hypothesis, e.kind, threshold)]
        decided = [e for e in matches if e.status != "failed"]
        if decided:
            return decided[-1]
        failed = [e for e in matches if e.status == "failed"]
        return failed[-1] if len(failed) >= 2 else None

    def kind_stats(self) -> list[str]:
        """Итог по видам гипотез: сколько пробовали, принято (лучший выигрыш), отклонено, упало (почему)."""
        groups: dict[str, list[Experiment]] = {}
        for exp in self.experiments:
            groups.setdefault(exp.kind, []).append(exp)
        lines = []
        for kind, exps in groups.items():
            accepted = [e for e in exps if e.status == "accepted"]
            gains = [e.gain for e in accepted if e.gain is not None and math.isfinite(e.gain)]
            parts = [f"{len(exps)} tried", f"{len(accepted)} accepted"
                     + (f" (best gain {max(gains):+.4g})" if gains else "")]
            rejected = sum(e.status == "rejected" for e in exps)
            if rejected:
                parts.append(f"{rejected} rejected")
            failed = [e for e in exps if e.status == "failed"]
            if failed:
                causes = list(dict.fromkeys(_failure_cause(e.error) for e in failed))
                parts.append(f"{len(failed)} failed ({', '.join(causes[:2])})")
            lines.append(f"{kind}: {', '.join(parts)}")
        return lines

    def coverage(self, all_columns: Iterable[str] | None = None, kinds: Iterable[str] = (), task_type: str = "",
                 max_columns: int = 15, exhausted_after: int = 3) -> dict:
        """Что ещё не пробовали: виды гипотез без попыток, исчерпанные виды (exhausted_after+ попыток без
        принятия), исходные колонки, которых не касался ни один эксперимент (по диффу и по названию)."""
        all_columns = list(self.columns if all_columns is None else all_columns)
        kinds = [k for k in kinds if k != "target_transform" or task_type in ("", "regression")]
        groups: dict[str, list[Experiment]] = {}
        used: set[str] = set()
        for exp in self.experiments:
            groups.setdefault(exp.kind, []).append(exp)
            used.update(exp.columns, mentioned_columns(exp.hypothesis, all_columns))
        untouched = [c for c in all_columns if c not in used]
        return {"untried_kinds": [k for k in kinds if k not in groups],
                "exhausted_kinds": [k for k, exps in groups.items()
                                    if len(exps) >= exhausted_after and all(e.status != "accepted" for e in exps)],
                "untouched_columns": untouched[:max_columns], "n_untouched_columns": len(untouched)}

    def planner_brief(self, max_chars: int = 1500, all_columns: Iterable[str] | None = None,
                      kinds: Iterable[str] = (), task_type: str = "") -> str:
        """Для планировщика: сводка + куда ещё не смотрели и что исчерпано (в конце — ближе к вопросу)."""
        if not self.experiments:
            return self.summary(max_chars)
        cov = self.coverage(all_columns, kinds, task_type)
        lines = []
        if cov["untried_kinds"]:
            lines.append("Kinds not tried yet: " + ", ".join(cov["untried_kinds"]))
        if cov["exhausted_kinds"]:
            lines.append("Exhausted (3+ tries, nothing accepted), propose only a clearly new idea: "
                         + ", ".join(cov["exhausted_kinds"]))
        if cov["untouched_columns"]:
            more = cov["n_untouched_columns"] - len(cov["untouched_columns"])
            lines.append("Columns no experiment has used yet: " + ", ".join(cov["untouched_columns"])
                         + (f" (+{more} more)" if more else ""))
        tail = "\n".join(lines)
        if len(tail) > max_chars // 3:
            tail = tail[: max_chars // 3 - 3] + "..."
        if not tail:
            return self.summary(max_chars)
        return self.summary(max_chars - len(tail) - 1) + "\n" + tail

    def summary(self, max_chars: int = 1500, recent: int = 8) -> str:
        """Сводка для промпта. Секции по приоритету, при нехватке места режется низший:
        принятые шаги (с диффом) > итог по видам > последние эксперименты > «уже пробовали» > заметки критика.
        Принятые шаги и самый свежий эксперимент не выкидываются; длина не больше max_chars."""
        if not self.experiments and not self.notes:
            return "No experiments yet."
        accepted = [e for e in self.experiments if e.status == "accepted"]
        others = [e for e in self.experiments if e.status != "accepted"]
        latest = others[-recent:] if recent > 0 else []
        older = others[: len(others) - len(latest)]
        diffs = {e.id: _diff_preview(e.diff) for e in accepted}
        by_kind = self.kind_stats() if len(self.experiments) > recent else []
        tried = [f"{_one_line(e.hypothesis, 50)} ({e.status})" for e in older]
        notes = list(self.notes)

        def render() -> str:
            blocks = []
            if accepted:
                lines = ["Accepted steps (already in the current best):"]
                for exp in accepted:
                    lines += [_format(exp), *("    " + line for line in diffs[exp.id])]
                blocks.append("\n".join(lines))
            if by_kind:
                blocks.append("By kind: " + "; ".join(by_kind))
            if latest:
                blocks.append("\n".join(["Recent experiments (oldest first):", *map(_format, latest)]))
            if tried:
                blocks.append("Already tried, do not repeat: " + "; ".join(tried))
            if notes:
                blocks.append("\n".join(["Critic notes:", *(f"- {note}" for note in notes)]))
            return "\n".join(blocks)

        text = render()
        while len(text) > max_chars:
            if notes:
                notes.pop(0)
            elif tried:
                tried.pop(0)
            elif len(latest) > 1:
                latest.pop(0)
            elif any(diffs.values()):
                diffs[next(i for i, lines in diffs.items() if lines)] = []
            elif by_kind:
                by_kind.clear()
            else:
                break
            text = render()
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def dump_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for exp in self.experiments:
                row = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                       for k, v in asdict(exp).items()}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load_jsonl(cls, path: str | Path, columns: Iterable[str] = ()) -> "ExperimentMemory":
        """Обратно к dump_jsonl; старый формат (без новых полей) и битые строки не мешают."""
        memory = cls(columns)
        path = Path(path)
        if not path.exists():
            return memory
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                exp = Experiment(**{k: v for k, v in row.items() if k in _FIELDS})
                memory.add(exp)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return memory
