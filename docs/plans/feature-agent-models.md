# План ветки `feature/agent-models` (P4, @egrsid)

> Ветка от `feature/agent-strategies` (`7ebe7a9`). Ниже, в разделе «Статус выполнения», — что сделано, как проверить
> и известные ограничения.

## 1. Зачем

Разбор кода показал четыре слабых места агента:
1. **Модель почти не выбирается.** Старт всегда LightGBM (`BASE_CODE`); другая модель пробуется, только если LLM
   сама предложит гипотезу `model`; `tune` подбирает параметры лишь текущей модели.
2. **В боевом запуске агент ничего не делает:** без `evaluator.py` P1 (его пока нет ни в одной ветке) `research()`
   возвращает `[]`. Эффект агента виден только в `tools/try_agent.py` со `StubEvaluator` из `tests/`.
3. **Macro F1 не дожимается постобработкой:** класс = argmax вероятностей; при recall Backdoor 0.11 / DoS 0.17
   подбор множителей вероятностей по классам обычно даёт несколько пунктов F1 без LLM.
4. **Текстовые колонки** превращаются только в длину и число слов. Предобученные эмбеддинги запрещены правилами
   («предобученные модели нельзя»), поэтому — TF-IDF + SVD, обученные на данных задачи.

Решения: перебор только 3 бустингов (контракт `evaluator` с P1 не меняется); запасной evaluator, пороги под F1 и
TF-IDF — в этой ветке. Семантическая разметка категорий через Qwen — не здесь.

Вне кода (сообщить команде): в `requirements.txt` есть и установлены AutoGluon / FLAML / LightAutoML — кандидат на
сильный fallback для P3/P1.

## 2. Шаги (каждый — отдельный коммит, все тесты зелёные после каждого)

0. Этот файл и раздел ветки в `CLAUDE.md`.
1. **Свой evaluator агента — `agent_eval.py`.** `StubEvaluator` и подсчёт метрик переезжают из `tests/mocks.py` в
   плоский модуль как `LocalEvaluator` (тот же контракт `(spec, build_features, cfg, deadline) -> Candidate`):
   поиск файлов задачи (пути в `task.md` вложенные), `meta["classes"]`, остановка по дедлайну между фолдами,
   все ядра. `research()`: `evaluate` → `evaluator.evaluate` P1 → `LocalEvaluator` (если `AgentConfig.local_evaluator`).
   `tests/mocks.StubEvaluator` — подкласс с прежними аргументами.
2. **Вид гипотезы `model_search`** (по образцу `tune`): признаки текущего лучшего кода строятся один раз, каждая
   другая модель из `MODELS` оценивается тем же evaluator в пределах бюджета (дорогие модели пропускаются по оценке
   времени), лучший кандидат идёт в обычное правило принятия без повторной оценки. Общие параметры переносятся
   между моделями (`n_estimators ↔ iterations`, `class_weight ↔ auto_class_weights`, `max_depth ↔ depth`).
   Авто-запуск перед остановкой: сначала `model_search`, потом `tune` (тюнить уже выбранную модель).
3. **Пороги под F1 — `postprocess.py`.** Множители вероятностей по классам максимизируют метрику на
   `argmax(p · w)` (координатный подъём по лог-сетке). Только метрики по меткам (`macro_f1`, `f1`, `weighted_f1`,
   `accuracy`, ...). Честная оценка: веса подбираются на K−1 частях OOF и проверяются на оставшейся; решение —
   существующий `decide()` по парным фолдам. В конце исследования применяется к текущему лучшему.
4. **Текст — TF-IDF + SVD** в `BASE_CODE` для `spec["text_cols"]`; эвристика текстовых колонок, пока нет
   `task_understanding` P2; строка про текст в промптах; синтетическая текстовая задача в моках.
5. **Замеры и статус:** dry-run обеих public-задач без передачи evaluator (агент на своём), время `model_search`,
   выигрыш постобработки, размер промпта ≤ 4k токенов.

## 3. Контракты

```python
# agent_eval.py (P4)
LocalEvaluator(n_splits=5, seed=42, n_estimators=300, learning_rate=0.05, threads=None)
    (spec, build_features, cfg: {"model", "params", "target_transform"}, deadline) -> Candidate
    # meta: fold_scores, classes (порядок столбцов вероятностей), model, params, target_transform, n_splits, seed
score(metric, y, pred, task_type) -> float            # ValueError на неизвестной метрике
find_file(spec, name) -> Path

# postprocess.py (P4)
LABEL_METRICS                                          # метрики, которые считаются по argmax
fit_class_weights(y, proba, metric, n_splits=5, seed=42) -> PostResult | None
apply_weights(proba, weights) -> np.ndarray            # перевзвешенные и нормированные вероятности
```

Для P1/P3: у кандидата с `meta["postprocess"]` `oof`/`test_pred` — уже перевзвешенные вероятности, класс в сабмите =
argmax. Настоящий `evaluator.evaluate` P1 приоритетнее `LocalEvaluator`; от него агенту нужны `meta["fold_scores"]` и
`meta["classes"]`.

## 4. Как проверять

```sh
LLM_DISABLED=1 .venv/bin/python -m pytest tests/ -q
.venv/bin/python tools/try_agent.py "Многоклассовая классификация/task.md" --dry-run
.venv/bin/python tools/try_agent.py "Регрессия/task.md" --dry-run
```

## 5. Definition of Done

- Все прежние тесты зелёные; без новых стратегий (флаги `AgentConfig`) агент ведёт себя как раньше.
- Агент работает без `evaluator.py` P1 и по-прежнему не трогает `submission.csv`; с `local_evaluator=False` — `[]`.
- `model_search`: принят/отклонён/модель не влезла по времени/перенос параметров — покрыто тестами.
- Постобработка улучшает F1 на синтетике с дисбалансом, не применяется к RMSE/AUC/logloss, на шуме не принимается.
- TF-IDF-признаки появляются для текстовых колонок, train/test согласованы.
- Новые модули плоские, импортируются без модулей команды; sklearn и бустинги — только внутри функций.

## 6. Риски

| Риск | Мера |
|---|---|
| `model_search` съедает окно (catboost на мультиклассе медленный) | бюджет `min(seconds, share · остаток)`, пропуск модели по оценке времени, проверка дедлайна между фолдами |
| Пороги переобучаются под OOF | подбор на K−1 частях, проверка на оставшейся, тот же порог шума |
| Свой evaluator расходится с P1 | P1 приоритетнее; контракт тот же; сплиты/сид фиксированы внутри одного запуска |
| TF-IDF тяжёлый на длинных текстах | `max_features`, `min_df`, SVD до 20 компонент, таймаут песочницы |
