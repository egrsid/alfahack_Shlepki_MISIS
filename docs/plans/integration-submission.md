# Ветка `integration/submission` — сборка сабмита из веток команды (P4, @egrsid)

> Локальная ветка интеграции: `feature/agent-models` (P4) + `origin/main` (P5 `sandbox.py`) +
> `origin/feature/fallback-pipeline` (P3, включает `metrics-submission`, по `1cde55c`) + `origin/feature/llm-client` (P2).
> Модулей P1 (`tuning`, `ensemble`, `solution.py`), P2 (`task_understanding`) и P5 (`guards`, тестовые задачи) ни в
> одной ветке не было — они написаны здесь P4, **владельцы могут переписать** (контракты — из плана команды в `main`).

## Что в ZIP (`tools/make_zip.py` → `dist/solution_<коммит>.zip`, ~115 КБ, 18 файлов в корне)

| Файл | Владелец | Роль |
|---|---|---|
| `solution.py` | P1 (черновик P4) | точка входа: понимание задачи → fallback → агент → бленд → guards → проверка файла |
| `fallback.py`, `preprocess.py`, `metrics.py`, `submission.py` | P3 | честная схема CV, бустинги, бленд, веса классов, запись и проверка сабмита |
| `llm_client.py` | P2 | клиент Qwen: ретраи, кэш, учёт токенов, бюджет → `LLMUnavailable` |
| `task_understanding.py` | P2 (написано P4) | `understand_task`: эвристики + проверенные подсказки LLM |
| `sandbox.py` | P5 | песочница для кода LLM |
| `guards.py` | P5 (написано P4) | `check_candidate`: сверка кандидата с fallback перед записью |
| `tuning.py` | P1 (написано P4) | `tune`: optuna по контракту агента |
| `ensemble.py` | P1 (написано P4) | `blend`: Caruana по OOF fallback + агента |
| `agent.py`, `agent_eval.py`, `prompts_agent.py`, `memory.py`, `error_analysis.py`, `postprocess.py` | P4 | агент |
| `requirements.txt` | P1 | зависимости платформы (не менялся) |

## Как работает `solution.py`

0. LLM-клиент (если есть `OPENAI_API_KEY` и не `LLM_DISABLED=1`) → `task_understanding.understand_task`:
   `fallback.guess_spec` (таргет и id — из образца, метрика — из `task.md`) + заметки `data_dictionary.csv` + один
   короткий запрос к LLM. Из ответа берётся только проверенное по данным: метрика (лишь если в `task.md` её не нашли),
   колонки дат (парсятся ≥ 90%), текста (длина ≥ 20, ≥ 3 слов), строковых категорий, утечек (≤ 3 и упомянуты в
   `task.md`). Любая ошибка — эвристики.
1. `fallback.run(spec, deadline)`: сабмит-константа сразу, потом после каждой лучшей модели (бюджет fallback — 900 с).
2. Агент (окно — до T+2ч30): `LocalEvaluator` на **фолдах и строках кандидата fallback** и с его обучением моделей.
   Вид гипотезы `tune` теперь работает: `tuning.tune` подбирает параметры тем же evaluator на первых 3 фолдах.
   LLM — `llm_client.LLMClient` через адаптер `AgentLLM`.
3. `pick`: кандидаты агента + `ensemble.blend([fallback, *агент])`; пишется лучший по CV из тех, что
   `fallback.clearly_better` (выигрыш по парным фолдам > SE) **и** прошли `guards.check_candidate` против fallback.
   Иначе остаётся fallback.
4. `submission.check_file`; испорчен — переписывается кандидатом fallback, в крайнем случае константой.

## Проверено

- Все тесты вместе без LLM: 400 (P2, P3, P4, P5, `tests/test_solution.py`, новые `test_tuning`, `test_ensemble`,
  `test_task_understanding`, `test_guards`). Тесты агента изолированы от настоящего `tuning.py` (conftest).
- `tools/make_datasets.py` + `tools/run_all.py`: 12 синтетических задач + 2 public, ZIP распакован в чистую папку,
  `python3 solution.py task.md` без LLM — **14/14 корректный файл**, все лучше константы (таблица в «Статусе»).

## Как проверить самому

```sh
.venv/bin/python tools/make_zip.py                                        # собрать
.venv/bin/python tools/make_datasets.py                                   # задачи → .agent_work/datasets/
# public-задачи в тот же формат (ответов нет — только проверка файла):
mkdir -p .agent_work/datasets/public_reg/task .agent_work/datasets/public_cls/task
cp Регрессия/{*.csv,task.md} .agent_work/datasets/public_reg/task/
cp "Многоклассовая классификация"/{*.csv,task.md} .agent_work/datasets/public_cls/task/
.venv/bin/python tools/run_all.py dist/solution_<коммит>.zip              # без LLM, ~20 мин
# короткий живой прогон агента (тратит общий бюджет): ключ в окружении, 4 итерации, окно до T+25 мин
set -a; source .env; set +a
SOLUTION_AGENT_ITERS=4 SOLUTION_RESEARCH_END=1500 \
  .venv/bin/python tools/run_all.py dist/solution_<коммит>.zip --only public_reg,public_cls --llm
# логи: .agent_work/runs/<задача>/run.log
```

## Известные ограничения

- Живой прогон агента с Qwen внутри ZIP ещё не делался (ключ только в `.env`; команда — выше).
- Docker-проверка (Linux, `--network none`) не делалась — Docker не установлен; остаётся за P5.
- Мульти-таргет — только fallback (`run_multi`), агент и понимание задачи через LLM пропускаются.
- `tuning` выбирает параметры на тех же фолдах, на которых агент их потом оценивает, — оценка оптимистична;
  защита — правило принятия по парным фолдам и `clearly_better` в `solution.py`.
- `ensemble` подбирает веса на тех же OOF — так же, как бленд fallback; решение — `clearly_better`.
- `guards` сверяет с fallback, а не с ответами: при сильном сдвиге train/test честный кандидат тоже может
  отличаться от fallback; пороги консервативные (TV долей классов 0.2, корреляция регрессии 0.5).

## Статус

Прогон ZIP `3699e62` **без LLM** (`tools/run_all.py`, Mac, 10 потоков):

| задача | метрика | время | итог | константа |
|---|---|---|---|---|
| public регрессия | RMSE ↓ (CV) | 174 с | **41.37** (было 43.29: признаки сущности P3 `a594aea`) | — |
| public мультикласс | Macro F1 ↑ (CV) | 560 с | **0.649** (бленд + веса классов) | — |
| bin_auc (рак груди) | ROC-AUC ↑ | 4 с | 0.9985 | 0.5 |
| bin_f1_str (таргет yes/no, 9:1) | F1 ↑ | 6 с | 0.769 | 0 |
| digits_weighted (10 классов) | weighted F1 ↑ | 93 с | 0.979 | 0.015 |
| multi_imbalance (6 классов, 100:1, 800 уровней) | Macro F1 ↑ | 78 с | 0.824 | 0.118 |
| multi_target (2 таргета) | RMSE ↓ | 21 с | 25.2 | 150.1 |
| reg_mae_en (442 строки, англ.) | MAE ↓ | 2 с | 43.7 | 65.1 |
| reg_mape_period (новый год в test) | MAPE ↓ | 15 с | 0.048 | 0.097 |
| reg_no_id (без id) | R² ↑ | 12 с | 0.983 | 0.0 |
| reg_rmsle_skewed (дата, текст, пропуски) | RMSLE ↓ | 14 с | 0.338 | 0.782 |
| small_wine (178 строк, без словаря) | accuracy ↑ | 7 с | 0.979 | 0.447 |
| text_cls (свободный текст) | accuracy ↑ | 7 с | 0.900 | 0.509 |
| wide_logloss (300 признаков) | logloss ↓ | 84 с | 0.204 | 0.693 |

Агент с живым Qwen внутри ZIP не запускался; с MockLLM — `tests/test_solution.py`.
