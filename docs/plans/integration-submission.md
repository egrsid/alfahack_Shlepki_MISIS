# Ветка `integration/submission` — сборка сабмита из веток команды (P4, @egrsid)

> Локальная ветка интеграции: `feature/agent-models` (P4) + `origin/main` (P5 `sandbox.py`) +
> `origin/feature/fallback-pipeline` (P3, включает `metrics-submission`) + `origin/feature/llm-client` (P2).
> P1 (сборка, `solution.py`, сабмиты) своей работы не выложил — `solution.py` здесь черновик для команды.

## Что в ZIP (`tools/make_zip.py` → `dist/solution_<коммит>.zip`, ~100 КБ, 14 файлов в корне)

| Файл | Владелец | Роль |
|---|---|---|
| `solution.py` | P1 (черновик P4) | точка входа: fallback → агент → проверка файла |
| `fallback.py`, `preprocess.py`, `metrics.py`, `submission.py` | P3 | честная схема CV, бустинги, бленд, веса классов, запись и проверка сабмита |
| `llm_client.py` | P2 | клиент Qwen: ретраи, кэш, учёт токенов, бюджет → `LLMUnavailable` |
| `sandbox.py` | P5 | песочница для кода LLM |
| `agent.py`, `agent_eval.py`, `prompts_agent.py`, `memory.py`, `error_analysis.py`, `postprocess.py` | P4 | агент |
| `requirements.txt` | P1 | зависимости платформы (не менялся) |

## Как работает `solution.py`

1. `fallback.guess_spec(task.md)` → `fallback.run(spec, deadline)`: сабмит-константа сразу, потом после каждой
   лучшей модели (бюджет fallback — 900 с, как настроил P3).
2. Агент (если есть `OPENAI_API_KEY` и не `LLM_DISABLED=1`; окно — до T+2ч30, 3 часа — лимит платформы):
   `LocalEvaluator` на **фолдах и строках кандидата fallback** (`meta["folds"]`, `meta["train_rows"]`) и с его
   **обучением моделей** (параметры, ранняя остановка) — CV моделей совпадает с fallback до 4 знаков.
   LLM — `llm_client.LLMClient` через адаптер `AgentLLM`: без `chat_json` (на плохом JSON клиент бросает
   `LLMUnavailable`, что остановило бы агента), HTTP 400/422 — исходной ошибкой (агент повторит без `extra_body`).
3. Кандидат агента пишется, только если `fallback.clearly_better` (выигрыш по парным фолдам > SE) против того, что
   fallback уже записал (с весами классов). Иначе остаётся fallback.
4. `submission.check_file`; испорчен — переписывается кандидатом fallback, в крайнем случае константой.

## Проверено

- Все тесты команды вместе: 375 (P2, P3, P4, P5 + `tests/test_solution.py`), без LLM.
- `tests/test_solution.py`: только fallback; агент честно лучше → пишет; агент не лучше / падает → fallback;
  нет окна → агент не запускается; адаптер LLM; ZIP плоский и запускается в чистой папке (`python3 solution.py task.md`).
- Прогон ZIP на public-задачах без LLM — см. «Статус».

## Как проверить самому (живой Qwen тратит общий бюджет)

```sh
.venv/bin/python tools/make_zip.py                                   # собрать
mkdir /tmp/t && unzip dist/solution_*.zip -d /tmp/t && cp "Регрессия"/{*.csv,task.md} /tmp/t
cd /tmp/t && LLM_DISABLED=1 python3 solution.py task.md               # только fallback
# короткий прогон агента: ключ в окружении, 3 итерации, окно до T+25 мин
cd /tmp/t && SOLUTION_AGENT_ITERS=3 SOLUTION_RESEARCH_END=1500 python3 solution.py task.md
```

## Известные ограничения

- Живой прогон агента в составе ZIP не делался (запуск с ключом из `.env` требует подтверждения пользователя).
- Мульти-таргет — только fallback (`run_multi`), агент пропускается.
- Агенту трудно обогнать fallback: тот уже делает бленд трёх бустингов и веса классов; агент начинает с одного
  LightGBM на своих признаках. Выигрыш возможен за счёт новых признаков, выбора модели и постобработки.

## Статус

Прогон ZIP на public-задачах **без LLM** (распакован в чистую папку рядом с данными, `python3 solution.py task.md`,
Mac, 8+ ядер):

| задача | время | схема CV | итог fallback | файл |
|---|---|---|---|---|
| регрессия, RMSE ↓ | 173 с | по годам | бленд xgboost 0.75 + lightgbm без года 0.25: **43.29** | 22 449 строк, id по порядку, без NaN — ok |
| мультикласс, macro F1 ↑ | 486 с | группы дублей | бленд balanced-lightgbm + xgboost с весами классов: **0.649** | 41 166 строк, классы ⊆ train — ok |

Числа совпадают с отчётом P3 (`docs/plans/feature-fallback-pipeline.md`, v4). Агент с живым Qwen внутри ZIP не
запускался (см. ограничения); с MockLLM — `tests/test_solution.py`.
