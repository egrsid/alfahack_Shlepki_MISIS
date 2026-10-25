# alfahack — AI-агент для разработки ML-моделей (Альфа-Банк × Changellenge, трек AutoML)

Агент получает `task.md` и CSV задачи (`train.csv`, `test.csv`, `sample_submission.csv`, иногда `data_dictionary.csv`),
сам строит модель и пишет `submission.csv` рядом с `task.md`. Запуск на платформе:

```sh
python3 solution.py task.md
```

Ключ и адрес LLM — только из переменных окружения (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`); без них
решение работает без LLM. Общий план команды, правила и ограничения — `CLAUDE.md`.

## Как устроено решение (`solution.py`)

1. **Понимание задачи** (`task_understanding.py`): таргет и id — из `sample_submission.csv`, метрика — из `task.md`,
   плюс проверенные подсказки LLM (даты, текст, утечки).
2. **Fallback** (`fallback.py`): честная схема CV по данным (периоды / группы дублей / случайная), LightGBM, XGBoost,
   RandomForest, CatBoost, признаки повторяющейся сущности, бленд, веса классов под F1, усреднение по сидам.
   `submission.csv` пишется в первые секунды и улучшается дальше.
3. **AutoGluon** (`automl.py`): по умолчанию выключен (проигрывал fallback; включение — `SOLUTION_AUTOML_SECONDS`).
4. **Агент** (`agent.py`): LLM предлагает гипотезы и пишет код признаков; код исполняется в песочнице; оценка — на рецепте
   лучшего кандидата fallback с новыми колонками (`agent_eval.RecipeEvaluator`), принятие — только при выигрыше больше шума;
   в конце лучший кандидат доводится до тех же 3 сидов, что итог fallback.
5. **Бленд и запись** (`ensemble.py`, `guards.py`): честный бленд всех моделей; файл переписывается, только если
   кандидат лучше уже записанного по парным фолдам и прошёл проверки прогноза. Оценки честные с обеих сторон: у бленда
   fallback веса для фолда подбираются на остальных фолдах (`fallback.honest_view`).

## Структура

Корень — ровно то, что уходит в ZIP (платформа копирует только корневые `.py` и `requirements.txt`).

| Файл | Роль | Владелец |
|---|---|---|
| `solution.py` | точка входа, порядок этапов, время | P1 (черновик P4) |
| `fallback.py`, `preprocess.py`, `metrics.py`, `submission.py` | ML-ядро: CV, модели, метрики, запись сабмита | P3 |
| `llm_client.py` | клиент Qwen: ретраи, кэш, учёт токенов, бюджет | P2 |
| `task_understanding.py` | понимание задачи | P2 (написано P4) |
| `sandbox.py` | песочница для кода LLM | P5 |
| `guards.py` | проверка кандидата перед записью | P5 (написано P4) |
| `tuning.py`, `ensemble.py`, `automl.py` | optuna, бленд, AutoGluon | P1 (написано P4) |
| `agent.py`, `agent_eval.py`, `prompts_agent.py`, `memory.py`, `error_analysis.py`, `postprocess.py` | агент | P4 |
| `requirements.txt` | зависимости платформы (не менять) | организаторы |

| Папка | Что там |
|---|---|
| `tools/` | вне ZIP: `make_zip.py` (сборка + проверки + `--smoke`), `package.py` (сборка из распакованного архива), `make_datasets.py` + `run_all.py` (тестовые задачи и прогон ZIP), `try_agent.py`, `baseline.py` (организаторов), `check_llm.py` |
| `tests/` | pytest всех модулей, без LLM |
| `docs/plans/` | план и итоги каждой ветки; `integration-submission.md` — сборка и замеры |
| `docs/submissions.md` | таблица сабмитов на платформу |

## Как проверить

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
LLM_DISABLED=1 .venv/bin/python -m pytest tests/ -q             # все тесты, без LLM
.venv/bin/python tools/make_zip.py --smoke                        # ZIP в dist/ + дымовой прогон
.venv/bin/python tools/make_datasets.py                           # 12 тестовых задач → .agent_work/datasets/
.venv/bin/python tools/run_all.py dist/solution_<коммит>.zip      # прогон ZIP на них (тяжело для ноутбука)
```

Данные public-задач (`Регрессия/`, `Многоклассовая классификация/`), `.env`, `dist/`, `.agent_work/` в git не идут.

## Git

Ветки `feature/<имя>` от свежего `main`, в `main` — через PR. Не коммитить ключи, `.env`, `*.csv`, `*.zip`.
Подробно — `CLAUDE.md`, раздел «Git-процесс».
