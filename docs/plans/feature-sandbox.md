# План ветки `feature/sandbox` (P5)

## Цель

Безопасный запуск Python-кода, сгенерированного LLM (P4) или взятым из других
источников, с гарантией: жёсткий таймаут, убийство всего дерева процессов,
отсутствие секретов в окружении, лимит вывода, передача результата через файл.

## Контракт (зафиксирован в CLAUDE.md, менять только через согласие всех)

```python
sandbox.run_code(code, cwd, timeout, memory_limit) -> ExecResult
```

`ExecResult(stdout, stderr, returncode, artifacts)` — dataclass из `sandbox.py`.

## Входы / выходы

- Вход: строка с произвольным Python-кодом, рабочий каталог (подкаталог задачи или `/tmp`).
- Выход: `ExecResult` — stdout/stderr (усечённые до лимита), код возврата,
  флаг таймаута, распарсенный результат из файла, список созданных артефактов.

## Шаги

1. [x] `sandbox.py`: `run_code` + `ExecResult`.
   - отдельный процесс `sys.executable -I -u -c <code>` (isolated mode: без `PYTHON*` env, без user site);
   - `preexec_fn`: `os.setsid()` (своя группа процессов) + `resource.setrlimit(RLIMIT_AS)` при заданном `memory_limit_mb`;
   - таймаут: мягкое завершение дерева процессов (`psutil`, recursive children),
     затем `SIGKILL` всей группе; процесс не может пережить sandbox;
   - env — белый список (PATH, LANG, TMPDIR, PYTHONIOENCODING, OMP_NUM_THREADS и т.п.),
     всё содержащее `KEY/TOKEN/SECRET/PASS` — вырезано принудительно;
   - stdout/stderr усечены до `max_output_chars`;
   - результат: код пишет JSON в путь из env `SANDBOX_RESULT` → sandbox парсит и отдаёт в `ExecResult.result`;
   - артефакты: файлы, созданные в `cwd`, кроме result-файла.
2. [x] `tests/test_sandbox.py`: юнит-тесты (без LLM, `LLM` не требуется).
3. [ ] PR в `main`, ревью — P4 (первый клиент) или P1.

## Как проверить

```sh
.venv/bin/python -m pytest tests/test_sandbox.py -v
```

## Известные ограничения и риски

- **Сеть и pip не отсекаются процессно** — на платформе это не страшно (нет
  интернета кроме LLM-прокси), локально доверяем честности кода. Полная изоляция —
  только Docker (этап `feature/docker-check`).
- `memory_limit_mb` через `RLIMIT_AS` — best-effort: на Linux (платформа)
  срабатывает надёжно, на macOS отдельные numpy-операции могут падать иначе.
  По умолчанию лимит не задаётся.
- Код внутри песочницы может читать файлы каталога задачи — это допустимо
  (agent пишет туда фичи), запрет на чтение таргета test — зона `guards.py`.
- Дочерние процессы, отсоединившиеся через double-fork, теоретически могут
  уцелеть; практически `setsid` + killpg + recursive kill закрывают сценарии LLM-кода.

## Зависимости

Только `psutil` (есть в `requirements.txt`) + стандартная библиотека.
