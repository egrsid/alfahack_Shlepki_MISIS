import os
import sys
from pathlib import Path

# модули решения лежат плоско в корне репозитория
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# в тестах реальный Qwen не вызывается никогда; MockLLM помечен offline и этим флагом не отключается
os.environ.setdefault("LLM_DISABLED", "1")
# AutoGluon в solution.solve — только там, где тест включает его сам (SOLUTION_AUTOML_SECONDS)
os.environ.setdefault("SOLUTION_AUTOML_SECONDS", "0")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _agent_tests_without_real_tuning(request, monkeypatch):
    """Тесты агента — без настоящего tuning.py (optuna): им нужен «модуля нет» или StubTuner из моков."""
    if request.node.path.name.startswith("test_agent_"):
        monkeypatch.setitem(sys.modules, "tuning", None)
