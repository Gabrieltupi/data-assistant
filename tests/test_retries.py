import unittest
from unittest.mock import patch

from src.agent.graph import _route_after_execute, _route_after_guard
from src.agent.nodes import _guard_repair, node_execute_sql, node_repair_sql
from src.config import settings


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _LLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, messages: list) -> _Resp:
        return _Resp(self.content)


class RetryCounterTest(unittest.TestCase):
    def test_three_repairs_enter_the_fourth_does_not(self) -> None:
        self.assertEqual(settings.sql_max_retries, 3)
        for retries in (0, 1, 2):
            state = {"status": "reflect_needed", "retries": retries}
            self.assertEqual(_route_after_guard(state), "reflect_sql")
            state = {"status": "exec_error", "retries": retries}
            self.assertEqual(_route_after_execute(state), "reflect_sql")
        blocked = {"status": "reflect_needed", "retries": 3}
        self.assertEqual(_route_after_guard(blocked), "fail_graceful")
        exhausted = {"status": "exec_error", "retries": 3}
        self.assertEqual(_route_after_execute(exhausted), "fail_graceful")

    def test_success_leaves_retries_at_zero(self) -> None:
        state = {"sql": "SELECT 1", "retries": 0, "trace": []}
        with patch("src.agent.nodes.run_sql", return_value=([{"a": 1}], None)):
            out = node_execute_sql(state)
        self.assertEqual(out["status"], "success")
        self.assertEqual(out.get("retries", 0), 0)

    def test_guard_block_does_not_increment(self) -> None:
        state = {"retries": 0, "trace": []}
        out = _guard_repair(state, "SELECT 1", "fanout", "bloqueado")
        self.assertEqual(out.get("retries", 0), 0)
        self.assertEqual(out["error_source"], "guard")

    def test_repair_increments_once(self) -> None:
        state = {
            "question": "q",
            "schema_text": "schema",
            "sql": "SELECT 1",
            "sql_error": "erro",
            "retries": 0,
            "error_source": "sqlite",
            "reflection": "diag",
            "trace": [],
        }
        with patch("src.agent.nodes.get_chat_model", return_value=_LLM("SELECT 2")):
            out = node_repair_sql(state)
        self.assertEqual(out["retries"], 1)
        self.assertEqual(out["status"], "sql_ready")
