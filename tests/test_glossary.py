import unittest
from unittest.mock import patch

from src.agent.nodes import (
    GENERATE_SQL_SYSTEM,
    REPAIR_SQL_SYSTEM,
    _GLOSSARY_LINE,
    node_load_schema,
    repair_prompt_guard,
    repair_prompt_sqlite,
)
from src.db.schema import get_schema_text
from src.glossary import render_glossary


class GlossaryTest(unittest.TestCase):
    def test_render_contains_spend_formula_without_database(self) -> None:
        text = render_glossary()
        self.assertIn("SUM da medida de valor na fato de compras", text)
        self.assertIn("clientes.valor_total_gasto", text)

    def test_schema_text_has_no_snapshot_note(self) -> None:
        text, _ = get_schema_text()
        self.assertNotIn("SNAPSHOT", text)
        self.assertNotIn("NÃO usar para gasto", text)

    def test_load_schema_appends_glossary(self) -> None:
        with patch("src.agent.nodes.get_schema_text", return_value=("Table: x", "abc")):
            out = node_load_schema({"trace": []})
        self.assertTrue(out["schema_text"].startswith("Table: x"))
        self.assertIn("## Glossário", out["schema_text"])
        self.assertIn("SUM da medida de valor na fato de compras", out["schema_text"])

    def test_prompts_only_conduct(self) -> None:
        self.assertIn(_GLOSSARY_LINE, GENERATE_SQL_SYSTEM)
        self.assertIn(_GLOSSARY_LINE, REPAIR_SQL_SYSTEM)
        self.assertIn("lado A", GENERATE_SQL_SYSTEM)
        self.assertIn("lado B", GENERATE_SQL_SYSTEM)
        for prompt in (GENERATE_SQL_SYSTEM, REPAIR_SQL_SYSTEM):
            self.assertNotIn("compras_2025", prompt)
            self.assertNotIn("tickets_nao_resolvidos", prompt)
            self.assertNotIn("valor_total_gasto", prompt)
            self.assertNotIn("SEMPRE têm SQL", prompt)


class RepairPromptTest(unittest.TestCase):
    def test_guard_door_omits_failed_sql(self) -> None:
        state = {
            "question": "quanto gastou",
            "schema_text": "Table: compras",
            "sql": "SELECT segredo_barrado FROM compras JOIN suporte",
            "sql_error": "JOIN entre fatos",
            "reflection": "separar as fatos",
            "error_source": "guard",
        }
        text = repair_prompt_guard(state)
        self.assertNotIn("SQL que falhou", text)
        self.assertNotIn("segredo_barrado", text)

    def test_sqlite_door_keeps_failed_sql(self) -> None:
        state = {
            "question": "por mês",
            "schema_text": "Table: compras",
            "sql": "SELECT ORDER BY antes",
            "sql_error": "ORDER BY clause should come after UNION",
            "reflection": "envolver cada lado",
            "error_source": "sqlite",
        }
        text = repair_prompt_sqlite(state)
        self.assertIn("SQL que falhou", text)
        self.assertIn("SELECT ORDER BY antes", text)
        self.assertIn("envolver cada lado", text)

    def test_identical_sql_stops(self) -> None:
        from src.agent.nodes import node_repair_sql

        state = {
            "question": "q",
            "schema_text": "s",
            "sql": "SELECT 1",
            "sql_error": "near UNION",
            "retries": 1,
            "error_source": "sqlite",
            "reflection": "diag",
            "trace": [],
        }

        class _Resp:
            content = "select 1"

        class _LLM:
            def invoke(self, messages: list) -> _Resp:
                return _Resp()

        with patch("src.agent.nodes.get_chat_model", return_value=_LLM()):
            out = node_repair_sql(state)
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["retries"], 2)
