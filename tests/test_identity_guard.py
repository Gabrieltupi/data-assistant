import unittest

from src.db.executor import detect_identity_aggregation
from src.db.schema import _column_role


class ColumnRoleTest(unittest.TestCase):
    def test_pk_is_identity(self) -> None:
        self.assertEqual(_column_role("id", "INTEGER", True, set()), "identidade")

    def test_fk_is_identity(self) -> None:
        self.assertEqual(_column_role("cliente_id", "INTEGER", False, {"cliente_id"}), "identidade")

    def test_valor_is_measure(self) -> None:
        self.assertEqual(_column_role("valor", "REAL", False, set()), "medida")

    def test_nome_has_no_role(self) -> None:
        self.assertIsNone(_column_role("nome", "TEXT", False, set()))


class IdentityAggTest(unittest.TestCase):
    def test_blocks_avg_id(self) -> None:
        err = detect_identity_aggregation(
            "SELECT AVG(id) FROM suporte",
            identities={"id"},
            measures={"valor"},
        )
        self.assertIsNotNone(err)
        self.assertIn("identidade", err.lower())

    def test_allows_count_id(self) -> None:
        err = detect_identity_aggregation(
            "SELECT COUNT(id) FROM suporte",
            identities={"id"},
            measures={"valor"},
        )
        self.assertIsNone(err)

    def test_allows_avg_valor(self) -> None:
        err = detect_identity_aggregation(
            "SELECT AVG(valor) FROM compras",
            identities={"id", "cliente_id"},
            measures={"valor"},
        )
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
