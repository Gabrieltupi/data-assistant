import unittest

from src.db.executor import detect_fact_fanout

FACTS = {"compras", "suporte", "campanhas_marketing"}


class FactFanoutTest(unittest.TestCase):
    def test_blocks_join_two_facts(self) -> None:
        sql = """
        SELECT AVG(compras.valor) FROM suporte
        JOIN clientes ON suporte.cliente_id = clientes.id
        JOIN compras ON clientes.id = compras.cliente_id
        """
        self.assertIsNotNone(detect_fact_fanout(sql, FACTS))

    def test_blocks_union_that_still_joins_two_facts(self) -> None:
        sql = """
        SELECT AVG(compras.valor) FROM suporte
        JOIN clientes ON suporte.cliente_id = clientes.id
        JOIN compras ON clientes.id = compras.cliente_id
        WHERE suporte.resolvido = 1
        UNION ALL
        SELECT AVG(compras.valor) FROM suporte
        JOIN clientes ON suporte.cliente_id = clientes.id
        JOIN compras ON clientes.id = compras.cliente_id
        WHERE suporte.resolvido = 0
        """
        self.assertIsNotNone(detect_fact_fanout(sql, FACTS))

    def test_allows_union_one_fact_per_side(self) -> None:
        sql = """
        SELECT 'compra' AS tipo, valor FROM compras
        UNION ALL
        SELECT 'ticket' AS tipo, NULL FROM suporte
        """
        self.assertIsNone(detect_fact_fanout(sql, FACTS))

    def test_allows_single_fact(self) -> None:
        sql = "SELECT resolvido, COUNT(*) FROM suporte GROUP BY resolvido"
        self.assertIsNone(detect_fact_fanout(sql, FACTS))

    def test_blocks_unaggregated_subquery_join(self) -> None:
        sql = "SELECT * FROM suporte JOIN (SELECT * FROM compras) c"
        self.assertIsNotNone(detect_fact_fanout(sql, FACTS))

    def test_allows_aggregated_subquery_join(self) -> None:
        sql = """
        SELECT * FROM clientes
        JOIN (
            SELECT cliente_id, SUM(valor) FROM compras GROUP BY cliente_id
        ) c ON clientes.id = c.cliente_id
        """
        self.assertIsNone(detect_fact_fanout(sql, FACTS))

    def test_blocks_unaggregated_cte_join(self) -> None:
        sql = "WITH c AS (SELECT * FROM compras) SELECT * FROM c JOIN suporte"
        self.assertIsNotNone(detect_fact_fanout(sql, FACTS))

    def test_allows_not_in_other_fact(self) -> None:
        sql = """
        SELECT * FROM compras
        WHERE cliente_id NOT IN (SELECT cliente_id FROM suporte)
        """
        self.assertIsNone(detect_fact_fanout(sql, FACTS))


if __name__ == "__main__":
    unittest.main()
