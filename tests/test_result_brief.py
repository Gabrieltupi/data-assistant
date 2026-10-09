import unittest

from src.agent.result_brief import (
    ResultShape,
    _detect_shape,
    _to_df,
    build_narrate_context,
    inspect_result,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _rows_union_named() -> list[dict]:
    """UNION: compras 2025 + tickets não resolvidos — mesmo bug que ocorreu."""
    compras = [
        {"tipo": "compras 2025", "nome": f"Cliente {i}", "total_compras": 1000 * i}
        for i in range(1, 6)
    ]
    tickets = [
        {"tipo": "tickets_nao_resolvidos", "nome": f"Cliente {i}", "total_compras": i % 3 + 1}
        for i in range(1, 6)
    ]
    return compras + tickets


def _rows_ranking() -> list[dict]:
    return [
        {"nome": "João Pedro Castro", "total_gasto": 13938.27},
        {"nome": "Rafael Dias", "total_gasto": 13516.21},
        {"nome": "Enzo Nascimento", "total_gasto": 13184.55},
    ]


def _rows_distribution() -> list[dict]:
    return [
        {"canal": "App", "quantidade": 337},
        {"canal": "Loja Física", "quantidade": 313},
        {"canal": "Site", "quantidade": 296},
    ]


def _rows_time_series() -> list[dict]:
    return [
        {"mes": "2025-01", "total": 3200.0},
        {"mes": "2025-02", "total": 4100.0},
        {"mes": "2025-03", "total": 12400.0},
        {"mes": "2025-04", "total": 5000.0},
    ]


def _rows_scalar() -> list[dict]:
    return [{"total_clientes": 200}]


# ── testes de detecção de forma ───────────────────────────────────────────────

class ShapeDetectionTest(unittest.TestCase):
    def test_detects_multi_section(self) -> None:
        df = _to_df(_rows_union_named())
        self.assertEqual(_detect_shape(df, len(df)), ResultShape.MULTI_SECTION)

    def test_detects_ranking(self) -> None:
        df = _to_df(_rows_ranking())
        self.assertEqual(_detect_shape(df, len(df)), ResultShape.RANKED_LIST)

    def test_detects_distribution(self) -> None:
        df = _to_df(_rows_distribution())
        self.assertEqual(_detect_shape(df, len(df)), ResultShape.DISTRIBUTION)

    def test_detects_time_series(self) -> None:
        df = _to_df(_rows_time_series())
        self.assertEqual(_detect_shape(df, len(df)), ResultShape.TIME_SERIES)

    def test_long_series_does_not_suggest_month(self) -> None:
        rows = [{"mes": f"2024-01-{i:02d}", "total": 1000.0 + i * 17} for i in range(1, 31)]
        text = build_narrate_context(rows, len(rows))
        self.assertNotIn("agrupar", text.lower())
        self.assertNotIn("strftime", text.lower())
        self.assertIn("Total acumulado", text)
        self.assertIn("Média por ponto", text)

    def test_detects_scalar(self) -> None:
        df = _to_df(_rows_scalar())
        self.assertEqual(_detect_shape(df, 1), ResultShape.SCALAR)

    def test_detects_empty(self) -> None:
        df = _to_df([])
        self.assertEqual(_detect_shape(df, 0), ResultShape.EMPTY)

    def test_inspect_skips_sequential_id(self) -> None:
        rows = [{"nome": "A", "id": 1, "total": 10.0}, {"nome": "B", "id": 2, "total": 20.0}]
        info = inspect_result(rows)
        self.assertNotIn("id", info["numeric_cols"])
        self.assertIn("total", info["numeric_cols"])


# ── testes de conteúdo do briefing ────────────────────────────────────────────

class BriefContentTest(unittest.TestCase):
    def test_multi_section_without_tipo_splits_by_order(self) -> None:
        """SQL sem coluna tipo: nome repete 2x. Brief separa em duas listas."""
        rows = [
            {"nome": "Enzo Nascimento", "total_compras": 8021.78},
            {"nome": "Fernanda Carvalho", "total_compras": 6287.30},
            {"nome": "João Pedro Castro", "total_compras": 9598.63},
            {"nome": "Nicolas Ferreira", "total_compras": 5004.73},
            {"nome": "Rafael Dias", "total_compras": 7430.34},
            {"nome": "Enzo Nascimento", "total_compras": 1.0},
            {"nome": "Fernanda Carvalho", "total_compras": 1.0},
            {"nome": "João Pedro Castro", "total_compras": 1.0},
            {"nome": "Nicolas Ferreira", "total_compras": 3.0},
            {"nome": "Rafael Dias", "total_compras": 1.0},
        ]
        ctx = build_narrate_context(rows, len(rows))
        # Não deve dizer "2" como número de tickets
        self.assertNotIn("→ 2", ctx)
        # Deve mostrar os valores reais
        self.assertIn("9.598,63", ctx)
        self.assertIn("3,00", ctx)
        # Deve separar em listas
        self.assertIn("Lista 1", ctx)
        self.assertIn("Lista 2", ctx)

    def test_ranking_shows_position_and_value(self) -> None:
        ctx = build_narrate_context(_rows_ranking(), 3)
        self.assertIn("João Pedro Castro", ctx)
        self.assertIn("13.938,27", ctx)
        self.assertIn("Ranking", ctx)

    def test_distribution_shows_percentage(self) -> None:
        ctx = build_narrate_context(_rows_distribution(), 3)
        self.assertIn("App", ctx)
        self.assertIn("337", ctx)
        self.assertIn("fatia do total", ctx)
        self.assertIn("%", ctx)

    def test_rate_column_is_not_divided_by_its_sum(self) -> None:
        rows = [
            {"canal": "E-mail", "percentual_resolvidos": 52.3},
            {"canal": "Telefone", "percentual_resolvidos": 50.0},
            {"canal": "Chat", "percentual_resolvidos": 46.3},
        ]
        ctx = build_narrate_context(rows, len(rows))
        self.assertIn("52,3", ctx)
        self.assertNotIn("35,2", ctx)
        self.assertNotIn("35.2", ctx)
        self.assertNotIn("31,2", ctx)
        self.assertNotIn("31.2", ctx)

    def test_several_metrics_have_no_unlabeled_share(self) -> None:
        rows = [
            {
                "categoria": "Viagens",
                "valor_2024": 45859.0,
                "valor_2025": 72097.51,
                "variação_percentual": 57.22,
            },
            {
                "categoria": "Outros",
                "valor_2024": 281705.29,
                "valor_2025": 1.0,
                "variação_percentual": 0.0,
            },
        ]
        ctx = build_narrate_context(rows, len(rows))
        self.assertIn("57,22", ctx)
        self.assertNotIn("14,0", ctx)
        self.assertNotIn("14.0%", ctx)
        self.assertNotIn("fatia do total", ctx)

    def test_record_states_variation_against_previous(self) -> None:
        rows = [
            {
                "mes": "2025-06",
                "valor_compras": 77526.10,
                "valor_compras_anterior": 60539.26,
            }
        ]
        ctx = build_narrate_context(rows, 1)
        self.assertIn("28,06", ctx)
        self.assertIn("acima do valor anterior", ctx)

    def test_time_series_shows_peak(self) -> None:
        ctx = build_narrate_context(_rows_time_series(), 4)
        self.assertIn("Pico", ctx)
        self.assertIn("2025-03", ctx)

    def test_time_series_marks_direction_vs_previous(self) -> None:
        ctx = build_narrate_context(_rows_time_series(), 4)
        self.assertIn("[subiu]", ctx)
        self.assertIn("[desceu]", ctx)
        first_point = next(ln for ln in ctx.splitlines() if "3.200,00" in ln)
        self.assertNotIn("[", first_point)

    def test_scalar_shows_value(self) -> None:
        ctx = build_narrate_context(_rows_scalar(), 1)
        self.assertIn("200", ctx)

    def test_empty_result(self) -> None:
        ctx = build_narrate_context([], 0)
        self.assertIn("nenhum registro", ctx.lower())


if __name__ == "__main__":
    unittest.main()
