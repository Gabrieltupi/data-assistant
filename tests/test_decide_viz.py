import pathlib
import unittest

from src.viz.decide import decide_viz


class DecideVizTest(unittest.TestCase):
    def test_empty_is_none(self) -> None:
        spec = decide_viz([])
        self.assertEqual(spec["mark"], "none")

    def test_scalar_is_none(self) -> None:
        spec = decide_viz([{"total": 24}])
        self.assertEqual(spec["mark"], "none")

    def test_ranking_is_bar(self) -> None:
        rows = [{"nome": f"P{i}", "total": float(100 - i)} for i in range(5)]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "bar")
        self.assertEqual(spec["label"], "nome")
        self.assertEqual(spec["values"], ["total"])

    def test_distribution_is_bar(self) -> None:
        rows = [
            {"canal": "A", "qtd": 10},
            {"canal": "B", "qtd": 20},
            {"canal": "C", "qtd": 30},
        ]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "bar")
        self.assertEqual(spec["values"], ["qtd"])

    def test_time_series_is_line(self) -> None:
        rows = [{"mes": f"2025-{m:02d}", "total": 100.0 * m} for m in range(1, 8)]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "line")
        self.assertEqual(spec["label"], "mes")
        self.assertEqual(spec["values"], ["total"])

    def test_two_metrics_is_grouped_bar(self) -> None:
        rows = [{"cat": f"G{i}", "a": 1000.0 + i, "b": 2000.0 + i} for i in range(6)]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "grouped_bar")
        self.assertEqual(spec["label"], "cat")
        self.assertEqual(spec["values"], ["a", "b"])

    def test_four_metrics_is_table(self) -> None:
        rows = [
            {"x": 1.0, "y": 2.0, "z": 3.0, "w": 4.0, "extra": 5.0},
            {"x": 2.0, "y": 3.0, "z": 4.0, "w": 5.0, "extra": 6.0},
        ]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "table")

    def test_two_text_cols_is_bar_with_label_cols(self) -> None:
        rows = [
            {"c1": "App", "c2": "A", "metric": 10.0},
            {"c1": "Site", "c2": "B", "metric": 20.0},
        ]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "bar")
        self.assertEqual(spec["values"], ["metric"])
        self.assertEqual(spec["label_cols"], ["c1", "c2"])

    def test_scale_filter_drops_small_series(self) -> None:
        rows = [
            {"cat": "A", "a": 50000.0, "b": 60000.0, "pct": 22.0},
            {"cat": "B", "a": 40000.0, "b": 55000.0, "pct": 15.0},
        ]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "grouped_bar")
        self.assertEqual(spec["values"], ["a", "b"])
        self.assertNotIn("pct", spec["values"])

    def test_section_bars_when_tipo_splits_lists(self) -> None:
        rows = [
            {"nome": "João", "tipo": "compras", "valor": 13938.0},
            {"nome": "Rafael", "tipo": "compras", "valor": 13516.0},
            {"nome": "João", "tipo": "tickets", "valor": 2.0},
            {"nome": "Rafael", "tipo": "tickets", "valor": 4.0},
        ]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "section_bars")
        self.assertEqual(spec["section"], "tipo")
        self.assertEqual(spec["label"], "nome")
        self.assertEqual(spec["values"], ["valor"])

    def test_repeated_names_without_section_label_is_table(self) -> None:
        rows = [
            {"nome": "João", "valor": 13938.0},
            {"nome": "João", "valor": 2.0},
            {"nome": "Rafael", "valor": 13516.0},
            {"nome": "Rafael", "valor": 2.0},
        ]
        spec = decide_viz(rows)
        self.assertEqual(spec["mark"], "table")

    def test_decide_py_has_no_question_keywords(self) -> None:
        path = pathlib.Path(__file__).resolve().parents[1] / "src" / "viz" / "decide.py"
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("comparar", text)
        self.assertNotIn("evolução", text)
        self.assertNotIn("question", text.lower().replace("notrequired", ""))


if __name__ == "__main__":
    unittest.main()
