import unittest

import src.db.schema as schema
from src.db.schema import get_schema_text


def _column_line(text: str, name: str) -> str:
    prefix = f"{name} "
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped
    raise AssertionError(f"coluna ausente: {name}")


class EnumSampleTest(unittest.TestCase):
    def test_cardinality_samples_and_no_name_list(self) -> None:
        self.assertFalse(hasattr(schema, "CATEGORICAL_COLUMNS"))
        self.assertFalse(hasattr(schema, "COLUMN_NOTES"))
        text, _ = get_schema_text()
        canal_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("canal ")]
        canal = " ".join(canal_lines)
        self.assertIn("App", canal)
        self.assertIn("Loja Física", canal)
        self.assertIn("Site", canal)
        self.assertNotIn("valores", _column_line(text, "nome"))
        self.assertNotIn("valores", _column_line(text, "cidade"))
        self.assertIn("valores", _column_line(text, "estado"))
        self.assertIn("valores", _column_line(text, "resolvido"))
        self.assertIn("valores", _column_line(text, "interagiu"))
