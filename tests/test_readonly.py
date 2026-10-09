import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import settings
from src.db.executor import run_sql


class ReadOnlySqlTest(unittest.TestCase):
    def test_select_returns_rows_and_delete_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prova.db"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            conn.close()

            with patch.object(settings, "sqlite_path", str(path)):
                rows, error = run_sql("SELECT id FROM t")
                self.assertIsNone(error)
                self.assertEqual(rows, [{"id": 1}])

                _, delete_error = run_sql("DELETE FROM t")
                self.assertIsNotNone(delete_error)

            check = sqlite3.connect(path)
            count = check.execute("SELECT COUNT(*) FROM t").fetchone()[0]
            check.close()
            self.assertEqual(count, 1)
