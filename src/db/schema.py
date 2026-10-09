import hashlib
import sqlite3
from pathlib import Path

from src.config import settings

_NUMERIC_TYPES = ("INT", "REAL", "NUM", "FLOAT", "DOUBLE", "DEC")
_ENUM_MAX_DISTINCT = 25
_ENUM_SAMPLE_LIMIT = 15

# Cache em memória (por sessão de processo)
_cache: dict = {
    "fingerprint": None,
    "text": None,
    "identity_columns": frozenset(),
    "measure_columns": frozenset(),
    "fact_tables": frozenset(),
    "dimension_tables": frozenset(),
}


def get_schema_text() -> tuple[str, str]:
    """
    Retorna (schema_text, fingerprint).
    Usa cache se o arquivo .db não mudou desde a última leitura.
    """
    _refresh_cache_if_needed()
    return _cache["text"], _cache["fingerprint"]


def get_identity_columns() -> frozenset[str]:
    """Nomes de colunas identidade (PK/FK/id), em minúsculas."""
    _refresh_cache_if_needed()
    return _cache["identity_columns"]


def get_measure_columns() -> frozenset[str]:
    """Nomes de colunas numéricas que não são identidade."""
    _refresh_cache_if_needed()
    return _cache["measure_columns"]


def get_fact_tables() -> frozenset[str]:
    """Tabelas com FK (fatos). Inferido do schema, não de lista fixa."""
    _refresh_cache_if_needed()
    return _cache["fact_tables"]


def get_dimension_tables() -> frozenset[str]:
    """Tabelas sem FK (dimensões)."""
    _refresh_cache_if_needed()
    return _cache["dimension_tables"]


def _refresh_cache_if_needed() -> None:
    db_path = Path(settings.sqlite_path)
    fingerprint = _fingerprint(db_path)
    if _cache["fingerprint"] == fingerprint:
        return
    text, identities, measures, facts, dims = _build_schema_text(db_path)
    _cache["fingerprint"] = fingerprint
    _cache["text"] = text
    _cache["identity_columns"] = identities
    _cache["measure_columns"] = measures
    _cache["fact_tables"] = facts
    _cache["dimension_tables"] = dims


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.md5(f"{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()


def _build_schema_text(
    db_path: Path,
) -> tuple[str, frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        tables = _get_tables(conn)
        chunks: list[str] = []
        identities: set[str] = set()
        measures: set[str] = set()
        facts: set[str] = set()
        dims: set[str] = set()
        for table in tables:
            text, ids, meas, is_fact = _format_table(conn, table)
            chunks.append(text)
            identities.update(ids)
            measures.update(meas)
            if is_fact:
                facts.add(table.lower())
            else:
                dims.add(table.lower())
        return (
            "\n\n".join(chunks),
            frozenset(identities),
            frozenset(measures),
            frozenset(facts),
            frozenset(dims),
        )
    finally:
        conn.close()


def _get_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows if not r[0].startswith("sqlite_")]


def _format_table(
    conn: sqlite3.Connection, table: str
) -> tuple[str, set[str], set[str], bool]:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    fks = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    fk_names = {fk[3].lower() for fk in fks}
    enum_samples = _get_enum_samples(conn, table, cols)
    identities: set[str] = set()
    measures: set[str] = set()

    is_fact = bool(fks)
    kind = "fato" if is_fact else "dimensão"
    lines = [f"Table: {table} -- {kind}"]
    for col in cols:
        # col: (cid, name, type, notnull, dflt_value, pk)
        name, ctype, notnull, pk = col[1], col[2], col[3], col[5]
        role = _column_role(name, ctype, bool(pk), fk_names)
        if role == "identidade":
            identities.add(name.lower())
        elif role == "medida":
            measures.add(name.lower())
        markers = []
        if pk:
            markers.append("PK")
        if notnull:
            markers.append("NOT NULL")
        marker_str = f" ({', '.join(markers)})" if markers else ""
        enum_hint = f" -- valores: {enum_samples[name]}" if name in enum_samples else ""
        role_hint = ""
        if role == "identidade":
            role_hint = " -- identidade, não somar/média (COUNT pode)"
        elif role == "medida":
            role_hint = " -- medida, pode SUM/AVG"
        lines.append(f"  {name} {ctype}{marker_str}{enum_hint}{role_hint}")

    for fk in fks:
        # fk: (id, seq, table, from, to, on_update, on_delete, match)
        lines.append(f"  FK: {fk[3]} → {fk[2]}.{fk[4]}")

    return "\n".join(lines), identities, measures, is_fact


def _column_role(name: str, ctype: str, is_pk: bool, fk_names: set[str]) -> str | None:
    n = name.lower()
    if is_pk or n in fk_names or n == "id" or n.endswith("_id"):
        return "identidade"
    if any(ctype.upper().startswith(t) for t in _NUMERIC_TYPES):
        return "medida"
    return None


def _is_text_or_bool(ctype: str) -> bool:
    t = (ctype or "").upper()
    return "TEXT" in t or "CHAR" in t or "CLOB" in t or "BOOL" in t


def _get_enum_samples(conn: sqlite3.Connection, table: str, cols: list) -> dict:
    samples = {}
    for col in cols:
        col_name, ctype = col[1], col[2]
        if not _is_text_or_bool(ctype):
            continue
        try:
            distinct = conn.execute(
                f"SELECT COUNT(DISTINCT [{col_name}]) FROM [{table}]"
            ).fetchone()[0]
            if distinct is None or distinct < 2 or distinct > _ENUM_MAX_DISTINCT:
                continue
            rows = conn.execute(
                f"SELECT [{col_name}], COUNT(*) AS cnt "
                f"FROM [{table}] "
                f"GROUP BY [{col_name}] "
                f"ORDER BY cnt DESC "
                f"LIMIT {_ENUM_SAMPLE_LIMIT}"
            ).fetchall()
            values = [r[0] for r in rows if r[0] is not None]
            if values:
                samples[col_name] = values
        except Exception:
            pass
    return samples
