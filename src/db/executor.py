import re
import sqlite3
from pathlib import Path
from urllib.parse import quote

import sqlglot
from sqlglot import exp

from src.config import settings
from src.db.schema import get_fact_tables, get_identity_columns, get_measure_columns

_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|DETACH|TRUNCATE|REPLACE)\b",
    re.IGNORECASE,
)

_AGG_ON_COLUMN = re.compile(
    r"\b(SUM|AVG|AVERAGE|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?(?:\w+\.)?([A-Za-z_][\w]*)\s*\)",
    re.IGNORECASE,
)

FANOUT_ERROR = (
    "JOIN entre tabelas fato bloqueado. "
    "Cada SELECT só pode referenciar UMA tabela fato no FROM/JOIN direto. "
    "Para correlacionar dois fatos use: (1) EXISTS/IN com subquery isolada, "
    "ou (2) pré-agregue um fato em subquery antes do JOIN principal."
)


def guard_sql(sql: str) -> tuple[bool, str | None]:
    """
    Valida se o SQL é permitido (somente SELECT/WITH).
    Faz strip de trailing ; antes de validar.

    Retorna (is_allowed, error_message).
    error_message é None se permitido.
    """
    clean = sql.strip().rstrip(";").strip()

    # Multi-statement (; no meio)
    if ";" in clean:
        return False, "Multi-statement SQL não permitido (ponto e vírgula interno detectado)."

    # Palavras-chave proibidas
    match = _BLOCKED.search(clean)
    if match:
        return False, f"Instrução '{match.group().upper()}' bloqueada. Apenas SELECT é permitido."

    # Deve iniciar com SELECT ou WITH
    upper = clean.upper().lstrip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "SQL deve iniciar com SELECT ou WITH."

    return True, None


def detect_fact_fanout(
    sql: str,
    fact_tables: frozenset[str] | set[str] | None = None,
) -> str | None:
    """
    Cada SELECT pode ter no máximo uma fato no FROM/JOIN.
    Fatos vêm do schema (tabela com FK), não de lista fixa.
    EXISTS/IN não contam para o SELECT de fora. Subquery ou CTE sem agregar conta.
    """
    facts = {t.lower() for t in (fact_tables if fact_tables is not None else get_fact_tables())}
    if len(facts) < 2:
        return None

    try:
        trees = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return None

    for tree in trees:
        if tree is None:
            continue
        ctes = _cte_bodies(tree)
        for select in tree.find_all(exp.Select):
            if _direct_fact_count(select, facts, ctes) >= 2:
                return FANOUT_ERROR
    return None


def _cte_bodies(tree: exp.Expression) -> dict[str, exp.Expression]:
    bodies: dict[str, exp.Expression] = {}
    for cte in tree.find_all(exp.CTE):
        alias = cte.args.get("alias")
        if not isinstance(alias, exp.Expression):
            continue
        bodies[alias.name.lower()] = cte.this
    return bodies


def _direct_fact_count(
    select: exp.Select,
    facts: set[str],
    ctes: dict[str, exp.Expression],
) -> int:
    sources: list[exp.Expression] = []
    frm = select.args.get("from_")
    if isinstance(frm, exp.From):
        sources.append(frm.this)
    for join in select.args.get("joins") or []:
        if isinstance(join, exp.Join):
            sources.append(join.this)
    return sum(1 for source in sources if _counts_as_fact(source, facts, ctes))


def _counts_as_fact(
    node: exp.Expression,
    facts: set[str],
    ctes: dict[str, exp.Expression],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(node, exp.Table):
        name = node.name.lower()
        if name in facts:
            return True
        body = ctes.get(name)
        if body is None or name in seen:
            return False
        return _multiplies_fact(body, facts, ctes, seen | {name})
    if isinstance(node, exp.Subquery):
        return _multiplies_fact(node.this, facts, ctes, seen)
    return False


def _multiplies_fact(
    query: exp.Expression,
    facts: set[str],
    ctes: dict[str, exp.Expression],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(query, exp.Union):
        return _multiplies_fact(query.left, facts, ctes, seen) or _multiplies_fact(
            query.right, facts, ctes, seen
        )
    if isinstance(query, exp.Subquery):
        return _multiplies_fact(query.this, facts, ctes, seen)
    if not isinstance(query, exp.Select) or _is_aggregated(query):
        return False
    return _direct_fact_count(query, facts, ctes) >= 1


def _is_aggregated(select: exp.Select) -> bool:
    if select.args.get("group"):
        return True
    for expr in select.args.get("expressions") or []:
        if isinstance(expr, exp.Expression) and _has_agg(expr):
            return True
    return False


def _has_agg(node: exp.Expression) -> bool:
    stack: list[exp.Expression] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (exp.Subquery, exp.Select)):
            continue
        if isinstance(cur, exp.AggFunc):
            return True
        for value in cur.args.values():
            if isinstance(value, exp.Expression):
                stack.append(value)
            elif isinstance(value, list):
                stack.extend(item for item in value if isinstance(item, exp.Expression))
    return False


def detect_identity_aggregation(
    sql: str,
    identities: frozenset[str] | set[str] | None = None,
    measures: frozenset[str] | set[str] | None = None,
) -> str | None:
    """
    Bloqueia SUM/AVG/MIN/MAX em colunas identidade (PK, FK, id).
    COUNT(id) é permitido. Papéis vêm do schema, não da pergunta.
    """
    forbidden = {c.lower() for c in (identities if identities is not None else get_identity_columns())}
    measure_names = sorted(measures if measures is not None else get_measure_columns())
    for match in _AGG_ON_COLUMN.finditer(sql):
        fn = match.group(1).upper()
        col = match.group(2).lower()
        if col not in forbidden:
            continue
        medidas = ", ".join(measure_names) if measure_names else "uma coluna numérica de fato (ex.: valor)"
        return (
            f"{fn}({col}) é inválido: {col} é identidade, não métrica. "
            f"Para quantidade use COUNT(*). Para média/total use uma medida: {medidas}."
        )
    return None


def inject_limit(sql: str, max_rows: int | None = None) -> str:
    """
    Garante que o SQL tenha LIMIT <= max_rows.
    Se LIMIT ausente → adiciona. Se maior que teto → substitui.
    """
    limit = max_rows or settings.sql_max_rows
    clean = sql.strip().rstrip(";").strip()
    match = re.search(r"\bLIMIT\s+(\d+)", clean, re.IGNORECASE)
    if match:
        if int(match.group(1)) > limit:
            return re.sub(r"\bLIMIT\s+\d+", f"LIMIT {limit}", clean, flags=re.IGNORECASE)
        return clean
    return f"{clean} LIMIT {limit}"


def run_sql(sql: str) -> tuple[list[dict], str | None]:
    """
    Executa o SQL contra o banco configurado.
    Retorna (rows, error). rows=[] em caso de erro.
    """
    conn = None
    try:
        conn = sqlite3.connect(_readonly_uri(settings.sqlite_path), uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = [dict(r) for r in cursor.fetchall()]
        return rows, None
    except Exception as exc:
        return [], str(exc)
    finally:
        if conn is not None:
            conn.close()


def _readonly_uri(path: str) -> str:
    """URI absoluta, barras normalizadas, espaço codificado, mode=ro."""
    posix = Path(path).resolve().as_posix()
    encoded = quote(posix, safe="/:")
    return f"file:{encoded}?mode=ro"
