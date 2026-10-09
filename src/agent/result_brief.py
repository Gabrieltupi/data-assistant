"""Briefing determinístico do resultado SQL — preparação de contexto para o narrate.

Padrão: resultado SQL → detecta forma → resume em código → LLM só redige.
O modelo nunca conta linhas, nunca agrega. Código fecha os números.

Formas detectadas (por estrutura dos dados, sem conhecer o schema):
  EMPTY          — sem linhas
  SCALAR         — 1 linha, 1-2 colunas
  RECORD         — 1 linha, muitas colunas
  MULTI_SECTION  — coluna de seção presente (tipo, kind etc.)
  TIME_SERIES    — coluna temporal + coluna numérica
  MULTI_DIM      — 2+ colunas de texto + numérico (ex.: canal, categoria, valor)
  RANKED_LIST    — coluna chave + coluna numérica, cardinalidade alta
  DISTRIBUTION   — coluna categórica + coluna numérica/contagem
  TEXT_LIST      — só texto, sem numérico
  GENERIC        — fallback
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import pandas as pd

# ── constantes ────────────────────────────────────────────────────────────────

# Prefixos/sufixos que identificam coluna de seção (UNION ALL marker)
_SECTION_HINTS = ("tipo", "kind", "categoria_resultado", "secao", "section", "group")

# Nomes de colunas que são chave/identificador — nunca agregar
_KEY_HINTS = ("id", "_id", "uuid", "codigo", "code")

# Máximo de valores distintos para classificar como categórico
_MAX_DISTINCT_CATEGORY = 15

# Máximo de linhas na amostra
_SAMPLE_MAX = 20

# Máximo de itens mostrados por seção no MULTI_SECTION
_SECTION_MAX_ITEMS = 10


# ── forma do resultado ────────────────────────────────────────────────────────

class ResultShape(Enum):
    EMPTY = auto()
    SCALAR = auto()
    RECORD = auto()
    MULTI_SECTION = auto()
    TIME_SERIES = auto()
    MULTI_DIM = auto()      # 2+ colunas de texto + numérico (ex.: canal, categoria, valor)
    RANKED_LIST = auto()
    DISTRIBUTION = auto()
    TEXT_LIST = auto()      # só texto, sem numérico (ex.: lista de nomes)
    GENERIC = auto()


# ── API pública ───────────────────────────────────────────────────────────────

def build_narrate_context(rows: list[dict], row_count: int) -> str:
    """
    Transforma o resultado SQL em texto estruturado para o LLM narrar.
    Detecta a forma dos dados e aplica o resumo adequado.
    Nunca conta ocorrências de identificadores — reporta valores reais.
    """
    if not rows:
        return "RESULTADO: nenhum registro encontrado."

    df = _to_df(rows)
    shape = _detect_shape(df, row_count)

    header = f"RESULTADO ({row_count} linha(s)) — forma: {shape.name}\n"
    body = _summarize(df, shape, row_count)
    instruction = (
        "\nINSTRUÇÃO: use SOMENTE os valores acima. "
        "Não some, não calcule médias, não invente. "
        "Se um valor está ausente no resumo, não mencione. "
        "Na resposta ao usuário não cite RESULTADO, INSTRUÇÃO, forma nem briefing. "
        "Série: pico, menor e as marcas subiu/desceu/igual; não invente tendência."
    )
    return header + body + instruction


def inspect_result(rows: list[dict]) -> dict:
    """
    Inspeção estrutural do resultado SQL para o viz (e testes).

    Retorna shape, colunas de texto, métricas plotáveis (sem id sequencial)
    e coluna temporal se houver. Não conhece nomes de negócio.
    """
    if not rows:
        return {
            "shape": ResultShape.EMPTY,
            "text_cols": [],
            "numeric_cols": [],
            "temporal_col": None,
            "section_col": None,
        }
    df = _to_df(rows)
    return {
        "shape": _detect_shape(df, len(df)),
        "text_cols": [c for c in df.columns if _is_text_col(df[c])],
        "numeric_cols": _plottable_numeric_cols(df),
        "temporal_col": _find_temporal_col(df),
        "section_col": _find_section_col(df),
    }


# ── detecção de forma ─────────────────────────────────────────────────────────

def _detect_shape(df: pd.DataFrame, row_count: int) -> ResultShape:
    if row_count == 0:
        return ResultShape.EMPTY

    cols = list(df.columns)

    if row_count == 1:
        return ResultShape.SCALAR if len(cols) <= 2 else ResultShape.RECORD

    # MULTI_SECTION: coluna que funciona como marcador de seção
    section_col = _find_section_col(df)
    if section_col is not None:
        return ResultShape.MULTI_SECTION

    # TIME_SERIES: coluna temporal + coluna numérica
    temporal_col = _find_temporal_col(df)
    numeric_cols = _numeric_cols(df)
    if temporal_col and numeric_cols:
        return ResultShape.TIME_SERIES

    # TEXT_LIST: sem nenhuma coluna numérica (lista pura de texto)
    if not numeric_cols:
        return ResultShape.TEXT_LIST

    # MULTI_DIM: 2+ colunas de texto + numérico (ex.: canal, categoria, valor)
    text_cols = [c for c in df.columns if _is_text_col(df[c])]
    if len(text_cols) >= 2:
        return ResultShape.MULTI_DIM

    # DISTRIBUTION vem antes de RANKED_LIST:
    # categoria conhecida (canal, estado, categoria…) + numérico
    cat_col = _find_category_col(df)
    if cat_col and numeric_cols:
        return ResultShape.DISTRIBUTION

    # RANKED_LIST: coluna de entidade (alta cardinalidade) + numérico
    key_col = _find_key_col(df)
    if key_col and numeric_cols:
        return ResultShape.RANKED_LIST

    return ResultShape.GENERIC


# ── resumidores por forma ─────────────────────────────────────────────────────

def _summarize(df: pd.DataFrame, shape: ResultShape, row_count: int) -> str:
    if shape == ResultShape.EMPTY:
        return "Nenhum registro encontrado."
    if shape == ResultShape.SCALAR:
        return _summarize_scalar(df)
    if shape == ResultShape.RECORD:
        return _summarize_record(df)
    if shape == ResultShape.MULTI_SECTION:
        return _summarize_multi_section(df, row_count)
    if shape == ResultShape.TIME_SERIES:
        return _summarize_time_series(df)
    if shape == ResultShape.MULTI_DIM:
        return _summarize_multi_dim(df, row_count)
    if shape == ResultShape.RANKED_LIST:
        return _summarize_ranked(df, row_count)
    if shape == ResultShape.DISTRIBUTION:
        return _summarize_distribution(df)
    if shape == ResultShape.TEXT_LIST:
        return _summarize_text_list(df, row_count)
    return _summarize_generic(df, row_count)


def _summarize_scalar(df: pd.DataFrame) -> str:
    row = df.iloc[0].to_dict()
    parts = [f"{col} = {_fmt(val)}" for col, val in row.items()]
    return "Valor: " + " | ".join(parts)


def _summarize_record(df: pd.DataFrame) -> str:
    row = df.iloc[0].to_dict()
    lines = [f"  {col}: {_fmt(val)}" for col, val in row.items()]
    lines.extend(_variations_vs_previous(row))
    return "Registro:\n" + "\n".join(lines)


def _summarize_multi_section(df: pd.DataFrame, row_count: int) -> str:
    section_col = _find_section_col(df)
    other_cols = [c for c in df.columns if c != section_col]

    # Caso padrão: seção é um rótulo (tipo, kind…), chave é outra coluna
    key_col = _find_key_col(df[other_cols]) if other_cols else None
    metric_col = _find_best_metric_col(df[other_cols]) if other_cols else None

    # Caso UNION sem coluna tipo: section_col É a chave, cada nome aparece N vezes.
    # Agrupa por posição de linha (primeira metade = seção 1, segunda = seção 2).
    if key_col is None and metric_col is not None:
        n = len(df)
        half = n // 2
        # Tenta detectar as duas seções pela coluna numérica (duas faixas de valor distintas)
        col_name = metric_col
        first_half = df.iloc[:half]
        second_half = df.iloc[half:]
        lines = [
            f"Lista 1 ({col_name}):",
        ]
        for _, row in first_half.iterrows():
            lines.append(f"  {_fmt(row[section_col])} → {_fmt(row[col_name])}")
        lines.append(f"\nLista 2 ({col_name}):")
        for _, row in second_half.iterrows():
            lines.append(f"  {_fmt(row[section_col])} → {_fmt(row[col_name])}")
        return "\n".join(lines)

    lines: list[str] = []
    for section_val, group in df.groupby(section_col, sort=False):
        lines.append(f'\nSeção "{section_val}" ({len(group)} linha(s)):')
        subset = group.reset_index(drop=True)
        items_shown = 0
        for _, row in subset.iterrows():
            if items_shown >= _SECTION_MAX_ITEMS:
                lines.append(f"  ... (mais {len(subset) - items_shown} itens)")
                break
            if key_col and metric_col:
                lines.append(f"  {_fmt(row[key_col])} → {_fmt(row[metric_col])}")
            elif metric_col:
                lines.append(f"  {_fmt(row[metric_col])}")
            else:
                lines.append(f"  {dict(row)}")
            items_shown += 1

    return "\n".join(lines)


def _summarize_time_series(df: pd.DataFrame) -> str:
    temporal_col = _find_temporal_col(df)
    metric_col = _find_best_metric_col(df)
    if not temporal_col or not metric_col:
        return _summarize_generic(df, len(df))

    series = df[[temporal_col, metric_col]].dropna()
    if series.empty:
        return "Série temporal sem dados."

    n = len(series)
    first_period = _fmt(series.iloc[0][temporal_col])
    last_period = _fmt(series.iloc[-1][temporal_col])
    max_row = series.loc[series[metric_col].idxmax()]
    min_row = series.loc[series[metric_col].idxmin()]

    lines = [
        f"Período: {first_period} → {last_period} ({n} pontos)",
        f"Pico: {_fmt(max_row[temporal_col])} = {_fmt(max_row[metric_col])}",
        f"Menor: {_fmt(min_row[temporal_col])} = {_fmt(min_row[metric_col])}",
        "",
    ]

    # ≤ 24 pontos: lista todos. > 24: pico, menor, amostra, total e média.
    _TIME_SERIES_MAX = 24
    if n <= _TIME_SERIES_MAX:
        lines.append(f"Todos os pontos ({temporal_col} → {metric_col}):")
        prev = None
        for _, row in series.iterrows():
            mark = _direction_vs_prev(row[metric_col], prev)
            suffix = f" [{mark}]" if mark else ""
            lines.append(
                f"  {_fmt(row[temporal_col])} → {_fmt(row[metric_col])}{suffix}"
            )
            prev = row[metric_col]
    else:
        lines.append(f"Primeiros 5 ({temporal_col} → {metric_col}):")
        for _, row in series.head(5).iterrows():
            lines.append(f"  {_fmt(row[temporal_col])} → {_fmt(row[metric_col])}")
        lines.append(f"Últimos 5:")
        for _, row in series.tail(5).iterrows():
            lines.append(f"  {_fmt(row[temporal_col])} → {_fmt(row[metric_col])}")
        total = series[metric_col].sum()
        media = series[metric_col].mean()
        lines.append(f"Total acumulado: {_fmt(total)} | Média por ponto: {_fmt(media)}")

    return "\n".join(lines)


def _summarize_ranked(df: pd.DataFrame, row_count: int) -> str:
    key_col = _find_key_col(df)
    num_cols = _numeric_cols(df)
    if not key_col or not num_cols:
        return _summarize_generic(df, row_count)

    lines = [f"Ranking ({', '.join(num_cols)}):"]
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i > _SAMPLE_MAX:
            lines.append(f"  ... (mais {row_count - _SAMPLE_MAX} itens)")
            break
        if len(num_cols) == 1:
            lines.append(f"  {i}. {_fmt(row[key_col])} → {_fmt(row[num_cols[0]])}")
        else:
            metrics = " | ".join(f"{c}={_fmt(row[c])}" for c in num_cols)
            lines.append(f"  {i}. {_fmt(row[key_col])} → {metrics}")
    return "\n".join(lines)


def _summarize_distribution(df: pd.DataFrame) -> str:
    cat_col = _find_category_col(df)
    num_cols = _numeric_cols(df)
    if not cat_col or not num_cols:
        return _summarize_generic(df, len(df))

    lines = [f"Distribuição ({cat_col}):"]
    for _, row in df.iterrows():
        if len(num_cols) == 1:
            col = num_cols[0]
            val = row[col]
            suffix = "" if _is_rate_name(col) else _share_of_total(val, df[col].sum())
            lines.append(f"  {_fmt(row[cat_col])}: {_fmt(val)}{suffix}")
        else:
            metrics = " | ".join(f"{c}={_fmt(row[c])}" for c in num_cols)
            lines.append(f"  {_fmt(row[cat_col])}: {metrics}")
    return "\n".join(lines)


def _summarize_multi_dim(df: pd.DataFrame, row_count: int) -> str:
    """2+ colunas de texto + numérico — chave composta com ' | ' entre as dimensões."""
    text_cols = [c for c in df.columns if _is_text_col(df[c])]
    num_cols = _numeric_cols(df)

    dim_label = " | ".join(text_cols)
    metric_label = ", ".join(num_cols)
    lines = [f"Resultado ({dim_label} → {metric_label}):"]

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i > _SAMPLE_MAX:
            lines.append(f"  ... (mais {row_count - _SAMPLE_MAX} itens)")
            break
        key = " | ".join(_fmt(row[c]) for c in text_cols)
        if len(num_cols) == 1:
            lines.append(f"  {i}. {key} → {_fmt(row[num_cols[0]])}")
        else:
            metrics = " | ".join(f"{c}={_fmt(row[c])}" for c in num_cols)
            lines.append(f"  {i}. {key} → {metrics}")
    return "\n".join(lines)


def _summarize_text_list(df: pd.DataFrame, row_count: int) -> str:
    """Sem colunas numéricas — lista simples de texto."""
    text_cols = [c for c in df.columns if _is_text_col(df[c])]
    if not text_cols:
        return _summarize_generic(df, row_count)

    lines = [f"Lista ({', '.join(text_cols)}):"]
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i > _SAMPLE_MAX:
            lines.append(f"  ... (mais {row_count - _SAMPLE_MAX} itens)")
            break
        if len(text_cols) == 1:
            lines.append(f"  {i}. {_fmt(row[text_cols[0]])}")
        else:
            values = " | ".join(_fmt(row[c]) for c in text_cols)
            lines.append(f"  {i}. {values}")
    return "\n".join(lines)


def _summarize_generic(df: pd.DataFrame, row_count: int) -> str:
    cols = list(df.columns)
    sample = df.head(_SAMPLE_MAX)
    lines = [f"Colunas: {', '.join(cols)}"]
    for _, row in sample.iterrows():
        lines.append("  " + " | ".join(f"{c}={_fmt(row[c])}" for c in cols))
    if row_count > _SAMPLE_MAX:
        lines.append(f"  ... (amostra de {_SAMPLE_MAX} de {row_count} linhas)")
    return "\n".join(lines)


# ── detectores de coluna ──────────────────────────────────────────────────────

def _find_section_col(df: pd.DataFrame) -> str | None:
    """Coluna de seção: texto, poucos distintos, nome sugere agrupamento,
    OU padrão de repetição uniforme de chave (UNION ALL sem coluna tipo)."""
    for col in df.columns:
        if not _is_text_col(df[col]):
            continue
        n_distinct = df[col].nunique()
        if n_distinct < 2 or n_distinct > 10:
            continue
        name = col.lower()
        if any(hint in name for hint in _SECTION_HINTS):
            return col
        if n_distinct <= 5 and n_distinct < len(df) / 2:
            return col

    # Padrão de UNION ALL sem coluna tipo:
    # cada valor de chave aparece exatamente N>=2 vezes (N arms do UNION)
    for col in df.columns:
        if not _is_text_col(df[col]):
            continue
        counts = df[col].value_counts()
        n_distinct = len(counts)
        if (
            n_distinct >= 2
            and counts.nunique() == 1   # todos os valores repetem igual
            and counts.iloc[0] >= 2     # cada um aparece >= 2 vezes
            and n_distinct * counts.iloc[0] == len(df)  # cobre todas as linhas
        ):
            return col  # coluna-chave funcionará como eixo de seção

    return None


def _find_temporal_col(df: pd.DataFrame) -> str | None:
    """Coluna temporal: nome sugere data/período."""
    hints = ("data", "date", "mes", "month", "ano", "year", "periodo", "period", "dt_")
    for col in df.columns:
        if any(h in col.lower() for h in hints):
            return col
    return None


def _find_key_col(df: pd.DataFrame) -> str | None:
    """Coluna chave/entidade: texto, alta cardinalidade (≈ número de linhas)."""
    n = len(df)
    for col in df.columns:
        if not _is_text_col(df[col]):
            continue
        # Chave de negócio por nome
        if any(col.lower().endswith(h) for h in _KEY_HINTS):
            continue  # PK/FK, não é label
        n_distinct = df[col].nunique()
        if n_distinct >= max(2, n * 0.6):  # 60%+ das linhas são únicas
            return col
    return None


def _find_category_col(df: pd.DataFrame) -> str | None:
    """Coluna categórica: texto, baixa cardinalidade OU nome típico de categoria."""
    _CATEGORY_HINTS = (
        "canal", "categoria", "estado", "genero", "tipo_contato",
        "regiao", "status", "resolvido", "interagiu", "profissao",
        "channel", "category", "region", "gender",
    )
    n = len(df)
    # Prioridade 1: nome da coluna é um hint de categoria
    for col in df.columns:
        if col.lower() in _CATEGORY_HINTS and _is_text_col(df[col]):
            return col
    # Prioridade 2: texto com baixa cardinalidade (≤ metade das linhas)
    for col in df.columns:
        if not _is_text_col(df[col]):
            continue
        n_distinct = df[col].nunique()
        if 2 <= n_distinct <= min(_MAX_DISTINCT_CATEGORY, n // 2 + 1):
            return col
    return None


def _plottable_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Numéricas que não parecem identidade sequencial (1, 2, 3, ...)."""
    out: list[str] = []
    for col in _numeric_cols(df):
        series = df[col].dropna()
        if series.empty:
            continue
        if series.is_monotonic_increasing and series.min() == 1 and series.max() == len(df):
            continue
        out.append(col)
    return out


def _find_best_metric_col(df: pd.DataFrame) -> str | None:
    """Melhor coluna numérica: não é identidade sequencial (1,2,3...)."""
    cols = _plottable_numeric_cols(df)
    return cols[0] if cols else None


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _is_text_col(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


_RATE_HINTS = ("percent", "pct", "taxa", "varia", "ratio")


def _is_rate_name(name: str) -> bool:
    folded = name.lower()
    return any(hint in folded for hint in _RATE_HINTS)


def _share_of_total(val: Any, total: Any) -> str:
    try:
        number = float(val)
        whole = float(total)
    except (TypeError, ValueError):
        return ""
    if pd.isna(number) or pd.isna(whole) or whole == 0:
        return ""
    text = f"{number / whole * 100:.1f}".replace(".", ",")
    return f" (fatia do total: {text}%)"


def _variations_vs_previous(row: dict) -> list[str]:
    lines: list[str] = []
    keys = {str(key).lower(): key for key in row}
    for key in row:
        folded = str(key).lower()
        if not folded.endswith("_anterior"):
            continue
        sibling = keys.get(folded[: -len("_anterior")])
        if sibling is None:
            continue
        try:
            current = float(row[sibling])
            previous = float(row[key])
        except (TypeError, ValueError):
            continue
        if pd.isna(current) or pd.isna(previous) or previous == 0:
            continue
        pct = (current - previous) / previous * 100
        direction = "acima" if pct >= 0 else "abaixo"
        lines.append(
            f"  variação de {sibling}: {_fmt(abs(pct))}% {direction} do valor anterior"
        )
    return lines


def _direction_vs_prev(curr: Any, prev: Any) -> str | None:
    if prev is None:
        return None
    try:
        a, b = float(prev), float(curr)
    except (TypeError, ValueError):
        return None
    if b > a:
        return "subiu"
    if b < a:
        return "desceu"
    return "igual"


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "(vazio)"
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(value)
