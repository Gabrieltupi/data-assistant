"""Escolha de visualização a partir da forma do resultado — não da pergunta."""

from __future__ import annotations

from typing import Literal, TypedDict

import pandas as pd

from src.agent.result_brief import ResultShape, inspect_result

VizMark = Literal["none", "table", "bar", "line", "grouped_bar", "section_bars"]

_MAX_SERIES = 3


class VizSpec(TypedDict, total=False):
    mark: VizMark
    label: str | None
    values: list[str]
    label_cols: list[str]
    section: str | None


def _empty_spec() -> VizSpec:
    return {"mark": "none", "label": None, "values": [], "label_cols": []}


def _spec(
    mark: VizMark,
    *,
    label: str | None = None,
    values: list[str] | None = None,
    label_cols: list[str] | None = None,
    section: str | None = None,
) -> VizSpec:
    vals = values or []
    cols = label_cols if label_cols is not None else ([label] if label else [])
    spec: VizSpec = {"mark": mark, "label": label, "values": vals, "label_cols": cols}
    if section:
        spec["section"] = section
    return spec


def _filter_scale(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Drop séries 'pequenas' quando convivem com séries grandes (ex.: % vs reais)."""
    if len(cols) < 2:
        return cols
    maxima = {c: float(df[c].max()) for c in cols if pd.notna(df[c].max())}
    if not maxima:
        return cols
    has_large = any(m >= 1000 for m in maxima.values())
    if not has_large:
        return cols
    kept = [c for c in cols if maxima.get(c, 0) > 100]
    return kept or cols


def decide_viz(rows: list[dict]) -> VizSpec:
    """
    Encoding a partir da estrutura do dataframe.

    Não lê a pergunta. Não usa nomes de coluna de negócio.
    Filtro de escala: se uma métrica tem max ≤ 100 e outra ≥ 1000, a pequena sai.
    Mais de 3 métricas → tabela.
    """
    if not rows:
        return _empty_spec()

    info = inspect_result(rows)
    shape = info["shape"]
    text_cols: list[str] = info["text_cols"]
    raw_nums: list[str] = info["numeric_cols"]
    temporal = info["temporal_col"]
    n = len(rows)
    df = pd.DataFrame(rows)

    if n == 1 or shape in (
        ResultShape.EMPTY,
        ResultShape.SCALAR,
        ResultShape.RECORD,
        ResultShape.TEXT_LIST,
    ):
        return _empty_spec()

    if len(raw_nums) > _MAX_SERIES and shape != ResultShape.TIME_SERIES:
        return _spec("table")

    nums = _filter_scale(df, raw_nums)

    if shape == ResultShape.MULTI_SECTION and len(nums) == 1:
        section = info.get("section_col")
        entities = [c for c in text_cols if c != section]
        if section and len(entities) == 1:
            return _spec(
                "section_bars",
                label=entities[0],
                values=nums[:1],
                label_cols=[entities[0]],
                section=section,
            )
        return _spec("table")

    if shape == ResultShape.TIME_SERIES and temporal and nums:
        return _spec("line", label=temporal, values=nums[:1], label_cols=[temporal])

    if len(text_cols) >= 1 and 2 <= len(nums) <= _MAX_SERIES:
        return _spec(
            "grouped_bar",
            label=text_cols[0],
            values=nums[:_MAX_SERIES],
            label_cols=[text_cols[0]],
        )

    if len(text_cols) >= 2 and len(nums) == 1:
        return _spec("bar", label=None, values=nums, label_cols=text_cols)

    if len(text_cols) >= 1 and len(nums) == 1 and n > 1:
        return _spec("bar", label=text_cols[0], values=nums, label_cols=[text_cols[0]])

    return _spec("table")
