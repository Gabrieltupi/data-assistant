"""Renderiza o spec de viz no Streamlit. Não escolhe o gráfico — só executa o encoding."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.viz.decide import VizSpec


def render_viz(rows: list[dict], spec: VizSpec | str | None) -> None:
    if not rows:
        return
    spec = _normalize(spec)
    mark = spec.get("mark") or "none"
    if mark == "none":
        return

    df = pd.DataFrame(rows)
    if mark == "table":
        st.dataframe(df, width="stretch")
        return

    values = spec.get("values") or []
    if not values:
        st.dataframe(df, width="stretch")
        return

    label = _ensure_label(df, spec)
    if label is None or label not in df.columns:
        st.dataframe(df, width="stretch")
        return

    y_scale = alt.Scale(zero=True)

    if mark == "section_bars":
        section = spec.get("section")
        value_col = values[0]
        if not section or section not in df.columns:
            st.dataframe(df, width="stretch")
            return
        for section_val, group in df.groupby(section, sort=False):
            st.caption(str(section_val))
            chart = (
                alt.Chart(group)
                .mark_bar()
                .encode(
                    x=alt.X(label, sort="-y", title=label),
                    y=alt.Y(value_col, scale=y_scale, title=value_col),
                    tooltip=[label, value_col],
                )
            )
            st.altair_chart(chart, width="stretch")
        st.dataframe(df, width="stretch")
        return

    if mark == "grouped_bar" and len(values) >= 2:
        long = df.melt(
            id_vars=[label],
            value_vars=values,
            var_name="serie",
            value_name="valor",
        )
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X(label, sort="-y", title=label),
                y=alt.Y("valor", scale=y_scale, title=""),
                color=alt.Color("serie", title=""),
                xOffset="serie",
                tooltip=[label, "serie", "valor"],
            )
        )
        st.altair_chart(chart, width="stretch")
        return

    value_col = values[0]
    if mark == "line":
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(label, sort=None, title=label),
                y=alt.Y(value_col, scale=y_scale, title=value_col),
                tooltip=[label, value_col],
            )
        )
        st.altair_chart(chart, width="stretch")
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(label, sort="-y", title=label),
            y=alt.Y(value_col, scale=y_scale, title=value_col),
            tooltip=[label, value_col],
        )
    )
    st.altair_chart(chart, width="stretch")


def _normalize(spec: VizSpec | str | None) -> VizSpec:
    if spec is None:
        return {"mark": "none", "label": None, "values": [], "label_cols": []}
    if isinstance(spec, str):
        return {"mark": spec, "label": None, "values": [], "label_cols": []}  # type: ignore[typeddict-item]
    return spec


def _ensure_label(df: pd.DataFrame, spec: VizSpec) -> str | None:
    label_cols = spec.get("label_cols") or []
    if len(label_cols) > 1:
        df["_label"] = df[label_cols].astype(str).agg(" | ".join, axis=1)
        return "_label"
    if spec.get("label") and spec["label"] in df.columns:
        return spec["label"]
    if len(label_cols) == 1 and label_cols[0] in df.columns:
        return label_cols[0]
    return None
