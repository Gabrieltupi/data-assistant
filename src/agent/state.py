from typing import NotRequired, TypedDict


class TraceEvent(TypedDict, total=False):
    step: int
    node: str
    status: str
    detail: str
    sql: str
    error: str
    row_count: int
    ms: float


class AgentState(TypedDict):
    question: str
    messages: list[dict]        # [{role: "user"|"assistant", content: str}]
    schema_text: str
    schema_fingerprint: str
    sql: str | None
    sql_error: str | None
    retries: int
    error_source: NotRequired[str | None]  # guard | sqlite
    rows: list[dict] | None
    row_count: int
    status: str                 # pending|schema_loaded|sql_ready|blocked|reflect_needed|exec_error|empty|success|out_of_scope|failed
    viz_type: str               # none|table|bar|line|grouped_bar (mark)
    viz_spec: NotRequired[dict | None]
    answer: str
    trace: list[TraceEvent]
    reflection: NotRequired[str | None]  # diagnóstico JSON do node_reflect_sql
