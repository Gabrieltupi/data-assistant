from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    node_decide_viz,
    node_execute_sql,
    node_fail_graceful,
    node_generate_sql,
    node_guard_sql,
    node_load_schema,
    node_narrate,
    node_reflect_sql,
    node_repair_sql,
)
from src.agent.state import AgentState
from src.config import settings


# ── roteadores condicionais ───────────────────────────────────────────────────

def _route_after_generate(state: AgentState) -> str:
    if state["status"] == "out_of_scope":
        return "narrate"
    return "guard_sql"


def _route_after_guard(state: AgentState) -> str:
    if state["status"] == "blocked":
        return "fail_graceful"
    if state["status"] == "reflect_needed":
        if state["retries"] < settings.sql_max_retries:
            return "reflect_sql"
        return "fail_graceful"
    return "execute_sql"


def _route_after_execute(state: AgentState) -> str:
    status = state["status"]
    if status == "exec_error":
        if state["retries"] < settings.sql_max_retries:
            return "reflect_sql"
        return "fail_graceful"
    if status == "failed":
        return "fail_graceful"
    if status == "empty":
        return "narrate"
    if status == "success":
        return "decide_viz"
    return "fail_graceful"


def _route_after_repair(state: AgentState) -> str:
    # Correção: repair pode retornar status "failed" (OUT_OF_SCOPE no repair).
    # Nesse caso vai direto para fail_graceful — não passa pelo guard com sql=None.
    if state["status"] == "failed":
        return "fail_graceful"
    return "guard_sql"


# ── compilação do grafo ───────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("load_schema",   node_load_schema)
    g.add_node("generate_sql",  node_generate_sql)
    g.add_node("guard_sql",     node_guard_sql)
    g.add_node("execute_sql",   node_execute_sql)
    g.add_node("reflect_sql",   node_reflect_sql)
    g.add_node("repair_sql",    node_repair_sql)
    g.add_node("decide_viz",    node_decide_viz)
    g.add_node("narrate",       node_narrate)
    g.add_node("fail_graceful", node_fail_graceful)

    g.set_entry_point("load_schema")
    g.add_edge("load_schema", "generate_sql")

    g.add_conditional_edges("generate_sql", _route_after_generate, {
        "narrate":   "narrate",
        "guard_sql": "guard_sql",
    })

    g.add_conditional_edges("guard_sql", _route_after_guard, {
        "fail_graceful": "fail_graceful",
        "reflect_sql":   "reflect_sql",
        "execute_sql":   "execute_sql",
    })

    # reflect sempre encaminha para repair (sem condicional)
    g.add_edge("reflect_sql", "repair_sql")

    g.add_conditional_edges("execute_sql", _route_after_execute, {
        "reflect_sql":   "reflect_sql",
        "fail_graceful": "fail_graceful",
        "narrate":       "narrate",
        "decide_viz":    "decide_viz",
    })

    # repair: volta para guard se corrigiu, vai para fail_graceful se desistiu
    g.add_conditional_edges("repair_sql", _route_after_repair, {
        "guard_sql":     "guard_sql",
        "fail_graceful": "fail_graceful",
    })

    g.add_edge("decide_viz",    "narrate")
    g.add_edge("narrate",       END)
    g.add_edge("fail_graceful", END)

    return g.compile()
