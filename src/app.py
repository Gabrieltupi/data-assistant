import streamlit as st

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.viz.render import render_viz

# ── configuração da página ────────────────────────────────────────────────────

st.set_page_config(page_title="Data Assistant", page_icon="🔍", layout="centered")


def _render_trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander("🔎 Raciocínio"):
        for event in trace:
            step = event.get("step", "?")
            node = event.get("node", "")
            status = event.get("status", "")
            detail = event.get("detail", "")
            if status == "ok":
                icon = "✅"
            elif status in ("blocked", "failed", "exec_error"):
                icon = "❌"
            else:
                icon = "ℹ️"
            st.markdown(f"**{icon} Passo {step} — `{node}`**: {detail}")
            if "sql" in event:
                st.code(event["sql"], language="sql")
            if "error" in event:
                st.error(event["error"])
            if "row_count" in event:
                st.caption(f"{event['row_count']} linhas retornadas")
            if "ms" in event:
                st.caption(f"⏱ {event['ms']}ms")


def _run_question(question: str) -> None:
    initial_state: AgentState = {
        "question": question,
        "messages": st.session_state["messages"],
        "schema_text": "",
        "schema_fingerprint": "",
        "sql": None,
        "sql_error": None,
        "retries": 0,
        "rows": None,
        "row_count": 0,
        "status": "pending",
        "viz_type": "none",
        "viz_spec": None,
        "answer": "",
        "trace": [],
        "reflection": None,
    }
    with st.spinner("Analisando..."):
        result = graph.invoke(initial_state)

    st.session_state["messages"] = result["messages"]
    st.session_state["history_display"].append(
        (
            question,
            result["answer"],
            result.get("rows") or [],
            result.get("viz_spec") or result.get("viz_type", "none"),
            result.get("trace", []),
        )
    )
    st.session_state["pending_question"] = None
    st.rerun()


def _reset_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["history_display"] = []
    st.session_state["pending_question"] = None
    st.rerun()


# ── inicialização do session_state ────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "history_display" not in st.session_state:
    st.session_state["history_display"] = []  # [(question, answer, rows, viz_spec, trace)]

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

# ── grafo compilado uma vez por sessão ───────────────────────────────────────


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

# ── exemplos do PDF ───────────────────────────────────────────────────────────

EXEMPLOS = [
    "Quais os 5 clientes que mais gastaram?",
    "Qual a distribuição de compras por canal?",
    "Quantos tickets de suporte foram resolvidos vs não resolvidos?",
    "Qual campanha teve maior taxa de interação?",
    "Qual a evolução do valor de compras ao longo dos meses?",
]

# ── header ────────────────────────────────────────────────────────────────────

st.title("🔍 Data Assistant")
st.caption("Faça perguntas em português sobre os dados de clientes, compras, suporte e campanhas.")

pending = st.session_state["pending_question"]
busy = bool(pending)

# ── histórico da thread ───────────────────────────────────────────────────────

for item in st.session_state["history_display"]:
    q, answer, rows, viz_spec, trace = item
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(answer)
        render_viz(rows, viz_spec)
        _render_trace(trace)

if pending:
    with st.chat_message("user"):
        st.write(pending)

# ── atalhos ───────────────────────────────────────────────────────────────────

if not busy:
    with st.expander("Perguntas atalho"):
        st.caption(
            "Clique em um exemplo para enviar. Também pode perguntar qualquer "
            "coisa sobre os dados — o assistente responde."
        )
        for i, ex in enumerate(EXEMPLOS):
            if st.button(ex, key=f"shortcut_{i}", width="stretch"):
                st.session_state["pending_question"] = ex
                st.rerun()

# ── formulário de pergunta ────────────────────────────────────────────────────

st.divider()
submitted = False
user_input = ""
if busy:
    st.text_input("Sua pergunta:", value=pending, disabled=True)
else:
    with st.form("question_form", clear_on_submit=True):
        user_input = st.text_input(
            "Sua pergunta:",
            placeholder="Ex: Quais os clientes que mais compraram em SP?",
        )
        submitted = st.form_submit_button("Enviar", type="primary", width="stretch")

if st.button("Novo chat", type="secondary", width="stretch"):
    _reset_chat()

if submitted and user_input.strip():
    st.session_state["pending_question"] = user_input.strip()
    st.rerun()

if pending:
    _run_question(pending)
