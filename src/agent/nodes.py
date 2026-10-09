import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.result_brief import build_narrate_context
from src.agent.state import AgentState, TraceEvent
from src.db.executor import (
    detect_fact_fanout,
    detect_identity_aggregation,
    guard_sql,
    inject_limit,
    run_sql,
)
from src.db.schema import get_dimension_tables, get_fact_tables, get_schema_text
from src.glossary import render_glossary
from src.llm.factory import get_chat_model
from src.llm.parser import extract_sql_from_response
from src.viz.decide import decide_viz


# ── prompts ───────────────────────────────────────────────────────────────────

_GLOSSARY_LINE = "O bloco Glossário do schema é a fonte da métrica. Obedeça esse bloco."

GENERATE_SQL_SYSTEM = f"""\
Você é um analista de dados especialista em SQL SQLite.
Recebe um schema de banco de dados e uma pergunta em português.

Banco de dados: SQLite.
Funções de data no SQLite (NUNCA use MONTH(), DATE_TRUNC(), EXTRACT() — não existem no SQLite):
- Mês   : strftime('%Y-%m', coluna_data) AS mes
- Ano   : strftime('%Y', coluna_data) AS ano
- Dia   : coluna_data diretamente (já é texto 'YYYY-MM-DD')
- Trimestre: strftime('%Y', coluna_data) || '-Q' || ((CAST(strftime('%m', coluna_data) AS INTEGER) - 1) / 3 + 1) AS trimestre

Regras para perguntas temporais:
- "evolução mensal" / "por mês" / "mensal" → GROUP BY strftime('%Y-%m', coluna_data) AS mes
- "por ano" / "anual" → GROUP BY strftime('%Y', coluna_data) AS ano
- "por trimestre" / "trimestral" → GROUP BY com a fórmula de trimestre acima
- NUNCA agrupe pela coluna de data bruta quando a pergunta pedir agregação por período.

Regras obrigatórias:
- Responda APENAS com SQL puro SELECT, sem markdown, sem explicação, sem comentários.
- {_GLOSSARY_LINE}
- OUT_OF_SCOPE somente quando o bloco Glossário indicar que o assunto não existe nas tabelas.
- Nunca invente tabelas, colunas ou valores que não existam no schema.
- Nunca use INSERT, UPDATE, DELETE, DROP, ALTER ou qualquer instrução que modifique dados.
- No schema, tabela -- fato tem FK; tabela -- dimensão não tem. Fato só faz JOIN com dimensão.
- Cada SELECT (cada lado do UNION) usa no máximo UMA fato. UNION ALL empilha listas; não cruza fatos.
- Em UNION ALL que empilha listas diferentes, cada lado ganha uma coluna de rótulo. Mantenha GROUP BY se a pergunta pedir por entidade. Exemplo de rótulo: 'lado A' e 'lado B'.
- Coluna marcada identidade: NUNCA SUM/AVG/MIN/MAX. COUNT(*) pode. Totais e médias só em coluna marcada medida.
"""

REPAIR_SQL_SYSTEM = f"""\
Você é um especialista em SQL SQLite.
Um SQL foi gerado e retornou erro ao ser executado no banco.
Corrija o SQL para responder corretamente a pergunta original.

Regras obrigatórias:
- Responda APENAS com SQL puro SELECT corrigido, sem markdown, sem explicação.
- {_GLOSSARY_LINE}
- Se não for possível corrigir com as tabelas disponíveis, responda exatamente: OUT_OF_SCOPE
- Nunca invente tabelas, colunas ou valores que não existam no schema.
- Se o erro falar em JOIN entre fatos: use EXISTS/IN com subquery isolada,
  ou pré-agregue um fato em subquery antes do JOIN principal. Nunca junte dois fatos diretamente.
- Nunca SUM/AVG em identidade. Use COUNT(*) ou coluna medida.
"""

REFLECT_SQL_SYSTEM = """\
Você é um especialista em SQL SQLite. Um SQL foi bloqueado pela validação do sistema.
Sua única tarefa é DIAGNOSTICAR o problema — não gere SQL.

IMPORTANTE — hierarquia de contexto:
- PERGUNTA ORIGINAL = fonte de verdade (golden context). Derive a intenção SOMENTE dela.
- SQL BLOQUEADO = artefato com erros semânticos. Use apenas para entender o tipo de erro técnico.
  Nunca incorpore condições ou filtros do SQL na interpretação da intenção.

Analise o SQL bloqueado, o erro recebido e a pergunta original.
Responda SOMENTE com JSON válido, sem markdown, sem explicação extra:

{
  "problema": "o que está errado no SQL em uma frase curta",
  "intencao": "o que a PERGUNTA ORIGINAL quer obter — derivado apenas do texto da pergunta",
  "estrategia": "como reescrever o SQL para evitar o problema — em uma ou duas frases objetivas"
}
"""

NARRATE_SUCCESS_SYSTEM = """\
Você é um analista de dados respondendo perguntas sobre um banco de dados.
Escreva uma resposta clara e objetiva em português.

Regras obrigatórias:
- Use APENAS os dados fornecidos. Não invente valores.
- Não acrescente informações além do que está nos dados.
- Seja direto. Máximo 3 parágrafos curtos.
- Não repita os dados em tabela — eles serão exibidos separadamente na interface.
- Copie os números do briefing. Não some, não calcule, não invente.
- Nunca afirme que uma categoria não existe se o briefing mostrar valor para ela.
- Nunca escreva as palavras RESULTADO, INSTRUÇÃO, forma ou briefing — isso é bastidor.
- Série temporal: cite pico, menor e extremos. Só fale subiu/desceu/igual se o briefing marcou isso no ponto. Não invente tendência mês a mês.
"""


# ── nós do grafo ──────────────────────────────────────────────────────────────

def node_load_schema(state: AgentState) -> dict:
    """Lê schema do banco (com cache) e anexa o glossário no mesmo texto."""
    schema_text, fingerprint = get_schema_text()
    schema_text = f"{schema_text}\n\n## Glossário\n{render_glossary()}"
    trace_event: TraceEvent = {
        "step": len(state["trace"]) + 1,
        "node": "load_schema",
        "status": "ok",
        "detail": f"Schema carregado. Fingerprint: {fingerprint[:8]}...",
    }
    return {
        "schema_text": schema_text,
        "schema_fingerprint": fingerprint,
        "status": "schema_loaded",
        "trace": state["trace"] + [trace_event],
    }


def node_generate_sql(state: AgentState) -> dict:
    """Chama LLM para gerar SQL. Detecta OUT_OF_SCOPE via parser."""
    t0 = time.monotonic()
    llm = get_chat_model()

    historico = _format_history(state["messages"])
    human_text = (
        f"Schema do banco:\n{state['schema_text']}\n\n"
        f"{historico}"
        f"Pergunta: {state['question']}"
    )

    response = llm.invoke([
        SystemMessage(content=GENERATE_SQL_SYSTEM),
        HumanMessage(content=human_text),
    ])
    sql = extract_sql_from_response(response.content)
    ms = round((time.monotonic() - t0) * 1000)

    if sql == "OUT_OF_SCOPE":
        trace_event: TraceEvent = {
            "step": len(state["trace"]) + 1,
            "node": "generate_sql",
            "status": "out_of_scope",
            "detail": "Modelo indicou que a pergunta está fora do escopo dos dados.",
            "ms": ms,
        }
        return {"sql": None, "status": "out_of_scope", "trace": state["trace"] + [trace_event]}

    trace_event = {
        "step": len(state["trace"]) + 1,
        "node": "generate_sql",
        "status": "ok",
        "detail": "SQL gerado.",
        "sql": sql,
        "ms": ms,
    }
    return {"sql": sql, "status": "sql_ready", "trace": state["trace"] + [trace_event]}


def node_guard_sql(state: AgentState) -> dict:
    """Valida se o SQL é read-only. Injeta LIMIT se necessário."""
    sql = state["sql"]
    allowed, error = guard_sql(sql)

    if not allowed:
        trace_event: TraceEvent = {
            "step": len(state["trace"]) + 1,
            "node": "guard_sql",
            "status": "blocked",
            "detail": f"SQL bloqueado: {error}",
            "sql": sql,
            "error": error,
        }
        return {"status": "blocked", "sql_error": error, "trace": state["trace"] + [trace_event]}

    fanout_error = detect_fact_fanout(sql)
    if fanout_error:
        return _guard_repair(state, sql, fanout_error, "JOIN entre tabelas fato bloqueado.")

    identity_error = detect_identity_aggregation(sql)
    if identity_error:
        return _guard_repair(state, sql, identity_error, "Agregação em coluna identidade bloqueada.")

    safe_sql = inject_limit(sql)
    trace_event = {
        "step": len(state["trace"]) + 1,
        "node": "guard_sql",
        "status": "ok",
        "detail": "SQL aprovado pelo guard.",
        "sql": safe_sql,
    }
    return {"sql": safe_sql, "status": "sql_ready", "trace": state["trace"] + [trace_event]}


def node_execute_sql(state: AgentState) -> dict:
    """Executa o SQL contra o banco. Sucesso e erro não somam retries."""
    t0 = time.monotonic()
    rows, error = run_sql(state["sql"])
    ms = round((time.monotonic() - t0) * 1000)

    if error:
        trace_event: TraceEvent = {
            "step": len(state["trace"]) + 1,
            "node": "execute_sql",
            "status": "exec_error",
            "detail": "Erro na execução.",
            "sql": state["sql"],
            "error": error,
            "ms": ms,
        }
        return {
            "rows": [],
            "row_count": 0,
            "sql_error": error,
            "error_source": "sqlite",
            "status": "exec_error",
            "trace": state["trace"] + [trace_event],
        }

    status = "empty" if len(rows) == 0 else "success"
    trace_event = {
        "step": len(state["trace"]) + 1,
        "node": "execute_sql",
        "status": status,
        "detail": f"{len(rows)} linhas retornadas.",
        "row_count": len(rows),
        "ms": ms,
    }
    return {
        "rows": rows,
        "row_count": len(rows),
        "sql_error": None,
        "status": status,
        "trace": state["trace"] + [trace_event],
    }


def node_reflect_sql(state: AgentState) -> dict:
    """
    Diagnostica o SQL bloqueado pelo guard.
    Gera JSON {problema, intencao, estrategia} para guiar o repair.
    Resultado é visível no trace e passado ao repair como contexto.
    """
    t0 = time.monotonic()
    llm = get_chat_model()

    human_text = (
        f"PERGUNTA ORIGINAL (fonte de verdade):\n{state['question']}\n\n"
        f"Schema do banco:\n{state['schema_text']}\n\n"
        f"SQL bloqueado (contém erros — use só para identificar o tipo de problema técnico):\n{state['sql']}\n\n"
        f"Erro do sistema:\n{state['sql_error']}\n\n"
        f"Diagnostique o problema."
    )

    response = llm.invoke([
        SystemMessage(content=REFLECT_SQL_SYSTEM),
        HumanMessage(content=human_text),
    ])
    ms = round((time.monotonic() - t0) * 1000)
    reflection = response.content.strip()

    trace_event: TraceEvent = {
        "step": len(state["trace"]) + 1,
        "node": "reflect_sql",
        "status": "ok",
        "detail": reflection,   # visível no trace da UI
        "ms": ms,
    }
    return {
        "reflection": reflection,
        "status": "exec_error",   # encaminha para repair_sql
        "trace": state["trace"] + [trace_event],
    }


def node_repair_sql(state: AgentState) -> dict:
    """Chama LLM para gerar SQL corrigido. Soma 1 em retries neste nó."""
    retries = state["retries"] + 1
    t0 = time.monotonic()
    llm = get_chat_model()

    if state.get("error_source") == "guard":
        human_text = repair_prompt_guard(state)
    else:
        human_text = repair_prompt_sqlite(state)

    response = llm.invoke([
        SystemMessage(content=REPAIR_SQL_SYSTEM),
        HumanMessage(content=human_text),
    ])
    sql = extract_sql_from_response(response.content)
    ms = round((time.monotonic() - t0) * 1000)

    if sql == "OUT_OF_SCOPE":
        trace_event: TraceEvent = {
            "step": len(state["trace"]) + 1,
            "node": "repair_sql",
            "status": "failed",
            "detail": "Modelo não conseguiu corrigir o SQL.",
            "ms": ms,
        }
        return {
            "sql": None,
            "retries": retries,
            "status": "failed",
            "reflection": None,
            "trace": state["trace"] + [trace_event],
        }

    if _same_sql(sql, state.get("sql")):
        trace_event = {
            "step": len(state["trace"]) + 1,
            "node": "repair_sql",
            "status": "failed",
            "detail": "SQL repetido. Correção encerrada.",
            "sql": sql,
            "ms": ms,
        }
        return {
            "sql": sql,
            "retries": retries,
            "status": "failed",
            "reflection": None,
            "trace": state["trace"] + [trace_event],
        }

    trace_event = {
        "step": len(state["trace"]) + 1,
        "node": "repair_sql",
        "status": "ok",
        "detail": f"SQL corrigido (tentativa {retries}).",
        "sql": sql,
        "ms": ms,
    }
    return {
        "sql": sql,
        "retries": retries,
        "status": "sql_ready",
        "reflection": None,
        "trace": state["trace"] + [trace_event],
    }


def node_decide_viz(state: AgentState) -> dict:
    """Escolhe encoding visual a partir da forma do resultado (não da pergunta)."""
    spec = decide_viz(state["rows"] or [])
    mark = spec["mark"]
    trace_event: TraceEvent = {
        "step": len(state["trace"]) + 1,
        "node": "decide_viz",
        "status": "ok",
        "detail": f"Visualização escolhida: {mark}",
    }
    return {"viz_spec": spec, "viz_type": mark, "trace": state["trace"] + [trace_event]}


def node_narrate(state: AgentState) -> dict:
    """
    Gera a resposta final.
    Success → LLM ancora no briefing em código (`result_brief`).
    Empty / Out of scope / Failed → template determinístico.
    Empilha pergunta + resposta em messages ao final.
    """
    status = state["status"]

    if status == "success":
        answer = _narrate_llm(state)
    elif status == "empty":
        answer = "Não encontrei registros para esse filtro com os dados disponíveis."
    elif status == "out_of_scope":
        tables = ", ".join(sorted(get_dimension_tables() | get_fact_tables()))
        answer = (
            f"Com as tabelas disponíveis ({tables}) "
            f"não é possível responder: \"{state['question']}\""
        )
    else:
        answer = f"Não consegui montar uma consulta válida após {state['retries']} tentativa(s)."

    trace_event: TraceEvent = {
        "step": len(state["trace"]) + 1,
        "node": "narrate",
        "status": "ok",
        "detail": "Resposta gerada.",
    }

    updated_messages = state["messages"] + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]

    return {
        "answer": answer,
        "messages": updated_messages,
        "trace": state["trace"] + [trace_event],
    }


def node_fail_graceful(state: AgentState) -> dict:
    """Encerra o grafo com mensagem honesta de falha."""
    reason = state.get("sql_error") or "motivo desconhecido"
    answer = f"Não consegui processar essa consulta. Motivo: {reason}"
    trace_event: TraceEvent = {
        "step": len(state["trace"]) + 1,
        "node": "fail_graceful",
        "status": "failed",
        "detail": f"Encerrado com falha: {reason}",
        "error": reason,
    }
    updated_messages = state["messages"] + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]
    return {
        "answer": answer,
        "status": "failed",
        "viz_type": "none",
        "viz_spec": {"mark": "none", "label": None, "values": [], "label_cols": []},
        "messages": updated_messages,
        "trace": state["trace"] + [trace_event],
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def repair_prompt_guard(state: AgentState) -> str:
    """Porta do guard: o SQL barrado não volta para o reparo."""
    return (
        f"PERGUNTA ORIGINAL (fonte de verdade):\n{state['question']}\n\n"
        f"Schema do banco:\n{state['schema_text']}\n\n"
        f"Erro do sistema (tipo de problema estrutural):\n{state['sql_error']}\n\n"
        f"Diagnóstico (use como guia principal):\n{state['reflection']}\n\n"
        f"Escreva o SQL correto do zero, sem reutilizar condições do SQL anterior."
    )


def repair_prompt_sqlite(state: AgentState) -> str:
    """Porta do SQLite: a mensagem aponta o texto, então o SQL quebrado fica visível."""
    return (
        f"Schema do banco:\n{state['schema_text']}\n\n"
        f"Pergunta original: {state['question']}\n\n"
        f"SQL que falhou:\n{state['sql']}\n\n"
        f"Erro retornado pelo SQLite:\n{state['sql_error']}\n\n"
        f"Diagnóstico (use como guia principal):\n{state.get('reflection') or ''}\n\n"
        f"Corrija o SQL."
    )


def _same_sql(new_sql: str | None, previous_sql: str | None) -> bool:
    if not new_sql or not previous_sql:
        return False
    return new_sql.strip().lower() == previous_sql.strip().lower()


def _guard_repair(state: AgentState, sql: str, error: str, detail: str) -> dict:
    """Guard que encaminha para reflect → repair (não é DML bloqueado). Não soma retries."""
    trace_event: TraceEvent = {
        "step": len(state["trace"]) + 1,
        "node": "guard_sql",
        "status": "exec_error",
        "detail": detail,
        "sql": sql,
        "error": error,
    }
    return {
        "status": "reflect_needed",
        "sql_error": error,
        "error_source": "guard",
        "trace": state["trace"] + [trace_event],
    }


def _format_history(messages: list[dict]) -> str:
    """Formata as últimas 3 trocas do histórico para o prompt."""
    if not messages:
        return ""
    lines = ["Histórico da conversa:"]
    for m in messages[-6:]:  # últimas 3 trocas (6 mensagens)
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:300]}")
    return "\n".join(lines) + "\n\n"


def _narrate_llm(state: AgentState) -> str:
    """Chama LLM para narrar o resultado com dados ancorados no briefing."""
    llm = get_chat_model()
    context = build_narrate_context(state["rows"] or [], state["row_count"])
    human_text = (
        f"Pergunta: {state['question']}\n\n"
        f"{context}\n\n"
        f"Escreva a resposta."
    )
    response = llm.invoke([
        SystemMessage(content=NARRATE_SUCCESS_SYSTEM),
        HumanMessage(content=human_text),
    ])
    return response.content.strip()
