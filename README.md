# Data Assistant

Assistente de dados inteligente que responde perguntas em linguagem natural sobre um banco SQLite.

Desenvolvido como desafio técnico para a vaga de **Engenheiro de IA Pleno** na [Franq Open Banking](https://franq.com.br).

Tratei o desafio como um produto. Quem pergunta em português precisa sair com uma resposta que dá para usar: o número que o banco sustenta, a tabela ou o gráfico na forma do resultado, e o raciocínio aberto para conferir o SQL. O agente é assertivo no que os dados sustentam e direto quando a pergunta sai desse banco. É assim que eu trabalho: a experiência de quem usa, e um agente que se comporta como ferramenta de verdade.

---

## O que ele faz

- Recebe perguntas em português sobre dados de clientes, compras, suporte e campanhas
- Gera SQL automaticamente usando LLM
- Corrige SQL com erro via loop de repair (até 3 tentativas)
- Exibe resposta em texto + tabela ou gráfico conforme a forma do resultado
- Mostra o raciocínio completo (schema lido, SQL gerado, erros, correções)
- Mantém o histórico da sessão na tela ("Novo chat" zera). A pergunta seguinte recebe a prosa recente, não a tabela da consulta anterior

---

## Arquitetura

```
Pergunta
  → load_schema (SQLite introspection com cache)
  → generate_sql (LLM → OUT_OF_SCOPE ou SQL puro)
  → guard_sql (read-only, LIMIT, fanout, identidade)
      ├── DML / multi-statement     → fail_graceful
      ├── fanout / identidade       → reflect_sql → repair_sql → guard_sql
      └── ok                        → execute_sql
  → execute_sql
      ├── ok + rows     → decide_viz → narrate → resposta
      ├── ok + 0 rows   → narrate ("não encontrei")
      ├── erro SQLite   → reflect_sql → repair_sql → guard_sql
      └── limite retry  → fail_graceful
```

O grafo é construído com **LangGraph**. O modelo LLM é plugável via `.env` (`ollama`, `google`, `openai` ou `anthropic`).

O LLM **não conta** o resultado. Depois do `execute_sql`, um briefing em código (`src/agent/result_brief.py`) detecta a forma dos dados e fecha os números; o narrate só redige. O gráfico segue a mesma forma (barra, linha, barras agrupadas) — não a palavra da pergunta.

`reflect_sql` diagnostica antes do reparo, em JSON visível no rastro, e não gera SQL. No bloqueio do guard, o reparo não recebe o SQL barrado. No erro do SQLite, o reparo recebe o SQL e a mensagem, porque o erro aponta o texto. Se o reparo devolver a mesma consulta, o fluxo para.

Papéis vêm do SQLite: PK/FK/`id` = identidade; número que não é chave = medida; tabela com FK = fato. `SUM`/`AVG` em identidade é barrado. Dois fatos no `FROM`/`JOIN` direto são barrados; `EXISTS`/`IN` e derived tables passam. Tabela nova com FK entra sozinha.

---

## Instalação

### Pré-requisito

- Python 3.11+

O LLM **não** é pré-requisito fixo. No `.env` vale Ollama, Gemini, OpenAI ou Claude.

### 1. Clone o repositório e entre na pasta do projeto

### 2. Instale as dependências

```bash
pip install -e .
# opcional — só se for rodar os testes:
pip install -e ".[dev]"
```

### 3. Configure o ambiente

```bash
cp .env.example .env
```

Escolha o provider (veja a seção abaixo). Se for Ollama, instale o [Ollama](https://ollama.com), suba o serviço e baixe um modelo, por exemplo `ollama pull qwen2.5-coder:7b`. Se for Gemini, OpenAI ou Claude, só a chave no `.env`.

### 4. Execute

```bash
streamlit run src/app.py
```

Acesse `http://localhost:8501`

---

## Provedores de LLM

O grafo não muda. No `.env`, `LLM_PROVIDER` pode ser:

| Valor | Modelo |
|---|---|
| `ollama` | Ollama local (padrão, sem chave) |
| `google` | Gemini |
| `openai` | OpenAI |
| `anthropic` | Claude (`claude` também vale) |

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b

# LLM_PROVIDER=google
# GOOGLE_API_KEY=sua-chave
# GOOGLE_MODEL=gemini-2.0-flash

# LLM_PROVIDER=openai
# OPENAI_API_KEY=sua-chave
# OPENAI_MODEL=gpt-4o-mini

# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sua-chave
# ANTHROPIC_MODEL=claude-sonnet-4-5
```

Reinicie o Streamlit depois de trocar o provider.

---

## Banco de dados

O banco `data/anexo_desafio_1.db` está incluído no repositório. Contém 4 tabelas:

| Tabela | Registros | Descrição |
|---|---|---|
| `clientes` | 100 | Dados cadastrais e valor total gasto |
| `compras` | 946 | Histórico de compras por categoria e canal |
| `suporte` | 273 | Tickets de atendimento |
| `campanhas_marketing` | 248 | Campanhas enviadas e interações |

---

## Estrutura do projeto

```
data-assistant/
├── src/
│   ├── app.py              # Interface Streamlit
│   ├── config.py           # Settings via .env
│   ├── glossary.py         # Contrato de métrica (texto para o schema)
│   ├── llm/
│   │   ├── factory.py      # Provider pluggable (Ollama/Gemini/OpenAI/Claude)
│   │   └── parser.py       # Extrai SQL de respostas markdown do modelo
│   ├── db/
│   │   ├── schema.py       # Introspection SQLite com cache
│   │   └── executor.py     # Guard read-only + fanout + identidade + execução
│   ├── agent/
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── nodes.py        # Nós do grafo + prompts
│   │   ├── graph.py        # Compilação LangGraph
│   │   └── result_brief.py # Forma do resultado → contexto do narrate
│   └── viz/
│       ├── decide.py       # Forma do resultado → spec de gráfico
│       └── render.py       # Altair a partir do spec
├── tests/
├── glossary/
│   └── metrics.yaml        # Gasto, ticket, vocabulário do banco
├── data/
│   └── anexo_desafio_1.db  # Banco SQLite do desafio
├── examples/
│   └── perguntas.md        # Exemplos testados
├── .env.example
└── pyproject.toml
```

---

## Exemplos de perguntas

```
Quais os 5 clientes que mais gastaram?
Qual a distribuição de compras por canal?
Quantos tickets foram resolvidos vs não resolvidos?
Qual campanha teve maior taxa de interação?
Qual a evolução do valor de compras ao longo dos meses?
```

Veja mais em [`examples/perguntas.md`](examples/perguntas.md).

---

## Testes

Os testes não chamam LLM nem sobem o Streamlit. Cobrem guard, schema, glossário, retries, factory e gráfico.

```bash
pip install -e ".[dev]"
python -m pytest
```

---

## Decisões técnicas

| Decisão | Escolha |
|---|---|
| LLM | Plugável: `ollama`, `google`, `openai`, `anthropic` |
| Observabilidade | Expander **Raciocínio** na UI (passos e SQL) — o que o desafio pede |
| Guard SQL | Allowlist SELECT/WITH; banco aberto em modo somente leitura |
| Schema | Introspection dinâmica a cada pergunta (cache por hash do arquivo) |
| Histórico | A tela guarda a thread da sessão. O generate recebe a prosa recente, não as linhas da consulta anterior |

---

## Sugestões de melhoria

### Como produto, em produção

- Resposta em streaming, com o passo atual na tela, para a espera do modelo não parecer travamento.
- Pergunta de volta quando uma palavra aponta para duas métricas. O agente confirma qual delas, em vez de escolher.
- A pergunta seguinte poder usar a tabela da anterior, não só a prosa recente.
- Gráfico na forma que a pessoa pedir, além da forma que o resultado já sugere.
- Sessão salva, para continuar no dia seguinte.
- Acesso por usuário, timeout e teto de custo do modelo.
- Suite de avaliação nas perguntas conhecidas, rodada antes de cada mudança de prompt.

### Técnicas

- LangFuse com o mesmo rastro de hoje: SQL, reparo e latência.
- API (FastAPI) na frente do grafo, para o Streamlit não ser a única porta.
- Tool calling só com `run_sql` e `refuse`. O guard continua fora do modelo, e isso vale quando o modelo devolve tool call estável.
- Planejador para pergunta que precisa de mais de uma consulta, com resultado intermediário.
- O briefing marcar o topo e o fundo da lista, para a redação não reinterpretar um ranking que o código já tem.
- Taxa de interação de campanha no glossário, com o denominador explícito.
