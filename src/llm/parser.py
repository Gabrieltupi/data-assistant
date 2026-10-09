import re


def extract_sql_from_response(text: str) -> str:
    """
    Extrai SQL puro de resposta do modelo (que pode vir em markdown).
    Retorna "OUT_OF_SCOPE" se não encontrar SQL válido.

    Ordem de tentativas:
    1. Token OUT_OF_SCOPE explícito
    2. Bloco ```sql ... ```
    3. Bloco genérico ``` ... ```
    4. Texto direto (fallback)
    """
    text = text.strip()

    # 1. Token de recusa explícito
    if "OUT_OF_SCOPE" in text.upper():
        return "OUT_OF_SCOPE"

    # 2. Bloco ```sql ... ```
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return _clean(match.group(1))

    # 3. Bloco genérico ``` ... ```
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return _clean(match.group(1))

    # 4. Texto direto (fallback)
    return _clean(text)


def _clean(sql: str) -> str:
    """Remove trailing ; e valida que começa com SELECT ou WITH."""
    sql = sql.strip().rstrip(";").strip()
    upper = sql.upper().lstrip()
    if upper.startswith("SELECT") or upper.startswith("WITH"):
        return sql
    return "OUT_OF_SCOPE"
