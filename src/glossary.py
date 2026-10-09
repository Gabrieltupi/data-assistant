from pathlib import Path

import yaml

_GLOSSARY_PATH = Path(__file__).resolve().parents[1] / "glossary" / "metrics.yaml"
_REQUIRED = ("nome", "sinonimos", "formula")


def render_glossary() -> str:
    """Lê o contrato de métrica e devolve o bloco de texto. Não abre banco."""
    if not _GLOSSARY_PATH.is_file():
        raise FileNotFoundError(f"Glossário ausente: {_GLOSSARY_PATH}")

    data = yaml.safe_load(_GLOSSARY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Glossário vazio.")

    lines: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Entrada do glossário inválida.")
        for key in _REQUIRED:
            value = item.get(key)
            if value is None or value == "" or value == []:
                raise ValueError(f"Campo obrigatório vazio: {key}")
        lines.append(f"- {item['nome']}")
        lines.append(f"  sinônimos: {', '.join(item['sinonimos'])}")
        lines.append(f"  fórmula: {item['formula']}")
        if item.get("nao_usar"):
            lines.append(f"  não usar: {item['nao_usar']}")
        if item.get("count_quando"):
            lines.append(f"  contar só quando: {item['count_quando']}")
    return "\n".join(lines)
