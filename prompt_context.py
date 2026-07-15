from __future__ import annotations


def apply_global_prompt_appendix(base_prompt: str | None, appendix: str | None) -> str:
    clean_appendix = (appendix or "").strip()
    clean_base = (base_prompt or "").strip()

    if not clean_appendix:
        return clean_base
    if not clean_base:
        return clean_appendix

    return f"{clean_base}\n\n[ОБЩИЙ БЛОК ИНСТРУКЦИЙ]\n{clean_appendix}"
