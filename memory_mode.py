MEMORY_MODE_RESET = "reset"
MEMORY_MODE_TOPIC = "topic"
MEMORY_MODE_GLOBAL = "global"

MEMORY_MODE_VALUES = {
    MEMORY_MODE_RESET,
    MEMORY_MODE_TOPIC,
    MEMORY_MODE_GLOBAL,
}


def normalize_memory_mode(ai_config) -> str:
    return get_memory_mode(ai_config)


def get_memory_mode(ai_config) -> str:
    mode = getattr(ai_config, "memory_mode", None)
    if mode in MEMORY_MODE_VALUES:
        return mode

    preserve_topic_context = bool(getattr(ai_config, "preserve_topic_context", False))
    return MEMORY_MODE_TOPIC if preserve_topic_context else MEMORY_MODE_RESET


def next_memory_mode(current_mode: str) -> str:
    if current_mode == MEMORY_MODE_RESET:
        return MEMORY_MODE_TOPIC
    if current_mode == MEMORY_MODE_TOPIC:
        return MEMORY_MODE_GLOBAL
    return MEMORY_MODE_RESET


def memory_mode_label(mode: str) -> str:
    if mode == MEMORY_MODE_TOPIC:
        return "По темам"
    if mode == MEMORY_MODE_GLOBAL:
        return "Глобальная"
    return "Сброс при смене"


def memory_mode_description(mode: str) -> str:
    if mode == MEMORY_MODE_TOPIC:
        return "Память сохраняется отдельно внутри каждой темы."
    if mode == MEMORY_MODE_GLOBAL:
        return "Контекст диалога сохраняется между темами, но используется промпт текущей темы."
    return "При переключении темы память сбрасывается."


def is_topic_memory_mode(mode: str) -> bool:
    return mode == MEMORY_MODE_TOPIC


def is_global_memory_mode(mode: str) -> bool:
    return mode == MEMORY_MODE_GLOBAL
