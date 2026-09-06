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


def build_history_scope(message_cls, user_id: int, dialogue_id, topic_id, memory_mode: str):
    """Return a SQLAlchemy WHERE condition for history queries based on memory mode."""
    from sqlalchemy import and_
    if memory_mode == MEMORY_MODE_GLOBAL:
        return and_(message_cls.user_id == user_id, message_cls.dialogue_id == (dialogue_id or 1))
    if memory_mode == MEMORY_MODE_TOPIC:
        if topic_id is not None:
            return and_(
                message_cls.user_id == user_id,
                message_cls.dialogue_id == (dialogue_id or 1),
                message_cls.topic_id == topic_id,
            )
        return and_(
            message_cls.user_id == user_id,
            message_cls.dialogue_id == (dialogue_id or 1),
            message_cls.topic_id.is_(None),
        )
    # MEMORY_MODE_RESET: scope by dialogue
    return and_(message_cls.user_id == user_id, message_cls.dialogue_id == (dialogue_id or 1))


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


async def start_new_dialogue(session, user, topic_id: int, memory_mode: str) -> None:
    """Increment the user's dialogue_id to start a fresh context window."""
    user.current_dialogue_id = (user.current_dialogue_id or 1) + 1


async def apply_memory_mode_topic_switch(session, user, topic_id: int, memory_mode: str) -> bool:
    """Handle memory context when the user switches topics.

    Returns True if a previously saved context for the new topic was restored.
    """
    from database import UserTopicState  # local import to avoid circular dependency

    if memory_mode == MEMORY_MODE_GLOBAL:
        return False

    # Save current topic's dialogue state before switching (key 0 for main dialogue).
    current_topic_id = getattr(user, "current_topic_id", None)
    current_key = current_topic_id if current_topic_id is not None else 0
    saved = await session.get(UserTopicState, (user.id, current_key))
    if saved:
        saved.dialogue_id = user.current_dialogue_id
    else:
        session.add(UserTopicState(
            user_id=user.id,
            topic_id=current_key,
            dialogue_id=user.current_dialogue_id,
        ))

    if memory_mode == MEMORY_MODE_RESET:
        user.current_dialogue_id = (user.current_dialogue_id or 1) + 1
        return False

    # MEMORY_MODE_TOPIC: restore previous dialogue state for this topic if available (key 0 for main dialogue).
    target_key = topic_id if topic_id is not None else 0
    saved_target = await session.get(UserTopicState, (user.id, target_key))
    if saved_target:
        user.current_dialogue_id = saved_target.dialogue_id
        return True

    user.current_dialogue_id = (user.current_dialogue_id or 1) + 1
    session.add(UserTopicState(
        user_id=user.id,
        topic_id=target_key,
        dialogue_id=user.current_dialogue_id,
    ))
    return False
