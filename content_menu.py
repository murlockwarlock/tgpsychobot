from sqlalchemy import select

from database import UserMenuBinding, async_session_maker


async def remember_menu(user_id, targets, *, session_maker=async_session_maker):
    if user_id is None:
        return
    async with session_maker() as session:
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
        for label, kind, identity in targets:
            statement = insert(UserMenuBinding).values(user_id=user_id, label=label, resource_kind=kind, resource_id=str(identity), ambiguous=False)
            await session.execute(statement.on_conflict_do_update(
                index_elements=["user_id", "label"],
                set_={"ambiguous": UserMenuBinding.ambiguous | (UserMenuBinding.resource_kind != kind) | (UserMenuBinding.resource_id != str(identity))},
            ))
        await session.commit()


async def resolve_menu(user_id, label, kind, *, session_maker=async_session_maker):
    async with session_maker() as session:
        row = await session.get(UserMenuBinding, (user_id, label))
        if row and not row.ambiguous and row.resource_kind == kind:
            return row.resource_id
    return None


async def needs_menu_refresh(user_id, label, *, session_maker=async_session_maker):
    from translation_service import translation_cache
    snapshot = translation_cache.snapshot
    candidates = {value for key, value in snapshot.sources.items() if (key.startswith("topic.") and key.endswith(".name")) or (key.startswith("content.") and key.endswith(".button_title"))}
    candidates.update(value for (_, key), value in snapshot.translations.items() if (key.startswith("topic.") and key.endswith(".name")) or (key.startswith("content.") and key.endswith(".button_title")))
    if not label or label not in candidates:
        return False
    async with session_maker() as session:
        row = await session.get(UserMenuBinding, (user_id, label))
        return row is None or row.ambiguous
