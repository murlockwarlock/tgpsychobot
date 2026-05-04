from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from .legacy import async_session_maker
from .logging_utils import get_bot_logger
from .time_utils import utc_now


StorageBase = declarative_base()
log = get_bot_logger("state")


class MaxState(StorageBase):
    __tablename__ = "max_bot_states"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)


class MaxContentMedia(StorageBase):
    __tablename__ = "max_content_media"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    media_type: Mapped[str] = mapped_column(String(32), nullable=False)
    token: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)


class MaxTopicMedia(StorageBase):
    __tablename__ = "max_topic_media"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    media_type: Mapped[str] = mapped_column(String(32), nullable=False)
    token: Mapped[str] = mapped_column(String(512), nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    category: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)


async def init_storage() -> None:
    from .legacy import engine

    async with engine.begin() as conn:
        await conn.run_sync(StorageBase.metadata.create_all)


@dataclass
class StateSnapshot:
    state: str
    data: dict


def _load_payload(payload_json: str | None, user_id: int) -> dict:
    try:
        loaded = json.loads(payload_json or "{}")
    except json.JSONDecodeError:
        log.exception("State payload decode failed user_id=%s payload=%r", user_id, payload_json)
        return {}
    return loaded if isinstance(loaded, dict) else {}


class StateStore:
    async def get(self, user_id: int) -> StateSnapshot | None:
        async with async_session_maker() as session:
            row = await session.get(MaxState, user_id)
            if not row:
                return None
            return StateSnapshot(state=row.state, data=_load_payload(row.payload_json, user_id))

    async def set(self, user_id: int, chat_id: int, state: str, data: dict | None = None) -> None:
        payload = json.dumps(data or {}, ensure_ascii=False)
        async with async_session_maker() as session:
            row = await session.get(MaxState, user_id)
            if row:
                row.chat_id = chat_id
                row.state = state
                row.payload_json = payload
                row.updated_at = utc_now()
            else:
                session.add(
                    MaxState(
                        user_id=user_id,
                        chat_id=chat_id,
                        state=state,
                        payload_json=payload,
                    )
                )
            await session.commit()
        log.info("State set user_id=%s chat_id=%s state=%s keys=%s", user_id, chat_id, state, sorted((data or {}).keys()))

    async def update(self, user_id: int, **data: object) -> None:
        async with async_session_maker() as session:
            row = await session.get(MaxState, user_id)
            if not row:
                return
            payload = _load_payload(row.payload_json, user_id)
            payload.update(data)
            row.payload_json = json.dumps(payload, ensure_ascii=False)
            row.updated_at = utc_now()
            await session.commit()
        log.info("State updated user_id=%s keys=%s", user_id, sorted(data.keys()))

    async def clear(self, user_id: int) -> None:
        async with async_session_maker() as session:
            row = await session.get(MaxState, user_id)
            if row:
                await session.delete(row)
                await session.commit()
                log.info("State cleared user_id=%s state=%s", user_id, row.state)
