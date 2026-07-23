"""Shared helpers for knowledge-base administration."""

from __future__ import annotations

import os

from sqlalchemy import delete, select

from database import IndexingQueue, KnowledgeBase, topic_knowledgebase_association


async def find_original_kb_file_id(session, entry: KnowledgeBase) -> str | None:
    """Return the latest completed Telegram upload matching a KB entry."""
    if not entry.filename:
        return None
    return await session.scalar(
        select(IndexingQueue.file_id)
        .where(
            IndexingQueue.filename == entry.filename,
            IndexingQueue.status == "completed",
        )
        .order_by(IndexingQueue.created_at.desc(), IndexingQueue.id.desc())
        .limit(1)
    )


def kb_text_export_filename(filename: str | None) -> str:
    stem = os.path.splitext(os.path.basename(filename or "knowledge_base"))[0].strip()
    return f"{stem or 'knowledge_base'}_БЗ.txt"


async def delete_knowledge_base_record(session, kb_id: int) -> str | None:
    """Delete a KB entry after removing its topic links."""
    entry = await session.get(KnowledgeBase, kb_id)
    if entry is None:
        return None

    filename = entry.filename or f"KB #{kb_id}"
    await session.execute(
        delete(topic_knowledgebase_association).where(
            topic_knowledgebase_association.c.knowledge_base_id == kb_id
        )
    )
    await session.execute(delete(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    return filename
