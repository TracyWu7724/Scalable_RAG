import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import User, Conversation, Message


# ── Users ──────────────────────────────────────────────────────────

async def create_user(session: AsyncSession, username: str, password_hash: str) -> User:
    user = User(username=username, password_hash=password_hash)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


# ── Conversations ──────────────────────────────────────────────────

async def create_conversation(
    session: AsyncSession,
    user_id: uuid.UUID,
    title: str,
    model_name: str,
    llm_model: str = "gemini-2.5-flash",
) -> Conversation:
    conv = Conversation(user_id=user_id, title=title, model_name=model_name, llm_model=llm_model)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


async def get_conversations_for_user(
    session: AsyncSession, user_id: uuid.UUID
) -> list[Conversation]:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID
) -> Conversation | None:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    return result.scalar_one_or_none()


# ── Messages ───────────────────────────────────────────────────────

async def add_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    session.add(msg)
    # Also bump conversation.updated_at
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if conv:
        conv.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(msg)
    return msg


async def get_history(
    session: AsyncSession, conversation_id: uuid.UUID, limit: int = 10
) -> list[Message]:
    """Return the last `limit` messages for a conversation, oldest first."""
    # Sub-query to get the last N messages (newest first), then re-order oldest first
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()  # oldest first
    return messages
