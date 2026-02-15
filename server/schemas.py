import uuid
from datetime import datetime
from pydantic import BaseModel


# ── Auth ───────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: uuid.UUID
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Ask ────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    model_name: str
    llm_model: str = "gemini-2.5-flash"
    conversation_id: uuid.UUID | None = None


class AskResponse(BaseModel):
    answer: str
    conversation_id: uuid.UUID


# ── Conversations ──────────────────────────────────────────────────

class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    model_name: str
    llm_model: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetail(BaseModel):
    id: uuid.UUID
    title: str
    model_name: str
    llm_model: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]

    model_config = {"from_attributes": True}
