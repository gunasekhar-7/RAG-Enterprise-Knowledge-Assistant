import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class RoleEnum(str):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    conversation_id: uuid.UUID
    role: str
    content: str
    token_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    title: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
