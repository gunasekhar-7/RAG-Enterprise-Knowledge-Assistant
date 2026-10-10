import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, EmailStr, Field


class Role(StrEnum):
    EMPLOYEE = "Employee"
    MANAGER = "Manager"
    ADMINISTRATOR = "Administrator"


class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: Role = Role.EMPLOYEE
    tenant_id: uuid.UUID


class UserInDB(UserBase):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserResponse(UserBase):
    id: uuid.UUID
    is_active: bool
