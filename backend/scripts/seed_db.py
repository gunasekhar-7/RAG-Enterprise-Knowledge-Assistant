import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_password_hash
from app.db.session import async_sessionmaker, engine
from app.models.db import UserModel
from app.models.user import Role


async def seed_users():
    async with engine.begin() as _conn:
        # We don't drop all tables in a normal seed, just insert.
        pass

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        # Create a common tenant
        tenant_id = uuid.uuid4()

        users_to_create = [
            UserModel(
                email="admin@example.com",
                full_name="Administrator",
                hashed_password=get_password_hash("password123"),
                role=Role.ADMINISTRATOR,
                tenant_id=tenant_id,
            ),
            UserModel(
                email="manager@example.com",
                full_name="Manager User",
                hashed_password=get_password_hash("password123"),
                role=Role.MANAGER,
                tenant_id=tenant_id,
            ),
            UserModel(
                email="employee@example.com",
                full_name="Employee User",
                hashed_password=get_password_hash("password123"),
                role=Role.EMPLOYEE,
                tenant_id=tenant_id,
            ),
        ]

        for user in users_to_create:
            session.add(user)

        await session.commit()
        print("Successfully seeded 3 users.")


if __name__ == "__main__":
    asyncio.run(seed_users())
