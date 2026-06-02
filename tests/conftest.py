"""
Shared test configuration.
Overrides the FastAPI get_db dependency with a fresh in-memory SQLite database
for every test, ensuring full isolation without touching the real DB file.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.database import Base
from app.main import app
import app.database as db_module


@pytest_asyncio.fixture(autouse=True)
async def isolated_db():
    """
    Replace the module-level engine and session factory with a fresh in-memory
    SQLite instance for each test. Override the FastAPI dependency so all
    endpoints use this clean DB.
    """
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", echo=False, future=True
    )
    test_session_factory = async_sessionmaker(
        test_engine, expire_on_commit=False, class_=AsyncSession
    )

    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Patch module-level references used by get_db()
    original_engine = db_module.engine
    original_factory = db_module.AsyncSessionLocal

    db_module.engine = test_engine
    db_module.AsyncSessionLocal = test_session_factory

    # Override the FastAPI dependency
    async def override_get_db():
        async with test_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    from app.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    yield

    # Restore originals
    app.dependency_overrides.clear()
    db_module.engine = original_engine
    db_module.AsyncSessionLocal = original_factory
    await test_engine.dispose()
