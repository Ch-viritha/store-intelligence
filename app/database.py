"""
SQLite database setup using SQLAlchemy async.
SQLite chosen for zero-config docker compose up. Swap DATABASE_URL env var for PostgreSQL
in production — the SQLAlchemy async layer is database-agnostic.
Documented in CHOICES.md.
"""

import os
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Index

_DB_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/store_intelligence.db")

# Module-level engine and session factory — can be replaced by tests
engine: AsyncEngine = create_async_engine(_DB_URL, echo=False, future=True)
AsyncSessionLocal: async_sessionmaker = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


def get_engine() -> AsyncEngine:
    """Return the current engine. Tests can replace the module-level `engine`."""
    return engine


def get_session_factory() -> async_sessionmaker:
    """Return the current session factory. Tests can replace `AsyncSessionLocal`."""
    return AsyncSessionLocal


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    store_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    visitor_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=True)
    dwell_ms: Mapped[int] = mapped_column(Integer, default=0)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    queue_depth: Mapped[int] = mapped_column(Integer, nullable=True)
    sku_zone: Mapped[str] = mapped_column(String(50), nullable=True)
    session_seq: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("ix_store_ts", "store_id", "timestamp"),
        Index("ix_store_type", "store_id", "event_type"),
        Index("ix_visitor", "visitor_id", "store_id"),
    )


async def init_db():
    """Create all tables. Called on application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
