"""Database session management for Dabba.

Provides a configured SQLAlchemy ``Engine``, session factory, and a
``get_db`` context-manager / generator for use in both synchronous
scripts and FastAPI dependency injection.

Usage (script):
    >>> from src.dabba.database.session import get_db, init_db
    >>> init_db()
    >>> with get_db() as db:
    ...     results = db.query(Restaurant).all()

Usage (FastAPI):
    .. code-block:: python

        from fastapi import Depends
        from sqlalchemy.orm import Session
        from src.dabba.database.session import get_db

        @app.get("/restaurants")
        def list_restaurants(db: Session = Depends(get_db)):
            return db.query(Restaurant).all()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from dabba.config import DabbaConfig, get_config
from dabba.database.models import Base

logger = logging.getLogger(__name__)

# Module-level engine & sessionmaker (lazy-initialized)
_engine = None
_SessionLocal = None


def _get_engine(config: DabbaConfig | None = None):
    """Create (or return) a configured SQLAlchemy Engine.

    For SQLite, enables WAL mode and foreign keys.
    """
    global _engine
    if _engine is not None:
        return _engine

    config = config or get_config()
    url = config.database_url

    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False  # Allow multi-threaded access

    _engine = create_engine(
        url,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    # Enable WAL mode and foreign keys for SQLite
    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    logger.info("Database engine created: %s", url)
    return _engine


def _get_sessionmaker(config: DabbaConfig | None = None):
    """Create (or return) a configured sessionmaker."""
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal

    engine = _get_engine(config)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def init_db(config: DabbaConfig | None = None) -> None:
    """Create all tables if they don't exist.

    Safe to call multiple times — SQLAlchemy's ``create_all`` is
    idempotent (uses ``IF NOT EXISTS`` internally).

    Args:
        config: Project configuration.
    """
    engine = _get_engine(config)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created")


def dispose_engine() -> None:
    """Dispose the engine (for testing / teardown)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
        logger.info("Database engine disposed")


@contextmanager
def get_db() -> Iterator[Session]:
    """Context manager that yields a database session.

    Automatically commits on success and rolls back on exception.

    Yields:
        SQLAlchemy Session.
    """
    session_maker = _get_sessionmaker()
    db: Session = session_maker()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_generator() -> Generator[Session, None, None]:
    """Generator-based get_db for FastAPI ``Depends(get_db_generator)``.

    Usage:
        .. code-block:: python

            from fastapi import Depends
            @app.get("/items")
            def list_items(db: Session = Depends(get_db_generator)):
                ...
    """
    session_maker = _get_sessionmaker()
    db: Session = session_maker()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
