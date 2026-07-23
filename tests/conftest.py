"""Pytest configuration and shared fixtures for the Dabba test suite.

Provides:
- ``test_db`` session-scoped fixture: an isolated temporary SQLite database
  that is created per test session and torn down afterward. This prevents
  cross-test pollution and keeps the dev database clean.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Silence noisy loggers during tests by default
logging.getLogger("dabba").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


@pytest.fixture(scope="session", autouse=True)
def test_db() -> Generator[Path, None, None]:
    """Create an isolated temporary SQLite database for the test session.

    The fixture:
      1. Creates a temp directory and a ``test_dabba.db`` file inside it.
      2. Overrides ``DABBA_DATABASE_URL`` in the environment so that
         ``DabbaConfig()`` reads the temporary DB path.
      3. Disposes any existing global engine (from a previous test run).
      4. Initializes the schema on the temp DB via ``init_db()``.
      5. After all tests in the session finish, disposes the engine and
         removes the temp directory.

    Any test that uses ``get_db()`` or ``DabbaConfig()`` will
    automatically pick up the isolated database — no per-test changes
    required.

    Yields:
        Path to the temporary database file.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="dabba_test_"))
    db_path = tmp_dir / "test_dabba.db"
    db_url = f"sqlite:///{db_path.as_posix()}"

    # Set environment variable so DabbaConfig() picks it up
    old_url = os.environ.get("DABBA_DATABASE_URL")
    os.environ["DABBA_DATABASE_URL"] = db_url

    # Import lazily inside fixture to avoid eager engine creation
    from dabba.database.session import dispose_engine, init_db

    # Dispose any cached engine from a previous session
    dispose_engine()

    # Create tables on the temp DB
    init_db()

    logger = logging.getLogger("dabba.tests.conftest")
    logger.info("Test DB created at %s", db_url)

    yield db_path

    # Teardown
    dispose_engine()
    if old_url is not None:
        os.environ["DABBA_DATABASE_URL"] = old_url
    else:
        del os.environ["DABBA_DATABASE_URL"]

    # Clean up temp directory
    try:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
