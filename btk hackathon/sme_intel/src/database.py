"""
database.py — Production-ready SQLAlchemy ORM setup for SME-Intel.

Environment detection
---------------------
The engine is configured from the ``DATABASE_URL`` environment variable when
present (production / cloud), and falls back to a local SQLite file otherwise
(local development).

PostgreSQL URL scheme fix
-------------------------
Neon.tech, Supabase, Railway, and older Heroku dynos all emit connection
strings that begin with ``postgres://``.  SQLAlchemy 1.4+ **rejects** that
scheme; it only accepts ``postgresql://``.  The ``_normalise_db_url()``
helper performs a safe, prefix-only replacement before the URL is handed to
``create_engine()``.

Connection pool tuning (PostgreSQL only)
-----------------------------------------
Serverless cloud databases impose strict connection limits (Neon free tier:
10 concurrent connections).  The pool parameters below are chosen to be safe
for a single-dyno / single-instance Streamlit Cloud deployment:

    pool_size=2       — Keep 2 connections warm.
    max_overflow=3    — Allow 3 burst connections above pool_size.
    pool_timeout=30   — Raise after 30 s if no connection is available.
    pool_recycle=1800 — Recycle connections every 30 min to avoid stale-TCP
                        timeouts from the cloud provider's firewall.
    pool_pre_ping=True — Test the connection health before use; transparent
                         reconnect on transient network blips.
"""

from __future__ import annotations

import logging
import os
from datetime import date

from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------


def _normalise_db_url(url: str) -> str:
    """
    Replace the deprecated ``postgres://`` scheme with ``postgresql://``.

    SQLAlchemy 1.4+ raises ``NoSuchModuleError`` on ``postgres://`` because
    the dialect alias was removed.  Cloud providers (Neon, Supabase, Heroku,
    Railway) all still emit the old scheme — this one-liner fixes it without
    touching the rest of the URL.

    Args:
        url: Raw database URL string from the environment.

    Returns:
        A URL string with the scheme normalised to ``postgresql://``.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
        logger.debug("Normalised DB URL scheme: postgres:// → postgresql://")
    return url


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _build_engine() -> Engine:
    """
    Construct the SQLAlchemy engine based on the runtime environment.

    Resolution order
    ----------------
    1. ``DATABASE_URL`` env var is set  →  PostgreSQL (production).
    2. Not set                          →  SQLite local file (development).

    Returns:
        A fully configured ``Engine`` instance.
    """
    raw_url = os.environ.get("DATABASE_URL", "")

    if raw_url:
        # ── Production: PostgreSQL ───────────────────────────────────────
        database_url = _normalise_db_url(raw_url)
        logger.info("Using PostgreSQL database (production mode).")

        return create_engine(
            database_url,
            # ── Pool tuning for serverless PostgreSQL ────────────────
            pool_size=2,
            max_overflow=3,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,          # Detect stale connections automatically
            echo=False,
        )
    else:
        # ── Development: SQLite fallback ─────────────────────────────────
        _db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(_db_dir, exist_ok=True)
        database_url = f"sqlite:///{os.path.join(_db_dir, 'kobi_zeka.db')}"
        logger.info("DATABASE_URL not set — using local SQLite: %s", database_url)

        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

engine: Engine = _build_engine()

# Expose the resolved URL for logging (credentials are in the URL, so we
# redact the password portion before logging it).
DATABASE_URL: str = str(engine.url.render_as_string(hide_password=True))

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Project-wide declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------


class SalesRecordModel(Base):
    """
    Represents a single B2B sales transaction row.

    Columns
    -------
    id               : Auto-incremented surrogate key.
    transaction_date : Date the transaction occurred.
    customer_name    : Real customer / company name (masked before leaving DB layer).
    category         : Product or service category.
    quantity         : Units sold in this transaction.
    unit_sales_price : Revenue per unit (TRY).
    unit_cost        : Cost of goods/services per unit (TRY).
    payment_term     : Agreed payment term string (e.g. ``"Net 30"``).
    """

    __tablename__ = "sales_records"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    transaction_date: date = Column(Date, nullable=False)
    customer_name: str = Column(String(255), nullable=False)
    category: str = Column(String(100), nullable=False)
    quantity: int = Column(Integer, nullable=False)
    unit_sales_price: float = Column(Float, nullable=False)
    unit_cost: float = Column(Float, nullable=False)
    payment_term: str = Column(String(50), nullable=False)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SalesRecordModel(id={self.id!r}, "
            f"date={self.transaction_date!r}, "
            f"customer={self.customer_name!r}, "
            f"qty={self.quantity!r})"
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def init_db() -> None:
    """
    Create all tables declared on ``Base`` if they do not already exist.

    Idempotent: safe to call on every application start-up.
    Works for both SQLite (development) and PostgreSQL (production).
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised. Tables verified at: %s", DATABASE_URL)


def health_check() -> bool:
    """
    Execute a trivial ``SELECT 1`` against the configured database.

    Returns:
        ``True`` if the database is reachable, ``False`` otherwise.
        Never raises — intended for use in start-up probes and UI banners.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Database health check failed: %s", exc)
        return False
