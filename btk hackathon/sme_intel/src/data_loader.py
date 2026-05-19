"""
data_loader.py — Database-backed data loader for SME-Intel.

SECURITY CONTRACT
-----------------
This module is the **only** place that touches real customer names.
After querying the database, ``customer_name`` is immediately replaced with
opaque tokens via ``DataMasker`` before the DataFrame leaves this function.
The returned DataFrame MUST NEVER contain real identifiers.

Public API
----------
load_and_anonymize(masker) -> pd.DataFrame
    Query all sales records and return a fully masked DataFrame.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
from sqlalchemy.orm import Session

from src.database import SalesRecordModel, SessionLocal
from src.security import DataMasker

logger = logging.getLogger(__name__)

# Canonical column name used throughout the rest of the pipeline.
CUSTOMER_NAME_COLUMN: str = "Customer Name"


def load_and_anonymize(masker: DataMasker) -> pd.DataFrame:
    """
    Query all rows from ``sales_records`` and return an anonymised DataFrame.

    Steps
    -----
    1. Open a SQLAlchemy session and fetch every ``SalesRecordModel`` row.
    2. Convert the ORM objects to a list of dicts, then to a ``pd.DataFrame``.
    3. **Immediately** replace the ``customer_name`` column values with tokens
       produced by the caller-supplied ``DataMasker`` instance.
    4. Rename the column to the canonical ``"Customer Name"`` expected by the
       rest of the pipeline (``graph.py``, agents, etc.).
    5. Close the session in a ``finally`` block — no connections leak.

    Args:
        masker: A ``DataMasker`` instance owned by the current Streamlit
                session.  The same instance must be kept alive so that the
                UI can call ``masker.unmask_text()`` on LLM output later.

    Returns:
        A pandas DataFrame with columns::

            Customer Name (masked token) | transaction_date | category |
            quantity | unit_sales_price | unit_cost | payment_term

    Raises:
        RuntimeError: If the database query fails unexpectedly.
        ValueError:   If the table is empty (database not seeded yet).
    """
    session: Session = SessionLocal()
    try:
        rows: List[SalesRecordModel] = session.query(SalesRecordModel).all()

        if not rows:
            from src.seed import seed_database
            logger.info("Database is empty. Auto-seeding mock data...")
            seed_database()
            rows = session.query(SalesRecordModel).all()
            if not rows:
                raise ValueError("Auto-seeding failed or generated no records.")

        records = [
            {
                "customer_name_raw": row.customer_name,
                "transaction_date": row.transaction_date,
                "category": row.category,
                "quantity": row.quantity,
                "unit_sales_price": row.unit_sales_price,
                "unit_cost": row.unit_cost,
                "payment_term": row.payment_term,
            }
            for row in rows
        ]

        logger.info("Loaded %d raw records from database.", len(records))

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Database query failed.")
        raise RuntimeError(f"Failed to load data from database: {exc}") from exc
    finally:
        session.close()
        logger.debug("Database session closed.")

    df: pd.DataFrame = pd.DataFrame(records)

    # ── SECURITY: mask real names immediately, before the DataFrame
    #             is passed to any external code (LLM, logs, UI). ──────────
    df[CUSTOMER_NAME_COLUMN] = df["customer_name_raw"].apply(masker.mask)
    df.drop(columns=["customer_name_raw"], inplace=True)

    logger.info(
        "Anonymisation complete. %d unique customer tokens assigned.",
        df[CUSTOMER_NAME_COLUMN].nunique(),
    )

    return df
