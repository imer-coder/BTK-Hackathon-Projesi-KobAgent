"""
seed.py — Mock B2B data seeder for SME-Intel.

Populates the ``sales_records`` table with realistic Turkish SME data that
deliberately encodes two detectable signal patterns:

1. **Churn Risk** — customers whose order quantities are monotonically or
   sharply declining across successive quarters, signalling disengagement.

2. **Margin Squeeze** — customers whose ``unit_cost`` has grown faster than
   ``unit_sales_price`` over time, eroding gross margin below a safe threshold.

Idempotency guarantee
---------------------
The script checks whether the table already contains rows before inserting.
If any rows exist the seed is skipped and the script exits cleanly.  This
means ``python -m src.seed`` can be called safely on every container restart.

Usage::

    python -m src.seed
"""

from __future__ import annotations

import logging
import sys
from datetime import date

from src.database import SessionLocal, SalesRecordModel, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock dataset
# ---------------------------------------------------------------------------

# fmt: off
_MOCK_RECORDS: list[dict] = [
    # ── CHURN RISK: Yıldız Tekstil A.Ş. ─────────────────────────────────
    # Quantity craters from 500 → 80 across four quarters.
    {"transaction_date": date(2024, 1, 15), "customer_name": "Yıldız Tekstil A.Ş.",    "category": "Ham Kumaş",          "quantity": 500, "unit_sales_price": 85.00,  "unit_cost": 52.00,  "payment_term": "Net 60"},
    {"transaction_date": date(2024, 4, 10), "customer_name": "Yıldız Tekstil A.Ş.",    "category": "Ham Kumaş",          "quantity": 320, "unit_sales_price": 85.00,  "unit_cost": 53.50,  "payment_term": "Net 60"},
    {"transaction_date": date(2024, 7, 22), "customer_name": "Yıldız Tekstil A.Ş.",    "category": "Ham Kumaş",          "quantity": 150, "unit_sales_price": 86.00,  "unit_cost": 55.00,  "payment_term": "Net 60"},
    {"transaction_date": date(2024, 10, 5), "customer_name": "Yıldız Tekstil A.Ş.",    "category": "Ham Kumaş",          "quantity": 80,  "unit_sales_price": 86.00,  "unit_cost": 55.00,  "payment_term": "Net 90"},

    # ── MARGIN SQUEEZE: Demirtaş Makine Ltd. ────────────────────────────
    # Costs rise 38 % while prices rise only 8 %; margin collapses from 35 % → 8 %.
    {"transaction_date": date(2024, 1, 20), "customer_name": "Demirtaş Makine Ltd.",    "category": "Endüstriyel Parça",  "quantity": 200, "unit_sales_price": 310.00, "unit_cost": 200.00, "payment_term": "Net 30"},
    {"transaction_date": date(2024, 4, 18), "customer_name": "Demirtaş Makine Ltd.",    "category": "Endüstriyel Parça",  "quantity": 210, "unit_sales_price": 318.00, "unit_cost": 225.00, "payment_term": "Net 30"},
    {"transaction_date": date(2024, 7, 14), "customer_name": "Demirtaş Makine Ltd.",    "category": "Endüstriyel Parça",  "quantity": 195, "unit_sales_price": 325.00, "unit_cost": 255.00, "payment_term": "Net 30"},
    {"transaction_date": date(2024, 10, 9), "customer_name": "Demirtaş Makine Ltd.",    "category": "Endüstriyel Parça",  "quantity": 205, "unit_sales_price": 335.00, "unit_cost": 308.00, "payment_term": "Net 30"},

    # ── BOTH RISKS: Karahan Gıda Paz. A.Ş. ──────────────────────────────
    # Quantity halved AND margin dropped below 10 % — the worst-case customer.
    {"transaction_date": date(2024, 2,  3), "customer_name": "Karahan Gıda Paz. A.Ş.", "category": "Ambalaj Malzemesi",  "quantity": 900, "unit_sales_price": 22.00,  "unit_cost": 12.00,  "payment_term": "Net 30"},
    {"transaction_date": date(2024, 5,  7), "customer_name": "Karahan Gıda Paz. A.Ş.", "category": "Ambalaj Malzemesi",  "quantity": 700, "unit_sales_price": 22.50,  "unit_cost": 14.50,  "payment_term": "Net 30"},
    {"transaction_date": date(2024, 8, 12), "customer_name": "Karahan Gıda Paz. A.Ş.", "category": "Ambalaj Malzemesi",  "quantity": 450, "unit_sales_price": 23.00,  "unit_cost": 18.50,  "payment_term": "Net 45"},
    {"transaction_date": date(2024, 11, 6), "customer_name": "Karahan Gıda Paz. A.Ş.", "category": "Ambalaj Malzemesi",  "quantity": 410, "unit_sales_price": 23.50,  "unit_cost": 21.40,  "payment_term": "Net 60"},

    # ── HEALTHY CONTROL: Anadolu Lojistik A.Ş. ──────────────────────────
    # Stable volume, healthy 30 %+ margin — should not be flagged.
    {"transaction_date": date(2024, 1, 25), "customer_name": "Anadolu Lojistik A.Ş.",  "category": "Taşıma Hizmetleri",  "quantity": 120, "unit_sales_price": 750.00, "unit_cost": 490.00, "payment_term": "Net 15"},
    {"transaction_date": date(2024, 4, 22), "customer_name": "Anadolu Lojistik A.Ş.",  "category": "Taşıma Hizmetleri",  "quantity": 125, "unit_sales_price": 760.00, "unit_cost": 495.00, "payment_term": "Net 15"},
    {"transaction_date": date(2024, 7, 30), "customer_name": "Anadolu Lojistik A.Ş.",  "category": "Taşıma Hizmetleri",  "quantity": 118, "unit_sales_price": 770.00, "unit_cost": 500.00, "payment_term": "Net 15"},
    {"transaction_date": date(2024, 10,28), "customer_name": "Anadolu Lojistik A.Ş.",  "category": "Taşıma Hizmetleri",  "quantity": 130, "unit_sales_price": 780.00, "unit_cost": 505.00, "payment_term": "Net 15"},

    # ── CHURN RISK: Bosphorus Kimya San. Ltd. ───────────────────────────
    # Sudden 70 % quantity drop in the final quarter — early warning signal.
    {"transaction_date": date(2024, 3,  8), "customer_name": "Bosphorus Kimya San. Ltd.", "category": "Kimyasal Hammadde", "quantity": 340, "unit_sales_price": 195.00, "unit_cost": 115.00, "payment_term": "Net 45"},
    {"transaction_date": date(2024, 6, 15), "customer_name": "Bosphorus Kimya San. Ltd.", "category": "Kimyasal Hammadde", "quantity": 355, "unit_sales_price": 198.00, "unit_cost": 117.00, "payment_term": "Net 45"},
    {"transaction_date": date(2024, 9, 20), "customer_name": "Bosphorus Kimya San. Ltd.", "category": "Kimyasal Hammadde", "quantity": 360, "unit_sales_price": 200.00, "unit_cost": 118.00, "payment_term": "Net 45"},
    {"transaction_date": date(2024, 12,10), "customer_name": "Bosphorus Kimya San. Ltd.", "category": "Kimyasal Hammadde", "quantity": 105, "unit_sales_price": 200.00, "unit_cost": 119.00, "payment_term": "Net 60"},
]
# fmt: on


# ---------------------------------------------------------------------------
# Seeder
# ---------------------------------------------------------------------------


def seed_database() -> None:
    """
    Insert mock records into ``sales_records`` if the table is empty.

    This function is **idempotent**: if any rows already exist, the insert
    is skipped entirely to prevent duplicate data across restarts.

    Raises:
        RuntimeError: If the database session encounters an unexpected error.
    """
    session = SessionLocal()
    try:
        existing_count: int = session.query(SalesRecordModel).count()
        if existing_count > 0:
            logger.info(
                "Seed skipped — %d records already exist in sales_records.",
                existing_count,
            )
            return

        records = [SalesRecordModel(**row) for row in _MOCK_RECORDS]
        session.add_all(records)
        session.commit()
        logger.info(
            "Seed complete — %d records inserted into sales_records.",
            len(records),
        )
    except Exception as exc:
        session.rollback()
        logger.exception("Seed failed; transaction rolled back.")
        raise RuntimeError(f"Database seed failed: {exc}") from exc
    finally:
        session.close()


if __name__ == "__main__":
    logger.info("Initialising database schema…")
    init_db()
    logger.info("Running seeder…")
    seed_database()
    logger.info("Done.")
    sys.exit(0)
