from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_engine(db_url: str | None = None) -> Engine:
    """
    Build SQLAlchemy engine.
    Priority:
    1) argument db_url
    2) env F1_DB_URL
    3) local sqlite fallback
    """
    url = db_url or os.getenv("F1_DB_URL", "sqlite:///f1_analysis.db")
    return create_engine(url, pool_pre_ping=True)

