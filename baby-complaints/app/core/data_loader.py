# app/core/data_loader.py
# Developer note: Singleton data store backed by pandas DataFrame.
# To swap in a database, replace the CSV load in `_load_data` with a DB query
# and keep the same public API (`get_df`, `reload`).

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.risk import compute_risk

logger = logging.getLogger(__name__)

_df: Optional[pd.DataFrame] = None

REQUIRED_COLUMNS = [
    "product_id", "product_name", "brand", "product_category",
    "baby_age_months", "issue_type", "return_reason", "severity",
    "frequency_score", "risk_tag", "report_date", "resolution_status",
]


def _load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(p)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Cast types
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(1)
    df["frequency_score"] = pd.to_numeric(df["frequency_score"], errors="coerce").fillna(1)
    df["baby_age_months"] = pd.to_numeric(df["baby_age_months"], errors="coerce").fillna(0)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    # Recompute risk tags deterministically
    risk_cols = df.apply(lambda row: compute_risk(row.to_dict()), axis=1, result_type="expand")
    df["risk_tag"] = risk_cols["risk_tag"]
    df["composite_score"] = risk_cols["composite_score"]
    df["normalized_severity"] = risk_cols["normalized_severity"]
    df["normalized_frequency"] = risk_cols["normalized_frequency"]
    df["risk_explanation"] = risk_cols["explanation"]

    # Age bucket for grouping
    df["baby_age_bucket"] = pd.cut(
        df["baby_age_months"],
        bins=[-1, 3, 6, 12, 24, float("inf")],
        labels=["0-3m", "4-6m", "7-12m", "13-24m", "24m+"],
    ).astype(str)

    logger.info("Loaded %d records from %s", len(df), path)
    return df


def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        settings = get_settings()
        _df = _load_data(settings.DATA_PATH)
    return _df


def reload() -> pd.DataFrame:
    """Force reload from disk (useful in tests or after file updates)."""
    global _df
    settings = get_settings()
    _df = _load_data(settings.DATA_PATH)
    return _df


def load_from_path(path: str) -> pd.DataFrame:
    """Load from an explicit path (used in tests)."""
    return _load_data(path)
