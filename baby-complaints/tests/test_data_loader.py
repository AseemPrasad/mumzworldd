# tests/test_data_loader.py
# Developer note: Tests for CSV loading and risk recomputation.
# Uses a temp CSV fixture so tests are self-contained.

import io
import pytest
import pandas as pd
import tempfile
import os

from app.core.data_loader import load_from_path
from app.core.risk import compute_risk


SAMPLE_CSV = """\
product_id,product_name,brand,product_category,baby_age_months,issue_type,return_reason,severity,frequency_score,risk_tag,report_date,resolution_status
p001,Baby Lotion,BrandA,Skincare,2,skin_irritation,Caused rash,6,4,low,2026-04-01,open
p002,Ring Teether,BrandB,Toys,6,choking_hazard,Small parts broke off,9,7,low,2026-03-15,returned
p003,Diapers,BrandD,Hygiene,0,poor_fit,Too loose,5,5,low,2026-01-10,exchanged
"""


@pytest.fixture
def sample_csv_path(tmp_path):
    p = tmp_path / "test_products.csv"
    p.write_text(SAMPLE_CSV)
    return str(p)


def test_load_returns_dataframe(sample_csv_path):
    df = load_from_path(sample_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_required_columns_present(sample_csv_path):
    df = load_from_path(sample_csv_path)
    for col in ["product_id", "product_name", "severity", "frequency_score", "risk_tag"]:
        assert col in df.columns


def test_risk_tags_recomputed(sample_csv_path):
    """
    The CSV has all risk_tags as 'low', but p002 (severity=9, freq=7)
    should be recomputed to 'high'.
    """
    df = load_from_path(sample_csv_path)
    p002 = df[df["product_id"] == "p002"].iloc[0]
    assert p002["risk_tag"] == "high", f"Expected 'high', got '{p002['risk_tag']}'"


def test_composite_score_column_exists(sample_csv_path):
    df = load_from_path(sample_csv_path)
    assert "composite_score" in df.columns
    assert df["composite_score"].notna().all()


def test_age_bucket_column(sample_csv_path):
    df = load_from_path(sample_csv_path)
    assert "baby_age_bucket" in df.columns


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_from_path("/nonexistent/path/file.csv")
