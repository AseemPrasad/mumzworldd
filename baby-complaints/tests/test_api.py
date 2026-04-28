# tests/test_api.py
# Developer note: Integration tests using FastAPI TestClient with a patched data loader.
# To test against the real CSV, remove the monkeypatch and set DATA_PATH.

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


SAMPLE_CSV_CONTENT = """\
product_id,product_name,brand,product_category,baby_age_months,issue_type,return_reason,severity,frequency_score,risk_tag,report_date,resolution_status
p001,Baby Lotion,BrandA,Skincare,2,skin_irritation,Caused rash,6,4,medium,2026-04-01,open
p002,Ring Teether,BrandB,Toys,6,choking_hazard,Small parts broke off,9,7,high,2026-03-15,returned
p003,Diapers,BrandD,Hygiene,0,poor_fit,Too loose,5,5,medium,2026-01-10,exchanged
p004,Baby Monitor,BrandF,Electronics,4,connectivity,Disconnects at night,6,5,medium,2026-02-14,open
p005,Stroller Clip,BrandE,Accessories,8,breakage,Clip snapped,3,3,low,2026-04-05,open
"""


@pytest.fixture(autouse=True)
def patch_data(tmp_path, monkeypatch):
    """Write a temp CSV and patch DATA_PATH so the app uses it."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(SAMPLE_CSV_CONTENT)

    monkeypatch.setenv("DATA_PATH", str(csv_file))

    # Reset the singleton so it reloads with the patched path
    import app.core.data_loader as dl
    dl._df = None

    # Also patch settings cache
    from app.core.config import get_settings
    get_settings.cache_clear()

    yield

    dl._df = None
    get_settings.cache_clear()


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["status"] == "ok"


# ── Products ──────────────────────────────────────────────────────────────────

def test_products_list(client):
    r = client.get("/products")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert len(body["data"]) == 5


def test_products_filter_category(client):
    r = client.get("/products?category=Skincare")
    assert r.status_code == 200
    body = r.json()
    assert all(p["product_category"] == "Skincare" for p in body["data"])


def test_products_filter_risk_tag(client):
    r = client.get("/products?risk_tag=high")
    assert r.status_code == 200
    body = r.json()
    assert all(p["risk_tag"] == "high" for p in body["data"])


def test_products_pagination(client):
    r = client.get("/products?limit=2&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 2
    assert body["meta"]["limit"] == 2
    assert body["meta"]["total"] == 5


def test_products_invalid_sort(client):
    r = client.get("/products?sort_by=invalid_field")
    assert r.status_code == 400


def test_products_limit_max(client):
    r = client.get("/products?limit=201")
    assert r.status_code == 422  # Pydantic validation


def test_product_detail(client):
    r = client.get("/products/p001")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["record"]["product_id"] == "p001"
    assert "issue_summary" in body["data"]


def test_product_not_found(client):
    r = client.get("/products/p999")
    assert r.status_code == 404


def test_products_age_filter(client):
    r = client.get("/products?min_age=5&max_age=8")
    body = r.json()
    for p in body["data"]:
        assert 5 <= p["baby_age_months"] <= 8


# ── Issues ────────────────────────────────────────────────────────────────────

def test_issues_by_issue_type(client):
    r = client.get("/issues?group_by=issue_type")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert len(body["data"]) > 0
    first = body["data"][0]
    assert "group_key" in first
    assert "count" in first
    assert "avg_severity" in first


def test_issues_by_category(client):
    r = client.get("/issues?group_by=product_category")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["group_by"] == "product_category"


def test_issues_top_n(client):
    r = client.get("/issues?top_n=2")
    body = r.json()
    assert len(body["data"]) <= 2


# ── Risks ─────────────────────────────────────────────────────────────────────

def test_risks_list(client):
    r = client.get("/risks")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    scores = [item["composite_score"] for item in body["data"]]
    assert scores == sorted(scores, reverse=True)


def test_risks_threshold_high(client):
    r = client.get("/risks?threshold=high")
    body = r.json()
    assert all(item["risk_tag"] == "high" for item in body["data"])


def test_risks_threshold_low(client):
    r = client.get("/risks?threshold=low")
    body = r.json()
    assert all(item["risk_tag"] == "low" for item in body["data"])


def test_risks_explanation_present(client):
    r = client.get("/risks")
    body = r.json()
    for item in body["data"]:
        assert "explanation" in item
        assert len(item["explanation"]) > 0


# ── LLM ───────────────────────────────────────────────────────────────────────

def test_llm_no_api_key(client):
    """Graceful degradation when OPENROUTER_API_KEY is not set."""
    r = client.post("/llm/summary", json={"product_ids": ["p001", "p002"]})
    assert r.status_code == 501
    body = r.json()
    assert body["ok"] is False
    assert "not configured" in body["error"].lower() or "OPENROUTER_API_KEY" in body["error"]


def test_llm_mocked_success(client, monkeypatch):
    """Mock the LLM call and assert correct response structure."""
    mock_result = {"summary": "Product p001 poses a skin irritation risk.", "model": "test-model"}

    monkeypatch.setattr("app.main.call_llm", lambda **kwargs: mock_result)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")
    from app.core.config import get_settings
    get_settings.cache_clear()

    r = client.post("/llm/summary", json={
        "product_ids": ["p001"],
        "prompt_template": "Summarize safety concerns."
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["summary"] is not None
    assert body["data"]["products_analyzed"] == 1


def test_llm_missing_product_ids(client):
    r = client.post("/llm/summary", json={"product_ids": ["p999"]})
    assert r.status_code == 404
