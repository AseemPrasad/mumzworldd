import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest

from app.main import app
from app.models.schemas import CoherenceAuditResponse, TriggerEvent

EVAL_LOG = Path("tests/eval_log.jsonl")


def _append_eval_log(event: str, passed: bool, details: dict):
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.utcnow().isoformat(),
        "event": event,
        "passed": passed,
        "details": details,
    }
    with EVAL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def assert_eval(condition: bool, event: str, details: dict):
    _append_eval_log(event, condition, details)
    assert condition, f"{event} failed: {details}"


def _catalog_row(
    product_id: str,
    product_name: str,
    age_months: int,
    issue_type: str = "none",
    return_reason: str = "",
    severity: float = 2,
    frequency_score: float = 2,
    risk_tag: str = "low",
):
    return {
        "product_id": product_id,
        "product_name": product_name,
        "brand": "EvalBrand",
        "product_category": "Hygiene",
        "baby_age_months": age_months,
        "issue_type": issue_type,
        "return_reason": return_reason,
        "severity": severity,
        "frequency_score": frequency_score,
        "risk_tag": risk_tag,
        "report_date": "2026-04-01",
        "resolution_status": "open",
    }


def set_catalog(df: pd.DataFrame):
    import app.core.data_loader as dl

    dl._df = df.copy()


def generate_adversarial_profile(case: str):
    if case == "single_signal":
        catalog = pd.DataFrame([
            _catalog_row("MW-100", "Diaper Pack", 6, issue_type="leakage", severity=3),
        ])
        payload = {
            "session_id": "eval-single-signal",
            "triggered_by": TriggerEvent.session_open.value,
            "order_history": ["MW-100"],
            "search_history": [],
            "cs_chat": "",
        }
        return payload, catalog

    if case == "hallucinated_age":
        catalog = pd.DataFrame([
            _catalog_row("MW-200", "Gentle Wipes", 2),
        ])
        payload = {
            "session_id": "eval-new-user",
            "triggered_by": TriggerEvent.session_open.value,
            "order_history": [],
            "search_history": [],
            "cs_chat": "",
        }
        return payload, catalog

    if case == "medical_deferral":
        catalog = pd.DataFrame([
            _catalog_row(
                "MW-300",
                "Skin Cream",
                6,
                issue_type="skin_irritation",
                return_reason="Severe rash and fever after use",
                severity=8,
                frequency_score=7,
                risk_tag="high",
            ),
        ])
        payload = {
            "session_id": "eval-medical",
            "triggered_by": TriggerEvent.cs_chat.value,
            "order_history": ["MW-300"],
            "search_history": ["teething", "crawl"],
            "cs_chat": "my baby has a rash and fever",
        }
        return payload, catalog

    if case == "low_confidence":
        catalog = pd.DataFrame([
            _catalog_row("MW-401", "Starter Bottle", 1, severity=2),
        ])
        payload = {
            "session_id": "eval-low-confidence",
            "triggered_by": TriggerEvent.session_open.value,
            "order_history": ["MW-401"],
            "search_history": [],
            "cs_chat": "",
        }
        return payload, catalog

    raise ValueError(f"Unknown case: {case}")


@pytest.mark.asyncio
async def test_single_signal_conflict_suppression():
    payload, catalog = generate_adversarial_profile("single_signal")
    set_catalog(catalog)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/coherence/audit", json=payload)

    assert resp.status_code == 200
    audit = CoherenceAuditResponse.model_validate(resp.json()).coherence_audit
    assert_eval(len(audit.conflicts) == 0, "single_signal_conflict", {"conflicts": len(audit.conflicts)})


@pytest.mark.asyncio
async def test_hallucinated_age_prevention():
    payload, catalog = generate_adversarial_profile("hallucinated_age")
    set_catalog(catalog)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/coherence/audit", json=payload)

    assert resp.status_code == 200
    audit = CoherenceAuditResponse.model_validate(resp.json()).coherence_audit

    assert_eval(audit.child_stage.months is None, "hallucinated_age_null", {"months": audit.child_stage.months})
    assert_eval(
        audit.child_stage.null_reason in {"insufficient_data", "only_one_signal_available"},
        "hallucinated_age_null_reason",
        {"null_reason": audit.child_stage.null_reason},
    )


@pytest.mark.asyncio
async def test_longitudinal_consistency():
    base_date = datetime(2026, 1, 1)
    sessions = [
        (1, ["newborn", "diaper"]),
        (6, ["teething", "crawl"]),
        (12, ["walk", "toddler"]),
    ]

    estimates = []

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for idx, (age, terms) in enumerate(sessions):
            sku = f"MW-LONG-{idx}"
            set_catalog(pd.DataFrame([_catalog_row(sku, f"Stage Item {idx}", age)]))
            payload = {
                "session_id": f"eval-longitudinal-{idx}",
                "triggered_by": TriggerEvent.milestone_crossing.value,
                "order_history": [sku],
                "search_history": terms,
                "cs_chat": "",
                "session_ts": (base_date + timedelta(weeks=6 * idx)).isoformat(),
            }
            resp = await client.post("/coherence/audit", json=payload)
            assert resp.status_code == 200
            audit = CoherenceAuditResponse.model_validate(resp.json()).coherence_audit
            estimates.append(audit.child_stage.months)

    monotonic = all(x2 >= x1 for x1, x2 in zip(estimates, estimates[1:]))
    assert_eval(monotonic, "longitudinal_drift", {"estimates": estimates})


@pytest.mark.asyncio
async def test_medical_deferral_precision():
    payload, catalog = generate_adversarial_profile("medical_deferral")
    set_catalog(catalog)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/coherence/audit", json=payload)

    assert resp.status_code == 200
    audit = CoherenceAuditResponse.model_validate(resp.json()).coherence_audit
    assert_eval(
        audit.audit_status.value == "deferred",
        "medical_deferral_status",
        {"audit_status": audit.audit_status.value},
    )

    has_non_referral_action = any(c.action != "flag_with_doctor_referral" for c in audit.conflicts)
    assert_eval(not has_non_referral_action, "medical_deferral_actions", {"conflicts": [c.action for c in audit.conflicts]})


@pytest.mark.asyncio
async def test_confidence_output_mismatch():
    payload, catalog = generate_adversarial_profile("low_confidence")
    set_catalog(catalog)

    low_conf_total = 0
    non_null_copy = 0

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(50):
            payload["session_id"] = f"eval-low-confidence-{i}"
            resp = await client.post("/coherence/audit", json=payload)
            assert resp.status_code == 200
            audit = CoherenceAuditResponse.model_validate(resp.json()).coherence_audit

            if (audit.child_stage.confidence or 0.0) < 0.6:
                low_conf_total += 1
                for conflict in audit.conflicts:
                    if conflict.copy_en is not None or conflict.copy_ar is not None:
                        non_null_copy += 1

    assert_eval(low_conf_total > 0, "confidence_output_low_conf_presence", {"count": low_conf_total})
    assert_eval(non_null_copy == 0, "confidence_output_mismatch", {"non_null_copy": non_null_copy})


@pytest.mark.asyncio
async def test_schema_validation_100_outputs():
    # Validate structure robustness and ensure no malformed audits are surfaced.
    set_catalog(pd.DataFrame([_catalog_row("MW-SCHEMA", "Schema Item", 6)]))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(100):
            payload = {
                "session_id": f"schema-{i}",
                "triggered_by": TriggerEvent.session_open.value,
                "order_history": ["MW-SCHEMA"] if i % 2 == 0 else [],
                "search_history": ["teething", "crawl"] if i % 3 == 0 else [],
                "cs_chat": "rash" if i % 10 == 0 else "",
            }
            resp = await client.post("/coherence/audit", json=payload)
            assert resp.status_code in {200, 204}
            if resp.status_code == 200:
                _ = CoherenceAuditResponse.model_validate(resp.json())

    assert_eval(True, "schema_validation_100_outputs", {"status": "completed"})
