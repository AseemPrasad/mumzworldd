# tests/test_risk.py
# Developer note: Edge case and boundary tests for the deterministic risk formula.

import pytest
from app.core.risk import compute_risk, _normalize


def test_normalize_min():
    assert _normalize(1) == 0.0


def test_normalize_max():
    assert _normalize(10) == 1.0


def test_normalize_midpoint():
    result = _normalize(5.5)
    assert abs(result - 0.5) < 0.01


def test_normalize_clamps_below():
    assert _normalize(0) == 0.0


def test_normalize_clamps_above():
    assert _normalize(11) == 1.0


def test_high_risk():
    result = compute_risk({"severity": 10, "frequency_score": 10})
    assert result["risk_tag"] == "high"
    assert result["composite_score"] == 1.0


def test_low_risk():
    result = compute_risk({"severity": 1, "frequency_score": 1})
    assert result["risk_tag"] == "low"
    assert result["composite_score"] == 0.0


def test_medium_risk():
    # severity=5, frequency=5 → norm=0.444, composite ≈ 0.444 → "low"? let's calc
    # norm_sev = (5-1)/9 = 0.444, norm_freq = 0.444
    # composite = 0.6*0.444 + 0.4*0.444 = 0.444 → low
    result = compute_risk({"severity": 5, "frequency_score": 5})
    assert result["risk_tag"] == "low"


def test_medium_risk_boundary():
    # severity=7, frequency=4 → norm_sev=0.667, norm_freq=0.333
    # composite = 0.6*0.667 + 0.4*0.333 = 0.4 + 0.133 = 0.533 → medium
    result = compute_risk({"severity": 7, "frequency_score": 4})
    assert result["risk_tag"] == "medium"
    assert 0.45 <= result["composite_score"] < 0.75


def test_explanation_contains_key_info():
    result = compute_risk({"severity": 9, "frequency_score": 7})
    assert "severity" in result["explanation"]
    assert "frequency" in result["explanation"]
    assert "composite" in result["explanation"]


def test_bad_values_default_gracefully():
    result = compute_risk({"severity": None, "frequency_score": "bad"})
    assert result["risk_tag"] in ("low", "medium", "high")
    assert "composite_score" in result


def test_output_keys():
    result = compute_risk({"severity": 5, "frequency_score": 5})
    assert set(result.keys()) == {
        "normalized_severity", "normalized_frequency",
        "composite_score", "risk_tag", "explanation"
    }
