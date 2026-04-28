# tests/test_coherence_llm.py
# Tests for LLM copy generation, conflict analyzer integration,
# bilingual output quality, and graceful fallback behaviour.

import pytest
import json
from unittest.mock import patch, MagicMock

from app.core.llm_client import (
    generate_bilingual_copy,
    _fallback_copy,
    _extract_json_from_content,
)
from app.core.llm_conflict_analyzer import analyze_conflict, _heuristic_fallback
from app.models.schemas import AuditStatus


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_conflict(confidence: float = 0.85, product_name: str = "Test Product") -> dict:
    return {
        "product_sku": "p001",
        "product_name": product_name,
        "conflict_type": "ingredient_age_safety",
        "signals_supporting": 2,
        "confidence": confidence,
        "evidence_source": "who_guideline: honey risk before 12 months",
        "action": "flag_with_doctor_referral",
    }


def _make_child_stage(months: int = 3, confidence: float = 0.85) -> dict:
    return {
        "months": months,
        "confidence": confidence,
        "evidence": ["order_size_drift", "search_term_shift"],
        "null_reason": None,
    }


# ── _extract_json_from_content ──────────────────────────────────────────────────

def test_extract_json_plain():
    raw = '{"copy_en": "Hello", "copy_ar": "مرحبا"}'
    result = _extract_json_from_content(raw)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["copy_en"] == "Hello"


def test_extract_json_with_markdown_fence():
    raw = '```json\n{"copy_en": "A", "copy_ar": "ب"}\n```'
    result = _extract_json_from_content(raw)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["copy_ar"] == "ب"


def test_extract_json_embedded_in_prose():
    raw = 'Here is the result: {"copy_en": "Test", "copy_ar": "اختبار"} done.'
    result = _extract_json_from_content(raw)
    assert result is not None
    assert "Test" in result


def test_extract_json_returns_none_on_garbage():
    result = _extract_json_from_content("no json here at all")
    assert result is None


# ── _fallback_copy ─────────────────────────────────────────────────────────────

def test_fallback_copy_defer_true():
    conflict = _make_conflict()
    stage = _make_child_stage()
    result = _fallback_copy(conflict, stage, defer_true=True)
    assert "copy_en" in result
    assert "copy_ar" in result
    assert result["copy_en"] is not None
    assert result["copy_ar"] is not None
    assert "pediatrician" in result["copy_en"].lower()
    assert "طبيب" in result["copy_ar"]


def test_fallback_copy_defer_false():
    conflict = _make_conflict(confidence=0.85)
    stage = _make_child_stage(months=3)
    result = _fallback_copy(conflict, stage, defer_true=False)
    assert "copy_en" in result
    assert "copy_ar" in result
    assert result["copy_en"] is not None
    assert result["copy_ar"] is not None


def test_fallback_copy_low_confidence_hedge():
    conflict = _make_conflict(confidence=0.65)
    stage = _make_child_stage()
    result = _fallback_copy(conflict, stage, defer_true=False)
    assert "may be worth reviewing" in result["copy_en"]


def test_fallback_copy_high_confidence_no_hedge():
    conflict = _make_conflict(confidence=0.90)
    stage = _make_child_stage()
    result = _fallback_copy(conflict, stage, defer_true=False)
    assert "is likely no longer suitable" in result["copy_en"]


# ── generate_bilingual_copy ────────────────────────────────────────────────────

def test_bilingual_copy_below_threshold_returns_none():
    conflict = _make_conflict(confidence=0.50)
    stage = _make_child_stage()
    result = generate_bilingual_copy(conflict, stage, defer_true=False)
    assert result["copy_en"] is None
    assert result["copy_ar"] is None


def test_bilingual_copy_no_api_key_uses_fallback(monkeypatch):
    monkeypatch.setattr("app.core.llm_client.get_settings", lambda: MagicMock(
        OPENROUTER_API_KEY="",
        OPENROUTER_MODEL="openai/gpt-3.5-turbo",
        OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions",
    ))
    conflict = _make_conflict(confidence=0.85)
    stage = _make_child_stage()
    result = generate_bilingual_copy(conflict, stage, defer_true=False)
    assert result["copy_en"] is not None
    assert result["copy_ar"] is not None


def test_bilingual_copy_mocked_llm_success(monkeypatch):
    """Mock LLM to return valid bilingual JSON."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"copy_en": "English alert.", "copy_ar": "تنبيه عربي."}'}}]
    }
    mock_response.raise_for_status = MagicMock()

    monkeypatch.setattr("app.core.llm_client.get_settings", lambda: MagicMock(
        OPENROUTER_API_KEY="fake-key",
        OPENROUTER_MODEL="openai/gpt-3.5-turbo",
        OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions",
    ))

    with patch("app.core.llm_client.requests.post", return_value=mock_response):
        conflict = _make_conflict(confidence=0.85)
        stage = _make_child_stage()
        result = generate_bilingual_copy(conflict, stage, defer_true=False)

    assert result["copy_en"] == "English alert."
    assert result["copy_ar"] == "تنبيه عربي."


def test_bilingual_copy_mocked_llm_markdown_json(monkeypatch):
    """Mock LLM returning JSON wrapped in markdown fences — should still parse."""
    content = '```json\n{"copy_en": "Fence test EN.", "copy_ar": "اختبار."}\n```'
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock_response.raise_for_status = MagicMock()

    monkeypatch.setattr("app.core.llm_client.get_settings", lambda: MagicMock(
        OPENROUTER_API_KEY="fake-key",
        OPENROUTER_MODEL="openai/gpt-3.5-turbo",
        OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions",
    ))

    with patch("app.core.llm_client.requests.post", return_value=mock_response):
        result = generate_bilingual_copy(_make_conflict(0.85), _make_child_stage(), False)

    assert result["copy_en"] == "Fence test EN."
    assert result["copy_ar"] == "اختبار."


def test_bilingual_copy_llm_parse_failure_falls_back(monkeypatch):
    """When LLM returns invalid JSON, should gracefully fall back."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Not JSON at all"}}]}
    mock_response.raise_for_status = MagicMock()

    monkeypatch.setattr("app.core.llm_client.get_settings", lambda: MagicMock(
        OPENROUTER_API_KEY="fake-key",
        OPENROUTER_MODEL="openai/gpt-3.5-turbo",
        OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions",
    ))

    with patch("app.core.llm_client.requests.post", return_value=mock_response):
        result = generate_bilingual_copy(_make_conflict(0.85), _make_child_stage(), False)

    # Should have fallen back to deterministic copy
    assert result["copy_en"] is not None
    assert result["copy_ar"] is not None


def test_bilingual_copy_deferred_no_product_name(monkeypatch):
    """Defer copy must not expose product name in persuasive copy."""
    monkeypatch.setattr("app.core.llm_client.get_settings", lambda: MagicMock(
        OPENROUTER_API_KEY="",
        OPENROUTER_MODEL="openai/gpt-3.5-turbo",
        OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions",
    ))
    conflict = _make_conflict(confidence=0.90, product_name="Dangerous Honey Gel")
    stage = _make_child_stage()
    result = generate_bilingual_copy(conflict, stage, defer_true=True)
    # Deferred copy should not recommend any product
    assert "Dangerous Honey Gel" not in (result["copy_en"] or "")


# ── Arabic copy quality checks ─────────────────────────────────────────────────

def test_arabic_copy_contains_arabic_script():
    """Arabic copy must contain actual Arabic Unicode characters."""
    conflict = _make_conflict(confidence=0.85)
    stage = _make_child_stage()
    result = _fallback_copy(conflict, stage, defer_true=False)
    ar_text = result["copy_ar"] or ""
    has_arabic = any('\u0600' <= ch <= '\u06ff' for ch in ar_text)
    assert has_arabic, "Arabic copy does not contain Arabic script"


def test_arabic_defer_copy_contains_arabic_script():
    conflict = _make_conflict(confidence=0.85)
    stage = _make_child_stage()
    result = _fallback_copy(conflict, stage, defer_true=True)
    ar_text = result["copy_ar"] or ""
    has_arabic = any('\u0600' <= ch <= '\u06ff' for ch in ar_text)
    assert has_arabic, "Arabic deferral copy does not contain Arabic script"


def test_arabic_not_just_english_transliteration():
    """Arabic copy should differ from English copy (not a word-for-word transliteration)."""
    conflict = _make_conflict(confidence=0.85)
    stage = _make_child_stage()
    result = _fallback_copy(conflict, stage, defer_true=False)
    # Simple check: they should not be identical
    assert result["copy_en"] != result["copy_ar"]


# ── LLM Conflict Analyzer ─────────────────────────────────────────────────────

def test_heuristic_fallback_honey_flag():
    result = _heuristic_fallback("Honey Gel", "ingredient_age_safety", 2, ["honey"], 2)
    assert result["is_conflict"] is True
    assert result["confidence"] >= 0.85
    assert result["severity_level"] >= 8
    assert result["llm_analyzed"] is False


def test_heuristic_fallback_fragrance_flag():
    result = _heuristic_fallback("Scented Lotion", "ingredient_age_safety", 1, ["fragrance"], 2)
    assert result["is_conflict"] is True
    assert result["confidence"] >= 0.75


def test_heuristic_fallback_nut_flag():
    result = _heuristic_fallback("Peanut Puffs", "ingredient_age_safety", 3, ["nuts"], 2)
    assert result["is_conflict"] is True
    assert result["severity_level"] >= 8


def test_heuristic_fallback_single_signal_no_flag():
    result = _heuristic_fallback("Plain Bottle", "stage_mismatch", 6, [], 1)
    assert result["is_conflict"] is False


def test_heuristic_fallback_stage_mismatch_two_signals():
    result = _heuristic_fallback("Stage 1 Formula", "stage_mismatch", 7, [], 2)
    assert result["is_conflict"] is True


def test_analyze_conflict_no_api_key_uses_fallback(monkeypatch):
    monkeypatch.setattr("app.core.llm_conflict_analyzer.get_settings", lambda: MagicMock(
        OPENROUTER_API_KEY="",
        OPENROUTER_MODEL="openai/gpt-3.5-turbo",
        OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions",
    ))
    result = analyze_conflict(
        product_name="Honey Teething Gel",
        conflict_type="ingredient_age_safety",
        child_age_months=2,
        ingredient_flags=["honey"],
        evidence_text="WHO guideline: honey before 12 months — botulism risk",
        signals_supporting=2,
    )
    assert "is_conflict" in result
    assert "confidence" in result
    assert result["llm_analyzed"] is False


def test_analyze_conflict_mocked_llm_valid(monkeypatch):
    """Mock LLM returning valid JSON conflict analysis."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "is_conflict": True,
                    "confidence": 0.93,
                    "severity_level": 9,
                    "explanation": "Honey before 12 months carries botulism risk per WHO guidelines."
                })
            }
        }]
    }
    mock_response.raise_for_status = MagicMock()

    monkeypatch.setattr("app.core.llm_conflict_analyzer.get_settings", lambda: MagicMock(
        OPENROUTER_API_KEY="fake-key",
        OPENROUTER_MODEL="openai/gpt-3.5-turbo",
        OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions",
    ))

    with patch("app.core.llm_conflict_analyzer.requests.post", return_value=mock_response):
        result = analyze_conflict(
            product_name="Honey Teething Gel",
            conflict_type="ingredient_age_safety",
            child_age_months=2,
            ingredient_flags=["honey"],
            evidence_text="WHO: honey botulism risk",
            signals_supporting=2,
        )

    assert result["is_conflict"] is True
    assert result["confidence"] == 0.93
    assert result["llm_analyzed"] is True


def test_analyze_conflict_llm_timeout_falls_back(monkeypatch):
    """LLM timeout should produce heuristic result, not raise an exception."""
    import requests as req_lib
    monkeypatch.setattr("app.core.llm_conflict_analyzer.get_settings", lambda: MagicMock(
        OPENROUTER_API_KEY="fake-key",
        OPENROUTER_MODEL="openai/gpt-3.5-turbo",
        OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions",
    ))

    with patch("app.core.llm_conflict_analyzer.requests.post",
               side_effect=req_lib.exceptions.Timeout("timeout")):
        result = analyze_conflict(
            product_name="Ring Teether",
            conflict_type="stage_mismatch",
            child_age_months=4,
            ingredient_flags=[],
            evidence_text="Choking hazard for under 8 months",
            signals_supporting=2,
        )

    assert "is_conflict" in result
    assert result["llm_analyzed"] is False


# ── Conflict Loader ────────────────────────────────────────────────────────────

def test_conflict_rules_json_loads():
    from app.core.conflict_loader import load_conflict_rules
    rules = load_conflict_rules()
    assert len(rules) >= 5, "Expected at least 5 conflict rules"


def test_conflict_rules_have_required_fields():
    from app.core.conflict_loader import load_conflict_rules
    rules = load_conflict_rules()
    required = {"rule_id", "rule_name", "conflict_type", "severity_level",
                "age_safe_min_months", "ingredient_flags", "who_guideline",
                "aap_guideline", "description_en", "description_ar", "action"}
    for rule in rules:
        missing = required - set(rule.keys())
        assert not missing, f"Rule {rule.get('rule_id')} missing fields: {missing}"


def test_conflict_rules_bilingual_descriptions():
    from app.core.conflict_loader import load_conflict_rules
    rules = load_conflict_rules()
    for rule in rules:
        ar = rule.get("description_ar", "")
        en = rule.get("description_en", "")
        assert len(en) > 0, f"Rule {rule['rule_id']} has empty English description"
        assert len(ar) > 0, f"Rule {rule['rule_id']} has empty Arabic description"
        # Arabic description should contain Arabic script
        has_arabic = any('\u0600' <= ch <= '\u06ff' for ch in ar)
        assert has_arabic, f"Rule {rule['rule_id']} Arabic description lacks Arabic script"


def test_conflict_rules_severity_range():
    from app.core.conflict_loader import load_conflict_rules
    rules = load_conflict_rules()
    for rule in rules:
        sev = rule.get("severity_level", 0)
        assert 1 <= sev <= 10, f"Rule {rule['rule_id']} severity {sev} out of range"
