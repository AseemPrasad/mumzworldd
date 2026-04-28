# app/core/llm_conflict_analyzer.py
# Validates whether detected signals constitute a real conflict via OpenRouter LLM.
# Provides graceful fallback when API is unavailable.

import json
import logging
import requests
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 20
MAX_PROMPT_LEN = 1500

ANALYZER_SYSTEM_PROMPT = (
    "You are a pediatric product safety analyst for a baby e-commerce platform. "
    "Your task is to evaluate whether a reported product signal constitutes a genuine "
    "safety conflict for a baby of a given developmental stage. "
    "Be conservative: only confirm conflicts supported by established WHO or AAP guidelines. "
    "Respond ONLY with a raw JSON object — no markdown, no code blocks."
)


def analyze_conflict(
    product_name: str,
    conflict_type: str,
    child_age_months: int,
    ingredient_flags: list[str],
    evidence_text: str,
    signals_supporting: int,
) -> dict:
    """
    Validate whether detected signals constitute a real conflict.
    
    Returns:
        {
            "is_conflict": bool,
            "confidence": float (0.0-1.0),
            "severity_level": int (1-10),
            "explanation": str,
            "llm_analyzed": bool
        }
    """
    settings = get_settings()

    if not settings.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set — using heuristic conflict analysis fallback")
        return _heuristic_fallback(
            product_name, conflict_type, child_age_months, ingredient_flags, signals_supporting
        )

    user_message = _build_analyzer_prompt(
        product_name, conflict_type, child_age_months, ingredient_flags, evidence_text, signals_supporting
    )

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }

    if "openai" in settings.OPENROUTER_MODEL or "mistral" in settings.OPENROUTER_MODEL:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://baby-safety-dashboard.local",
        "X-Title": "Baby Product Safety Dashboard",
    }

    try:
        resp = requests.post(
            settings.OPENROUTER_API_URL,
            json=payload,
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result = json.loads(content)
        logger.info(
            "LLM conflict analysis for '%s': is_conflict=%s confidence=%.2f",
            product_name,
            result.get("is_conflict"),
            float(result.get("confidence", 0)),
        )
        return {
            "is_conflict": bool(result.get("is_conflict", False)),
            "confidence": float(result.get("confidence", 0.5)),
            "severity_level": int(result.get("severity_level", 5)),
            "explanation": str(result.get("explanation", "")),
            "llm_analyzed": True,
        }

    except requests.exceptions.Timeout:
        logger.error("LLM conflict analyzer timed out for product '%s'", product_name)
        return _heuristic_fallback(
            product_name, conflict_type, child_age_months, ingredient_flags, signals_supporting
        )
    except requests.exceptions.HTTPError as e:
        logger.error("LLM conflict analyzer HTTP error: %s", e)
        return _heuristic_fallback(
            product_name, conflict_type, child_age_months, ingredient_flags, signals_supporting
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error("LLM conflict analyzer response parse error: %s", e)
        return _heuristic_fallback(
            product_name, conflict_type, child_age_months, ingredient_flags, signals_supporting
        )
    except Exception as e:
        logger.error("LLM conflict analyzer unexpected error: %s", e)
        return _heuristic_fallback(
            product_name, conflict_type, child_age_months, ingredient_flags, signals_supporting
        )


def _build_analyzer_prompt(
    product_name: str,
    conflict_type: str,
    child_age_months: int,
    ingredient_flags: list[str],
    evidence_text: str,
    signals_supporting: int,
) -> str:
    flags_str = ", ".join(ingredient_flags) if ingredient_flags else "none"
    prompt = (
        f"Evaluate this potential product safety conflict:\n\n"
        f"Product: {product_name}\n"
        f"Child age: {child_age_months} months\n"
        f"Conflict type: {conflict_type}\n"
        f"Ingredient flags: {flags_str}\n"
        f"Supporting signals: {signals_supporting}\n"
        f"Evidence: {evidence_text[:MAX_PROMPT_LEN]}\n\n"
        "Return a JSON object with:\n"
        '  "is_conflict": true/false — is this a genuine safety conflict?\n'
        '  "confidence": 0.0-1.0 — your confidence in this assessment\n'
        '  "severity_level": 1-10 — severity of this conflict if real\n'
        '  "explanation": string — brief factual explanation (max 2 sentences)\n'
    )
    return prompt


def _heuristic_fallback(
    product_name: str,
    conflict_type: str,
    child_age_months: int,
    ingredient_flags: list[str],
    signals_supporting: int,
) -> dict:
    """
    Rule-based fallback when LLM is unavailable.
    Uses ingredient flags and conflict type to produce a deterministic assessment.
    """
    HIGH_RISK_FLAGS = {"honey", "nuts"}
    MEDIUM_RISK_FLAGS = {"fragrance"}

    is_conflict = signals_supporting >= 2
    severity = 5
    confidence = 0.70

    if any(f in HIGH_RISK_FLAGS for f in ingredient_flags):
        severity = 9
        confidence = 0.90
        is_conflict = True
    elif any(f in MEDIUM_RISK_FLAGS for f in ingredient_flags):
        severity = 7
        confidence = 0.80
        is_conflict = True
    elif conflict_type == "stage_mismatch":
        severity = 6
        confidence = 0.75
        is_conflict = signals_supporting >= 2
    elif conflict_type == "ingredient_age_safety":
        severity = 8
        confidence = 0.85
        is_conflict = True

    explanation = (
        f"{product_name} flagged as {conflict_type} for a {child_age_months}-month-old "
        f"based on {signals_supporting} supporting signal(s). "
        f"Heuristic assessment (LLM unavailable)."
    )

    logger.debug(
        "Heuristic fallback for '%s': is_conflict=%s confidence=%.2f",
        product_name, is_conflict, confidence,
    )

    return {
        "is_conflict": is_conflict,
        "confidence": confidence,
        "severity_level": severity,
        "explanation": explanation,
        "llm_analyzed": False,
    }
