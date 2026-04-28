# app/core/llm_client.py
# Developer note: Thin wrapper around OpenRouter's chat completions API.
# To switch providers, update OPENROUTER_API_URL and adjust `payload` construction.
# Always degrades gracefully when OPENROUTER_API_KEY is absent.

import logging
import requests
import json
import re
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

MAX_PROMPT_LEN = 2000
TIMEOUT_SECONDS = 20

SYSTEM_PROMPT = (
    "You are a product safety analyst reviewing baby product complaint data. "
    "Your job is to produce concise, parent-friendly summaries of reported issues. "
    "Be factual, empathetic, and highlight the most important safety concerns first."
)


def _build_user_message(prompt_template: str, product_summaries: list[dict]) -> str:
    lines = ["Here are the product issue summaries:\n"]
    for p in product_summaries:
        lines.append(
            f"- {p.get('product_name', 'Unknown')} ({p.get('product_id', '')}): "
            f"issue={p.get('issue_type', 'N/A')}, severity={p.get('severity', 'N/A')}, "
            f"reason=\"{p.get('return_reason', '')}\""
        )
    lines.append(f"\n{prompt_template}")
    return "\n".join(lines)


def _extract_json_from_content(content: str) -> Optional[str]:
    """
    Robustly extract a JSON object from LLM output.
    Handles markdown fences, leading/trailing whitespace, and inline JSON.
    """
    content = content.strip()

    # Strip ```json ... ``` or ``` ... ``` blocks
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # If content starts with '{', try directly
    if content.startswith("{"):
        return content

    # Try to find first { ... } block
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return match.group(0)

    return None


def call_llm(
    product_summaries: list[dict],
    prompt_template: str = "Summarize the key safety concerns and recommend actions.",
) -> dict:
    """
    Call OpenRouter to generate a summary.
    Returns dict with `summary` key on success, `error` key on failure.
    """
    settings = get_settings()

    if not settings.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set – LLM endpoint degraded")
        return {"error": "LLM integration not configured. Set OPENROUTER_API_KEY in .env."}

    # Sanitize prompt
    safe_prompt = prompt_template[:MAX_PROMPT_LEN]
    user_message = _build_user_message(safe_prompt, product_summaries)

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 512,
        "temperature": 0.4,
    }

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
        summary = data["choices"][0]["message"]["content"]
        logger.info("LLM summary generated (%d chars)", len(summary))
        return {"summary": summary, "model": settings.OPENROUTER_MODEL}
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        return {"error": "LLM request timed out. Try again later."}
    except requests.exceptions.HTTPError as e:
        logger.error("LLM HTTP error: %s", e)
        return {"error": f"LLM API error: {e.response.status_code}"}
    except Exception as e:
        logger.error("LLM unexpected error: %s", e)
        return {"error": "Unexpected LLM error. Check server logs."}


def _build_bilingual_prompt(conflict_data: dict, child_stage: dict, defer_true: bool) -> str:
    """Build a dynamic bilingual prompt based on conflict data."""
    product_name = conflict_data.get("product_name", "this product")
    months = child_stage.get("months", "unknown")
    confidence = float(conflict_data.get("confidence", 0.7))
    evidence = conflict_data.get("evidence_source", "")
    conflict_type = conflict_data.get("conflict_type", "safety concern")

    hedge = "may be worth checking" if confidence < 0.75 else "has been identified as"

    if defer_true:
        return (
            "Generate a bilingual EN/AR safety message that defers to a doctor. "
            "Do NOT name any product and do NOT suggest alternatives. "
            "English: clinical, specific, professional medical referral tone. "
            "Arabic: formal, warm, doctor-authority foregrounded. "
            "Return ONLY a raw JSON object with keys 'copy_en' and 'copy_ar'. No markdown."
        )

    return (
        f"Generate a bilingual EN/AR alert for a parent about a product safety conflict.\n\n"
        f"Product: {product_name}\n"
        f"Baby age: {months} months\n"
        f"Conflict type: {conflict_type} ({hedge} a conflict)\n"
        f"Evidence: {evidence[:200]}\n"
        f"Confidence: {confidence:.0%}\n\n"
        "English copy rules:\n"
        "- Authoritative and data-driven framing\n"
        "- Clinical-adjacent tone with specific evidence\n"
        "- 3-4 sentences maximum\n\n"
        "Arabic copy rules:\n"
        "- Warm, reassurance-first tone\n"
        "- Community and family framing\n"
        "- Doctor-authority invoked prominently\n"
        "- Native Arabic — NOT a translation of the English\n"
        "- 3-4 sentences maximum\n\n"
        "Return ONLY a raw JSON object with keys 'copy_en' and 'copy_ar'. No markdown."
    )


def generate_bilingual_copy(conflict_data: dict, child_stage: dict, defer_true: bool) -> dict:
    """
    Layer 5: Generate bilingual EN/AR copy using OpenRouter JSON mode.
    Falls back to deterministic strings if API key is missing or call fails.
    """
    settings = get_settings()

    # Hard guardrail: do not generate persuasive copy under low confidence.
    if float(conflict_data.get("confidence", 0.0)) < 0.60:
        return {"copy_en": None, "copy_ar": None}

    if not settings.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set – using fallback copy generator")
        return _fallback_copy(conflict_data, child_stage, defer_true)

    system_prompt = (
        "You are a parent-friendly AI for a baby e-commerce platform. "
        "Your job is to generate a specific alert regarding a product conflict. "
        "You must return ONLY a raw JSON object with two keys: 'copy_en' and 'copy_ar'. "
        "Do not use markdown blocks. Do not include any other text."
    )

    user_message = _build_bilingual_prompt(conflict_data, child_stage, defer_true)

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 600,
        "temperature": 0.3,
    }

    # Enable JSON mode if model supports it
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
        raw_content = data["choices"][0]["message"]["content"]

        json_str = _extract_json_from_content(raw_content)
        if json_str is None:
            logger.warning("Could not extract JSON from LLM copy response — using fallback")
            return _fallback_copy(conflict_data, child_stage, defer_true)

        result = json.loads(json_str)
        copy_en = result.get("copy_en")
        copy_ar = result.get("copy_ar")

        # Validate output — must have non-empty strings
        if not copy_en or not copy_ar:
            logger.warning("LLM returned empty copy fields — using fallback")
            return _fallback_copy(conflict_data, child_stage, defer_true)

        logger.info(
            "Bilingual copy generated via LLM (EN: %d chars, AR: %d chars)",
            len(copy_en), len(copy_ar),
        )
        return {"copy_en": copy_en, "copy_ar": copy_ar}

    except json.JSONDecodeError as e:
        logger.error("Bilingual copy JSON parse error: %s", e)
        return _fallback_copy(conflict_data, child_stage, defer_true)
    except requests.exceptions.Timeout:
        logger.error("Bilingual copy LLM request timed out")
        return _fallback_copy(conflict_data, child_stage, defer_true)
    except requests.exceptions.HTTPError as e:
        logger.error("Bilingual copy LLM HTTP error: %s", e)
        return _fallback_copy(conflict_data, child_stage, defer_true)
    except Exception as e:
        logger.error("Bilingual LLM generation error: %s", e)
        return _fallback_copy(conflict_data, child_stage, defer_true)


def _fallback_copy(c: dict, child_stage: dict, defer_true: bool) -> dict:
    if defer_true:
        return {
            "copy_en": (
                "Given the safety concern reported, please consult your pediatrician immediately. "
                "Do not continue using this product until you have spoken with a healthcare professional."
            ),
            "copy_ar": (
                "بناءً على المخاوف الأمنية المُبلَّغ عنها، يُرجى استشارة طبيب الأطفال فورًا. "
                "لا تستمري في استخدام هذا المنتج حتى تتحدثي مع متخصص رعاية صحية."
            ),
        }

    confidence = float(c.get("confidence", 0.7))
    confidence_hedge = "may be worth reviewing" if confidence < 0.75 else "is likely no longer suitable"
    months = child_stage.get("months", "this stage")
    product_name = c.get("product_name", "this product")
    evidence = c.get("evidence_source", "pediatric guidelines")

    return {
        "copy_en": (
            f"{product_name} {confidence_hedge} for your baby at {months} months. "
            f"Based on: {evidence[:100]}. "
            f"Most families make this transition between months {months} and {int(months) + 2 if isinstance(months, int) else months + 2}. "  # noqa: E501
            "Please consult your pediatrician if you have concerns."
        ),
        "copy_ar": (
            f"يبدو أن {product_name} لم يعد مناسبًا لطفلتك في هذه المرحلة ({months} أشهر). "
            "معظم العائلات في وضعك تجري هذا التحول في هذا الوقت. "
            "نوصي باستشارة طبيب الأطفال للتأكد من الخيار الأنسب لطفلتك."
        ),
    }
