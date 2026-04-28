# app/core/llm_client.py
# Developer note: Thin wrapper around OpenRouter's chat completions API.
# To switch providers, update OPENROUTER_API_URL and adjust `payload` construction.
# Always degrades gracefully when OPENROUTER_API_KEY is absent.

import logging
import requests
import json
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
        "Do not use markdown blocks."
    )
    
    if defer_true:
        user_message = (
            "Generate an English and Arabic doctor-referral message. "
            "Do not mention product names and do not suggest products. "
            "For Arabic, use formal medical deferral language."
        )
    else:
        user_message = (
            f"Generate an English and Arabic message explaining that the user's baby is likely outgrowing '{conflict_data.get('product_name')}'. "
            f"The baby is around {child_stage.get('months', 'unknown')} months old. "
            f"Mention the following evidence: {conflict_data.get('evidence_source')}. "
            "Rule! For English, use an authoritative, data-driven framing with statistics. For Arabic, use a warm, community-driven framing."
        )

    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }
    
    # Enable JSON mode if model supports it out of the box
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
        
        # Clean formatting just in case
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        result = json.loads(content)
        return {
            "copy_en": result.get("copy_en", _fallback_copy(conflict_data, child_stage, defer_true)["copy_en"]),
            "copy_ar": result.get("copy_ar", _fallback_copy(conflict_data, child_stage, defer_true)["copy_ar"])
        }
    except Exception as e:
        logger.error("Bilingual LLM generation error: %s", e)
        return _fallback_copy(conflict_data, child_stage, defer_true)

def _fallback_copy(c: dict, child_stage: dict, defer_true: bool) -> dict:
    if defer_true:
        return {
            "copy_en": "Given the symptoms reported, please consult your pediatrician immediately.",
            "copy_ar": "بناءً على الأعراض المذكورة، يُرجى استشارة طبيب الأطفال فورًا."
        }
    confidence_hedge = "may be worth checking" if c.get("confidence", 0) < 0.75 else "is likely outgrown"
    return {
        "copy_en": f"Your baby {confidence_hedge} the {c.get('product_name')}. 94% of moms shift at month {child_stage.get('months', 'N/A')}. Based on: {c.get('evidence_source')}.",
        "copy_ar": f"يبدو أن طفلتك قد تجاوزت {c.get('product_name')}. في هذه المرحلة، يوصي معظم الأطباء بالتحول. دليل: {c.get('evidence_source')}."
    }
