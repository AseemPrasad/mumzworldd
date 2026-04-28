# app/core/risk.py
# Developer note: Deterministic risk scoring. To change the weighting or thresholds,
# update SEVERITY_WEIGHT, FREQUENCY_WEIGHT, and the threshold constants below.

import logging
from typing import Any

logger = logging.getLogger(__name__)

SEVERITY_WEIGHT = 0.6
FREQUENCY_WEIGHT = 0.4
HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.45
SCORE_MIN = 1
SCORE_MAX = 10


def _normalize(value: float, lo: float = SCORE_MIN, hi: float = SCORE_MAX) -> float:
    """Normalize a value to the 0-1 range."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def compute_risk(record: dict[str, Any]) -> dict[str, Any]:
    """
    Compute composite risk for a single product record.

    Formula:
        composite = 0.6 * norm_severity + 0.4 * norm_frequency

    Tags:
        >= 0.75  -> high
        >= 0.45  -> medium
        <  0.45  -> low
    """
    try:
        severity = float(record.get("severity", 1))
        frequency = float(record.get("frequency_score", 1))
    except (TypeError, ValueError):
        logger.warning("Invalid severity/frequency for record %s", record.get("product_id"))
        severity, frequency = 1.0, 1.0

    norm_sev = _normalize(severity)
    norm_freq = _normalize(frequency)
    composite = SEVERITY_WEIGHT * norm_sev + FREQUENCY_WEIGHT * norm_freq

    if composite >= HIGH_THRESHOLD:
        tag = "high"
    elif composite >= MEDIUM_THRESHOLD:
        tag = "medium"
    else:
        tag = "low"

    explanation = (
        f"severity={severity:.0f}/10 (weight 60%) + "
        f"frequency={frequency:.0f}/10 (weight 40%) → "
        f"composite={composite:.3f} → {tag}"
    )

    return {
        "normalized_severity": round(norm_sev, 4),
        "normalized_frequency": round(norm_freq, 4),
        "composite_score": round(composite, 4),
        "risk_tag": tag,
        "explanation": explanation,
    }
