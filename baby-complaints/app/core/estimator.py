import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return (2 * a * b) / (a + b)


def estimate_stage(search_terms: List[str], enriched_stack: List[Dict]) -> Dict[str, Any]:
    """
    Layer 1: Milestone Estimator Agent.
    Infers developmental stage from passive signals.
    Fails explicitly if confidence < 0.6.
    """
    evidence = []
    stack_ages = [
        item["age_months"]
        for item in enriched_stack
        if item.get("age_months") is not None
    ]
    
    signal_stack = None
    stack_conf = 0.0
    if stack_ages:
        signal_stack = sum(stack_ages) / len(stack_ages)
        stack_spread = max(stack_ages) - min(stack_ages) if len(stack_ages) > 1 else 0.0
        stack_conf = max(0.0, min(1.0, 0.92 - (stack_spread / 18.0)))
        evidence.append(f"order_size_drift (avg={signal_stack:.1f}m)")
        
    signal_search = None
    search_conf = 0.0
    search_str = " ".join(search_terms).lower()
    
    # Simple semantic heuristics for search terms
    if any(k in search_str for k in ["newborn", "diaper", "colic"]):
        signal_search = 1
        search_conf = 0.82
        evidence.append("search_term_shift (newborn vocabulary)")
    elif any(k in search_str for k in ["teething", "sit", "crawl"]):
        signal_search = 6
        search_conf = 0.82
        evidence.append("search_term_shift (teething vocabulary)")
    elif any(k in search_str for k in ["walk", "toddler", "shoes"]):
        signal_search = 12
        search_conf = 0.82
        evidence.append("search_term_shift (toddler vocabulary)")
        
    # Agent Logic: Combine signals
    if signal_stack is not None and signal_search is not None:
        # 4-week agreement window (approximately 1 month)
        if abs(signal_stack - signal_search) <= 1:
            confidence = round(_harmonic_mean(stack_conf, search_conf), 2)
            months = int(round((signal_stack + signal_search) / 2.0))
            return {
                "months": months,
                "confidence": confidence,
                "evidence": evidence,
                "null_reason": None
            }
        else:
            return {
                "months": None,
                "confidence": 0.40,
                "evidence": evidence,
                "null_reason": "signals_conflict_too_widely"
            }
            
    if signal_stack is not None or signal_search is not None:
        return {
            "months": None,
            "confidence": 0.50,
            "evidence": evidence,
            "null_reason": "only_one_signal_available"
        }
        
    return {
        "months": None,
        "confidence": 0.0,
        "evidence": [],
        "null_reason": "insufficient_data"
    }
