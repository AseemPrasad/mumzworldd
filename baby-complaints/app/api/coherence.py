from fastapi import APIRouter, Response, status
from pathlib import Path
from app.models.schemas import AuditRequest, CoherenceAuditResponse, TriggerEvent
from app.core.stack_loader import load_stack
from app.core.estimator import estimate_stage
from app.core.rag import detect_conflicts
from app.core.coherence import safe_generate_audit

router = APIRouter()

# Path to the demo dataset
_DEMO_CSV = Path(__file__).parent.parent / "data" / "products_demo.csv"


@router.post("/audit", response_model=CoherenceAuditResponse)
async def run_coherence_audit(request: AuditRequest):
    # Layer 2: Stack Loader
    enriched_stack = load_stack(request.order_history)

    # Layer 1: Estimator
    child_stage = estimate_stage(request.search_history, enriched_stack)

    # Layer 3: Conflict Detector RAG
    conflicts = detect_conflicts(child_stage, enriched_stack)

    # Layers 4 & 5: Classifier + Copy Generator
    req_dict = {
        "cs_chat": request.cs_chat,
        "triggered_by": request.triggered_by
    }
    audit = safe_generate_audit(req_dict, child_stage, conflicts)
    if audit is None:
        # Schema/runtime errors are logged internally and suppressed from users.
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return CoherenceAuditResponse(coherence_audit=audit)


def _run_with_demo_data(request: AuditRequest) -> CoherenceAuditResponse:
    """
    Internal helper: run the coherence pipeline using products_demo.csv
    so demo scenarios always resolve the correct demo products regardless
    of the DATA_PATH configuration.
    """
    import app.core.data_loader as dl

    saved_df = dl._df
    try:
        if _DEMO_CSV.exists():
            dl._df = dl.load_from_path(str(_DEMO_CSV))

        enriched_stack = load_stack(request.order_history)
        child_stage = estimate_stage(request.search_history, enriched_stack)
        conflicts = detect_conflicts(child_stage, enriched_stack)
        req_dict = {
            "cs_chat": request.cs_chat,
            "triggered_by": request.triggered_by,
        }
        audit = safe_generate_audit(req_dict, child_stage, conflicts)
        if audit is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        return CoherenceAuditResponse(coherence_audit=audit)
    finally:
        dl._df = saved_df


# ── Demo Scenarios ─────────────────────────────────────────────────────────────
# Five hardcoded scenarios covering the five conflict rules in conflict_rules.json.
# Each uses products from products_demo.csv (p001–p005).

_DEMO_SCENARIOS: dict[str, AuditRequest] = {
    "honey_botulism": AuditRequest(
        session_id="demo_honey_botulism",
        triggered_by=TriggerEvent.milestone_crossing,
        order_history=["p001"],          # Honey Teething Gel — age 2 months
        search_history=["teething", "newborn"],
        cs_chat="",
    ),
    "choking_hazard": AuditRequest(
        session_id="demo_choking_hazard",
        triggered_by=TriggerEvent.session_open,
        order_history=["p002"],          # Ring Teether with Beads — age 6 months
        search_history=["teething", "sit"],
        cs_chat="",
    ),
    "fragrance_sensitivity": AuditRequest(
        session_id="demo_fragrance_sensitivity",
        triggered_by=TriggerEvent.milestone_crossing,
        order_history=["p003"],          # Lavender Scented Baby Lotion — age 1 month
        search_history=["newborn", "diaper"],
        cs_chat="Baby developed a rash after using the lotion",
    ),
    "formula_transition": AuditRequest(
        session_id="demo_formula_transition",
        triggered_by=TriggerEvent.milestone_crossing,
        order_history=["p004"],          # Stage 1 Infant Formula — age 6 months
        search_history=["teething", "sit", "crawl"],
        cs_chat="",
    ),
    "allergen_too_early": AuditRequest(
        session_id="demo_allergen_too_early",
        triggered_by=TriggerEvent.session_open,
        order_history=["p005"],          # Peanut Butter Baby Puffs — age 1 month
        search_history=["newborn", "colic"],
        cs_chat="",
    ),
}


@router.get("/demo", response_model=CoherenceAuditResponse)
async def demo_audit(scenario: str = "honey_botulism"):
    """
    Demo endpoint with 5 hardcoded scenarios showing real conflict detection.

    Available scenarios:
    - honey_botulism        : 2-month-old, honey teething gel (botulism risk, severity 9)
    - choking_hazard        : 6-month-old, ring teether with beads (choking risk, severity 8)
    - fragrance_sensitivity : 1-month-old, scented lotion (skin irritation + rash, severity 7)
    - formula_transition    : 6-month-old, Stage 1 formula (stage mismatch, severity 3)
    - allergen_too_early    : 1-month-old, peanut product (allergen risk, severity 8)
    """
    req = _DEMO_SCENARIOS.get(scenario, _DEMO_SCENARIOS["honey_botulism"])
    return _run_with_demo_data(req)


