from fastapi import APIRouter, Response, status
from app.models.schemas import AuditRequest, CoherenceAuditResponse, TriggerEvent
from app.core.stack_loader import load_stack
from app.core.estimator import estimate_stage
from app.core.rag import detect_conflicts
from app.core.coherence import safe_generate_audit

router = APIRouter()

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

@router.get("/demo", response_model=CoherenceAuditResponse)
async def demo_audit(scenario: str = "normal"):
    """
    Demo endpoint to test the engine without frontend wiring
    """
    if scenario == "medical_deferral":
        req = AuditRequest(
            session_id="mock_session_1",
            triggered_by=TriggerEvent.cs_chat,
            order_history=["p002"], # Teether
            search_history=["teething", "fever"],
            cs_chat="My baby has a rash after using the teether"
        )
    elif scenario == "insufficient_signal":
        req = AuditRequest(
            session_id="mock_session_2",
            triggered_by=TriggerEvent.session_open,
            order_history=[],
            search_history=[],
            cs_chat=""
        )
    else:
        req = AuditRequest(
            session_id="mock_session_3",
            triggered_by=TriggerEvent.session_open,
            order_history=["p004", "p002"], # Newborn Diapers + Teether
            search_history=["teething", "sit", "crawl"],
            cs_chat=""
        )
        
    return await run_coherence_audit(req)
