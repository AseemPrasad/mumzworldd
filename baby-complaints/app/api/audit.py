from fastapi import APIRouter, Response, status

from app.models.schemas import AuditRequest, CoherenceAuditResponse
from app.core.engine import CoherenceEngine

router = APIRouter()


@router.post("/audit", response_model=CoherenceAuditResponse, tags=["Audit"])
async def run_audit(body: AuditRequest):
    """Compatibility endpoint that delegates to the canonical coherence engine."""
    engine = CoherenceEngine()
    result = engine.run(
        session_id=body.session_id,
        triggered_by=body.triggered_by,
        order_history=body.order_history,
        search_history=body.search_history,
        cs_chat=body.cs_chat,
    )
    if result is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return CoherenceAuditResponse(coherence_audit=result)
