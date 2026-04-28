import logging
from typing import List

from app.core.stack_loader import load_stack
from app.core.estimator import estimate_stage
from app.core.rag import detect_conflicts
from app.core.coherence import safe_generate_audit
from app.models.schemas import CoherenceAudit, TriggerEvent

logger = logging.getLogger(__name__)


class CoherenceEngine:
    """Compatibility wrapper over the canonical 5-layer coherence pipeline."""

    def run(
        self,
        session_id: str,
        triggered_by: TriggerEvent,
        order_history: List[str],
        search_history: List[str],
        cs_chat: str | None = None,
    ) -> CoherenceAudit | None:
        enriched_stack = load_stack(order_history)
        child_stage = estimate_stage(search_history, enriched_stack)
        conflicts = detect_conflicts(child_stage, enriched_stack)

        request_data = {
            "session_id": session_id,
            "triggered_by": triggered_by,
            "cs_chat": cs_chat or "",
        }
        return safe_generate_audit(request_data, child_stage, conflicts)
