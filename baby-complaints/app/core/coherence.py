import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import ValidationError
from app.models.schemas import CoherenceAudit, ChildStage, ConflictDetail, TriggerEvent, AuditStatus
from app.core.llm_client import generate_bilingual_copy
import uuid

logger = logging.getLogger(__name__)

SEVERITY_KEYWORDS = ["rash", "fever", "vomit", "vomiting", "swelling", "blood", "allergy", "reaction"]
OPS_LOG_PATH = Path("app/data/ops_eval_log.txt")
OPS_METRICS = {
    "schema_errors": 0,
    "audit_runs": 0,
    "deferred_runs": 0,
    "insufficient_data_runs": 0,
}


def _append_ops_log(event: str, payload: str) -> None:
    OPS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OPS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{event}|{payload}\n")


def generate_audit(request_data: Dict[str, Any], child_stage: Dict[str, Any], conflicts: List[Dict]) -> CoherenceAudit:
    """Layers 4 & 5: Deferral Classifier + Copy Generator + status assignment."""
    cs_chat = request_data.get("cs_chat", "").lower() if request_data.get("cs_chat") else ""

    # Layer 4: deferral guardrails
    defer_audit = False
    if any(k in cs_chat for k in SEVERITY_KEYWORDS):
        defer_audit = True
    if float(child_stage.get("confidence", 0.0)) < 0.60:
        defer_audit = True

    conflict_details: List[ConflictDetail] = []
    has_defer_conflict = defer_audit

    for c in conflicts:
        defer_this_conflict = (
            defer_audit
            or float(c.get("confidence", 0.0)) < 0.6
            or c.get("conflict_type", "") == "medical_symptom"
        )
        if defer_this_conflict:
            has_defer_conflict = True

        copy_result = generate_bilingual_copy(c, child_stage, defer_this_conflict)

        conflict_details.append(
            ConflictDetail(
                product_sku=c["product_sku"],
                product_name=c["product_name"],
                conflict_type=c["conflict_type"],
                signals_supporting=int(c["signals_supporting"]),
                confidence=float(c["confidence"]),
                evidence_source=c["evidence_source"],
                action="flag_with_doctor_referral" if defer_this_conflict else c["action"],
                defer_to_doctor=defer_this_conflict,
                copy_en=copy_result.get("copy_en"),
                copy_ar=copy_result.get("copy_ar"),
            )
        )

    # Status resolution per blueprint failure states
    if has_defer_conflict:
        audit_status = AuditStatus.deferred
    elif child_stage.get("months") is None and len(child_stage.get("evidence", [])) < 2:
        audit_status = AuditStatus.insufficient_data
    elif child_stage.get("months") is None:
        audit_status = AuditStatus.partial
    else:
        audit_status = AuditStatus.complete

    null_reason = None
    if audit_status == AuditStatus.partial:
        null_reason = child_stage.get("null_reason") or "partial_missing_fields"
    elif audit_status == AuditStatus.insufficient_data:
        null_reason = child_stage.get("null_reason") or "insufficient_data"

    OPS_METRICS["audit_runs"] += 1
    if audit_status == AuditStatus.deferred:
        OPS_METRICS["deferred_runs"] += 1
    if audit_status == AuditStatus.insufficient_data:
        OPS_METRICS["insufficient_data_runs"] += 1

    return CoherenceAudit(
        run_id=str(uuid.uuid4()),
        triggered_by=request_data.get("triggered_by", TriggerEvent.session_open),
        child_stage=ChildStage(
            months=child_stage.get("months"),
            confidence=child_stage.get("confidence"),
            evidence=child_stage.get("evidence", []),
            null_reason=child_stage.get("null_reason"),
        ),
        conflicts=conflict_details,
        audit_status=audit_status,
        null_reason=null_reason,
        schema_version="1.0",
    )


def safe_generate_audit(request_data: Dict[str, Any], child_stage: Dict[str, Any], conflicts: List[Dict]) -> Optional[CoherenceAudit]:
    """Validate and suppress malformed output while logging schema errors."""
    try:
        return generate_audit(request_data, child_stage, conflicts)
    except ValidationError as exc:
        OPS_METRICS["schema_errors"] += 1
        _append_ops_log("schema_error", str(exc).replace("\n", " "))
        logger.exception("Coherence schema validation error: %s", exc)
        return None
    except Exception as exc:  # defensive fallback
        OPS_METRICS["schema_errors"] += 1
        _append_ops_log("runtime_error", str(exc).replace("\n", " "))
        logger.exception("Coherence audit generation failed: %s", exc)
        return None
