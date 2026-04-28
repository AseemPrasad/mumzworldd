# app/models/schemas.py
# Developer note: All request/response shapes live here.
# Extend by adding new fields to existing models or creating new ones.
# All endpoints wrap responses in ApiResponse for consistency.

from enum import Enum
from typing import Any, Optional, List, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime
T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    ok: bool
    data: Optional[T] = None
    meta: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ProductRecord(BaseModel):
    product_id: str
    product_name: str
    brand: str
    product_category: str
    baby_age_months: float
    issue_type: str
    return_reason: str
    severity: float
    frequency_score: float
    risk_tag: str
    report_date: Optional[str]
    resolution_status: str
    composite_score: Optional[float] = None
    normalized_severity: Optional[float] = None
    normalized_frequency: Optional[float] = None
    risk_explanation: Optional[str] = None
    baby_age_bucket: Optional[str] = None


class ProductDetail(BaseModel):
    record: ProductRecord
    issue_summary: dict[str, Any]


class IssueGroup(BaseModel):
    group_key: str
    count: int
    avg_severity: float
    avg_frequency: float
    sample_records: List[dict[str, Any]]


class RiskRecord(BaseModel):
    product_id: str
    product_name: str
    brand: str
    product_category: str
    composite_score: float
    risk_tag: str
    explanation: str
    severity: float
    frequency_score: float
    issue_type: str
    return_reason: str


class LLMRequest(BaseModel):
    product_ids: List[str]
    prompt_template: Optional[str] = "Summarize the key safety concerns and recommend actions."


class LLMResponse(BaseModel):
    summary: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    products_analyzed: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str


class TriggerEvent(str, Enum):
    milestone_crossing = "milestone_crossing"
    session_open = "session_open"
    cs_chat = "cs_chat"

class AuditStatus(str, Enum):
    complete = "complete"
    partial = "partial"
    insufficient_data = "insufficient_data"
    deferred = "deferred"
    schema_error = "schema_error"

class ChildStage(BaseModel):
    months: Optional[int] = None
    confidence: Optional[float] = None
    evidence: List[str] = Field(default_factory=list)
    null_reason: Optional[str] = None

class ConflictDetail(BaseModel):
    product_sku: str
    product_name: str
    conflict_type: str
    signals_supporting: int
    confidence: float
    evidence_source: str
    action: str
    defer_to_doctor: bool = False
    copy_en: Optional[str] = None
    copy_ar: Optional[str] = None

class CoherenceAudit(BaseModel):
    run_id: str
    triggered_by: TriggerEvent
    child_stage: ChildStage
    conflicts: List[ConflictDetail] = Field(default_factory=list)
    audit_status: AuditStatus
    null_reason: Optional[str] = None
    schema_version: str = "1.0"

class CoherenceAuditResponse(BaseModel):
    coherence_audit: CoherenceAudit

class AuditRequest(BaseModel):
    session_id: str
    triggered_by: TriggerEvent
    order_history: List[str] = Field(default_factory=list, description="List of SKUs/Product IDs")
    search_history: List[str] = Field(default_factory=list, description="Recent search terms")
    cs_chat: Optional[str] = None
