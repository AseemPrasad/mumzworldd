# app/api/products.py
# Developer note: Product listing and detail endpoints.
# To add new filter dimensions, add a query param and extend the filter chain.

import logging
from typing import Optional, Literal
from fastapi import APIRouter, Query, HTTPException

from app.core.data_loader import get_df
from app.models.schemas import ApiResponse, ProductRecord, ProductDetail

logger = logging.getLogger(__name__)
router = APIRouter()

SORT_FIELDS = {"severity", "frequency_score", "report_date", "composite_score"}
MAX_LIMIT = 200


def _row_to_record(row) -> dict:
    rec = row.to_dict()
    # Serialize datetime
    if hasattr(rec.get("report_date"), "isoformat"):
        rec["report_date"] = rec["report_date"].isoformat()[:10]
    # Convert numpy types for JSON
    for k, v in rec.items():
        if hasattr(v, "item"):
            rec[k] = v.item()
    return rec


@router.get("/products", response_model=ApiResponse[list[ProductRecord]])
async def list_products(
    category: Optional[str] = Query(None),
    min_age: Optional[float] = Query(None, ge=0),
    max_age: Optional[float] = Query(None, ge=0),
    risk_tag: Optional[Literal["low", "medium", "high"]] = Query(None),
    sort_by: Optional[str] = Query("report_date"),
    limit: int = Query(20, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
):
    if sort_by and sort_by not in SORT_FIELDS:
        raise HTTPException(status_code=400, detail=f"sort_by must be one of {SORT_FIELDS}")

    df = get_df().copy()

    if category:
        df = df[df["product_category"].str.lower() == category.lower()]
    if min_age is not None:
        df = df[df["baby_age_months"] >= min_age]
    if max_age is not None:
        df = df[df["baby_age_months"] <= max_age]
    if risk_tag:
        df = df[df["risk_tag"] == risk_tag]

    if sort_by:
        ascending = sort_by != "composite_score"
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")

    total = len(df)
    page = df.iloc[offset: offset + limit]
    records = [ProductRecord(**_row_to_record(row)) for _, row in page.iterrows()]

    return ApiResponse(
        ok=True,
        data=records,
        meta={"total": total, "limit": limit, "offset": offset, "count": len(records)},
    )


@router.get("/products/{product_id}", response_model=ApiResponse[ProductDetail])
async def get_product(product_id: str):
    df = get_df()
    match = df[df["product_id"] == product_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")

    row = match.iloc[0]
    record = ProductRecord(**_row_to_record(row))

    issue_summary = {
        "issue_type": record.issue_type,
        "severity": record.severity,
        "frequency_score": record.frequency_score,
        "composite_score": record.composite_score,
        "risk_tag": record.risk_tag,
        "resolution_status": record.resolution_status,
        "explanation": record.risk_explanation,
    }

    return ApiResponse(
        ok=True,
        data=ProductDetail(record=record, issue_summary=issue_summary),
        meta={"product_id": product_id},
    )
