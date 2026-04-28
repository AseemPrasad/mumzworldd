# app/api/risks.py
# Developer note: Risk-ranked product list with TTL cache.
# To change the cache TTL, update the `ttl` parameter in TTLCache.
# To add new risk fields to the response, extend RiskRecord in schemas.py.

import logging
import time
from typing import Optional, Literal
from fastapi import APIRouter, Query

from app.core.data_loader import get_df
from app.models.schemas import ApiResponse, RiskRecord

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple TTL cache: (threshold, top_n) -> (timestamp, data)
_risk_cache: dict = {}
CACHE_TTL = 60  # seconds


def _get_cached(key: tuple):
    entry = _risk_cache.get(key)
    if entry and (time.time() - entry["ts"] < CACHE_TTL):
        logger.debug("Risk cache hit for key %s", key)
        return entry["data"]
    return None


def _set_cached(key: tuple, data):
    _risk_cache[key] = {"ts": time.time(), "data": data}


@router.get("/risks", response_model=ApiResponse[list[RiskRecord]])
async def list_risks(
    threshold: Optional[Literal["low", "medium", "high"]] = Query(None),
    top_n: int = Query(10, ge=1, le=50),
):
    cache_key = (threshold, top_n)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    df = get_df().copy()

    if threshold:
        df = df[df["risk_tag"] == threshold]

    df = df.sort_values("composite_score", ascending=False).head(top_n)

    records = []
    for _, row in df.iterrows():
        records.append(RiskRecord(
            product_id=str(row["product_id"]),
            product_name=str(row["product_name"]),
            brand=str(row["brand"]),
            product_category=str(row["product_category"]),
            composite_score=round(float(row["composite_score"]), 4),
            risk_tag=str(row["risk_tag"]),
            explanation=str(row["risk_explanation"]),
            severity=float(row["severity"]),
            frequency_score=float(row["frequency_score"]),
            issue_type=str(row["issue_type"]),
            return_reason=str(row["return_reason"]),
        ))

    response = ApiResponse(
        ok=True,
        data=records,
        meta={
            "threshold": threshold,
            "top_n": top_n,
            "returned": len(records),
            "cache_ttl_seconds": CACHE_TTL,
        },
    )
    _set_cached(cache_key, response)
    return response
