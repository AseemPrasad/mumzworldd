# app/api/issues.py
# Developer note: Aggregates issue data by various dimensions.
# To add a new group dimension, add it to GROUP_FIELDS and handle in the groupby block.

import logging
from typing import Optional, Literal
from fastapi import APIRouter, Query, HTTPException

from app.core.data_loader import get_df
from app.models.schemas import ApiResponse, IssueGroup

logger = logging.getLogger(__name__)
router = APIRouter()

GROUP_FIELDS = {"issue_type", "product_category", "baby_age_bucket"}


@router.get("/issues", response_model=ApiResponse[list[IssueGroup]])
async def list_issues(
    group_by: Literal["issue_type", "product_category", "baby_age_bucket"] = Query("issue_type"),
    top_n: int = Query(10, ge=1, le=50),
):
    if group_by not in GROUP_FIELDS:
        raise HTTPException(status_code=400, detail=f"group_by must be one of {GROUP_FIELDS}")

    df = get_df()

    grouped = (
        df.groupby(group_by)
        .agg(
            count=(group_by, "count"),
            avg_severity=("severity", "mean"),
            avg_frequency=("frequency_score", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    results = []
    for _, grp_row in grouped.iterrows():
        key = grp_row[group_by]
        samples = df[df[group_by] == key].head(3)
        sample_list = []
        for _, s in samples.iterrows():
            sample_list.append({
                "product_id": s["product_id"],
                "product_name": s["product_name"],
                "return_reason": s["return_reason"],
                "severity": float(s["severity"]),
                "risk_tag": s["risk_tag"],
            })

        results.append(IssueGroup(
            group_key=str(key),
            count=int(grp_row["count"]),
            avg_severity=round(float(grp_row["avg_severity"]), 2),
            avg_frequency=round(float(grp_row["avg_frequency"]), 2),
            sample_records=sample_list,
        ))

    return ApiResponse(
        ok=True,
        data=results,
        meta={"group_by": group_by, "top_n": top_n, "groups_returned": len(results)},
    )
