import logging
from typing import List, Dict
import pandas as pd
from app.core.data_loader import get_df

logger = logging.getLogger(__name__)

INGREDIENT_FLAG_RULES = {
    "honey": ["honey", "raw honey"],
    "fragrance": ["fragrance", "perfume", "scented"],
    "nuts": ["nut", "peanut", "almond"],
    "latex": ["latex"],
}


def _extract_ingredient_flags(text: str) -> list[str]:
    source = (text or "").lower()
    flags = []
    for name, needles in INGREDIENT_FLAG_RULES.items():
        if any(n in source for n in needles):
            flags.append(name)
    return sorted(set(flags))

def load_stack(skus: List[str]) -> List[Dict]:
    """
    Layer 2: Retrieves the parent's full order history and enriches each SKU 
    with catalog metadata. Returns explicit warnings/flags for missing data.
    """
    df = get_df()
    stack = []
    
    for sku in skus:
        matches = df[df['product_id'] == sku]
        if matches.empty:
            logger.warning("StackLoader: Missing SKU metadata for %s", sku)
            stack.append({
                "sku": sku,
                "name": "Unknown Product",
                "age_min_months": None,
                "age_max_months": None,
                "ingredient_flags": [],
                "metadata_complete": False,
                "warning": "missing_sku_metadata",
            })
            continue
            
        row = matches.iloc[0]
        reason_text = str(row.get("return_reason", ""))
        product_text = str(row.get("product_name", ""))
        ingredient_flags = _extract_ingredient_flags(f"{product_text} {reason_text}")

        age_months = int(row['baby_age_months']) if not pd.isna(row['baby_age_months']) else None
        age_min_months = age_months
        age_max_months = age_months + 2 if age_months is not None else None

        metadata_complete = all(
            v is not None
            for v in [age_min_months, age_max_months]
        )

        stack.append({
            "sku": sku,
            "name": row['product_name'],
            "purchased_date": str(row.get("report_date")) if not pd.isna(row.get("report_date")) else None,
            "category": row['product_category'],
            "brand": row['brand'],
            "age_months": age_months,
            "age_min_months": age_min_months,
            "age_max_months": age_max_months,
            "ingredient_flags": ingredient_flags,
            "issue_type": row['issue_type'],
            "return_reason": row['return_reason'],
            "severity": float(row['severity']),
            "risk_tag": row['risk_tag'],
            "metadata_complete": metadata_complete,
            "warning": None if metadata_complete else "missing_age_range_metadata",
        })
        
    return stack
