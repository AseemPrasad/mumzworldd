import logging
from typing import List, Dict
import pandas as pd
from app.core.data_loader import get_df

logger = logging.getLogger(__name__)

INGREDIENT_FLAG_RULES = {
    "honey": ["honey", "raw honey"],
    "fragrance": ["fragrance", "perfume", "scented", "lavender", "rose", "jasmine"],
    "nuts": ["nut", "peanut", "almond", "sesame", "tree nut"],
    "latex": ["latex"],
}

# Issue types that indicate ingredient/safety concerns independent of age
SAFETY_ISSUE_TYPES = {
    "ingredient_age_safety",
    "choking_hazard",
    "skin_irritation",
    "allergen",
}


def _extract_ingredient_flags(text: str) -> list[str]:
    source = (text or "").lower()
    flags = []
    for name, needles in INGREDIENT_FLAG_RULES.items():
        if any(n in source for n in needles):
            flags.append(name)
    return sorted(set(flags))


def _get_conflict_rule_age_range(ingredient_flags: list[str], issue_type: str) -> tuple[int | None, int | None]:
    """
    Look up age range constraints from conflict_rules.json for this product's flags.
    Returns (age_safe_min_months, age_safe_max_months) or (None, None) if no rule matches.
    """
    try:
        from app.core.conflict_loader import load_conflict_rules
        rules = load_conflict_rules()
        flag_set = set(ingredient_flags)
        for rule in rules:
            rule_flags = set(rule.get("ingredient_flags", []))
            # Match on ingredient flags
            if flag_set & rule_flags:
                age_min = rule.get("age_safe_min_months")
                age_max = rule.get("age_safe_max_months")
                return (int(age_min) if age_min is not None else None,
                        int(age_max) if age_max is not None else None)
            # Match on issue type via conflict_type
            if issue_type in ("ingredient_age_safety", "stage_mismatch"):
                if rule.get("conflict_type") == issue_type:
                    age_min = rule.get("age_safe_min_months")
                    age_max = rule.get("age_safe_max_months")
                    return (int(age_min) if age_min is not None else None,
                            int(age_max) if age_max is not None else None)
    except Exception as e:
        logger.warning("Could not query conflict rules for age range: %s", e)
    return None, None


def load_stack(skus: List[str]) -> List[Dict]:
    """
    Layer 2: Retrieves the parent's full order history and enriches each SKU
    with catalog metadata. Returns explicit warnings/flags for missing data.

    Age range is enriched from conflict_rules.json when ingredient flags match.
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
        issue_type = str(row.get("issue_type", ""))
        ingredient_flags = _extract_ingredient_flags(f"{product_text} {reason_text}")

        age_months = int(row['baby_age_months']) if not pd.isna(row['baby_age_months']) else None

        # Use conflict rule age ranges when available (more accurate than CSV-derived ranges)
        rule_age_min, rule_age_max = _get_conflict_rule_age_range(ingredient_flags, issue_type)

        if rule_age_min is not None:
            age_min_months = rule_age_min
            age_max_months = rule_age_max  # May be None for open-ended upper bound
        else:
            age_min_months = age_months
            age_max_months = age_months + 2 if age_months is not None else None

        metadata_complete = all(
            v is not None
            for v in [age_min_months]
        ) and age_months is not None

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
            "issue_type": issue_type,
            "return_reason": row['return_reason'],
            "severity": float(row['severity']),
            "risk_tag": row['risk_tag'],
            "metadata_complete": metadata_complete,
            "warning": None if metadata_complete else "missing_age_range_metadata",
        })

    return stack
