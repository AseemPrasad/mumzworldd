#!/usr/bin/env python3
# scripts/generate_synthetic.py
# Developer note: Run this to regenerate the sample CSV with N rows.
# Usage: python scripts/generate_synthetic.py --rows 100 --seed 42 --output app/data/products_sample.csv

import argparse
import csv
import random
from datetime import date, timedelta

BRANDS = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "BrandF", "BrandG", "BrandH"]

CATEGORIES = {
    "Feeding": {
        "products": ["AntiColic Bottle", "Sippy Cup", "Breast Pump", "Baby Spoon Set", "Formula Dispenser"],
        "issues": ["leakage", "material_defect", "mold_growth", "breakage"],
    },
    "Toys": {
        "products": ["Ring Teether", "Soft Rattle", "Activity Mat", "Bath Toy", "Wooden Blocks"],
        "issues": ["choking_hazard", "breakage", "paint_toxicity", "sharp_edges"],
    },
    "Hygiene": {
        "products": ["Newborn Diapers", "Organic Baby Wipes", "Baby Shampoo", "Diaper Rash Cream"],
        "issues": ["skin_irritation", "poor_fit", "allergic_reaction", "leakage"],
    },
    "Skincare": {
        "products": ["Baby Soft Lotion", "Sunscreen SPF50", "Cradle Cap Oil", "Baby Powder"],
        "issues": ["skin_irritation", "allergic_reaction", "rash", "fragrance_sensitivity"],
    },
    "Accessories": {
        "products": ["Stroller Clip", "Car Seat Mirror", "Baby Carrier", "Pacifier Clip"],
        "issues": ["breakage", "sharp_edges", "choking_hazard", "material_defect"],
    },
    "Safety": {
        "products": ["Infant Car Seat", "Baby Gate", "Corner Guards", "Baby Monitor Cam"],
        "issues": ["harness_issue", "installation_failure", "breakage", "connectivity"],
    },
    "Clothing": {
        "products": ["Baby Sleep Sack", "Onesie Set", "Winter Mittens", "Sun Hat"],
        "issues": ["sizing", "dye_bleeding", "breakage", "allergic_reaction"],
    },
    "Electronics": {
        "products": ["Baby Monitor", "Sound Machine", "Night Light", "Video Monitor"],
        "issues": ["connectivity", "overheating", "battery_failure", "screen_defect"],
    },
}

REASON_TEMPLATES = {
    "skin_irritation": [
        "Caused rash after first use",
        "Baby developed red patches within hours",
        "Skin became dry and irritated overnight",
    ],
    "choking_hazard": [
        "Small parts broke off during normal use",
        "Piece detached unexpectedly – near miss",
        "Component cracked revealing small fragment",
    ],
    "leakage": [
        "Product leaks consistently during use",
        "Seal failed after third wash",
        "Liquid seeps from bottom seam",
    ],
    "poor_fit": [
        "Does not fit as described for age range",
        "Too loose even at tightest setting",
        "Sizing runs significantly small",
    ],
    "breakage": [
        "Broke after minimal use",
        "Snapped during first outing",
        "Plastic cracked under normal pressure",
    ],
    "allergic_reaction": [
        "Baby showed allergic response after contact",
        "Hives appeared within 30 minutes of use",
        "Pediatrician confirmed product-related allergy",
    ],
    "harness_issue": [
        "Buckle difficult to release under stress",
        "Strap adjustment mechanism failed",
        "Harness does not hold secure position",
    ],
    "material_defect": [
        "Material discolored after first wash",
        "Texture degraded unexpectedly",
        "Component separated from base",
    ],
    "connectivity": [
        "Device disconnects randomly overnight",
        "Signal drops in adjacent room",
        "Pairing fails after firmware update",
    ],
    "sizing": [
        "Runs two sizes small",
        "Zipper broke on first use",
        "Elastic too tight for stated age",
    ],
}

DEFAULT_REASON = "Product did not meet safety expectations for stated age group"


def random_reason(issue_type: str, rng: random.Random) -> str:
    templates = REASON_TEMPLATES.get(issue_type, [DEFAULT_REASON])
    return rng.choice(templates)


def random_age(rng: random.Random) -> int:
    """Skewed toward 0-12 months."""
    if rng.random() < 0.7:
        return rng.randint(0, 12)
    return rng.randint(13, 36)


def random_date(rng: random.Random) -> str:
    start = date(2025, 1, 1)
    offset = rng.randint(0, 480)
    return (start + timedelta(days=offset)).isoformat()


def compute_risk_tag(severity: int, frequency: int) -> str:
    norm_sev = (severity - 1) / 9
    norm_freq = (frequency - 1) / 9
    composite = 0.6 * norm_sev + 0.4 * norm_freq
    if composite >= 0.75:
        return "high"
    elif composite >= 0.45:
        return "medium"
    return "low"


def generate(rows: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    categories = list(CATEGORIES.keys())

    for i in range(rows):
        pid = f"p{i+1:04d}"
        cat = rng.choice(categories)
        cat_data = CATEGORIES[cat]
        product = rng.choice(cat_data["products"])
        brand = rng.choice(BRANDS)
        issue = rng.choice(cat_data["issues"])
        age = random_age(rng)
        severity = rng.randint(1, 10)
        frequency = rng.randint(1, 10)
        tag = compute_risk_tag(severity, frequency)
        status = rng.choice(["open", "returned", "exchanged", "resolved"])
        records.append({
            "product_id": pid,
            "product_name": product,
            "brand": brand,
            "product_category": cat,
            "baby_age_months": age,
            "issue_type": issue,
            "return_reason": random_reason(issue, rng),
            "severity": severity,
            "frequency_score": frequency,
            "risk_tag": tag,
            "report_date": random_date(rng),
            "resolution_status": status,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic baby product complaint CSV")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="app/data/products_sample.csv")
    args = parser.parse_args()

    records = generate(args.rows, args.seed)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"✓ Generated {args.rows} rows → {args.output}")


if __name__ == "__main__":
    main()
