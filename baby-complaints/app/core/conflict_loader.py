# app/core/conflict_loader.py
# Loads conflict_rules.json into ChromaDB on startup.
# Indexes both EN and AR document content separately.
# Preserves source attribution (who_guideline, aap_guideline).

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_conflict_collection = None
_conflict_rules: list[dict] = []

CONFLICT_RULES_PATH = Path(__file__).parent.parent / "data" / "conflict_rules.json"


def load_conflict_rules() -> list[dict]:
    """Load raw rules from JSON file. Returns empty list on missing file."""
    global _conflict_rules
    if _conflict_rules:
        return _conflict_rules

    if not CONFLICT_RULES_PATH.exists():
        logger.warning("conflict_rules.json not found at %s", CONFLICT_RULES_PATH)
        return []

    try:
        with CONFLICT_RULES_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        _conflict_rules = data.get("rules", [])
        logger.info("Loaded %d conflict rules from %s", len(_conflict_rules), CONFLICT_RULES_PATH)
        return _conflict_rules
    except Exception as e:
        logger.error("Failed to load conflict_rules.json: %s", e)
        return []


def get_conflict_collection():
    """Return the initialized ChromaDB collection, or None if unavailable."""
    return _conflict_collection


def init_conflict_loader() -> None:
    """
    Load conflict_rules.json into ChromaDB.
    Indexes EN and AR document content separately for bilingual retrieval.
    """
    global _conflict_collection

    rules = load_conflict_rules()
    if not rules:
        logger.warning("No conflict rules to index — skipping ChromaDB conflict loader init")
        return

    try:
        import chromadb
        client = chromadb.Client()
        _conflict_collection = client.get_or_create_collection(name="conflict_rules_corpus")
    except ImportError:
        logger.error("ChromaDB not installed — conflict loader unavailable")
        return
    except Exception as e:
        logger.error("Failed to init conflict_rules ChromaDB collection: %s", e)
        return

    # Check if already populated
    if _conflict_collection.count() >= len(rules) * 2:
        logger.info("Conflict rules collection already populated (%d docs)", _conflict_collection.count())
        return

    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for rule in rules:
        rule_id = rule["rule_id"]

        # English document
        doc_en = (
            f"Rule: {rule['rule_name'].replace('_', ' ')}. "
            f"WHO: {rule['who_guideline']} "
            f"AAP: {rule['aap_guideline']} "
            f"Description: {rule['description_en']}"
        )
        documents.append(doc_en)
        ids.append(f"{rule_id}_en")
        metadatas.append({
            "rule_id": rule_id,
            "rule_name": rule["rule_name"],
            "conflict_type": rule["conflict_type"],
            "severity_level": int(rule["severity_level"]),
            "age_safe_min_months": int(rule["age_safe_min_months"]),
            "age_safe_max_months": int(rule["age_safe_max_months"]) if rule["age_safe_max_months"] is not None else -1,
            "ingredient_flags": ",".join(rule.get("ingredient_flags", [])),
            "action": rule["action"],
            "language": "en",
            "source_type": "who_guideline",
        })

        # Arabic document
        doc_ar = (
            f"قاعدة: {rule['rule_name'].replace('_', ' ')}. "
            f"وصف: {rule['description_ar']}"
        )
        documents.append(doc_ar)
        ids.append(f"{rule_id}_ar")
        metadatas.append({
            "rule_id": rule_id,
            "rule_name": rule["rule_name"],
            "conflict_type": rule["conflict_type"],
            "severity_level": int(rule["severity_level"]),
            "age_safe_min_months": int(rule["age_safe_min_months"]),
            "age_safe_max_months": int(rule["age_safe_max_months"]) if rule["age_safe_max_months"] is not None else -1,
            "ingredient_flags": ",".join(rule.get("ingredient_flags", [])),
            "action": rule["action"],
            "language": "ar",
            "source_type": "aap_guideline",
        })

    try:
        _conflict_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Conflict rules indexed: %d documents (%d rules × EN+AR)", len(documents), len(rules))
    except Exception as e:
        logger.error("Failed to index conflict rules into ChromaDB: %s", e)


def query_conflict_rules(query_text: str, n_results: int = 3) -> list[dict]:
    """
    Query the conflict rules corpus for relevant rules.
    Returns list of (document, metadata, distance) dicts.
    """
    if _conflict_collection is None:
        logger.warning("Conflict collection not initialized — returning empty results")
        return []

    try:
        results = _conflict_collection.query(
            query_texts=[query_text],
            n_results=min(n_results, max(1, _conflict_collection.count())),
        )
    except Exception as e:
        logger.error("Conflict rules query failed: %s", e)
        return []

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        hits.append({"document": doc, "metadata": meta, "distance": float(dist)})

    return hits


def get_rule_by_id(rule_id: str) -> Optional[dict]:
    """Retrieve a specific rule from the loaded rules list."""
    for rule in _conflict_rules:
        if rule["rule_id"] == rule_id:
            return rule
    return None
