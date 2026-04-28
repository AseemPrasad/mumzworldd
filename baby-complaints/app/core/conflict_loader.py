# app/core/conflict_loader.py
# Loads conflict_rules.json into ChromaDB on startup.
# Indexes both EN and AR document content separately.
# Preserves source attribution (who_guideline, aap_guideline).

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_conflict_collection = None
_conflict_rules: list[dict] = []
_conflict_vocab: dict[str, int] = {}  # shared vocab between indexing and querying

CONFLICT_RULES_PATH = Path(__file__).parent.parent / "data" / "conflict_rules.json"


# ── Offline embedding function ────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\u0600-\u06ff]+", text.lower())


def _build_vocab(documents: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for doc in documents:
        for token in _tokenize(doc):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _bow_embed(text: str, vocab: dict[str, int]) -> list[float]:
    """Bag-of-words embedding — works offline with no model download."""
    import numpy as np
    vec = np.zeros(len(vocab), dtype=np.float32)
    for token in _tokenize(text):
        idx = vocab.get(token)
        if idx is not None:
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


class _OfflineEmbeddingFunction:
    """
    Bag-of-words embedding function for ChromaDB.
    Requires no internet access and no pre-downloaded models.
    """

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [_bow_embed(text, self._vocab) for text in input]


# ── Public API ─────────────────────────────────────────────────────────────────

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
    Uses an offline bag-of-words embedding function — no internet required.
    """
    global _conflict_collection

    rules = load_conflict_rules()
    if not rules:
        logger.warning("No conflict rules to index — skipping ChromaDB conflict loader init")
        return

    documents_en_ar = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for rule in rules:
        rule_id = rule["rule_id"]

        doc_en = (
            f"Rule: {rule['rule_name'].replace('_', ' ')}. "
            f"WHO: {rule['who_guideline']} "
            f"AAP: {rule['aap_guideline']} "
            f"Description: {rule['description_en']}"
        )
        doc_ar = (
            f"قاعدة: {rule['rule_name'].replace('_', ' ')}. "
            f"وصف: {rule['description_ar']}"
        )

        for lang, doc in (("en", doc_en), ("ar", doc_ar)):
            documents_en_ar.append(doc)
            ids.append(f"{rule_id}_{lang}")
            metadatas.append({
                "rule_id": rule_id,
                "rule_name": rule["rule_name"],
                "conflict_type": rule["conflict_type"],
                "severity_level": int(rule["severity_level"]),
                "age_safe_min_months": int(rule["age_safe_min_months"]),
                "age_safe_max_months": int(rule["age_safe_max_months"]) if rule["age_safe_max_months"] is not None else -1,
                "ingredient_flags": ",".join(rule.get("ingredient_flags", [])),
                "action": rule["action"],
                "language": lang,
                "source_type": "who_guideline" if lang == "en" else "aap_guideline",
            })

    # Build offline vocabulary from all documents
    vocab = _build_vocab(documents_en_ar)
    global _conflict_vocab
    _conflict_vocab = vocab

    try:
        import chromadb
        from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

        class _WrappedEF(EmbeddingFunction):
            def __init__(self, v):
                self._v = v
            def __call__(self, input: Documents) -> Embeddings:
                return [_bow_embed(t, self._v) for t in input]

        client = chromadb.EphemeralClient()
        ef = _WrappedEF(vocab)
        _conflict_collection = client.get_or_create_collection(
            name="conflict_rules_corpus",
            embedding_function=ef,
        )
    except Exception as e:
        logger.error("Failed to init conflict_rules ChromaDB collection: %s", e)
        return

    if _conflict_collection.count() >= len(rules) * 2:
        logger.info("Conflict rules collection already populated (%d docs)", _conflict_collection.count())
        return

    try:
        # Pre-compute embeddings
        embeddings = [_bow_embed(doc, vocab) for doc in documents_en_ar]
        _conflict_collection.add(
            documents=documents_en_ar,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(
            "Conflict rules indexed: %d documents (%d rules × EN+AR)",
            len(documents_en_ar), len(rules),
        )
    except Exception as e:
        logger.error("Failed to index conflict rules into ChromaDB: %s", e)
        _conflict_collection = None


def query_conflict_rules(query_text: str, n_results: int = 3) -> list[dict]:
    """
    Query the conflict rules corpus for relevant rules.
    Returns list of (document, metadata, distance) dicts.

    Uses keyword matching as the primary approach (reliable for a small corpus),
    with ChromaDB cosine similarity as an optional secondary enhancement.
    Lazy-initializes the ChromaDB collection on first call.
    """
    global _conflict_collection
    if _conflict_collection is None:
        init_conflict_loader()

    # Always run keyword matching — it is precise for a small rules corpus
    kw_results = _keyword_match_rules(query_text, n_results)
    if kw_results:
        return kw_results

    # Fall back to ChromaDB if keyword matching produced no hits
    if _conflict_collection is None:
        return []

    try:
        if not _conflict_vocab:
            return []
        query_emb = _bow_embed(query_text, _conflict_vocab)
        count = _conflict_collection.count()
        results = _conflict_collection.query(
            query_embeddings=[query_emb],
            n_results=min(n_results, max(1, count)),
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


def _keyword_match_rules(query_text: str, n_results: int) -> list[dict]:
    """
    Pure-keyword fallback when ChromaDB is unavailable.
    Scores rules by keyword overlap with query.
    """
    rules = load_conflict_rules()
    if not rules:
        return []

    query_tokens = set(_tokenize(query_text))
    scored = []
    for rule in rules:
        doc = (
            f"{rule.get('rule_name','')} {rule.get('description_en','')} "
            f"{' '.join(rule.get('ingredient_flags', []))}"
        )
        rule_tokens = set(_tokenize(doc))
        overlap = len(query_tokens & rule_tokens)
        if overlap > 0:
            scored.append((overlap, rule))

def _keyword_match_rules(query_text: str, n_results: int) -> list[dict]:
    """
    Pure-keyword fallback when ChromaDB is unavailable.
    Scores rules by keyword overlap with query.
    """
    rules = load_conflict_rules()
    if not rules:
        return []

    query_tokens = set(_tokenize(query_text))
    scored: list[tuple[int, dict]] = []
    for rule in rules:
        doc = (
            f"{rule.get('rule_name','')} {rule.get('description_en','')} "
            f"{' '.join(rule.get('ingredient_flags', []))}"
        )
        rule_tokens = set(_tokenize(doc))
        overlap = len(query_tokens & rule_tokens)
        if overlap > 0:
            scored.append((overlap, rule))

    scored.sort(key=lambda x: x[0], reverse=True)

    results_clean = []
    for i, (_, rule) in enumerate(scored[:n_results]):
        results_clean.append({
            "document": rule.get("description_en", ""),
            "metadata": {
                "rule_id": rule["rule_id"],
                "rule_name": rule["rule_name"],
                "conflict_type": rule["conflict_type"],
                "severity_level": int(rule["severity_level"]),
                "age_safe_min_months": int(rule["age_safe_min_months"]),
                "age_safe_max_months": int(rule["age_safe_max_months"]) if rule["age_safe_max_months"] is not None else -1,
                "ingredient_flags": ",".join(rule.get("ingredient_flags", [])),
                "action": rule["action"],
                "language": "en",
                "source_type": "who_guideline",
            },
            "distance": 0.1 * (i + 1),  # ascending distance for top-ranked hits
        })
    return results_clean


def get_rule_by_id(rule_id: str) -> Optional[dict]:
    """Retrieve a specific rule from the loaded rules list."""
    for rule in _conflict_rules:
        if rule["rule_id"] == rule_id:
            return rule
    return None

