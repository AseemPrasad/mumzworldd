import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)
AUTHORITATIVE_SOURCES = {"who_guideline", "aap_guideline", "catalog_rule"}

# Single chromadb client in-memory
chroma_client = None
collection = None
_rag_vocab: dict[str, int] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _bow_embed(text: str, vocab: dict[str, int]) -> list[float]:
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


def init_rag():
    """
    Initialize the ChromaDB RAG collection from the synthetic CSV data.
    Uses an offline bag-of-words embedding — no network access required.
    """
    global chroma_client, collection, _rag_vocab
    logger.info("Initializing RAG ChromaDB collection…")

    from app.core.data_loader import get_df
    df = get_df()

    docs_raw: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for _, row in df.iterrows():
        brand = row.get("brand", "")
        product = row.get("product_name", "")
        issue = row.get("issue_type", "")
        reason = row.get("return_reason", "")
        cat = row.get("product_category", "")
        age = row.get("baby_age_months", "N/A")

        chunk = (
            f"Product: {product} ({brand}, Category: {cat}). "
            f"Age Suitable: {age} months. "
            f"Conflict/Issue: {issue}. "
            f"Details/Reason: {reason}."
        )
        docs_raw.append(chunk)
        ids.append(row["product_id"])
        metadatas.append({
            "product_name": product,
            "issue_type": issue,
            "severity": float(row.get("severity", 0)),
            "age_suitability": int(age) if age != "N/A" and not pd.isna(age) else 0,
            "source_type": "catalog_rule",
        })

    # Build offline vocabulary
    vocab: dict[str, int] = {}
    for doc in docs_raw:
        for token in _tokenize(doc):
            if token not in vocab:
                vocab[token] = len(vocab)
    _rag_vocab = vocab

    try:
        import chromadb
        from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

        class _WrappedEF(EmbeddingFunction):
            def __init__(self, v: dict[str, int]):
                self._v = v
            def __call__(self, input: Documents) -> Embeddings:
                return [_bow_embed(t, self._v) for t in input]

        chroma_client = chromadb.EphemeralClient()
        ef = _WrappedEF(_rag_vocab)
        collection = chroma_client.get_or_create_collection(
            name="baby_safety_corpus",
            embedding_function=ef,
        )
    except ImportError:
        logger.error("ChromaDB not installed, RAG disabled.")
        return
    except Exception as e:
        logger.error("Failed to init chromadb: %s", e)
        return

    if collection.count() >= len(df):
        logger.info("RAG collection already populated")
        return

    embeddings = [_bow_embed(doc, _rag_vocab) for doc in docs_raw]
    try:
        collection.add(
            documents=docs_raw,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("RAG collection populated with %d items.", len(docs_raw))
    except Exception as e:
        logger.error("Failed to populate RAG collection: %s", e)
        collection = None


def _query_conflict_rules(query_text: str) -> list[dict]:
    """
    Query the conflict_rules.json ChromaDB corpus (via conflict_loader).
    Returns list of hits with document, metadata, distance.
    """
    try:
        from app.core.conflict_loader import query_conflict_rules
        return query_conflict_rules(query_text, n_results=3)
    except Exception as e:
        logger.error("Conflict rules query failed: %s", e)
        return []


def _count_signals(item: dict, months: int, rule_hits: list[dict]) -> tuple[int, str]:
    """
    Count independent signals supporting a conflict.
    Returns (signal_count, conflict_type).
    Requires >= 2 independent signals to surface a conflict.
    """
    supporting_signals = 0
    conflict_type = ""

    # Signal 1: ingredient flag matches a rule's ingredient_flags
    item_flags = set(item.get("ingredient_flags", []))
    for hit in rule_hits:
        rule_flags_str = hit["metadata"].get("ingredient_flags", "")
        rule_flags = set(f.strip() for f in rule_flags_str.split(",") if f.strip())
        if item_flags & rule_flags:
            supporting_signals += 1
            conflict_type = "ingredient_age_safety"
            break

    # Signal 2: age mismatch against rule's safe age range
    for hit in rule_hits:
        age_safe_min = int(hit["metadata"].get("age_safe_min_months", 0))
        age_safe_max = int(hit["metadata"].get("age_safe_max_months", -1))
        if age_safe_min > 0 and months < age_safe_min:
            supporting_signals += 1
            if not conflict_type:
                conflict_type = hit["metadata"].get("conflict_type", "stage_mismatch")
            break
        if age_safe_max != -1 and months > age_safe_max:
            supporting_signals += 1
            if not conflict_type:
                conflict_type = "stage_mismatch"
            break

    # Signal 3: high severity from catalog (independent of rule match)
    if item.get("severity", 0) >= 7 and supporting_signals < 2:
        supporting_signals += 1
        if not conflict_type:
            conflict_type = item.get("issue_type", "ingredient_age_safety")

    return supporting_signals, conflict_type


def detect_conflicts(child_stage: dict, enriched_stack: list) -> list:
    """
    Layer 3: Conflict Detector (RAG + conflict_rules.json)

    Cross-reference stack against child_stage using:
      1. conflict_rules.json corpus (WHO/AAP guidelines)
      2. Synthetic CSV RAG corpus (catalog rules)

    Requirements:
      - >= 2 independent signals to surface a conflict
      - Retrieved evidence must be from an authoritative source
      - Integrates LLM Conflict Analyzer for validation
    """
    from app.core.llm_conflict_analyzer import analyze_conflict

    global collection
    # Lazy-initialize RAG collection if not already done
    if collection is None:
        init_rag()

    conflicts = []
    months = child_stage.get("months")
    if months is None:
        return conflicts

    for item in enriched_stack:
        if not item.get("metadata_complete", False):
            continue

        # Build query combining product metadata and child age
        ingredient_str = " ".join(item.get("ingredient_flags", []))
        query = (
            f"Safety conflict for {item['name']} "
            f"category {item.get('category', '')} "
            f"ingredients {ingredient_str} "
            f"for {months} month old baby."
        )

        # Query conflict_rules.json corpus (primary authoritative source)
        rule_hits = _query_conflict_rules(query)
        best_rule_hit = rule_hits[0] if rule_hits else None

        # Query CSV-based RAG corpus (secondary source)
        csv_hit = None
        if collection is not None:
            try:
                query_emb = _bow_embed(query, _rag_vocab)
                results = collection.query(query_embeddings=[query_emb], n_results=1)
                if results.get("documents") and results["documents"][0]:
                    csv_distance = results["distances"][0][0] if results.get("distances") else 1.0
                    csv_relevance = max(0.0, 1.0 - (csv_distance / 2.0))
                    if csv_relevance > 0.50:
                        csv_hit = {
                            "document": results["documents"][0][0],
                            "metadata": results["metadatas"][0][0],
                            "relevance": csv_relevance,
                        }
            except Exception as e:
                logger.error("CSV RAG query failed: %s", e)

        # Determine evidence source and relevance
        evidence_source = None
        best_relevance = 0.0

        if best_rule_hit:
            rule_distance = best_rule_hit.get("distance", 1.0)
            rule_relevance = max(0.0, 1.0 - (rule_distance / 2.0))
            source_type = best_rule_hit["metadata"].get("source_type", "who_guideline")
            if rule_relevance > 0.50 and source_type in AUTHORITATIVE_SOURCES:
                evidence_source = f"{source_type}: {best_rule_hit['document'][:80]}..."
                best_relevance = rule_relevance

        if csv_hit and (best_relevance < 0.60):
            csv_source = csv_hit["metadata"].get("source_type", "catalog_rule")
            if csv_source in AUTHORITATIVE_SOURCES:
                evidence_source = f"{csv_source}: {csv_hit['document'][:80]}..."
                best_relevance = csv_hit["relevance"]

        if evidence_source is None or best_relevance < 0.50:
            continue

        # Count independent signals
        supporting_signals, conflict_type = _count_signals(item, months, rule_hits)

        # Enforce >= 2 independent signals gate
        if supporting_signals < 2:
            logger.debug(
                "Suppressing conflict for %s: only %d signal(s)",
                item["name"], supporting_signals,
            )
            continue

        # LLM Conflict Analyzer validation
        llm_result = analyze_conflict(
            product_name=item["name"],
            conflict_type=conflict_type,
            child_age_months=months,
            ingredient_flags=item.get("ingredient_flags", []),
            evidence_text=evidence_source,
            signals_supporting=supporting_signals,
        )

        if not llm_result.get("is_conflict", False):
            logger.info(
                "LLM analyzer rejected conflict for '%s' (confidence=%.2f)",
                item["name"], llm_result.get("confidence", 0),
            )
            continue

        confidence = round(llm_result.get("confidence", best_relevance), 2)
        severity = llm_result.get("severity_level", item.get("severity", 5))

        # Determine action from best matching rule
        action = "replace_product"
        if best_rule_hit:
            action = best_rule_hit["metadata"].get("action", action)
        if severity >= 8 or item.get("severity", 0) >= 8:
            action = "flag_with_doctor_referral"

        # Build evidence attribution
        rule_id = best_rule_hit["metadata"].get("rule_id", "") if best_rule_hit else ""
        full_evidence = evidence_source
        if rule_id:
            full_evidence = f"{rule_id}: {evidence_source}"

        conflicts.append({
            "product_sku": item["sku"],
            "product_name": item["name"],
            "conflict_type": conflict_type,
            "signals_supporting": supporting_signals,
            "confidence": confidence,
            "evidence_source": full_evidence,
            "action": action,
            "llm_analyzed": llm_result.get("llm_analyzed", False),
        })

    return conflicts
