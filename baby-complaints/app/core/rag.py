import logging
import pandas as pd

logger = logging.getLogger(__name__)
AUTHORITATIVE_SOURCES = {"who_guideline", "aap_guideline", "catalog_rule"}

# Single chromadb client in-memory
chroma_client = None
collection = None

def init_rag():
    """
    Initialize the ChromaDB RAG collection from the synthetic CSV data.
    """
    global chroma_client, collection
    logger.info("Initializing RAG ChromaDB collection...")
    
    try:
        import chromadb
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="baby_safety_corpus")
    except ImportError:
        logger.error("ChromaDB not installed, RAG will mock.")
        return
    except Exception as e:
        logger.error("Failed to init chromadb: %s", e)
        return
        
    from app.core.data_loader import get_df
    df = get_df()
    
    # Check if we already populated it
    if collection.count() >= len(df):
        logger.info("RAG collection already populated")
        return
        
    documents = []
    ids = []
    metadatas = []
    
    for idx, row in df.iterrows():
        # Build the RAG chunk text
        brand = row.get("brand", "")
        product = row.get("product_name", "")
        issue = row.get("issue_type", "")
        reason = row.get("return_reason", "")
        cat = row.get("product_category", "")
        age = row.get("baby_age_months", "N/A")
        
        chunk = f"Product: {product} ({brand}, Category: {cat}). Age Suitable: {age} months. Conflict/Issue: {issue}. Details/Reason: {reason}."
        documents.append(chunk)
        ids.append(row["product_id"])
        
        metadatas.append({
            "product_name": product,
            "issue_type": issue,
            "severity": float(row.get("severity", 0)),
            "age_suitability": int(age) if age != "N/A" and not pd.isna(age) else 0,
            "source_type": "catalog_rule",
        })
        
    if documents:
        # Add to chroma DB
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("RAG collection populated with %d items.", len(documents))

def detect_conflicts(child_stage: dict, enriched_stack: list) -> list:
    """
    Layer 3: Conflict Detector
    Cross-reference stack against child_stage using RAG corpus.
    Must have >= 2 signals to surface a conflict.
    """
    global collection
    conflicts = []
    months = child_stage.get("months")
    if months is None or collection is None:
        return conflicts
        
    for item in enriched_stack:
        if not item.get("metadata_complete", False):
            continue
            
        # Retrieval query
        query = f"Safety issue or conflict for {item['name']} ({item['category']}) regarding {months} month old baby."
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=1,
            )
        except Exception as e:
            logger.error("RAG Query Failed: %s", e)
            continue
            
        if not results.get("documents") or not results["documents"][0]:
            continue
            
        best_distance = results["distances"][0][0] if "distances" in results and results["distances"] else 1.0
        best_meta = results["metadatas"][0][0]
        
        # Simulate relevance 0.0 - 1.0. All-MiniLM L2 roughly < 1.0 is decent
        relevance = max(0, 1.0 - (best_distance / 2.0))
        
        source_type = str(best_meta.get("source_type", "unknown"))
        if relevance > 0.75 and source_type in AUTHORITATIVE_SOURCES:
            db_age = best_meta.get("age_suitability", 0)
            age_diff = abs(db_age - months)
            
            supporting_signals = 1
            conflict_type = ""
            
            # Second signal check
            if age_diff > 3:
                supporting_signals += 1
                conflict_type = "stage_mismatch"
            elif item.get("severity", 0) >= 6:
                supporting_signals += 1
                conflict_type = "ingredient_age_safety"
            elif item.get("ingredient_flags"):
                supporting_signals += 1
                conflict_type = "ingredient_age_safety"
                
            if supporting_signals >= 2:
                conflicts.append({
                    "product_sku": item["sku"],
                    "product_name": item["name"],
                    "conflict_type": conflict_type,
                    "signals_supporting": supporting_signals,
                    "confidence": round(0.80 + (relevance * 0.15), 2),
                    "evidence_source": f"{source_type}: {results['documents'][0][0][:60]}...",
                    "action": "flag_with_doctor_referral" if item.get("severity", 0) >= 8 else "replace_product"
                })
                
    return conflicts
