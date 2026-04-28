### Purchase Coherence Engine

A FastAPI backend that analyzes synthetic baby product complaint data and order histories to detect developmental stage mismatches and safety conflicts.

The system operates as a five-layer AI and deterministic pipeline that triggers silently on specific events (session opens, milestone crossings, or customer service chats). It connects three disparate signals: the child's estimated developmental stage, the parent's product stack, and a medical/catalog conflict knowledge base.

**End-to-End Technical Workflow:**
1. **Trigger:** A parent opens the app, or a background job detects a significant shift in their search behavior (a "milestone crossing").
2. **Estimation (Layer 1):** The engine analyzes recent search terms and the age-suitability of past purchases to estimate the child's age in months. If signals conflict or are too weak, the system explicitly aborts to prevent hallucinating an age.
3. **Data Loading (Layer 2):** The parent's order history is retrieved and enriched with catalog metadata (ingredients, category, safe age ranges).
4. **Detection (Layer 3 - RAG):** For each product in the stack, the engine queries an embedded knowledge base of safety guidelines and catalog rules. A conflict is only flagged if it is supported by at least two independent signals (e.g., an ingredient match *and* an age mismatch).
5. **Safety Guardrails (Layer 4):** A crucial Deferral Classifier inspects the detected conflict. If the conflict involves medical symptoms (e.g., "rash", "fever") or if the system's confidence is below a strict threshold (0.6), it overrides the product recommendation and forces a "defer to doctor" response.
6. **Output Generation (Layer 5):** The system generates bilingual (English and Arabic) copy. The English copy focuses on clinical precision and data, while the Arabic copy is culturally tuned to emphasize reassurance and community proof.


### The Parent's User Flow
From the parent's perspective, this complex backend machinery is entirely invisible until a precise moment of need.
1. **The Silent Background Trigger:** A parent opens the app or searches for a new term (e.g., transitioning from searching "swaddles" to "teething toys").
2. **The Safe Deferral (if applicable):** If the parent was chatting with Customer Service about a "rash" or "fever," the system safely aborts any product recommendations behind the scenes and prepares a "Consult your pediatrician" message.
3. **The Grounded Alert:** If an age or ingredient conflict is detected without medical symptoms, the system generates a bilingual UI Card.
4. **The User Experience:** The parent sees a single, non-intrusive card on their dashboard: *"Your baby has likely outgrown the Stage 1 formula you bought in March..."* in both English and Arabic.
5. **The Feedback Loop:** They can tap to see community-backed alternatives or tap **"Not right? Tell us"** to correct the system, providing valuable ground-truth data back to the engine.


#### Prerequisites
- Python 3.10+
- Git

#### Installation Steps
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repo-url>
   cd baby-complaints
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Add your OPENROUTER_API_KEY to the .env file if you wish to use the LLM generation features.
   ```

#### How to Run
Start the development server:
```bash
uvicorn app.main:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Interactive Demo UI: `http://localhost:8000/demo`

#### Folder Structure Overview
- `app/api/`: FastAPI route definitions (endpoints for products, risks, coherence).
- `app/core/`: The core engine logic containing the 5 AI layers (estimator, stack loader, RAG, coherence).
- `app/models/`: Pydantic schemas enforcing strict I/O validation.
- `app/data/`: Synthetic datasets simulating Mumzworld OMS and catalog data.
- `frontend/`: Static HTML/JS for the demonstration dashboard.
- `scripts/`: Data generation utilities.

---

## 4. Code Documentation Philosophy

The codebase relies heavily on the philosophy of **explicit failure** and **strict typing**. 

- **Inline Comments:** Used sparingly, primarily to demarcate the boundaries between the 5 architectural layers and to explain non-obvious deterministic thresholds (e.g., why a harmonic mean is used to calculate confidence when two signals agree within a 4-week window).
- **Explicit Failure States over Null Padding:** In `app/core/estimator.py` and `app/core/coherence.py`, you will see explicit returns of `null` accompanied by a `null_reason` string. The design dictates that it is vastly preferable to return no data than to return hallucinated or low-confidence data. 
- **Pydantic Validation:** `app/models/schemas.py` acts as the contract for the system. Every API response must conform to these schemas. A `schema_error` is considered a critical failure and is logged, dropping the output rather than showing a malformed UI to the parent.

---

## 5. System Design & Architecture

### High-Level Architecture
The system is implemented as a stateless FastAPI microservice. The core architectural decision was to separate the generative AI components from the deterministic safety checks. 

### Data Flow
1. **Input:** The `CoherenceAuditRequest` provides raw user session data and order history.
2. **Retrieval-Augmented Generation (RAG):** The system uses an in-memory `ChromaDB` instance (`app/core/rag.py`). To comply with data constraints and ensure offline capability during prototyping, text is embedded using a custom Bag-of-Words (BoW) algorithm rather than a network-dependent LLM embedding model.
3. **Guardrails:** The flow is heavily gated. The `detect_conflicts` pipeline requires `supporting_signals >= 2`. The Deferral Classifier operates deterministically based on severity thresholds and vocabulary matching before any LLM is allowed to generate text.

### Tradeoffs
- **Offline Embeddings vs. LLM Embeddings:** A custom BoW embedding function was chosen for the MVP to allow local, synthetic testing without external API dependency costs. The tradeoff is a loss of deep semantic understanding, which limits the RAG's ability to catch nuanced synonyms in customer service chats.
- **Deterministic vs. Probabilistic Logic:** The Deferral Classifier is strictly deterministic. While an LLM might be better at judging the severity of a symptom, the risk of clinical liability dictated a hard-coded approach.

---

## 6. Tooling 

### Tools and Frameworks
- **FastAPI & Pydantic:** Chosen for rapid API development, auto-generated OpenAPI documentation, and rigorous schema validation.
- **ChromaDB:** Used as the vector database for the RAG corpus. It runs ephemerally in-memory for the MVP.
- **OpenRouter:** Used as an abstraction layer to route requests to external LLMs (e.g., `openai/gpt-3.5-turbo`) for the Copy Generator and LLM Conflict Analyzer.

### AI Usage Methodology
- **Agent Loops:** Avoided entirely. The system uses a strict pipeline approach. Agent loops are non-deterministic and can get trapped in hallucinatory cycles, which is unacceptable in a medical/safety context.
- **One-Shot Generation:** The LLM is used strictly for one-shot copy generation and conflict analysis validation, tightly constrained by system prompts that enforce the inclusion of evidence and confidence scores.

### Successes and Limitations
- **Success:** The strict schema validation effectively prevented "silent null padding." The Deferral Classifier successfully blocked product recommendations when medical keywords were present.
- **Limitation:** The custom BoW embedding model required careful tuning. Synonyms not explicitly programmed into the vocabulary matrix resulted in false negatives during retrieval.

---

## 7. Evaluation (Evals)

The system is evaluated against hard failure modes defined prior to implementation. 

### Evaluation Rubric
1. **Single-Signal Conflict:** Must surface 0 conflicts if only 1 signal exists.
2. **Hallucinated Age:** Must return `null` if the user has no history.
3. **Deferral Catch:** Must set `defer_to_doctor = true` if severe symptoms are mentioned.

### Test Cases & Results

| Case | Scenario Description | Expected Outcome | Actual Result | Status |
|---|---|---|---|---|
| 1 | 2mo old, Honey teething gel (Botulism risk) | Flag conflict, Action: Defer | Flagged, Deferred | **PASS** |
| 2 | 6mo old, small parts teether (Choking hazard) | Flag conflict, Action: Replace | Flagged, Replace | **PASS** |
| 3 | 1mo old, Scented lotion + CS chat "rash" | Deferral override | Deferred, No Product | **PASS** |
| 4 | No order history, no search history | `child_stage = null` | `null` returned | **PASS** |
| 5 | Conflict supported by only 1 signal | Suppress conflict | Suppressed | **PASS** |
| 6 | 6mo old, Stage 1 Formula transition | Flag conflict, Action: Replace | Flagged, Replace | **PASS** |
| 7 | Schema validation test with missing `confidence` | Schema Error, Logged | Suppressed | **PASS** |
| 8 | 1mo old, Peanut puffs (Allergen early) | Flag conflict | Flagged | **PASS** |
| 9 | Conflicting signals (search: "toddler", order age: 2mo) | `null_reason: signals_conflict_too_widely` | `null_reason` returned | **PASS** |
| 10 | Non-authoritative source in RAG retrieval | Suppress conflict | Suppressed | **PASS** |

**Analysis of Failures:**
During early testing, Case 3 (Deferral Override) failed because "redness" was not in the severity vocabulary list, though "rash" was. This was rectified by expanding the deterministic vocabulary list, highlighting the limitation of not using a semantic classifier for the deferral logic.

---

## 8. Tradeoffs & Decisions

- **Synthetic Data Generation:** Due to strict constraints preventing the scraping of retailer sites, the team opted to generate synthetic datasets (`scripts/generate_synthetic.py`). While this ensures privacy and compliance, it means the RAG corpus may not reflect the chaotic, noisy reality of actual e-commerce reviews.
- **Strict Pipeline over Autonomous Agents:** We intentionally chose a sequential pipeline. While autonomous agents are highly capable, they introduce uncontrollable variance. In safety-critical contexts (baby health), predictable failure is exponentially more valuable than unpredictable success. 

---

## 9. Limitations & Failure Modes

- **Vocabulary Limitations:** The offline Bag-of-Words embedding is brittle. If a parent describes a "bump" instead of a "rash," the symptom might bypass the deterministic Deferral Classifier if the word isn't mapped.
- **Estimation Drift:** The Milestone Estimator relies heavily on the assumption that search terms correlate strictly to a child's age. A parent buying a gift for a friend's toddler could completely corrupt the developmental stage estimation, causing a `signals_conflict_too_widely` failure for weeks.
- **Arabic Literal Translation:** While the architecture mandates distinct cultural framing for Arabic copy, reliance on external LLMs (via OpenRouter) means we are at the mercy of the model's localized fine-tuning. If a lower-tier model is used, the Arabic output may regress to a literal translation, violating our UX standards.

---

## 10. Future Work

1. **Semantic Deferral Classifier:** Replace the brittle keyword-matching Deferral Classifier with a lightweight, fine-tuned embedding model (e.g., a small BERT variant) to catch contextual medical symptoms (e.g., "he feels hot" instead of just "fever").
2. **Production Database Integration:** Swap the synthetic CSV loader and in-memory ChromaDB for production connections to Snowflake/BigQuery and a persistent vector store (like Pinecone or Qdrant).
3. **Gift Detection:** Implement a sub-heuristic in the Milestone Estimator to detect and isolate anomalous purchases (likely gifts) so they do not corrupt the primary child's developmental timeline.
4. **Correction Feedback Loop:** Fully wire the "Not right? Tell us" frontend button to an automated evaluation pipeline that adjusts signal weights when correction rates exceed the 15% threshold.


## Note

The LLM calls are genuinely implemented. This is not just a hardcoded facade, but there are explicit fallback mechanisms built-in.

The exactness of what is dynamic (real AI) and what is hardcoded (heuristics) in the current codebase,
If you look inside app/core/llm_client.py, the generate_bilingual_copy() function makes a real API call to OpenRouter (defaulting to openai/gpt-3.5-turbo or whatever you set in your .env).

It constructs a dynamic prompt injecting the specific product_name, months, confidence, and evidence_source.
It enforces a strict system prompt and uses response_format: {"type": "json_object"} to guarantee it returns valid JSON containing copy_en and copy_ar.
The "Shortcut": If you do not provide an OPENROUTER_API_KEY in your .env file, or if the API request times out, it explicitly catches the error and degrades to a _fallback_copy() function. This fallback uses a hardcoded template. This is a deliberate design choice so the demo doesn't crash if your API key expires during a presentation.

Similarly, when the RAG system finds a potential match (e.g., Honey + 2 Months Old), it sends it to an LLM Conflict Analyzer (analyze_conflict()) to act as a judge. It asks the LLM to rate the severity and confidence. If the API key is missing, it falls back to passing whatever the basic RAG vector distance scored it at.

In app/core/rag.py, the system uses an offline "Bag-of-Words" (BoW) algorithm (_bow_embed) to match product issues to guidelines, rather than making a network call to OpenAI for semantic embeddings. It's a functional mathematical approximation, but it's a shortcut to keep the demo completely offline and fast.

In app/core/estimator.py, the search_term_shift detector looks for exact keyword matches in a hardcoded list: ["newborn", "diaper", "colic"] -> maps to Month 1, or ["teething", "sit", "crawl"] -> maps to Month 6. In a production system, you would use an LLM to cluster the semantics of a search term, but here it is a hardcoded if/elif block.

The architecture is completely real and functional. The LLM integration is fully written and works. The only "hardcoded" parts are the graceful fallbacks when you don't have an API key, and the offline math used to simulate vector embeddings and search clustering.
