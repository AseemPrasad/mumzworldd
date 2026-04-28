# 🍼 Baby Product Safety Dashboard

A FastAPI backend that analyzes synthetic baby product complaint data, with a static frontend dashboard.

---

## Quick Start

```bash
# 1. Clone / unzip the project, then:
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Optional: set OPENROUTER_API_KEY=sk-... to enable /llm/summary

# 3. Start API
uvicorn app.main:app --reload --port 8000

# 4. Open dashboard (no build step needed)
open frontend/index.html           # or just open in browser
```

API docs auto-generated at: http://localhost:8000/docs

---

## Project Structure

```
baby-complaints/
├── app/
│   ├── main.py              # FastAPI app & LLM endpoint
│   ├── api/
│   │   ├── health.py        # GET /health
│   │   ├── products.py      # GET /products, /products/{id}
│   │   ├── issues.py        # GET /issues
│   │   └── risks.py         # GET /risks (cached)
│   ├── core/
│   │   ├── config.py        # Settings from .env
│   │   ├── data_loader.py   # CSV → DataFrame (singleton)
│   │   ├── risk.py          # Deterministic risk scoring
│   │   └── llm_client.py    # OpenRouter wrapper
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── data/
│       └── products_sample.csv
├── scripts/
│   └── generate_synthetic.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_risk.py
│   └── test_api.py
├── frontend/
│   └── index.html
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## API Reference

### GET /health
```bash
curl http://localhost:8000/health
```

### GET /products
```bash
curl "http://localhost:8000/products?category=Toys&risk_tag=high&limit=5"
curl "http://localhost:8000/products?sort_by=severity&min_age=0&max_age=6"
```

### GET /products/{product_id}
```bash
curl http://localhost:8000/products/p001
```

### GET /issues
```bash
curl "http://localhost:8000/issues?group_by=issue_type&top_n=5"
curl "http://localhost:8000/issues?group_by=product_category"
curl "http://localhost:8000/issues?group_by=baby_age_bucket"
```

### GET /risks
```bash
curl "http://localhost:8000/risks?threshold=high&top_n=5"
curl "http://localhost:8000/risks"
```

### POST /llm/summary
```bash
curl -X POST http://localhost:8000/llm/summary \
  -H "Content-Type: application/json" \
  -d '{"product_ids": ["p001","p002"], "prompt_template": "What are the main safety concerns?"}'
```
> Returns 501 if OPENROUTER_API_KEY is not set.

---

## Generate More Synthetic Data

```bash
python scripts/generate_synthetic.py --rows 500 --seed 99 --output app/data/products_sample.csv
```

---

## Docker

```bash
docker build -t baby-safety .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=sk-... baby-safety
```

---

## Risk Scoring Formula

```
norm_severity  = (severity - 1) / 9          # 1–10 → 0–1
norm_frequency = (frequency_score - 1) / 9   # 1–10 → 0–1
composite      = 0.6 × norm_severity + 0.4 × norm_frequency

composite ≥ 0.75 → high
composite ≥ 0.45 → medium
composite  < 0.45 → low
```

Risk tags are **recomputed on every startup** from the raw severity/frequency values.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `./app/data/products_sample.csv` | Path to complaint CSV |
| `OPENROUTER_API_KEY` | *(empty)* | Optional – enables /llm/summary |
| `OPENROUTER_API_URL` | `https://openrouter.ai/api/v1/...` | OpenRouter endpoint |
| `OPENROUTER_MODEL` | `openai/gpt-3.5-turbo` | Model slug |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Checklist of Files

- [x] `app/main.py`
- [x] `app/api/health.py`
- [x] `app/api/products.py`
- [x] `app/api/issues.py`
- [x] `app/api/risks.py`
- [x] `app/models/schemas.py`
- [x] `app/core/config.py`
- [x] `app/core/data_loader.py`
- [x] `app/core/risk.py`
- [x] `app/core/llm_client.py`
- [x] `app/data/products_sample.csv`
- [x] `scripts/generate_synthetic.py`
- [x] `tests/test_data_loader.py`
- [x] `tests/test_risk.py`
- [x] `tests/test_api.py`
- [x] `requirements.txt`
- [x] `Dockerfile`
- [x] `.env.example`
- [x] `frontend/index.html`
