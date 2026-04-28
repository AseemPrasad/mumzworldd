# app/main.py
# Developer note: Application entry point. Register new routers here.
# To add middleware (auth, CORS origin control), extend the middleware block below.

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.data_loader import get_df
from app.api import health, products, issues, risks, coherence
from app.models.schemas import ApiResponse

# ── Logging ───────────────────────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Baby Product Safety Dashboard API",
    description="Analyzes synthetic baby product complaint data.",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles

app.include_router(health.router, tags=["Health"])
app.include_router(products.router, tags=["Products"])
app.include_router(issues.router, tags=["Issues"])
app.include_router(risks.router, tags=["Risks"])
app.include_router(coherence.router, prefix="/coherence", tags=["Coherence Engine"])

# Mount frontend static files so they can be accessed directly from the server
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

# Mount demo UI so /demo serves the coherence demo HTML
app.mount("/demo", StaticFiles(directory="frontend", html=True), name="demo")


# ── LLM endpoint ──────────────────────────────────────────────────────────────
from app.models.schemas import LLMRequest, LLMResponse
from app.core.llm_client import call_llm
from fastapi import status


@app.post("/llm/summary", response_model=ApiResponse[LLMResponse], tags=["LLM"])
async def llm_summary(body: LLMRequest):
    """Generate an LLM summary for selected product IDs via OpenRouter."""
    df = get_df()
    matched = df[df["product_id"].isin(body.product_ids)]

    if matched.empty:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"ok": False, "error": "No products matched the given product_ids"},
        )

    product_summaries = matched[[
        "product_id", "product_name", "issue_type", "severity", "return_reason"
    ]].to_dict(orient="records")

    result = call_llm(
        product_summaries=product_summaries,
        prompt_template=body.prompt_template or "",
    )

    if "error" in result and "not configured" in result["error"]:
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content={"ok": False, "error": result["error"]},
        )

    if "error" in result:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"ok": False, "error": result["error"]},
        )

    return ApiResponse(
        ok=True,
        data=LLMResponse(
            summary=result.get("summary"),
            model=result.get("model"),
            products_analyzed=len(matched),
        ),
        meta={"product_ids": body.product_ids},
    )


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "Internal server error"},
    )


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    logger.info("Loading data on startup…")
    get_df()

    logger.info("Initializing conflict rules loader…")
    from app.core.conflict_loader import init_conflict_loader
    init_conflict_loader()

    logger.info("Initializing RAG corpus…")
    from app.core.rag import init_rag
    init_rag()

    logger.info("Startup complete.")
