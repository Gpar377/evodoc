"""
main.py — FastAPI entry point for EvoDoc Clinical Drug Safety Engine.

Endpoints:
  POST /check   — Drug safety assessment
  GET  /health  — Health check
  GET  /cache/stats — Cache diagnostics
  DELETE /cache — Clear cache (dev/admin)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cache import drug_safety_cache
from engine import run_safety_check
from models import SafetyCheckRequest, SafetyCheckResponse

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EvoDoc Drug Safety Engine starting…")
    yield
    logger.info("EvoDoc shutting down.")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="EvoDoc Clinical Drug Safety Engine",
    description=(
        "Medical-grade drug interaction checker for Indian clinics. "
        "Powered by Meditron LLM with rule-based fallback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to clinic IPs in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please contact EvoDoc support."},
    )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.post(
    "/check",
    response_model=SafetyCheckResponse,
    summary="Run drug safety assessment",
    description=(
        "Accept a list of proposed medicines and patient history. "
        "Returns structured safety assessment with interactions, allergy alerts, "
        "contraindication flags, and a 0–100 patient risk score."
    ),
    responses={
        200: {"description": "Safety assessment returned"},
        400: {"description": "Invalid input"},
        422: {"description": "Validation error"},
    },
)
async def check_drug_safety(request: SafetyCheckRequest) -> SafetyCheckResponse:
    start = time.monotonic()

    # ── Cache lookup ──────────────────────────────────────────
    cache_key = drug_safety_cache.build_key(
        request.proposed_medicines,
        request.patient_history.current_medications,
    )
    cached = await drug_safety_cache.get(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        logger.info("Cache HIT for key %s…", cache_key[:12])
        return SafetyCheckResponse(**cached)

    # ── Run safety engine ─────────────────────────────────────
    logger.info(
        "Cache MISS — running safety check for %d medicines",
        len(request.proposed_medicines),
    )
    try:
        result = await run_safety_check(
            proposed_medicines=request.proposed_medicines,
            patient_history=request.patient_history,
        )
    except Exception as exc:
        logger.error("Safety engine failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Safety engine encountered an unexpected error.",
        )

    # ── Cache the result ──────────────────────────────────────
    await drug_safety_cache.set(cache_key, result)

    response = SafetyCheckResponse(**result)
    elapsed = int((time.monotonic() - start) * 1000)
    logger.info(
        "Safety check completed in %d ms | risk=%s | safe=%s | source=%s",
        elapsed,
        response.overall_risk_level,
        response.safe_to_prescribe,
        response.source,
    )
    return response


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "service": "EvoDoc Drug Safety Engine",
        "version": "1.0.0",
    }


@app.get("/cache/stats", summary="Cache diagnostics")
async def cache_stats():
    return drug_safety_cache.stats


@app.delete("/cache", summary="Clear cache (admin/dev only)")
async def clear_cache():
    await drug_safety_cache.clear()
    return {"message": "Cache cleared"}
