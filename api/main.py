"""
api/main.py
──────────────────────────────────────────────────────────────────────────────
FastAPI application that exposes:
  POST /api/v1/score           — Score a single customer
  GET  /api/v1/customers       — List customers with risk data
  GET  /api/v1/customers/{id}  — Customer detail with full behavioral profile
  GET  /api/v1/portfolio       — Portfolio-level risk metrics
  POST /api/v1/auth/login      — JWT authentication
  GET  /health                 — Health check
──────────────────────────────────────────────────────────────────────────────
"""
from contextlib import asynccontextmanager
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from api.routers import auth, customers, portfolio, scoring
from config.logging_config import setup_logging
from config.settings import get_settings
 
setup_logging()
settings = get_settings()
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load scoring models into memory
    from serving.bentoml_service.pulse_scorer import get_scorer
    get_scorer()
    yield
    # Shutdown: cleanup if needed
 
 
app = FastAPI(
    title="Sentinel API",
    description="Pre-Delinquency Intervention Engine — Real-Time Risk Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)
 
# CORS — allow React dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Register all routers
app.include_router(auth.router,      prefix="/api/v1/auth",      tags=["Authentication"])
app.include_router(scoring.router,   prefix="/api/v1",            tags=["Scoring"])
app.include_router(customers.router, prefix="/api/v1/customers",  tags=["Customers"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio",  tags=["Portfolio"])
 
 
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "sentinel-api", "version": "1.0.0"}
