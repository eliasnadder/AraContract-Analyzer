"""
Main FastAPI application for AraContract Analyzer.

This module sets up the FastAPI app, includes all routers,
configures middleware, and handles startup/shutdown events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.exceptions import setup_exception_handlers
from app.routers import upload, segment, classify, summarize, compare, analyze

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info("Starting AraContract Analyzer API...")

    # Initialize resources here if needed
    # e.g., load models, connect to databases, etc.

    yield

    logger.info("Shutting down AraContract Analyzer API...")
    # Cleanup resources here


# Create FastAPI app instance
app = FastAPI(
    title="AraContract Analyzer API",
    description="AI-powered Arabic contract analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(upload.router, prefix="/api/contract", tags=["upload"])
app.include_router(segment.router, prefix="/api/contract", tags=["segmentation"])
app.include_router(classify.router, prefix="/api/contract", tags=["classification"])
app.include_router(summarize.router, prefix="/api/contract", tags=["summarization"])
app.include_router(analyze.router, prefix="/api/contract", tags=["analysis"])
app.include_router(compare.router, prefix="/api/contract", tags=["comparison"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "AraContract Analyzer API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}