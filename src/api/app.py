#!/usr/bin/env python3
"""
src/api/app.py
Main FastAPI application for the MLOps platform
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
from fastapi import HTTPException

import logging

from src.api.middleware import RateLimitMiddleware

from src.api.routes.feedback import router as feedback_router
from src.api.routes.inference import router as inference_router
from src.orchestrator.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLOps Platform API", version="1.0.0")

# Initialize the Orchestrator
orchestrator_instance = Orchestrator()

@app.on_event("startup")
async def startup_event():
    app.state.orchestrator = orchestrator_instance
    logger.info("Orchestrator initialized and ready.")


# Simulated API keys (replace with proper key management in production)


# Add middleware

app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

# Include routers
app.include_router(feedback_router)
app.include_router(inference_router)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler."""
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)