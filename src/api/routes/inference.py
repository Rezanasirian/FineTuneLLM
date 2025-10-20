#!/usr/bin/env python3
"""
src/api/inference.py
Inference API endpoints for running model predictions
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
import time
from typing import Optional, List
from datetime import datetime



logger = logging.getLogger(__name__)

class QueryInput(BaseModel):
    user_query: str
    conversation_history: Optional[List[Dict[str, Any]]] = []
    current_system_state: Optional[Dict[str, Any]] = {}

class OrchestratorResponse(BaseModel):
    mdx_query: Optional[str] = None
    query_results: Optional[Dict[str, Any]] = None
    insights: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    recommendations: Optional[List[str]] = None
    chart_type: Optional[str] = None
    error: Optional[bool] = False
    error_type: Optional[str] = None
    message: Optional[str] = None
    resolution_details: Optional[Dict[str, Any]] = None


router = APIRouter(prefix="/inference", tags=["inference"])








# Simulated model inference (replace with actual model in production)



@router.post("/query", response_model=OrchestratorResponse)
async def process_query(input_data: QueryInput, request: Request):
    """Run model inference with given prompt."""
    try:
        start_time = time.time()
        orchestrator = request.app.state.orchestrator
        response = await orchestrator.process_user_request(
            user_query=input_data.user_query,
            conversation_history=input_data.conversation_history,
            current_system_state=input_data.current_system_state
        )
        return OrchestratorResponse(**response)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))