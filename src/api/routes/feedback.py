#!/usr/bin/env python3
"""
src/api/feedback.py
Feedback API endpoints for collecting and retrieving user feedback
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import logging

from src.services.feedback_database import FeedbackDatabase, FeedbackRecord
from src.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackInput(BaseModel):
    session_id: str
    rating: Optional[int] = None  # 1-5
    comment: Optional[str] = None
    is_error: bool = False
    error_type: Optional[str] = None
    response_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackInput, request: Request):
    """Submit user feedback for a model response."""
    try:
        db = FeedbackDatabase()
        registry = ModelRegistry(registry_dir="data/model_registry")

        # Get latest model version
        latest_model = registry.get_latest_version()
        if not latest_model:
            raise HTTPException(status_code=500, detail="No active model found")

        feedback_record = FeedbackRecord(
            session_id=feedback.session_id,
            model_version=latest_model.version,
            rating=feedback.rating,
            comment=feedback.comment,
            is_error=feedback.is_error,
            error_type=feedback.error_type,
            response_time_ms=feedback.response_time_ms,
            tokens_used=feedback.tokens_used,
            user_id=getattr(request.state, "user_id", None),
            created_at=datetime.now().isoformat()
        )

        success = db.add_feedback(feedback_record)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save feedback")

        # Get feedback ID
        feedback_id = db.get_recent_feedback(days=1, limit=1)[0].id

        # Update model registry with new feedback stats
        registry.update_model_feedback(latest_model.version, db.get_feedback_stats())

        logger.info(f"Feedback submitted for session {feedback.session_id}")
        return FeedbackResponse(status="success", feedback_id=feedback_id)

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_feedback_stats(days: int = 14):
    """Get feedback statistics for specified time period."""
    try:
        db = FeedbackDatabase()
        stats = db.get_feedback_stats(days=days)
        return {
            "total_feedback": stats.total_feedback,
            "average_rating": stats.average_rating,
            "error_rate": stats.error_rate,
            "new_training_samples": stats.new_training_samples,
            "satisfaction_trend": stats.satisfaction_trend,
            "top_error_types": stats.top_error_types,
            "version_performance": stats.version_performance
        }
    except Exception as e:
        logger.error(f"Failed to retrieve stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))