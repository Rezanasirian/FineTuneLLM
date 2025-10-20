#!/usr/bin/env python3
"""
src/services/feedback_database.py
Feedback Database for Multi-Agent LLM Platform

Handles:
- Storing user feedback (ratings, comments, errors)
- Generating training samples from feedback
- Retrieving statistics for retraining decisions
- Integration with ModelRegistry
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Data class for feedback records."""
    id: Optional[int]
    session_id: str
    model_version: str
    rating: Optional[int] = None  # 1-5 stars
    comment: Optional[str] = None
    is_error: bool = False
    error_type: Optional[str] = None
    response_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    user_id: Optional[str] = None
    created_at: str = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackStats:
    """Data class for feedback statistics."""
    total_feedback: int
    average_rating: float
    error_rate: float
    new_training_samples: int
    satisfaction_trend: float  # 7-day vs 14-day avg rating difference
    top_error_types: List[Tuple[str, int]]
    version_performance: Dict[str, Dict[str, float]]


class FeedbackDatabase:
    """SQLite-based feedback database."""

    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"FeedbackDatabase initialized at {self.db_path}")

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Feedback table
        cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                rating INTEGER,
                comment TEXT,
                is_error BOOLEAN DEFAULT 0,
                error_type TEXT,
                response_time_ms INTEGER,
                tokens_used INTEGER,
                user_id TEXT,
                created_at TEXT NOT NULL
            )
                       """)

        # Training samples table (generated from feedback)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_samples (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            feedback_id INTEGER,
                            prompt TEXT NOT NULL,
                            response TEXT NOT NULL,
                            expected_response TEXT,
                            difficulty_level INTEGER,
                            sample_type TEXT,  -- 'error_correction', 'improvement', 'positive'
                            created_at TEXT NOT NULL,
                            FOREIGN KEY (feedback_id) REFERENCES feedback (id)
                        )
                       """)

        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_version ON feedback(model_version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_date ON feedback(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")

        conn.commit()
        conn.close()

    def add_feedback(self, feedback: FeedbackRecord) -> bool:
        """Add new feedback record."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                           INSERT INTO feedback
                           (session_id, model_version, rating, comment, is_error, error_type,
                            response_time_ms, tokens_used, user_id, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """, (
                               feedback.session_id, feedback.model_version, feedback.rating,
                               feedback.comment, feedback.is_error, feedback.error_type,
                               feedback.response_time_ms, feedback.tokens_used, feedback.user_id,
                               feedback.created_at
                           ))

            feedback_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.info(f"Added feedback #{feedback_id} for {feedback.model_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False

    def add_training_sample(self, feedback_id: int, prompt: str, response: str,
                            expected_response: Optional[str] = None,
                            sample_type: str = "improvement") -> bool:
        """Generate and store training sample from feedback."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get feedback details
            cursor.execute("SELECT comment FROM feedback WHERE id = ?", (feedback_id,))
            result = cursor.fetchone()
            if not result:
                return False

            difficulty = self._calculate_difficulty(feedback_id)

            cursor.execute("""
                           INSERT INTO training_samples
                           (feedback_id, prompt, response, expected_response, difficulty_level, sample_type, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           """, (
                               feedback_id, prompt, response, expected_response,
                               difficulty, sample_type, datetime.now().isoformat()
                           ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to add training sample: {e}")
            return False

    def _calculate_difficulty(self, feedback_id: int) -> int:
        """Calculate sample difficulty based on feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT rating, is_error, tokens_used
                       FROM feedback
                       WHERE id = ?
                       """, (feedback_id,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            return 3  # Default difficulty

        rating, is_error, tokens = result

        # Simple heuristic: low rating + error + high tokens = high difficulty
        difficulty = 3
        if rating and rating < 3:
            difficulty += 1
        if is_error:
            difficulty += 2
        if tokens and tokens > 1000:
            difficulty += 1

        return min(difficulty, 5)

    def get_feedback_stats(self, days: int = 14) -> FeedbackStats:
        """Get comprehensive feedback statistics."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)

        # Total feedback
        df = pd.read_sql_query(
            f"SELECT * FROM feedback WHERE created_at >= '{cutoff_date}'",
            conn
        )

        total_feedback = len(df)

        # Average rating
        avg_rating = df['rating'].mean() if 'rating' in df.columns and not df['rating'].isna().all() else 0.0

        # Error rate
        error_rate = (df['is_error'].sum() / total_feedback * 100) if total_feedback > 0 else 0.0

        # New training samples
        training_df = pd.read_sql_query(
            f"SELECT * FROM training_samples WHERE created_at >= '{cutoff_date}'",
            conn
        )
        new_samples = len(training_df)

        # Satisfaction trend (7-day vs 14-day)
        cutoff_7d = (datetime.now() - timedelta(days=7)).isoformat()
        df_7d = df[df['created_at'] >= cutoff_7d]
        avg_rating_7d = df_7d['rating'].mean() if len(df_7d) > 0 else avg_rating
        satisfaction_trend = (avg_rating_7d - avg_rating) * 100  # percentage points

        # Top error types
        error_df = df[df['is_error'] == 1]
        top_errors = error_df['error_type'].value_counts().head(5).to_dict()
        top_error_types = [(k, v) for k, v in top_errors.items()]

        # Version performance
        version_perf = df.groupby('model_version').agg({
            'rating': 'mean',
            'is_error': 'mean',
            'response_time_ms': 'mean'
        }).round(2).to_dict('index')

        conn.close()

        return FeedbackStats(
            total_feedback=total_feedback,
            average_rating=round(avg_rating, 2),
            error_rate=round(error_rate, 2),
            new_training_samples=new_samples,
            satisfaction_trend=round(satisfaction_trend, 2),
            top_error_types=top_error_types,
            version_performance=version_perf
        )

    def get_feedback_by_version(self, version: str, limit: int = 100) -> List[FeedbackRecord]:
        """Get feedback for specific model version."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM feedback WHERE model_version = ? ORDER BY created_at DESC LIMIT ?",
            conn, params=(version, limit)
        )
        conn.close()

        return [FeedbackRecord(**row) for _, row in df.iterrows()]

    def get_recent_feedback(self, days: int = 7, limit: int = 50) -> List[FeedbackRecord]:
        """Get most recent feedback."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM feedback WHERE created_at >= '{cutoff_date}' ORDER BY created_at DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()

        return [FeedbackRecord(**row) for _, row in df.iterrows()]

    def generate_training_data(self, min_rating_threshold: int = 3,
                               include_errors: bool = True) -> List[Dict[str, Any]]:
        """Generate training data from feedback."""
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT f.*, ts.prompt, ts.response, ts.expected_response
                FROM feedback f
                         LEFT JOIN training_samples ts ON f.id = ts.feedback_id
                WHERE (f.rating <= ? OR f.is_error = ?)
                ORDER BY f.created_at DESC \
                """

        df = pd.read_sql_query(
            query, conn, params=(min_rating_threshold, include_errors)
        )
        conn.close()

        training_data = []
        for _, row in df.iterrows():
            if pd.notna(row['prompt']) and pd.notna(row['response']):
                sample = {
                    "prompt": row['prompt'],
                    "response": row['response'],
                    "expected_response": row['expected_response'],
                    "rating": row['rating'],
                    "is_error": row['is_error'],
                    "difficulty": row['difficulty_level'],
                    "session_id": row['session_id'],
                    "created_at": row['created_at']
                }
                training_data.append(sample)

        logger.info(f"Generated {len(training_data)} training samples")
        return training_data

    def get_retraining_feedback(self, version: str) -> List[Dict[str, Any]]:
        """Get feedback suitable for retraining current version."""
        stats = self.get_feedback_stats(days=14)
        feedback_list = self.get_feedback_by_version(version, limit=200)

        retraining_feedback = []
        for fb in feedback_list:
            if (fb.rating and fb.rating <= 3) or fb.is_error:
                retraining_feedback.append({
                    "session_id": fb.session_id,
                    "rating": fb.rating,
                    "comment": fb.comment,
                    "is_error": fb.is_error,
                    "error_type": fb.error_type,
                    "created_at": fb.created_at
                })

        return retraining_feedback

    def clear_old_data(self, days: int = 90):
        """Clean up old feedback data."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"DELETE FROM feedback WHERE created_at < '{cutoff_date}'")
        cursor.execute(f"DELETE FROM training_samples WHERE created_at < '{cutoff_date}'")

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Deleted {deleted} old feedback records")
        return deleted

    def backup(self, backup_path: str) -> bool:
        """Create database backup."""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False


# Example usage and testing
# if __name__ == "__main__":
#     db = FeedbackDatabase()
#
#     # Add sample feedback
#     sample_feedback = FeedbackRecord(
#         id= 1,
#         session_id="test-001",
#         model_version="v1.0.0",
#         rating=4,
#         comment="Good response, but could be more concise",
#         is_error=False,
#         response_time_ms=250,
#         tokens_used=150,
#         created_at=datetime.now().isoformat()
#     )
#
#     db.add_feedback(sample_feedback)
#
#     # Add training sample
#     db.add_training_sample(
#         feedback_id=1,
#         prompt="What is machine learning?",
#         response="Machine learning is a subset of AI...",
#         sample_type="improvement"
#     )
#
#     # Get stats
#     stats = db.get_feedback_stats(days=30)
#     print(f"Stats: {stats.total_feedback} feedbacks, avg rating: {stats.average_rating}")