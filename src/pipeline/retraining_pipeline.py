#!/usr/bin/env python3
"""
src/pipeline/retraining_pipeline.py
Automated Retraining Pipeline for Multi-Agent LLM Platform

Handles:
1. Intelligent retraining decisions based on feedback
2. Data preparation from feedback database
3. Model fine-tuning with new samples
4. Evaluation and comparison
5. Automated deployment of improved versions
6. Rollback capability if performance degrades
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import get_config
from src.services.model_registry import ModelRegistry
from src.services.feedback_service import FeedbackDatabase, FeedbackStats
from src.pipeline.training_pipeline import TrainingPipeline
from src.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


@dataclass
class RetrainingDecision:
    """Data class for retraining decisions."""
    should_retrain: bool
    reasons: List[str]
    urgency: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # 0.0 to 1.0
    stats: FeedbackStats
    predicted_improvement: Optional[float] = None


@dataclass
class RetrainingResult:
    """Data class for retraining results."""
    status: str  # 'success', 'failed', 'skipped', 'rollback'
    new_version: Optional[str] = None
    duration_hours: float = 0.0
    performance_improvement: Optional[float] = None
    old_version: str = ""
    metrics_comparison: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class RetrainingPipeline:
    """Automated retraining pipeline with intelligent decision-making."""

    def __init__(self, config_dir: str = "config"):
        self.config = get_config(config_dir)
        self.registry = ModelRegistry(registry_dir=self.config.model_registry)
        self.feedback_db = FeedbackDatabase()
        self.evaluator = ModelEvaluator()
        self.training_pipeline = TrainingPipeline()

        # Thresholds for retraining decisions
        self.decision_thresholds = {
            'min_feedback_count': 50,
            'error_rate_threshold': 5.0,
            'rating_drop_threshold': 0.5,
            'satisfaction_trend_threshold': -10.0,
            'days_since_training': 14,
            'min_improvement_required': 2.0  # percentage points
        }

        logger.info("RetrainingPipeline initialized")

    def should_retrain(self) -> RetrainingDecision:
        """
        Intelligent decision engine for retraining.
        Returns comprehensive decision with reasons and confidence score.
        """
        logger.info("Evaluating retraining need...")

        # Get current model and stats
        current_model = self.registry.get_latest_version()
        if not current_model:
            return RetrainingDecision(
                should_retrain=True,
                reasons=["No model deployed"],
                urgency="critical",
                confidence=1.0,
                stats=FeedbackStats(0, 0.0, 0.0, 0, 0.0, [], {}),
                predicted_improvement=100.0
            )

        stats = self.feedback_db.get_feedback_stats(days=14)
        created = datetime.fromisoformat(current_model.created_at)
        days_since = (datetime.now() - created).days

        reasons = []
        urgency_score = 0
        confidence = 0.5

        # Reason 1: Minimum feedback threshold
        if stats.total_feedback >= self.decision_thresholds['min_feedback_count']:
            reasons.append(f"Sufficient feedback collected ({stats.total_feedback})")
            confidence += 0.2

        # Reason 2: High error rate
        if stats.error_rate > self.decision_thresholds['error_rate_threshold']:
            reasons.append(f"High error rate: {stats.error_rate}%")
            urgency_score += 30
            confidence += 0.3

        # Reason 3: Rating drop
        if stats.average_rating < 4.0:
            reasons.append(f"Low average rating: {stats.average_rating}/5")
            urgency_score += 25
            confidence += 0.25

        # Reason 4: Negative satisfaction trend
        if stats.satisfaction_trend < self.decision_thresholds['satisfaction_trend_threshold']:
            reasons.append(f"Declining satisfaction: {stats.satisfaction_trend}pp")
            urgency_score += 20
            confidence += 0.2

        # Reason 5: Time-based (scheduled)
        if days_since >= self.decision_thresholds['days_since_training']:
            reasons.append(f"Scheduled retraining ({days_since} days)")
            confidence += 0.1

        # Reason 6: Many new training samples
        if stats.new_training_samples > 100:
            reasons.append(f"Many new samples available ({stats.new_training_samples})")
            confidence += 0.15

        # Decision logic
        should_retrain = len(reasons) >= 2 or urgency_score >= 40

        # Urgency mapping
        if urgency_score >= 60:
            urgency = "critical"
        elif urgency_score >= 40:
            urgency = "high"
        elif urgency_score >= 20:
            urgency = "medium"
        else:
            urgency = "low"

        # Predict improvement based on error rate and samples
        predicted_improvement = min(stats.error_rate * 0.8, 25.0) if should_retrain else 0.0

        decision = RetrainingDecision(
            should_retrain=should_retrain,
            reasons=reasons,
            urgency=urgency,
            confidence=min(confidence, 1.0),
            stats=stats,
            predicted_improvement=predicted_improvement
        )

        logger.info(f"Retraining decision: {should_retrain} (confidence: {decision.confidence:.2f})")
        return decision

    def run_retraining(self) -> RetrainingResult:
        """
        Execute complete retraining pipeline.
        Returns detailed result with metrics comparison.
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("AUTOMATED RETRAINING PIPELINE")
        logger.info("=" * 80)

        try:
            # Step 1: Decision
            decision = self.should_retrain()
            logger.info(f"\n[STEP 1/6] Decision: {'RETRAIN' if decision.should_retrain else 'SKIP'}")
            logger.info(f"Urgency: {decision.urgency} | Confidence: {decision.confidence:.1%}")

            if not decision.should_retrain:
                logger.info("Skipping retraining - system performing well")
                return RetrainingResult(
                    status="skipped",
                    new_version=None,
                    duration_hours=0.0
                )

            # Step 2: Backup current model
            current_version = self.registry.get_latest_version()
            old_version = current_version.version if current_version else "none"

            logger.info(f"\n[STEP 2/6] Backing up current model v{old_version}")
            self.registry.create_backup(old_version)

            # Step 3: Prepare training data
            logger.info("\n[STEP 3/6] Preparing training data...")
            training_data = self._prepare_training_data(
                min_samples=200,
                include_current_version=True
            )

            if len(training_data) < 50:
                logger.warning("Insufficient training data, skipping retraining")
                return RetrainingResult(status="skipped", new_version=None)

            logger.info(f"Prepared {len(training_data)} training samples")

            # Step 4: Train new model
            logger.info("\n[STEP 4/6] Training new model...")
            train_result = self.training_pipeline.run(
                training_data=training_data,
                use_existing_data=False,
                skip_evaluation=False,
                register_model=True,
                base_version=old_version
            )

            if train_result["status"] != "SUCCESS":
                logger.error("Training failed!")
                return RetrainingResult(
                    status="failed",
                    new_version=None,
                    error_message=train_result.get("error")
                )

            new_version = train_result["model_version"]
            logger.info(f"✓ New model trained: v{new_version}")

            # Step 5: Evaluate and compare
            logger.info("\n[STEP 5/6] Evaluating new model...")
            comparison = self._compare_models(old_version, new_version)

            # Step 6: Deploy or rollback
            logger.info("\n[STEP 6/6] Deploying new model...")
            deploy_result = self._deploy_if_improved(
                old_version, new_version, comparison
            )

            duration = (time.time() - start_time) / 3600  # hours

            result = RetrainingResult(
                status=deploy_result["status"],
                new_version=new_version if deploy_result["deployed"] else None,
                duration_hours=duration,
                performance_improvement=comparison.get("improvement_percent"),
                old_version=old_version,
                metrics_comparison=comparison
            )

            self._log_result(result)
            return result

        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}", exc_info=True)
            duration = (time.time() - start_time) / 3600
            return RetrainingResult(
                status="failed",
                new_version=None,
                duration_hours=duration,
                error_message=str(e)
            )

    def _prepare_training_data(self, min_samples: int = 200,
                               include_current_version: bool = True) -> List[Dict[str, Any]]:
        """Prepare high-quality training data from feedback."""
        logger.info("Generating training data from feedback...")

        # Get retraining feedback
        current_version = self.registry.get_latest_version()
        if current_version:
            feedback_data = self.feedback_db.get_retraining_feedback(current_version.version)
        else:
            feedback_data = []

        # Generate comprehensive training samples
        training_data = []

        for feedback in feedback_data[:min_samples]:
            # Create diverse sample types
            sample_types = ["error_correction", "improvement", "augmentation"]

            for sample_type in sample_types[:2]:  # Limit variety
                sample = self._create_training_sample(feedback, sample_type)
                if sample:
                    training_data.append(sample)

        # Add synthetic data augmentation if needed
        if len(training_data) < min_samples:
            logger.info(f"Augmenting data: {min_samples - len(training_data)} samples needed")
            training_data.extend(self._augment_data(training_data,
                                                    min_samples - len(training_data)))

        # Shuffle and limit
        np.random.shuffle(training_data)
        return training_data[:min_samples]

    def _create_training_sample(self, feedback: Dict[str, Any],
                                sample_type: str) -> Optional[Dict[str, Any]]:
        """Create specific training sample based on feedback type."""
        # Simulate prompt/response generation based on feedback
        # In production, this would extract from actual session logs

        base_prompts = [
            "Explain {topic} in simple terms",
            "What are the benefits of {topic}?",
            "How does {topic} work?",
            "Give me examples of {topic}"
        ]

        topics = ["machine learning", "neural networks", "transformers",
                  "fine-tuning", "MLOps", "LLM deployment"]

        prompt = np.random.choice(base_prompts).format(
            topic=np.random.choice(topics)
        )

        # Generate response based on sample type
        if sample_type == "error_correction" and feedback.get("is_error"):
            response = "This is a corrected response based on user feedback."
            expected = f"Improved response for '{feedback.get('comment', '')}'"
        elif sample_type == "improvement" and feedback.get("rating", 5) < 4:
            response = "Enhanced response with better structure and detail."
            expected = f"Improved version addressing '{feedback.get('comment', '')}'"
        else:
            response = "High-quality baseline response."
            expected = None

        return {
            "prompt": prompt,
            "response": response,
            "expected_response": expected,
            "rating": feedback.get("rating", 5),
            "is_error": feedback.get("is_error", False),
            "difficulty": 3,
            "sample_type": sample_type,
            "session_id": feedback["session_id"],
            "created_at": feedback["created_at"]
        }

    def _augment_data(self, existing_data: List[Dict], count: int) -> List[Dict]:
        """Generate synthetic data to augment training set."""
        augmented = []

        for _ in range(count):
            base_sample = np.random.choice(existing_data)
            augmented.append({
                "prompt": base_sample["prompt"].replace("machine learning", "AI"),
                "response": base_sample["response"],
                "expected_response": base_sample.get("expected_response"),
                "rating": base_sample["rating"],
                "is_error": False,
                "difficulty": base_sample["difficulty"],
                "sample_type": "augmentation",
                "session_id": f"aug-{int(time.time())}",
                "created_at": datetime.now().isoformat()
            })

        return augmented

    def _compare_models(self, old_version: str, new_version: str) -> Dict[str, Any]:
        """Comprehensive model comparison using evaluator."""
        logger.info(f"Comparing v{old_version} vs v{new_version}")

        # Get evaluation data
        eval_data = self.feedback_db.generate_training_data(min_rating_threshold=3)
        if len(eval_data) < 20:
            eval_data.extend(self._generate_eval_samples(50))

        old_scores = {}
        new_scores = {}

        # Evaluate both models
        for sample in eval_data[:100]:  # Limit for speed
            old_score = self.evaluator.evaluate_response(
                old_version, sample["prompt"], sample["response"]
            )
            new_score = self.evaluator.evaluate_response(
                new_version, sample["prompt"], sample["response"]
            )

            for metric in old_score.keys():
                old_scores.setdefault(metric, []).append(old_score[metric])
                new_scores.setdefault(metric, []).append(new_score[metric])

        # Calculate averages
        comparison = {}
        for metric in old_scores:
            old_avg = np.mean(old_scores[metric])
            new_avg = np.mean(new_scores[metric])
            improvement = (new_avg - old_avg) / old_avg * 100 if old_avg > 0 else 0

            comparison[f"{metric}_old"] = round(old_avg, 3)
            comparison[f"{metric}_new"] = round(new_avg, 3)
            comparison[f"{metric}_improvement"] = round(improvement, 2)

        overall_improvement = np.mean([
            comparison.get(f"{m}_improvement", 0)
            for m in ['accuracy', 'relevance', 'coherence']
            if f"{m}_improvement" in comparison
        ])

        comparison["overall_improvement"] = round(overall_improvement, 2)
        comparison["improvement_percent"] = overall_improvement

        logger.info(f"Overall improvement: {overall_improvement:.1f}%")
        return comparison

    def _deploy_if_improved(self, old_version: str, new_version: str,
                            comparison: Dict[str, Any]) -> Dict[str, bool]:
        """Deploy new model only if it shows improvement."""
        improvement = comparison.get("overall_improvement", 0)

        if improvement >= self.decision_thresholds['min_improvement_required']:
            # Deploy new version
            self.registry.set_active_version(new_version)
            logger.info(f"✓ Deployed v{new_version} (+{improvement:.1f}% improvement)")

            # Update feedback database
            self.feedback_db.add_training_sample(
                feedback_id=1,  # Placeholder
                prompt="Deployment marker",
                response=f"Model upgraded to v{new_version}",
                sample_type="deployment"
            )

            return {"status": "success", "deployed": True}

        else:
            # Rollback - keep old version
            self.registry.set_active_version(old_version)
            logger.warning(f"✗ Rolled back to v{old_version} (no improvement)")
            return {"status": "rollback", "deployed": False}

    def _generate_eval_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate evaluation samples for model comparison."""
        prompts = [
            "Explain transformers in simple terms",
            "What are the benefits of fine-tuning LLMs?",
            "How does attention mechanism work?",
            "Compare supervised vs unsupervised learning",
            "Steps to deploy an LLM API"
        ]

        return [{"prompt": np.random.choice(prompts), "response": ""} for _ in range(count)]

    def _log_result(self, result: RetrainingResult):
        """Log retraining result to file and database."""
        log_entry = asdict(result)
        log_entry["timestamp"] = datetime.now().isoformat()

        # Write to log file
        log_file = Path("logs/retraining_history.jsonl")
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(f"Retraining completed: {result.status}")


# Standalone scheduler script
def run_scheduler():
    """Background scheduler for automatic retraining."""
    pipeline = RetrainingPipeline()

    logger.info("Starting retraining scheduler...")

    while True:
        try:
            logger.info("Checking for retraining need...")
            decision = pipeline.should_retrain()

            if decision.should_retrain:
                logger.info(f"Triggering retraining (urgency: {decision.urgency})")
                result = pipeline.run_retraining()

                if result.status == "success":
                    logger.info(f"✓ Successfully deployed v{result.new_version}")
                else:
                    logger.warning(f"Retraining failed: {result.status}")
            else:
                logger.info("No retraining needed")

            # Sleep for 24 hours (adjustable)
            time.sleep(24 * 3600)

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(3600)  # Wait 1 hour on error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retraining Pipeline")
    parser.add_argument("--mode", choices=["check", "run", "scheduler"], default="check")

    args = parser.parse_args()

    pipeline = RetrainingPipeline()

    if args.mode == "check":
        decision = pipeline.should_retrain()
        print(f"Should retrain: {decision.should_retrain}")
        print(f"Reasons: {', '.join(decision.reasons)}")
        print(f"Urgency: {decision.urgency}")

    elif args.mode == "run":
        result = pipeline.run_retraining()
        print(f"Result: {result.status}")
        if result.new_version:
            print(f"New version: {result.new_version}")
            print(f"Improvement: {result.performance_improvement}%")

    elif args.mode == "scheduler":
        run_scheduler()