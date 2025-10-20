
"""
src/data/data_processor.py
DataProcessor for extracting and processing training data from FeedbackDatabase

Handles:
1. Extract raw feedback data from database
2. Generate diverse training samples (error correction, improvement, augmentation)
3. Data cleaning, validation, and quality scoring
4. Format conversion for model training (JSONL, CSV, HuggingFace datasets)
5. Balancing and sampling strategies
6. Synthetic data generation for underrepresented cases
"""

import json
import logging
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from sklearn.utils import resample

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.feedback_service import FeedbackDatabase, FeedbackRecord
from src.core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Data class for processed training samples."""
    prompt: str
    response: str
    expected_response: Optional[str] = None
    sample_type: str = "general"  # 'error_correction', 'improvement', 'positive', 'augmentation'
    rating: Optional[int] = None
    difficulty: int = 1  # 1-5
    session_id: str = ""
    model_version: str = ""
    created_at: str = ""
    quality_score: float = 1.0  # 0.0-1.0
    tokens_prompt: int = 0
    tokens_response: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        """Format for JSONL training files."""
        data = self.to_dict()
        data["text"] = f"<|prompt|>{self.prompt}<|response|>{self.response}"
        return json.dumps(data)


@dataclass
class ProcessingStats:
    """Statistics for data processing results."""
    total_raw_feedback: int
    valid_samples: int
    error_correction: int
    improvement: int
    positive: int
    augmentation: int
    avg_quality_score: float
    avg_tokens: int
    class_balance: Dict[str, int]
    duplicates_removed: int


class DataProcessor:
    """Processes feedback data into high-quality training samples."""

    def __init__(self, config_dir: str = "config"):
        self.config = get_config(config_dir)
        self.feedback_db = FeedbackDatabase()
        self.min_sample_length = 10
        self.max_sample_length = 2000
        self.quality_threshold = 0.7

        # Sample type weights for balancing
        self.sample_weights = {
            'error_correction': 0.4,
            'improvement': 0.3,
            'positive': 0.2,
            'augmentation': 0.1
        }

        logger.info("DataProcessor initialized")

    def get_training_data(self,
                          target_version: str,
                          min_samples: int = 500,
                          days_back: int = 30,
                          balance_classes: bool = True,
                          include_synthetic: bool = True) -> Tuple[List[TrainingSample], ProcessingStats]:
        """
        Main method: Get processed training data ready for model training.

        Args:
            target_version: Model version to process feedback for
            min_samples: Minimum number of samples to generate
            days_back: Days of feedback to include
            balance_classes: Balance sample types
            include_synthetic: Add synthetic data if needed

        Returns:
            (training_samples, processing_stats)
        """
        logger.info(f"Generating training data for {target_version} (min_samples={min_samples})")

        start_time = datetime.now()

        # Step 1: Extract raw feedback
        raw_feedback = self._extract_feedback(target_version, days_back)
        logger.info(f"Extracted {len(raw_feedback)} raw feedback records")

        # Step 2: Generate base samples
        base_samples = self._generate_base_samples(raw_feedback)

        # Step 3: Clean and validate
        valid_samples, duplicates_removed = self._clean_and_validate(base_samples)

        # Step 4: Balance classes
        if balance_classes:
            valid_samples = self._balance_samples(valid_samples)

        # Step 5: Augment if needed
        if include_synthetic and len(valid_samples) < min_samples:
            synthetic_count = min_samples - len(valid_samples)
            synthetic_samples = self._generate_synthetic_data(valid_samples, synthetic_count)
            valid_samples.extend(synthetic_samples)

        # Step 6: Quality filter and sample
        final_samples = self._quality_filter_and_sample(valid_samples, min_samples)

        # Step 7: Calculate stats
        stats = self._calculate_stats(final_samples, raw_feedback, duplicates_removed)

        duration = (datetime.now() - start_time).total_seconds() / 60
        logger.info(f"✓ Generated {len(final_samples)} samples in {duration:.1f}min")

        return final_samples, stats

    def _extract_feedback(self, version: str, days_back: int) -> List[FeedbackRecord]:
        """Extract relevant feedback from database."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        # Get all feedback for version
        all_feedback = self.feedback_db.get_feedback_by_version(version, limit=10000)

        # Filter by date
        recent_feedback = [
            f for f in all_feedback
            if datetime.fromisoformat(f.created_at) >= datetime.fromisoformat(cutoff_date)
        ]

        # Prioritize low-rated and error feedback
        priority_feedback = [
            f for f in recent_feedback
            if (f.rating and f.rating <= 3) or f.is_error
        ]

        # Add some positive samples for balance
        positive_feedback = [
            f for f in recent_feedback
            if f.rating and f.rating >= 4 and not f.is_error
        ]

        # Sample positive feedback to avoid imbalance
        if len(positive_feedback) > len(priority_feedback) * 2:
            positive_feedback = random.sample(positive_feedback, len(priority_feedback) * 2)

        return priority_feedback + positive_feedback

    def _generate_base_samples(self, feedback: List[FeedbackRecord]) -> List[TrainingSample]:
        """Generate training samples from feedback records."""
        samples = []

        for fb in feedback:
            # Create prompt from feedback context
            prompt = self._create_prompt_from_feedback(fb)
            if not prompt:
                continue

            # Create response (actual vs expected)
            if fb.is_error or (fb.rating and fb.rating < 4):
                # Error correction or improvement sample
                sample_type = "error_correction" if fb.is_error else "improvement"
                response = self._generate_improved_response(fb)
                expected = self._create_expected_response(fb)
            else:
                # Positive sample
                sample_type = "positive"
                response = fb.comment or "High quality response"
                expected = None

            # Create sample
            sample = TrainingSample(
                prompt=prompt,
                response=response,
                expected_response=expected,
                sample_type=sample_type,
                rating=fb.rating,
                difficulty=self._calculate_difficulty(fb),
                session_id=fb.session_id,
                model_version=fb.model_version,
                created_at=fb.created_at,
                quality_score=self._calculate_quality_score(fb)
            )

            samples.append(sample)

        return samples

    def _create_prompt_from_feedback(self, fb: FeedbackRecord) -> Optional[str]:
        """Generate training prompt from feedback."""
        # In production, extract from session logs
        # For now, use feedback patterns
        prompt_templates = [
            f"User asked: {fb.comment or 'complex question'}",
            f"Explain: {fb.error_type or 'technical concept'}",
            f"Based on user feedback: {fb.comment}",
        ]

        return random.choice(prompt_templates)

    def _generate_improved_response(self, fb: FeedbackRecord) -> str:
        """Generate improved response based on feedback."""
        base_responses = [
            "Here's a comprehensive, accurate response addressing your query.",
            "Let me provide a detailed and correct explanation.",
            "Based on your feedback, here's an improved answer:"
        ]

        if fb.comment:
            return f"{random.choice(base_responses)} {fb.comment[:100]}"
        elif fb.error_type:
            return f"{random.choice(base_responses)} Fixed {fb.error_type}."
        else:
            return random.choice(base_responses)

    def _create_expected_response(self, fb: FeedbackRecord) -> Optional[str]:
        """Create expected response for supervised training."""
        if fb.comment and len(fb.comment) > 20:
            return f"Corrected version: {fb.comment[:200]}"
        return None

    def _calculate_difficulty(self, fb: FeedbackRecord) -> int:
        """Calculate sample difficulty (1-5)."""
        difficulty = 1

        if fb.rating and fb.rating <= 2:
            difficulty += 2
        elif fb.rating and fb.rating <= 3:
            difficulty += 1

        if fb.is_error:
            difficulty += 2

        if fb.response_time_ms and fb.response_time_ms > 1000:
            difficulty += 1

        return min(difficulty, 5)

    def _calculate_quality_score(self, fb: FeedbackRecord) -> float:
        """Calculate quality score (0.0-1.0) for sample."""
        score = 1.0

        if fb.rating:
            score *= (fb.rating / 5.0)

        if fb.is_error:
            score *= 0.8  # Penalize errors but keep useful

        if fb.comment and len(fb.comment.split()) < 5:
            score *= 0.9  # Short comments less informative

        return round(max(score, 0.3), 2)  # Minimum 0.3

    def _clean_and_validate(self, samples: List[TrainingSample]) -> Tuple[List[TrainingSample], int]:
        """Remove invalid samples and duplicates."""
        valid_samples = []
        seen_prompts = set()
        duplicates = 0

        for sample in samples:
            # Validation
            if (len(sample.prompt) < self.min_sample_length or
                    len(sample.response) < self.min_sample_length or
                    len(sample.prompt) > self.max_sample_length):
                continue

            # Remove duplicates
            prompt_hash = hash(sample.prompt.lower().strip())
            if prompt_hash in seen_prompts:
                duplicates += 1
                continue
            seen_prompts.add(prompt_hash)

            # Basic text cleaning
            sample.prompt = self._clean_text(sample.prompt)
            sample.response = self._clean_text(sample.response)

            valid_samples.append(sample)

        return valid_samples, duplicates

    def _clean_text(self, text: str) -> str:
        """Clean text for training."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        return text

    def _balance_samples(self, samples: List[TrainingSample]) -> List[TrainingSample]:
        """Balance sample types using weighted sampling."""
        if len(samples) < 10:
            return samples

        # Group by type
        type_groups = {}
        for sample in samples:
            type_groups.setdefault(sample.sample_type, []).append(sample)

        balanced = []

        for sample_type, weight in self.sample_weights.items():
            if sample_type in type_groups:
                group_size = len(samples) * weight
                group_samples = type_groups[sample_type]

                if len(group_samples) > group_size:
                    # Downsample
                    balanced.extend(random.sample(group_samples, int(group_size)))
                else:
                    # Upsample
                    balanced.extend(resample(group_samples, n_samples=int(group_size), random_state=42))

        return balanced[:len(samples)]  # Don't increase total size

    def _generate_synthetic_data(self, base_samples: List[TrainingSample],
                                 count: int) -> List[TrainingSample]:
        """Generate synthetic samples for data augmentation."""
        synthetic = []

        for _ in range(count):
            base = random.choice(base_samples)

            # Simple paraphrasing
            prompt_variants = [
                base.prompt.replace("Explain", "Describe"),
                base.prompt.replace("What is", "Define"),
                f"Simple explanation: {base.prompt}",
            ]

            synthetic_sample = TrainingSample(
                prompt=random.choice(prompt_variants),
                response=base.response,
                expected_response=base.expected_response,
                sample_type="augmentation",
                rating=base.rating,
                difficulty=base.difficulty,
                session_id=f"synth-{random.randint(1000, 9999)}",
                model_version=base.model_version,
                created_at=datetime.now().isoformat(),
                quality_score=base.quality_score * 0.9
            )

            synthetic.append(synthetic_sample)

        return synthetic

    def _quality_filter_and_sample(self, samples: List[TrainingSample],
                                   min_samples: int) -> List[TrainingSample]:
        """Filter by quality and sample to target size."""
        # Filter by quality
        high_quality = [s for s in samples if s.quality_score >= self.quality_threshold]

        if len(high_quality) >= min_samples:
            # Sample randomly
            return random.sample(high_quality, min_samples)
        else:
            # Use all high quality + some lower quality
            needed = min_samples - len(high_quality)
            remaining = [s for s in samples if s not in high_quality]

            if needed > 0 and remaining:
                additional = random.sample(remaining, min(needed, len(remaining)))
                high_quality.extend(additional)

            return high_quality[:min_samples]

    def _calculate_stats(self, samples: List[TrainingSample],
                         raw_feedback: List[FeedbackRecord],
                         duplicates: int) -> ProcessingStats:
        """Calculate comprehensive processing statistics."""
        if not samples:
            return ProcessingStats(0, 0, 0, 0, 0, 0, 0.0, 0, {}, 0)

        df = pd.DataFrame([s.to_dict() for s in samples])

        return ProcessingStats(
            total_raw_feedback=len(raw_feedback),
            valid_samples=len(samples),
            error_correction=len(df[df['sample_type'] == 'error_correction']),
            improvement=len(df[df['sample_type'] == 'improvement']),
            positive=len(df[df['sample_type'] == 'positive']),
            augmentation=len(df[df['sample_type'] == 'augmentation']),
            avg_quality_score=float(df['quality_score'].mean()),
            avg_tokens=int(df['tokens_prompt'].fillna(0).sum() + df['tokens_response'].fillna(0).sum()) // len(samples),
            class_balance=df['sample_type'].value_counts().to_dict(),
            duplicates_removed=duplicates
        )

    # ========== EXPORT METHODS ==========

    def save_to_jsonl(self, samples: List[TrainingSample], filepath: str):
        """Save samples as JSONL for training."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for sample in samples:
                f.write(sample.to_jsonl() + '\n')
        logger.info(f"Saved {len(samples)} samples to {filepath}")

    def save_to_csv(self, samples: List[TrainingSample], filepath: str):
        """Save samples as CSV."""
        df = pd.DataFrame([s.to_dict() for s in samples])
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(samples)} samples to {filepath}")

    def save_huggingface_dataset(self, samples: List[TrainingSample], output_dir: str):
        """Save as HuggingFace dataset format."""
        from datasets import Dataset

        data = [s.to_dict() for s in samples]
        dataset = Dataset.from_list(data)
        dataset.save_to_disk(output_dir)
        logger.info(f"Saved dataset to {output_dir}")


# ========== USAGE EXAMPLE & TESTING ==========
if __name__ == "__main__":
    processor = DataProcessor()

    # Generate training data
    samples, stats = processor.get_training_data(
        target_version="v_20241020_143022",
        min_samples=200,
        days_back=30
    )

    print("TRAINING DATA STATS:")
    print(f"Valid Samples: {stats.valid_samples}")
    print(f"Error Correction: {stats.error_correction}")
    print(f"Improvement: {stats.improvement}")
    print(f"Positive: {stats.positive}")
    print(f"Augmentation: {stats.augmentation}")
    print(f"Avg Quality: {stats.avg_quality_score:.3f}")
    print(f"Class Balance: {stats.class_balance}")

    # Save data
    processor.save_to_jsonl(samples, "data/training_samples.jsonl")
    processor.save_to_csv(samples, "data/training_samples.csv")