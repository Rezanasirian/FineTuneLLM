#!/usr/bin/env python3
"""
src/data/data_augmentor.py
DataAugmentor for generating synthetic training data

Advanced augmentation techniques:
1. Paraphrasing & synonym replacement
2. Back-translation
3. Question reformulation
4. Context injection
5. Error injection (for robustness)
6. Style transfer
7. Length variation
8. Multi-turn conversation simulation
"""

import json
import logging
import random
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import spacy
from transformers import pipeline, MarianMTModel, MarianTokenizer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model (install: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Run: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize translation models
try:
    back_translate_pipeline = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=-1  # CPU
    )
except Exception:
    back_translate_pipeline = None
    logging.warning("Back-translation not available (install transformers)")

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    max_samples: int = 500
    augmentation_ratio: float = 0.5  # 50% augmentation
    techniques: List[str] = None
    quality_threshold: float = 0.8
    diversity_score: float = 0.7

    def __post_init__(self):
        if self.techniques is None:
            self.techniques = [
                "paraphrase", "synonym", "back_translate",
                "question_reform", "context_inject", "style_transfer",
                "length_vary", "error_inject"
            ]


@dataclass
class AugmentedSample:
    """Data class for augmented training samples."""
    original_prompt: str
    original_response: str
    augmented_prompt: str
    technique: str
    quality_score: float = 1.0
    diversity_score: float = 1.0
    created_at: str = None
    augmented_response: Optional[str] = None


    def to_training_sample(self) -> Dict[str, Any]:
        """Convert to training format."""
        return {
            "prompt": self.augmented_prompt,
            "response": self.augmented_response or self.original_response,
            "expected_response": self.augmented_response,
            "sample_type": f"aug_{self.technique}",
            "quality_score": self.quality_score,
            "diversity_score": self.diversity_score,
            "created_at": self.created_at
        }


class DataAugmentor:
    """Advanced data augmentation for LLM training."""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.synonym_map = self._build_synonym_map()
        self.question_starters = [
            "What is", "Explain", "Describe", "Define", "How does",
            "Why do", "Tell me about", "Give examples of", "Compare",
            "Steps to", "Benefits of", "Differences between"
        ]
        self.response_styles = {
            "formal": ["Certainly", "Indeed", "To elaborate", "In summary"],
            "casual": ["Sure", "Yeah", "Here's the deal", "Bottom line"],
            "technical": ["From a technical perspective", "In terms of implementation"],
            "simple": ["In simple terms", "Basically", "To put it simply"]
        }

        logger.info("DataAugmentor initialized")

    def augment_dataset(self,
                        samples: List[Dict[str, Any]],
                        target_size: int) -> List[Dict[str, Any]]:
        """
        Augment dataset to reach target size.

        Args:
            samples: Original training samples
            target_size: Desired total dataset size

        Returns:
            Augmented dataset
        """
        current_size = len(samples)
        needed = max(0, target_size - current_size)

        if needed == 0:
            logger.info("Dataset size sufficient, no augmentation needed")
            return samples

        logger.info(f"Augmenting {needed} samples using {len(self.config.techniques)} techniques")

        augmented_samples = []
        techniques_used = {t: 0 for t in self.config.techniques}

        for sample in samples:
            if len(augmented_samples) >= needed:
                break

            # Apply multiple techniques per sample
            for technique in random.sample(self.config.techniques, k=min(3, len(self.config.techniques))):
                if len(augmented_samples) >= needed:
                    break

                aug_sample = self._apply_technique(sample, technique)
                if aug_sample and aug_sample.quality_score >= self.config.quality_threshold:
                    aug_sample.created_at = datetime.now().isoformat()
                    augmented_samples.append(aug_sample.to_training_sample())
                    techniques_used[technique] += 1

        # Combine original + augmented
        final_dataset = samples + augmented_samples[:needed]
        random.shuffle(final_dataset)

        logger.info(f"Augmentation complete: {len(final_dataset)} total samples")
        logger.info(f"Techniques used: {techniques_used}")

        return final_dataset

    def _apply_technique(self, sample: Dict[str, Any], technique: str) -> Optional[AugmentedSample]:
        """Apply specific augmentation technique."""
        prompt = sample["prompt"]
        response = sample["response"]

        if technique == "paraphrase":
            return self._paraphrase_sample(prompt, response)
        elif technique == "synonym":
            return self._synonym_replacement(prompt, response)
        elif technique == "back_translate":
            return self._back_translate(prompt, response)
        elif technique == "question_reform":
            return self._reformulate_question(prompt, response)
        elif technique == "context_inject":
            return self._inject_context(prompt, response)
        elif technique == "style_transfer":
            return self._style_transfer(prompt, response)
        elif technique == "length_vary":
            return self._vary_length(prompt, response)
        elif technique == "error_inject":
            return self._inject_errors(prompt, response)

        return None

    # ========== AUGMENTATION TECHNIQUES ==========

    def _paraphrase_sample(self, prompt: str, response: str) -> Optional[AugmentedSample]:
        """Paraphrase prompt and response."""
        # Simple rule-based paraphrasing
        prompt_words = word_tokenize(prompt.lower())
        paraphrased_prompt = self._paraphrase_words(prompt_words)
        paraphrased_response = self._paraphrase_words(word_tokenize(response.lower()))

        diversity = self._calculate_diversity(prompt, paraphrased_prompt)
        if diversity < self.config.diversity_score:
            return None

        return AugmentedSample(
            original_prompt=prompt,
            original_response=response,
            augmented_prompt=" ".join(paraphrased_prompt),
            augmented_response=" ".join(paraphrased_response),
            technique="paraphrase",
            quality_score=0.9,
            diversity_score=diversity
        )

    def _synonym_replacement(self, prompt: str, response: str) -> Optional[AugmentedSample]:
        """Replace 20-30% of words with synonyms."""

        def replace_synonyms(text_words, replace_ratio=0.25):
            new_words = []
            for word in text_words:
                if random.random() < replace_ratio and word in self.synonym_map:
                    synonym = random.choice(self.synonym_map[word])
                    new_words.append(synonym)
                else:
                    new_words.append(word)
            return new_words

        prompt_words = word_tokenize(prompt.lower())
        response_words = word_tokenize(response.lower())

        aug_prompt = " ".join(replace_synonyms(prompt_words))
        aug_response = " ".join(replace_synonyms(response_words))

        return AugmentedSample(
            original_prompt=prompt,
            original_response=response,
            augmented_prompt=aug_prompt,
            augmented_response=aug_response,
            technique="synonym",
            quality_score=0.85,
            diversity_score=self._calculate_diversity(prompt, aug_prompt)
        )

    def _back_translate(self, prompt: str, response: str) -> Optional[AugmentedSample]:
        """Back-translation via French."""
        if not back_translate_pipeline:
            return None

        try:
            # English -> French -> English
            fr_prompt = back_translate_pipeline(prompt, max_length=100)[0]['translation_text']
            aug_prompt = back_translate_pipeline(fr_prompt, model="Helsinki-NLP/opus-mt-fr-en")[0]['translation_text']

            fr_response = back_translate_pipeline(response, max_length=200)[0]['translation_text']
            aug_response = back_translate_pipeline(fr_response, model="Helsinki-NLP/opus-mt-fr-en")[0][
                'translation_text']

            return AugmentedSample(
                original_prompt=prompt,
                original_response=response,
                augmented_prompt=aug_prompt,
                augmented_response=aug_response,
                technique="back_translate",
                quality_score=0.95,
                diversity_score=self._calculate_diversity(prompt, aug_prompt)
            )
        except Exception as e:
            logger.debug(f"Back-translation failed: {e}")
            return None

    def _reformulate_question(self, prompt: str, response: str) -> Optional[AugmentedSample]:
        """Reformulate question starters."""
        starter = random.choice(self.question_starters)
        aug_prompt = f"{starter} {prompt.split(':')[-1] if ':' in prompt else prompt}"

        return AugmentedSample(
            original_prompt=prompt,
            original_response=response,
            augmented_prompt=aug_prompt,
            technique="question_reform",
            quality_score=0.9,
            diversity_score=self._calculate_diversity(prompt, aug_prompt)
        )

    def _inject_context(self, prompt: str, response: str) -> Optional[AugmentedSample]:
        """Inject additional context."""
        contexts = [
            "In the context of machine learning,",
            "From a technical perspective,",
            "For beginners,",
            "In production environments,"
        ]
        aug_prompt = f"{random.choice(contexts)} {prompt}"

        return AugmentedSample(
            original_prompt=prompt,
            original_response=response,
            augmented_prompt=aug_prompt,
            technique="context_inject",
            quality_score=0.85,
            diversity_score=self._calculate_diversity(prompt, aug_prompt)
        )

    def _style_transfer(self, prompt: str, response: str) -> Optional[AugmentedSample]:
        """Transfer response style."""
        style = random.choice(list(self.response_styles.keys()))
        style_starters = self.response_styles[style]

        aug_response = f"{random.choice(style_starters)} {response}"

        return AugmentedSample(
            original_prompt=prompt,
            original_response=response,
            augmented_prompt=prompt,
            augmented_response=aug_response,
            technique=f"style_{style}",
            quality_score=0.9,
            diversity_score=0.8
        )

    def _vary_length(self, prompt: str, response: str) -> Optional[AugmentedSample]:
        """Vary response length (short/long)."""
        words = word_tokenize(response)

        if len(words) > 20:  # Shorten
            aug_response = " ".join(words[:15]) + "..."
        else:  # Lengthen
            aug_response = response + " This comprehensive explanation covers all key aspects."

        return AugmentedSample(
            original_prompt=prompt,
            original_response=response,
            augmented_prompt=prompt,
            augmented_response=aug_response,
            technique="length_vary",
            quality_score=0.8,
            diversity_score=0.7
        )

    def _inject_errors(self, prompt: str, response: str) -> Optional[AugmentedSample]:
        """Inject controlled errors for robustness training."""
        # Add typos, grammar errors
        words = word_tokenize(response)
        error_types = ["typo", "grammar", "omission"]
        error = random.choice(error_types)

        if error == "typo" and len(words) > 5:
            idx = random.randint(0, len(words) - 1)
            misspelled = self._misspell_word(words[idx])
            words[idx] = misspelled

        aug_response = " ".join(words)

        return AugmentedSample(
            original_prompt=prompt,
            original_response=response,
            augmented_prompt=prompt,
            augmented_response=aug_response,
            technique="error_inject",
            quality_score=0.7,
            diversity_score=0.9
        )

    # ========== UTILITY METHODS ==========

    def _build_synonym_map(self) -> Dict[str, List[str]]:
        """Build synonym dictionary from WordNet."""
        synonyms = {}
        for synset in wordnet.all_synsets('n'):
            for lemma in synset.lemmas():
                word = lemma.name().replace('_', ' ')
                synonyms.setdefault(word, []).extend([l.name().replace('_', ' ') for l in synset.lemmas()])

        # Clean and deduplicate
        for word in synonyms:
            synonyms[word] = list(set([w for w in synonyms[word] if w != word]))

        return {k: v for k, v in synonyms.items() if len(v) > 0}

    def _paraphrase_words(self, words: List[str]) -> List[str]:
        """Paraphrase list of words."""
        paraphrased = []
        for word in words:
            if word in self.synonym_map and random.random() < 0.3:
                synonym = random.choice(self.synonym_map[word])
                paraphrased.append(synonym)
            else:
                paraphrased.append(word)
        return paraphrased

    def _misspell_word(self, word: str) -> str:
        """Simple misspelling."""
        if len(word) < 3:
            return word
        # Swap two adjacent letters
        idx = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return ''.join(chars)

    def _calculate_diversity(self, original: str, augmented: str) -> float:
        """Calculate text diversity score."""
        orig_words = set(word_tokenize(original.lower()))
        aug_words = set(word_tokenize(augmented.lower()))
        overlap = len(orig_words.intersection(aug_words))
        total = len(orig_words.union(aug_words))
        return 1.0 - (overlap / total) if total > 0 else 0.0

    def save_augmentation_report(self, samples: List[AugmentedSample], filepath: str):
        """Save augmentation report."""
        report = {
            "total_augmented": len(samples),
            "techniques_used": {},
            "quality_distribution": {}
        }

        for sample in samples:
            report["techniques_used"][sample.technique] = report["techniques_used"].get(sample.technique, 0) + 1
            quality_bin = f"{int(sample.quality_score * 10)}-{int((sample.quality_score + 0.1) * 10)}"
            report["quality_distribution"][quality_bin] = report["quality_distribution"].get(quality_bin, 0) + 1

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Augmentation report saved: {filepath}")


# ========== INTEGRATION WITH DataProcessor ==========
def integrate_with_dataprocessor():
    """Update DataProcessor to use DataAugmentor."""
    from src.pipeline.data_processor import DataProcessor

    class AugmentedDataProcessor(DataProcessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.augmentor = DataAugmentor()

        def get_training_data(self, *args, **kwargs):
            samples, stats = super().get_training_data(*args, **kwargs)

            # Augment if needed
            target_size = kwargs.get('min_samples', 500) * 2  # Double for augmentation
            augmented_samples = self.augmentor.augment_dataset(samples, target_size)

            # Save report
            self.augmentor.save_augmentation_report(
                [AugmentedSample(s['prompt'], s['response'], s['prompt'], technique="base")
                 for s in samples[:10]],  # Sample for report
                "logs/augmentation_report.json"
            )

            return augmented_samples, stats


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    # Sample data
    samples = [
        {
            "prompt": "Explain machine learning",
            "response": "Machine learning is a subset of AI..."
        },
        {
            "prompt": "What is neural network?",
            "response": "Neural networks are computational models..."
        }
    ]

    augmentor = DataAugmentor()
    augmented = augmentor.augment_dataset(samples, target_size=20)

    print(f"Original: {len(samples)}")
    print(f"Augmented: {len(augmented)}")
    print("\nSample augmented:")
    for sample in augmented[:3]:
        print(f"Prompt: {sample['prompt']}")
        print(f"Type: {sample['sample_type']}")
        print("---")