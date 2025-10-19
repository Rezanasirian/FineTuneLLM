"""
Evaluation Script for Fine-Tuned Multi-Agent LLM
Validates model performance on agent-specific tasks

Usage:
    python evaluate_model.py --model_path ./output/final_model --data_dir ./data

Estimated runtime: 15-30 minutes
"""

import os
import sys
import json
from src.core.logger import get_logger
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rouge_score import rouge_scorer

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger('evaluator')


class ModelEvaluator:
    """Evaluator for multi-agent LLM."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",  # For generation
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info("Model loaded successfully")

        # Initialize ROUGE scorer for text quality
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 2048) -> str:
        """Generate response for given messages."""
        # Format with chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for consistency
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def validate_json(self, text: str) -> Tuple[bool, Any]:
        """Validate if text is valid JSON and parse it."""
        try:
            # Try to find JSON in text (handle extra text)
            # Look for outermost { ... }
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                return False, None

            json_str = text[start:end]
            parsed = json.loads(json_str)
            return True, parsed
        except json.JSONDecodeError:
            return False, None

    def validate_mdx_syntax(self, mdx: str) -> bool:
        """Basic MDX syntax validation."""
        required_keywords = ['SELECT', 'FROM', '[ActualCPMDataCube]']
        mdx_upper = mdx.upper()

        # Check required keywords
        if not all(kw in mdx_upper for kw in required_keywords):
            return False

        # Check for proper member format: .&[
        if '.&[' not in mdx:
            return False

        # Check balanced brackets
        if mdx.count('[') != mdx.count(']') or mdx.count('(') != mdx.count(')'):
            return False

        return True

    def evaluate_query_understanding(self, examples: List[Dict]) -> Dict[str, float]:
        """Evaluate Query Understanding agent."""
        logger.info("Evaluating Query Understanding...")
        metrics = {
            "json_valid": 0,
            "has_required_fields": 0,
            "dimensions_correct": 0,
            "total": len(examples)
        }

        required_fields = ["question_type", "dimensions", "time", "filters", "measure"]

        for ex in tqdm(examples, desc="Query Understanding"):
            messages = ex["messages"][:2]  # System + User
            response = self.generate_response(messages)

            # Validate JSON
            is_valid, parsed = self.validate_json(response)
            if is_valid:
                metrics["json_valid"] += 1

                # Check required fields
                if all(field in parsed for field in required_fields):
                    metrics["has_required_fields"] += 1

                # Check dimensions presence
                if isinstance(parsed.get("dimensions"), list) and len(parsed["dimensions"]) > 0:
                    metrics["dimensions_correct"] += 1

        # Calculate percentages
        for key in ["json_valid", "has_required_fields", "dimensions_correct"]:
            metrics[f"{key}_percent"] = (metrics[key] / metrics["total"]) * 100

        return metrics

    def evaluate_mdx_generation(self, examples: List[Dict]) -> Dict[str, float]:
        """Evaluate MDX Generation agent."""
        logger.info("Evaluating MDX Generation...")
        metrics = {
            "json_valid": 0,
            "has_mdx_query": 0,
            "mdx_syntax_valid": 0,
            "has_proper_format": 0,
            "total": len(examples)
        }

        for ex in tqdm(examples, desc="MDX Generation"):
            messages = ex["messages"][:2]
            response = self.generate_response(messages)

            is_valid, parsed = self.validate_json(response)
            if is_valid:
                metrics["json_valid"] += 1

                if "mdx_query" in parsed:
                    metrics["has_mdx_query"] += 1
                    mdx = parsed["mdx_query"]

                    # Validate syntax
                    if self.validate_mdx_syntax(mdx):
                        metrics["mdx_syntax_valid"] += 1

                    # Check for proper member format
                    if ".&[" in mdx:
                        metrics["has_proper_format"] += 1

        for key in ["json_valid", "has_mdx_query", "mdx_syntax_valid", "has_proper_format"]:
            metrics[f"{key}_percent"] = (metrics[key] / metrics["total"]) * 100

        return metrics

    def evaluate_data_analysis(self, examples: List[Dict]) -> Dict[str, float]:
        """Evaluate Data Analysis agent."""
        logger.info("Evaluating Data Analysis...")
        metrics = {
            "json_valid": 0,
            "has_insights": 0,
            "has_summary": 0,
            "insight_types_valid": 0,
            "rouge1_f1": 0.0,
            "rougeL_f1": 0.0,
            "total": len(examples)
        }

        valid_types = {"trend", "outlier", "statistic", "comparison"}

        for ex in tqdm(examples, desc="Data Analysis"):
            messages = ex["messages"][:2]
            expected = ex["messages"][2]["content"]
            response = self.generate_response(messages)

            is_valid, parsed = self.validate_json(response)
            if is_valid:
                metrics["json_valid"] += 1

                if "insights" in parsed and isinstance(parsed["insights"], list) and len(parsed["insights"]) > 0:
                    metrics["has_insights"] += 1

                    # Validate insight types
                    all_valid = all(
                        isinstance(i, dict) and i.get("type") in valid_types
                        for i in parsed["insights"]
                    )
                    if all_valid:
                        metrics["insight_types_valid"] += 1

                if "summary" in parsed and parsed["summary"]:
                    metrics["has_summary"] += 1

                # Calculate ROUGE for summary quality
                if "summary" in parsed:
                    expected_parsed = self.validate_json(expected)[1]
                    if expected_parsed and "summary" in expected_parsed:
                        scores = self.rouge_scorer.score(
                            expected_parsed["summary"],
                            parsed["summary"]
                        )
                        metrics["rouge1_f1"] += scores['rouge1'].fmeasure
                        metrics["rougeL_f1"] += scores['rougeL'].fmeasure

        # Percentages and averages
        for key in ["json_valid", "has_insights", "has_summary", "insight_types_valid"]:
            metrics[f"{key}_percent"] = (metrics[key] / metrics["total"]) * 100

        metrics["rouge1_f1_avg"] = (metrics["rouge1_f1"] / metrics["total"]) * 100
        metrics["rougeL_f1_avg"] = (metrics["rougeL_f1"] / metrics["total"]) * 100

        return metrics

    def evaluate_visualization(self, examples: List[Dict]) -> Dict[str, float]:
        """Evaluate Visualization agent."""
        logger.info("Evaluating Visualization...")
        metrics = {
            "json_valid": 0,
            "has_chart_type": 0,
            "chart_type_valid": 0,
            "total": len(examples)
        }

        valid_charts = {
            "datalabel", "bar", "line", "pie", "scatter", "area",
            "radar", "heatmap", "bubble", "funnel", "gauge",
            "treemap", "sunburst", "waterfall", "boxplot"
        }

        for ex in tqdm(examples, desc="Visualization"):
            messages = ex["messages"][:2]
            response = self.generate_response(messages)

            is_valid, parsed = self.validate_json(response)
            if is_valid:
                metrics["json_valid"] += 1

                if "chart_type" in parsed:
                    metrics["has_chart_type"] += 1

                    if parsed["chart_type"] in valid_charts:
                        metrics["chart_type_valid"] += 1

        for key in ["json_valid", "has_chart_type", "chart_type_valid"]:
            metrics[f"{key}_percent"] = (metrics[key] / metrics["total"]) * 100

        return metrics

    def evaluate_all(self, data_dir: str, sample_size: int = 100) -> Dict[str, Any]:
        """Evaluate all agent types."""
        logger.info("=" * 60)
        logger.info("Starting Comprehensive Evaluation")
        logger.info("=" * 60)

        # Load validation dataset
        val_path = Path(data_dir) / "validation.jsonl"
        dataset = load_dataset("json", data_files={"validation": str(val_path)})["validation"]

        # Group by agent type
        by_agent = defaultdict(list)
        for ex in dataset:
            by_agent[ex["agent_type"]].append(ex)

        results = {}

        # Evaluate Query Understanding
        if "query_understanding" in by_agent:
            samples = by_agent["query_understanding"][:sample_size]
            results["query_understanding"] = self.evaluate_query_understanding(samples)

        # Evaluate MDX Generation
        if "mdx_generation" in by_agent:
            samples = by_agent["mdx_generation"][:sample_size]
            results["mdx_generation"] = self.evaluate_mdx_generation(samples)

        # Evaluate Data Analysis
        if "data_analysis" in by_agent:
            samples = by_agent["data_analysis"][:sample_size]
            results["data_analysis"] = self.evaluate_data_analysis(samples)

        # Evaluate Visualization
        if "visualization" in by_agent:
            samples = by_agent["visualization"][:sample_size]
            results["visualization"] = self.evaluate_visualization(samples)

        # Calculate overall score
        overall_scores = []
        for agent, metrics in results.items():
            # Primary metric per agent
            if agent == "query_understanding":
                overall_scores.append(metrics.get("has_required_fields_percent", 0))
            elif agent == "mdx_generation":
                overall_scores.append(metrics.get("mdx_syntax_valid_percent", 0))
            elif agent == "data_analysis":
                overall_scores.append(metrics.get("has_insights_percent", 0))
            elif agent == "visualization":
                overall_scores.append(metrics.get("chart_type_valid_percent", 0))

        results["overall_score"] = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for agent, metrics in results.items():
        if agent == "overall_score":
            continue

        print(f"\n{agent.upper().replace('_', ' ')}:")
        print("-" * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:.<50} {value:.2f}")
            else:
                print(f"  {key:.<50} {value}")

    print("\n" + "=" * 60)
    print(f"OVERALL SCORE: {results['overall_score']:.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing validation.jsonl")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of samples per agent type to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for results")
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = AgentEvaluator(args.model_path)

    # Run evaluation
    results = evaluator.evaluate_all(args.data_dir, args.sample_size)

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()