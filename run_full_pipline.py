"""
End-to-End Automated Pipeline for Multi-Agent LLM Fine-Tuning
Runs all steps sequentially: data generation → training → evaluation → merging → testing

Usage:
    python run_full_pipeline.py --base_model Qwen/Qwen2.5-7B-Instruct --output_dir ./pipeline_output

Estimated total runtime: 10-14 hours on A100 48GB
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for the full pipeline."""

    def __init__(self, args):
        self.base_model = args.base_model
        self.output_dir = Path(args.output_dir)
        self.num_samples = args.num_samples
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.skip_data_gen = args.skip_data_gen
        self.skip_training = args.skip_training
        self.skip_evaluation = args.skip_evaluation
        self.skip_merging = args.skip_merging
        self.skip_testing = args.skip_testing

        # Derived paths
        self.data_dir = self.output_dir / "data"
        self.training_dir = self.output_dir / "training"
        self.merged_model_dir = self.output_dir / "merged_model"
        self.eval_results = self.output_dir / "evaluation_results.json"
        self.pipeline_summary = self.output_dir / "pipeline_summary.json"


class PipelineRunner:
    """Orchestrates the full pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = None
        self.results = {
            "pipeline_version": "1.0",
            "base_model": config.base_model,
            "start_time": None,
            "end_time": None,
            "total_duration_hours": 0,
            "steps": {},
            "final_metrics": {},
        }

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd: list, step_name: str) -> bool:
        """Run a subprocess command and log results."""
        logger.info(f"Starting: {step_name}")
        logger.info(f"Command: {' '.join(cmd)}")

        step_start = time.time()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            step_duration = (time.time() - step_start) / 3600

            logger.info(f"✓ {step_name} completed in {step_duration:.2f} hours")
            logger.info(f"Output:\n{result.stdout[-500:]}")  # Last 500 chars

            self.results["steps"][step_name] = {
                "status": "success",
                "duration_hours": step_duration,
            }

            return True

        except subprocess.CalledProcessError as e:
            step_duration = (time.time() - step_start) / 3600

            logger.error(f"✗ {step_name} failed after {step_duration:.2f} hours")
            logger.error(f"Error:\n{e.stderr}")

            self.results["steps"][step_name] = {
                "status": "failed",
                "duration_hours": step_duration,
                "error": e.stderr,
            }

            return False

    def step_1_data_generation(self) -> bool:
        """Step 1: Generate synthetic training data."""
        if self.config.skip_data_gen:
            logger.info("Skipping data generation (--skip_data_gen)")
            return True

        logger.info("=" * 80)
        logger.info("STEP 1/6: DATA GENERATION")
        logger.info("=" * 80)

        cmd = [
            sys.executable,
            "data_generation.py",
            "--output_dir", str(self.config.data_dir),
            "--num_samples", str(self.config.num_samples),
        ]

        success = self.run_command(cmd, "data_generation")

        if success:
            # Verify data files
            train_file = self.config.data_dir / "train.jsonl"
            val_file = self.config.data_dir / "validation.jsonl"

            if train_file.exists() and val_file.exists():
                with open(train_file) as f:
                    train_count = sum(1 for _ in f)
                with open(val_file) as f:
                    val_count = sum(1 for _ in f)

                logger.info(f"✓ Generated {train_count} training samples")
                logger.info(f"✓ Generated {val_count} validation samples")

                self.results["steps"]["data_generation"]["train_samples"] = train_count
                self.results["steps"]["data_generation"]["val_samples"] = val_count
            else:
                logger.error("Data files not found after generation")
                return False

        return success

    def step_2_training(self) -> bool:
        """Step 2: Fine-tune model with QLoRA."""
        if self.config.skip_training:
            logger.info("Skipping training (--skip_training)")
            return True

        logger.info("=" * 80)
        logger.info("STEP 2/6: MODEL FINE-TUNING")
        logger.info("=" * 80)

        cmd = [
            sys.executable,
            "finetune_qlora.py",
            "--model_name", self.config.base_model,
            "--data_dir", str(self.config.data_dir),
            "--output_dir", str(self.config.training_dir),
            "--epochs", str(self.config.epochs),
            "--batch_size", str(self.config.batch_size),
            "--lora_r", str(self.config.lora_r),
            "--lora_alpha", str(self.config.lora_alpha),
        ]

        success = self.run_command(cmd, "training")

        if success:
            # Load training metrics
            metrics_file = self.config.training_dir / "training_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                self.results["steps"]["training"]["metrics"] = metrics
                logger.info(f"Final train loss: {metrics.get('final_train_loss')}")
                logger.info(f"Final eval loss: {metrics.get('final_eval_loss')}")

        return success

    def step_3_evaluation(self) -> bool:
        """Step 3: Evaluate fine-tuned model."""
        if self.config.skip_evaluation:
            logger.info("Skipping evaluation (--skip_evaluation)")
            return True

        logger.info("=" * 80)
        logger.info("STEP 3/6: MODEL EVALUATION")
        logger.info("=" * 80)

        model_path = self.config.training_dir / "final_model"

        cmd = [
            sys.executable,
            "evaluate_model.py",
            "--model_path", str(model_path),
            "--data_dir", str(self.config.data_dir),
            "--output", str(self.config.eval_results),
            "--sample_size", "50",  # Smaller for speed
        ]

        success = self.run_command(cmd, "evaluation")

        if success and self.config.eval_results.exists():
            with open(self.config.eval_results) as f:
                eval_results = json.load(f)
            self.results["final_metrics"] = eval_results
            logger.info(f"Overall Score: {eval_results.get('overall_score', 0):.2f}%")

        return success

    def step_4_merging(self) -> bool:
        """Step 4: Merge LoRA adapters into base model."""
        if self.config.skip_merging:
            logger.info("Skipping merging (--skip_merging)")
            return True

        logger.info("=" * 80)
        logger.info("STEP 4/6: MODEL MERGING")
        logger.info("=" * 80)

        adapter_path = self.config.training_dir / "final_model"

        cmd = [
            sys.executable,
            "merge_lora_weights.py",
            "--base_model", self.config.base_model,
            "--adapter_path", str(adapter_path),
            "--output_path", str(self.config.merged_model_dir),
            "--create_model_card",
        ]

        return self.run_command(cmd, "merging")

    def step_5_testing(self) -> bool:
        """Step 5: Run inference tests on merged model."""
        if self.config.skip_testing:
            logger.info("Skipping testing (--skip_testing)")
            return True

        logger.info("=" * 80)
        logger.info("STEP 5/6: INFERENCE TESTING")
        logger.info("=" * 80)

        cmd = [
            sys.executable,
            "inference_test.py",
            "--model_path", str(self.config.merged_model_dir),
        ]

        return self.run_command(cmd, "testing")

    def step_6_summary(self):
        """Step 6: Generate pipeline summary."""
        logger.info("=" * 80)
        logger.info("STEP 6/6: GENERATING SUMMARY")
        logger.info("=" * 80)

        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_duration_hours"] = (time.time() - self.start_time) / 3600

        # Save summary
        with open(self.config.pipeline_summary, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"✓ Pipeline summary saved: {self.config.pipeline_summary}")

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Base Model: {self.results['base_model']}")
        print(f"Total Duration: {self.results['total_duration_hours']:.2f} hours")
        print(f"Output Directory: {self.config.output_dir}")
        print("\nStep Results:")
        print("-" * 80)

        for step, info in self.results["steps"].items():
            status_icon = "✓" if info["status"] == "success" else "✗"
            print(f"{status_icon} {step:<25} {info['duration_hours']:.2f}h   {info['status'].upper()}")

        if self.results["final_metrics"]:
            print("\nFinal Metrics:")
            print("-" * 80)
            overall_score = self.results["final_metrics"].get("overall_score", 0)
            print(f"Overall Score: {overall_score:.2f}%")

            for agent, metrics in self.results["final_metrics"].items():
                if agent == "overall_score":
                    continue
                print(f"\n{agent.upper()}:")
                for key, value in metrics.items():
                    if isinstance(value, float) and "percent" in key:
                        print(f"  {key}: {value:.2f}%")

        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"\nMerged Model: {self.config.merged_model_dir}")
        print(f"Summary: {self.config.pipeline_summary}")
        print("\nNext Steps:")
        print("  1. Test model: python inference_test.py --model_path ./pipeline_output/merged_model --interactive")
        print("  2. Deploy model to your agent system")
        print("  3. Monitor performance and collect feedback for further fine-tuning")
        print("=" * 80)

    def run(self):
        """Execute the full pipeline."""
        logger.info("=" * 80)
        logger.info("MULTI-AGENT LLM FINE-TUNING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Base Model: {self.config.base_model}")
        logger.info(f"Output Directory: {self.config.output_dir}")
        logger.info(f"Training Samples: {self.config.num_samples}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info("=" * 80)

        self.start_time = time.time()
        self.results["start_time"] = datetime.now().isoformat()

        # Execute steps
        steps = [
            (self.step_1_data_generation, "Data Generation"),
            (self.step_2_training, "Training"),
            (self.step_3_evaluation, "Evaluation"),
            (self.step_4_merging, "Merging"),
            (self.step_5_testing, "Testing"),
        ]

        for step_func, step_name in steps:
            success = step_func()
            if not success:
                logger.error(f"Pipeline failed at: {step_name}")
                logger.error("Check logs for details. You can resume with --skip_* flags.")
                self.step_6_summary()
                return False

        # Generate summary
        self.step_6_summary()

        return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for multi-agent LLM fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example usage:
              # Full pipeline
              python run_full_pipeline.py --base_model Qwen/Qwen2.5-7B-Instruct --output_dir ./output

              # Resume from training (skip data generation)
              python run_full_pipeline.py --output_dir ./output --skip_data_gen

              # Only evaluate and merge (skip training)
              python run_full_pipeline.py --output_dir ./output --skip_data_gen --skip_training
        """
    )

    # Required
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./pipeline_output",
                        help="Output directory for all pipeline artifacts")

    # Data generation
    parser.add_argument("--num_samples", type=int, default=8000,
                        help="Number of training samples to generate")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--lora_r", type=int, default=128,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=256,
                        help="LoRA alpha")

    # Skip flags
    parser.add_argument("--skip_data_gen", action="store_true",
                        help="Skip data generation")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--skip_merging", action="store_true",
                        help="Skip merging")
    parser.add_argument("--skip_testing", action="store_true",
                        help="Skip testing")

    args = parser.parse_args()

    # Validate
    if not os.path.exists("data_generation.py"):
        logger.error("Required scripts not found. Ensure all pipeline scripts are in the current directory.")
        sys.exit(1)

    # Create config and run
    config = PipelineConfig(args)
    runner = PipelineRunner(config)

    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()