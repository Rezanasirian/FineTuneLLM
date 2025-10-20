"""
Automated Training Pipeline
Handles data generation, training, evaluation, and model registration
"""


import json
from src.core.logger import get_logger
from typing import Dict, Any, Tuple
from datetime import datetime


import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.core.config import get_config
from src.data.generator import SyntheticDataGenerator
from src.evaluation.evaluator import ModelEvaluator
from src.services.model_registry import ModelRegistry

logger = get_logger("training_pipline")


class TrainingPipeline:
    """Automated training pipeline for multi-agent LLM."""

    def __init__(self, config_dir: str = "config"):
        """Initialize training pipeline."""
        self.config = get_config(config_dir)
        self.registry = ModelRegistry(registry_dir=self.config.model_registry)

        # Pipeline state
        self.current_version = None
        self.training_metadata = {}

        logger.info("Training pipeline initialized")

    def run(
            self,
            use_existing_data: bool = False,
            skip_evaluation: bool = False,
            register_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.

        Args:
            use_existing_data: Use existing processed data instead of generating new
            skip_evaluation: Skip evaluation step
            register_model: Register model in registry after training

        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)

        start_time = datetime.now()
        results = {
            "pipeline_start": start_time.isoformat(),
            "config": self.config.to_dict(),
            "steps": {},
        }

        try:
            # Step 1: Data Generation/Loading
            if not use_existing_data:
                logger.info("\n[STEP 1/5] Generating training data...")
                data_result = self._generate_data()
                results["steps"]["data_generation"] = data_result
            else:
                logger.info("\n[STEP 1/5] Using existing data...")
                data_result = {"status": "skipped", "reason": "use_existing_data=True"}
                results["steps"]["data_generation"] = data_result

            # Step 2: Load and prepare datasets
            logger.info("\n[STEP 2/5] Loading datasets...")
            train_dataset, eval_dataset = self._load_datasets()
            results["steps"]["data_loading"] = {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
            }

            # Step 3: Training
            logger.info("\n[STEP 3/5] Training model...")
            training_result = self._train_model(train_dataset, eval_dataset)
            results["steps"]["training"] = training_result

            # Step 4: Evaluation
            if not skip_evaluation:
                logger.info("\n[STEP 4/5] Evaluating model...")
                eval_result = self._evaluate_model()
                results["steps"]["evaluation"] = eval_result

                # Check if model meets quality threshold
                if not self._check_quality_threshold(eval_result):
                    logger.warning("Model does not meet quality threshold!")
                    results["quality_check"] = "FAILED"
                    return results
            else:
                logger.info("\n[STEP 4/5] Skipping evaluation...")
                results["steps"]["evaluation"] = {"status": "skipped"}

            # Step 5: Model Registration
            if register_model:
                logger.info("\n[STEP 5/5] Registering model...")
                registration_result = self._register_model(results)
                results["steps"]["registration"] = registration_result
                results["model_version"] = self.current_version
            else:
                logger.info("\n[STEP 5/5] Skipping registration...")
                results["steps"]["registration"] = {"status": "skipped"}

            # Pipeline completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 3600
            results["pipeline_end"] = end_time.isoformat()
            results["duration_hours"] = duration
            results["status"] = "SUCCESS"

            logger.info("=" * 80)
            logger.info(f"✓ PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f} hours")
            logger.info("=" * 80)

            # Save results
            self._save_pipeline_results(results)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results["status"] = "FAILED"
            results["error"] = str(e)
            self._save_pipeline_results(results)
            raise

    def _generate_data(self) -> Dict[str, Any]:
        """Generate synthetic training data."""
        generator = SyntheticDataGenerator(seed=self.config.training.seed)

        # Generate examples
        generator.generate_all_examples(total_samples=self.config.training.num_train_samples)

        # Save datasets
        output_dir = self.config.data_dir / "processed"
        generator.save_datasets(str(output_dir), train_split=self.config.training.train_split)

        return {
            "status": "success",
            "total_examples": len(generator.examples),
            "output_dir": str(output_dir),
        }

    def _load_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load train and validation datasets."""
        data_files = {
            "train": str(self.config.data_dir / "processed" / "train.jsonl"),
            "validation": str(self.config.data_dir / "processed" / "validation.jsonl"),
        }

        dataset = load_dataset("json", data_files=data_files)
        return dataset["train"], dataset["validation"]

    def _train_model(self, train_dataset: Dataset, eval_dataset: Dataset) -> Dict[str, Any]:
        """Train model with QLoRA."""
        logger.info(f"Loading base model: {self.config.model.base_model}")

        # BitsAndBytes config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.model.use_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.model.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            attn_implementation=self.config.model.attn_implementation if self.config.hardware.use_flash_attention else None,
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.config.training.gradient_checkpointing
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side="right",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        # Apply LoRA
        if self.config.model.use_lora:
            lora_config = LoraConfig(
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                target_modules=self.config.model.lora_target_modules,
                lora_dropout=self.config.model.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        # Format datasets
        def format_chat(example):
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}

        train_dataset = train_dataset.map(format_chat, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(format_chat, remove_columns=eval_dataset.column_names)

        # Training arguments
        output_dir = self.config.models_dir / "checkpoints" / datetime.now().strftime("%Y%m%d_%H%M%S")

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            optim=self.config.training.optim,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy="steps",
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs=self.config.training.gradient_checkpointing_kwargs,
            report_to=["none"],
            seed=self.config.training.seed,
        )

        # Data collator
        response_template = "<|im_start|>assistant"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )

        # Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            max_seq_length=self.config.training.max_seq_length,
        )

        # Train
        train_result = trainer.train()

        # Save final model
        final_model_path = self.config.models_dir / "fine_tuned" / "latest"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))

        # Store for later steps
        self.training_metadata["model_path"] = str(final_model_path)
        self.training_metadata["checkpoint_dir"] = str(output_dir)

        return {
            "status": "success",
            "final_train_loss": train_result.training_loss,
            "model_path": str(final_model_path),
            "checkpoint_dir": str(output_dir),
        }

    def _evaluate_model(self) -> Dict[str, Any]:
        """Evaluate trained model."""
        evaluator = ModelEvaluator(
            model_path=self.training_metadata["model_path"],
            device=self.config.hardware.device,
        )

        data_dir = str(self.config.data_dir / "processed")
        results = evaluator.evaluate_all(
            data_dir=data_dir,
            sample_size=self.config.evaluation.eval_sample_size,
        )

        return results

    def _check_quality_threshold(self, eval_result: Dict[str, Any]) -> bool:
        """Check if model meets quality thresholds."""
        overall_score = eval_result.get("overall_score", 0)

        if overall_score < self.config.evaluation.min_overall_score * 100:
            logger.warning(
                f"Overall score {overall_score:.2f}% below threshold {self.config.evaluation.min_overall_score * 100}%")
            return False

        # Check agent-specific metrics
        for agent, metrics in eval_result.items():
            if agent == "overall_score":
                continue

            if agent == "mdx_generation":
                mdx_valid = metrics.get("mdx_syntax_valid_percent", 0)
                if mdx_valid < self.config.evaluation.min_mdx_syntax_validity * 100:
                    logger.warning(f"MDX syntax validity {mdx_valid:.2f}% below threshold")
                    return False

        return True

    def _register_model(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Register model in model registry."""
        version = self.registry.register_model(
            model_path=self.training_metadata["model_path"],
            metrics=results["steps"].get("evaluation", {}),
            config=self.config.to_dict(),
            training_results=results["steps"].get("training", {}),
        )

        self.current_version = version

        return {
            "status": "success",
            "version": version,
        }

    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file."""
        results_dir = self.config.logs_dir / "pipeline_runs"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"training_pipeline_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Pipeline results saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--config-dir", type=str, default="config", help="Configuration directory")
    parser.add_argument("--use-existing-data", action="store_true", help="Use existing data")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--no-register", action="store_true", help="Don't register model")

    args = parser.parse_args()

    pipeline = TrainingPipeline(config_dir=args.config_dir)
    results = pipeline.run(
        use_existing_data=args.use_existing_data,
        skip_evaluation=args.skip_evaluation,
        register_model=not args.no_register,
    )

    print("\n" + "=" * 80)
    print("PIPELINE RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2, ensure_ascii=False))