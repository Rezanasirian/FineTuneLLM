"""
QLoRA Fine-Tuning Script for Multi-Agent Financial LLM
Optimized for 48GB GPU (A100/RTX 6000 Ada)

Usage:
    python finetune_qlora.py --data_dir ./data --output_dir ./output --model_name Qwen/Qwen2.5-7B-Instruct

"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FineTuningConfig:
    """Configuration for fine-tuning."""

    def __init__(self, args):
        # Model
        self.model_name = args.model_name
        self.load_in_4bit = True
        self.bnb_4bit_compute_dtype = torch.bfloat16
        self.bnb_4bit_quant_type = "nf4"  # NormalFloat4
        self.bnb_4bit_use_double_quant = True

        # LoRA
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = 0.05
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # MLP
        ]

        # Training
        self.num_train_epochs = args.epochs
        self.per_device_train_batch_size = args.batch_size
        self.gradient_accumulation_steps = args.grad_accum
        self.learning_rate = args.learning_rate
        self.warmup_ratio = 0.03
        self.lr_scheduler_type = "cosine"
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.optim = "paged_adamw_8bit"  # Memory efficient

        # Memory optimization
        self.gradient_checkpointing = True
        self.max_seq_length = 4096

        # Logging & Checkpointing
        self.logging_steps = 50
        self.save_steps = 500
        self.eval_steps = 500
        self.save_total_limit = 3
        self.evaluation_strategy = "steps"
        self.load_best_model_at_end = True
        self.metric_for_best_model = "eval_loss"

        # Paths
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.seed = args.seed


class MemoryMonitorCallback(TrainerCallback):
    """Monitor GPU memory usage during training."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(
                f"Step {state.global_step}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def load_model_and_tokenizer(config: FineTuningConfig):
    """Load model with 4-bit quantization and tokenizer."""
    logger.info(f"Loading model: {config.model_name}")

    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",  # For training
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Model loaded: {model.dtype}, Vocab size: {tokenizer.vocab_size}")

    # Print model size
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model size: {all_params / 1e9:.2f}B parameters")
    logger.info(f"Trainable (before LoRA): {trainable_params / 1e6:.2f}M parameters")

    return model, tokenizer


def setup_lora(model, config: FineTuningConfig):
    """Apply LoRA adapters to model."""
    logger.info("Setting up LoRA configuration")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params

    logger.info(f"LoRA applied:")
    logger.info(f"  Trainable: {trainable_params / 1e6:.2f}M ({trainable_percent:.2f}%)")
    logger.info(f"  Total: {all_params / 1e9:.2f}B")

    return model


def load_datasets(config: FineTuningConfig):
    """Load train and validation datasets."""
    logger.info(f"Loading datasets from {config.data_dir}")

    data_files = {
        "train": str(Path(config.data_dir) / "train.jsonl"),
        "validation": str(Path(config.data_dir) / "validation.jsonl"),
    }

    dataset = load_dataset("json", data_files=data_files)

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")

    return dataset


def format_chat_template(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """Format example using chat template."""
    # Apply tokenizer's chat template
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train_model(config: FineTuningConfig):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Starting Multi-Agent LLM Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    logger.info("=" * 60)

    # Set seed
    torch.manual_seed(config.seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Apply LoRA
    model = setup_lora(model, config)

    # Load datasets
    dataset = load_datasets(config)

    # Format datasets
    logger.info("Formatting datasets with chat template...")
    dataset = dataset.map(
        lambda ex: format_chat_template(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy=config.evaluation_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=False,
        bf16=True,  # Use bfloat16 for training
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="none",  # Change to "wandb" for experiment tracking
        seed=config.seed,
    )

    # Data collator (for chat format)
    # This ensures we only compute loss on assistant responses
    response_template = "<|im_start|>assistant"  # Qwen2.5 format
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=config.max_seq_length,
        callbacks=[MemoryMonitorCallback()],
    )

    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    logger.info(f"Training completed in {duration:.2f} hours")

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(Path(config.output_dir) / "final_model")
    tokenizer.save_pretrained(Path(config.output_dir) / "final_model")

    # Save metrics
    metrics_path = Path(config.output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "final_train_loss": trainer.state.log_history[-1].get("train_loss"),
            "final_eval_loss": trainer.state.log_history[-1].get("eval_loss"),
            "total_steps": trainer.state.global_step,
            "training_hours": duration,
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("✓ Fine-tuning complete!")
    logger.info(f"Model saved to: {config.output_dir}/final_model")
    logger.info("=" * 60)

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train.jsonl and validation.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save fine-tuned model")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=128,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=256,
                        help="LoRA alpha (scaling factor)")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Validate GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires a GPU.")
        sys.exit(1)

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {gpu_memory:.1f}GB")

    if gpu_memory < 40:
        logger.warning(f"GPU memory ({gpu_memory:.1f}GB) is below recommended 48GB. Training may fail.")

    # Create config and train
    config = FineTuningConfig(args)
    train_model(config)


if __name__ == "__main__":
    main()