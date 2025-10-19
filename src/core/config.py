"""
Configuration Management System
Centralized configuration with environment variable support and validation
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger('app')


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    gpu_memory_gb: int = 48
    system_memory_gb: int = 100
    num_gpus: int = 1
    device: str = "cuda"
    use_flash_attention: bool = True

    def __post_init__(self):
        """Validate hardware config."""
        if self.gpu_memory_gb < 24:
            logger.warning(f"GPU memory ({self.gpu_memory_gb}GB) is below recommended 48GB")


@dataclass
class ModelConfig:
    """Model configuration."""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_type: str = "causal_lm"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    num_train_samples: int = 8000
    train_split: float = 0.9
    max_seq_length: int = 4096

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03

    # Optimizer
    optim: str = "paged_adamw_8bit"

    # Memory optimization
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict = field(default_factory=lambda: {"use_reentrant": False})

    # Logging and checkpointing
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Seed
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_batch_size: int = 8
    eval_sample_size: int = 100  # Samples per agent type

    # Metrics thresholds
    min_json_validity: float = 0.85
    min_mdx_syntax_validity: float = 0.90
    min_overall_score: float = 0.85

    # Evaluation frequency
    eval_every_n_epochs: int = 1


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Inference settings
    use_vllm: bool = True  # Use vLLM for production
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9

    # Model serving
    max_concurrent_requests: int = 100
    request_timeout: int = 60

    # Quantization for inference
    inference_quantization: Optional[str] = "8bit"  # None, "8bit", or "4bit"

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000


@dataclass
class FeedbackConfig:
    """Feedback collection configuration."""
    # Collection settings
    collect_all_interactions: bool = True
    collect_errors_only: bool = False

    # Feedback triggers
    ask_feedback_probability: float = 0.1  # 10% of requests

    # Storage
    feedback_db_path: str = "data/feedback/feedback.db"

    # Retraining triggers
    min_feedback_samples: int = 500
    retraining_interval_days: int = 14  # 2 weeks

    # Quality filters
    min_feedback_quality_score: float = 0.7


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_prometheus: bool = True
    prometheus_port: int = 9090

    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Performance tracking
    track_latency: bool = True
    track_token_usage: bool = True
    track_error_rate: bool = True

    # Alerts
    alert_on_high_latency: bool = True
    latency_threshold_seconds: float = 5.0
    alert_on_error_rate: bool = True
    error_rate_threshold: float = 0.05  # 5%


@dataclass
class PipelineConfig:
    """Pipeline orchestration configuration."""
    # Automated retraining
    enable_auto_retraining: bool = True
    retraining_schedule_cron: str = "0 2 * * 0"  # Every Sunday at 2 AM

    # Pipeline steps
    run_data_generation: bool = True
    run_training: bool = True
    run_evaluation: bool = True
    run_deployment: bool = True

    # Rollback settings
    enable_auto_rollback: bool = True
    rollback_on_performance_drop: bool = True
    max_performance_drop_percent: float = 10.0

    # Notifications
    notify_on_completion: bool = True
    notify_on_failure: bool = True
    notification_email: Optional[str] = None
    notification_webhook: Optional[str] = None


class Config:
    """Master configuration class."""

    def __init__(self, config_dir: str = "config"):
        """Initialize configuration from YAML files and environment variables."""
        self.config_dir = Path(config_dir)

        # Load configurations
        self.hardware = self._load_config("hardware", HardwareConfig)
        self.model = self._load_config("model", ModelConfig)
        self.training = self._load_config("training", TrainingConfig)
        self.evaluation = self._load_config("evaluation", EvaluationConfig)
        self.deployment = self._load_config("deployment", DeploymentConfig)
        self.feedback = self._load_config("feedback", FeedbackConfig)
        self.monitoring = self._load_config("monitoring", MonitoringConfig)
        self.pipeline = self._load_config("pipeline", PipelineConfig)

        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.model_registry  = self.project_root / "model_registry"

        # Ensure directories exist
        self._create_directories()

        # Validate configuration
        self._validate()

        logger.info("Configuration loaded successfully")

    def _load_config(self, config_name: str, config_class: type) -> Any:
        """Load configuration from YAML file and environment variables."""
        yaml_path = self.config_dir / f"{config_name}_config.yaml"

        # Default values from dataclass
        config_dict = {}

        # Load from YAML if exists
        if yaml_path.exists():
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f) or {}
                config_dict.update(yaml_config)

        # Override with environment variables
        env_prefix = f"{config_name.upper()}_"
        for key in config_class.__dataclass_fields__.keys():
            env_key = f"{env_prefix}{key.upper()}"
            if env_key in os.environ:
                config_dict[key] = self._parse_env_value(
                    os.environ[env_key],
                    config_class.__dataclass_fields__[key].type
                )

        return config_class(**config_dict)

    @staticmethod
    def _parse_env_value(value: str, target_type: type) -> Any:
        """Parse environment variable to target type."""
        if target_type == bool:
            return value.lower() in ("true", "1", "yes")
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        else:
            return value

    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "feedback",
            self.data_dir / "validation",
            self.models_dir / "base",
            self.models_dir / "checkpoints",
            self.models_dir / "fine_tuned" / "latest",
            self.logs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _validate(self):
        """Validate configuration."""
        # Check hardware
        if self.hardware.gpu_memory_gb < 24:
            logger.warning("GPU memory below 24GB may cause OOM during training")

        # Check model compatibility
        if self.model.use_4bit and self.training.per_device_train_batch_size > 8:
            logger.warning("Large batch size with 4-bit may cause memory issues")

        # Check training settings
        effective_batch_size = (
                self.training.per_device_train_batch_size *
                self.training.gradient_accumulation_steps *
                self.hardware.num_gpus
        )
        if effective_batch_size < 16:
            logger.warning(f"Effective batch size ({effective_batch_size}) is small")

        # Check deployment
        if self.deployment.use_vllm and self.deployment.api_workers > 1:
            logger.warning("vLLM works best with single worker process")

    def save(self, output_dir: Optional[str] = None):
        """Save current configuration to YAML files."""
        output_dir = Path(output_dir) if output_dir else self.config_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        configs = {
            "hardware": self.hardware,
            "model": self.model,
            "training": self.training,
            "evaluation": self.evaluation,
            "deployment": self.deployment,
            "feedback": self.feedback,
            "monitoring": self.monitoring,
            "pipeline": self.pipeline,
        }

        for name, config in configs.items():
            output_path = output_dir / f"{name}_config.yaml"
            with open(output_path, "w") as f:
                yaml.dump(config.__dict__, f, default_flow_style=False)

        logger.info(f"Configuration saved to {output_dir}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "hardware": self.hardware.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "deployment": self.deployment.__dict__,
            "feedback": self.feedback.__dict__,
            "monitoring": self.monitoring.__dict__,
            "pipeline": self.pipeline.__dict__,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(base_model={self.model.base_model}, epochs={self.training.num_train_epochs})"


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(config_dir: str = "config", force_reload: bool = False) -> Config:
    """Get or create configuration singleton."""
    global _config_instance

    if _config_instance is None or force_reload:
        _config_instance = Config(config_dir)

    return _config_instance


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(config)
    print(f"\nBase Model: {config.model.base_model}")
    print(f"Training Epochs: {config.training.num_train_epochs}")
    print(f"GPU Memory: {config.hardware.gpu_memory_gb}GB")
    print(f"Use vLLM: {config.deployment.use_vllm}")