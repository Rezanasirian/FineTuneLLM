"""
Merge LoRA Adapters into Base Model
Creates a standalone model without requiring PEFT at inference time

Usage:
    python merge_lora_weights.py --adapter_path ./output/final_model --output_path ./merged_model

Estimated runtime: 10-20 minutes
Output: Merged model ready for deployment (~14GB on disk for 7B model)
"""
import sys
from pathlib import Path
import argparse
import shutil

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.core.logger import get_logger
logger = get_logger("merger")

def merge_lora_weights(
        base_model_name: str,
        adapter_path: str,
        output_path: str,
        device: str = "cuda",
        push_to_hub: bool = False,
        hub_model_id: str = None,
):
    """
    Merge LoRA adapters into base model and save as standalone model.

    Args:
        base_model_name: Name or path of base model (e.g., "Qwen/Qwen2.5-7B-Instruct")
        adapter_path: Path to fine-tuned model with LoRA adapters
        output_path: Directory to save merged model
        device: Device to load model on
        push_to_hub: Whether to push to Hugging Face Hub
        hub_model_id: Model ID for Hugging Face Hub
    """
    logger.info("=" * 60)
    logger.info("Merging LoRA Adapters into Base Model")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Output path: {output_path}")
    logger.info("=" * 60)

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base model
    logger.info("Step 1/4: Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    logger.info(f"Base model loaded: {base_model.dtype}")

    # Step 2: Load LoRA adapters
    logger.info("Step 2/4: Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
    )
    logger.info("LoRA adapters loaded")

    # Step 3: Merge adapters into base model
    logger.info("Step 3/4: Merging adapters (this may take several minutes)...")
    model = model.merge_and_unload()
    logger.info("✓ Merge complete")

    # Step 4: Save merged model
    logger.info("Step 4/4: Saving merged model...")
    model.save_pretrained(
        output_path,
        safe_serialization=True,  # Save in safetensors format
        max_shard_size="5GB",
    )
    logger.info(f"✓ Model saved to: {output_path}")

    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_path)
    logger.info(f"✓ Tokenizer saved to: {output_path}")

    # Copy config files
    logger.info("Copying configuration files...")
    for config_file in ["config.json", "generation_config.json", "tokenizer_config.json"]:
        src = Path(adapter_path) / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)
            logger.info(f"  Copied {config_file}")

    # Calculate model size
    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    size_gb = total_size / (1024 ** 3)
    logger.info(f"Total model size: {size_gb:.2f} GB")

    # Optional: Push to Hub
    if push_to_hub and hub_model_id:
        logger.info(f"Pushing to Hugging Face Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        logger.info("✓ Model pushed to Hub")

    logger.info("=" * 60)
    logger.info("✓ Merge complete!")
    logger.info(f"Merged model ready at: {output_path}")
    logger.info("=" * 60)

    return str(output_path)


def create_model_card(output_path: str, base_model: str, training_info: dict = None):
    """Create a model card (README.md) for the merged model."""
    model_card = f"""# Fine-Tuned Multi-Agent Financial LLM

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) for multi-agent financial analysis tasks.

## Model Description

This model has been fine-tuned using QLoRA to handle multiple specialized agents:
- **Query Understanding**: Parse Persian financial queries into structured JSON
- **MDX Generation**: Generate OLAP queries from structured specifications
- **Data Analysis**: Analyze financial data and extract insights in Persian
- **Visualization**: Suggest appropriate chart types for data
- **Error Resolution**: Fix MDX query errors

## Training Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: QLoRA (4-bit quantization)
- **LoRA Config**: r=128, alpha=256
- **Training Data**: 8,000 synthetic examples across 5 agent types
- **GPU**: Single 48GB GPU (A100)
- **Training Time**: ~10 hours

## Intended Use

This model is designed for financial data analysis systems that need to:
1. Understand user queries in Persian
2. Generate OLAP (MDX) queries
3. Analyze financial data
4. Suggest visualizations

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "path/to/merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="bfloat16",
    trust_remote_code=True
)

# Example: Query Understanding
messages = [
    {{"role": "system", "content": "شما متخصص درک پرس‌وجوی مالی هستید."}},
    {{"role": "user", "content": "موجودی نقد شرکت در سال ۱۴۰۳ چقدر است؟"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Limitations

- Optimized for Persian financial domain
- Requires specific input formats (see agent prompts)
- May not generalize to other domains without additional fine-tuning

## License

Follows the license of the base model: {base_model}
"""

    readme_path = Path(output_path) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    logger.info(f"✓ Model card saved to: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to fine-tuned model with LoRA adapters")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save merged model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for merging (cuda or cpu)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push merged model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--create_model_card", action="store_true",
                        help="Create a model card (README.md)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.adapter_path).exists():
        logger.error(f"Adapter path does not exist: {args.adapter_path}")
        sys.exit(1)

    if args.push_to_hub and not args.hub_model_id:
        logger.error("--hub_model_id is required when --push_to_hub is set")
        sys.exit(1)

    # Merge model
    try:
        merged_path = merge_lora_weights(
            base_model_name=args.base_model,
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            device=args.device,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )

        # Create model card
        if args.create_model_card:
            create_model_card(merged_path, args.base_model)

        logger.info("\n✓ All done! Your model is ready for deployment.")

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise


if __name__ == "__main__":
    main()