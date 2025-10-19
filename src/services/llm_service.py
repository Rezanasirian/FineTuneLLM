import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from src.core.logger import get_logger

logger = get_logger("llm_service")


class FineTunedLLMService:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()
        logger.info("Model loaded successfully")

    def generate(self, messages: List[Dict], max_tokens: int = 2048,
                 temperature: float = 0.1) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=temperature > 0, pad_token_id=self.tokenizer.pad_token_id
            )

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


_llm_service = None


def get_llm_service(model_path: str = None):
    global _llm_service
    if _llm_service is None:
        if model_path is None:
            raise ValueError("model_path required for first initialization")
        _llm_service = FineTunedLLMService(model_path)
    return _llm_service