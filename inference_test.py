"""
Inference & Testing Script for Fine-Tuned Multi-Agent LLM
Test the model on sample inputs from each agent type

Usage:
    python inference_test.py --model_path ./merged_model --interactive

Modes:
    - Batch: Run pre-defined test cases
    - Interactive: Enter queries manually
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiAgentInference:
    """Inference engine for multi-agent LLM."""

    def __init__(self, model_path: str, device: str = "cuda", use_streaming: bool = False):
        self.device = device
        self.use_streaming = use_streaming

        logger.info(f"Loading model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model.eval()
        logger.info("✓ Model loaded successfully")

        if self.use_streaming:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            self.streamer = None

    def generate(
            self,
            messages: List[Dict[str, str]],
            max_new_tokens: int = 2048,
            temperature: float = 0.1,
            top_p: float = 0.9,
    ) -> str:
        """Generate response for given messages."""
        # Format with chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=self.streamer if self.use_streaming else None,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def query_understanding(self, user_query: str) -> str:
        """Test Query Understanding agent."""
        messages = [
            {
                "role": "system",
                "content": """شما متخصص درک پرس‌وجوی مالی هستید. پرس‌وجوهای فارسی را به JSON ساختاری تبدیل کنید.
باید بعدها (dimensions)، زمان (time)، فیلترها (filters)، و معیارها (measures) را استخراج کنید.
خروجی باید JSON معتبر باشد."""
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        return self.generate(messages)

    def mdx_generation(self, query_understanding_output: str) -> str:
        """Test MDX Generation agent."""
        messages = [
            {
                "role": "system",
                "content": """شما متخصص تولید پرس‌وجوی MDX هستید. از JSON ورودی، پرس‌وجوی MDX معتبر بسازید.
همیشه از فرمت [Dimension].[Level].&[ID] استفاده کنید.
خروجی باید JSON با کلید "mdx_query" باشد."""
            },
            {
                "role": "user",
                "content": f"QUERY UNDERSTANDING OUTPUT:\n{query_understanding_output}"
            }
        ]

        return self.generate(messages)

    def data_analysis(self, user_query: str, query_results: str) -> str:
        """Test Data Analysis agent."""
        messages = [
            {
                "role": "system",
                "content": """شما متخصص تحلیل داده مالی هستید. نتایج پرس‌وجو را تحلیل کنید و بینش‌های معنادار استخراج کنید.
خروجی باید JSON با کلیدهای "insights" و "summary" به فارسی باشد."""
            },
            {
                "role": "user",
                "content": f"User Query: {user_query}\nQuery Results: {query_results}"
            }
        ]

        return self.generate(messages, max_new_tokens=3072)

    def visualization(self, user_query: str, data_sample: str) -> str:
        """Test Visualization agent."""
        messages = [
            {
                "role": "system",
                "content": """شما متخصص انتخاب نمودار هستید. نوع نمودار مناسب را برای داده‌ها انتخاب کنید.
خروجی باید JSON با کلید "chart_type" باشد."""
            },
            {
                "role": "user",
                "content": f"User Query: {user_query}\nData Sample: {data_sample}"
            }
        ]

        return self.generate(messages)

    def mdx_error_resolution(self, error_msg: str, original_mdx: str) -> str:
        """Test MDX Error Resolution agent."""
        messages = [
            {
                "role": "system",
                "content": """شما متخصص رفع خطای MDX هستید. پرس‌وجوهای معیوب را تصحیح کنید.
خروجی باید JSON با کلید "mdx_query" باشد."""
            },
            {
                "role": "user",
                "content": f"Error Message: {error_msg}\nOriginal MDX:\n{original_mdx}"
            }
        ]

        return self.generate(messages)


def run_test_cases(inference: MultiAgentInference):
    """Run predefined test cases for all agents."""
    print("\n" + "=" * 80)
    print("RUNNING TEST CASES")
    print("=" * 80)

    # Test 1: Query Understanding
    print("\n" + "-" * 80)
    print("TEST 1: Query Understanding")
    print("-" * 80)
    test_query = "موجودی نقد نزد بانک ریالی شرکت احیا ریل ایرانیان در سال ۱۴۰۳ چقدر است؟"
    print(f"Input: {test_query}")
    print("\nOutput:")
    response = inference.query_understanding(test_query)
    print(response)

    # Validate JSON
    try:
        parsed = json.loads(response)
        print("\n✓ Valid JSON")
        print(f"Question Type: {parsed.get('question_type')}")
        print(f"Dimensions: {parsed.get('dimensions')}")
    except json.JSONDecodeError:
        print("\n✗ Invalid JSON")

    # Test 2: MDX Generation
    print("\n" + "-" * 80)
    print("TEST 2: MDX Generation")
    print("-" * 80)
    qu_output = json.dumps({
        "question_type": "data_retrieval",
        "dimensions": ["DimStandardAccount", "DimLevel", "DimDate"],
        "time": {"type": "year", "year": 1403, "breakdown_by": "none"},
        "filters": [
            {"dimension": "DimStandardAccount", "level": "DimStandardLedgerAccountSK", "operator": "=", "id": "1",
             "label": "موجودی نقد"},
            {"dimension": "DimLevel", "level": "DimLevelSK", "operator": "=", "id": "34", "label": "احیا ریل"}
        ],
        "measure": ["[Measures].[FactFinancail DebitBalance]"],
        "currency": "IRR",
        "scale": "unit"
    }, ensure_ascii=False, indent=2)

    print(f"Input (Query Understanding Output):\n{qu_output[:200]}...")
    print("\nOutput:")
    response = inference.mdx_generation(qu_output)
    print(response)

    try:
        parsed = json.loads(response)
        if "mdx_query" in parsed:
            print("\n✓ MDX query generated")
            mdx = parsed["mdx_query"]
            # Basic validation
            if "SELECT" in mdx and "FROM" in mdx and ".&[" in mdx:
                print("✓ Basic syntax check passed")
        else:
            print("\n✗ No mdx_query in response")
    except json.JSONDecodeError:
        print("\n✗ Invalid JSON")

    # Test 3: Data Analysis
    print("\n" + "-" * 80)
    print("TEST 3: Data Analysis")
    print("-" * 80)
    query_results = json.dumps([
        {"month": "140301", "value": 120000000, "formatted": "۱۲۰ میلیون ریال"},
        {"month": "140302", "value": 135000000, "formatted": "۱۳۵ میلیون ریال"},
        {"month": "140303", "value": 150000000, "formatted": "۱۵۰ میلیون ریال"},
    ], ensure_ascii=False)

    print(f"Input Query: هزینه‌های اداری سال ۱۴۰۳")
    print(f"Query Results:\n{query_results}")
    print("\nOutput:")
    response = inference.data_analysis("هزینه‌های اداری سال ۱۴۰۳", query_results)
    print(response)

    try:
        parsed = json.loads(response)
        if "insights" in parsed and "summary" in parsed:
            print(f"\n✓ Analysis complete")
            print(f"Number of insights: {len(parsed['insights'])}")
            print(f"Summary length: {len(parsed['summary'])} chars")
        else:
            print("\n✗ Missing insights or summary")
    except json.JSONDecodeError:
        print("\n✗ Invalid JSON")

    # Test 4: Visualization
    print("\n" + "-" * 80)
    print("TEST 4: Visualization")
    print("-" * 80)
    data_sample = json.dumps([
        {"month": 1, "value": 100}, {"month": 2, "value": 150},
        {"month": 3, "value": 130}, {"month": 4, "value": 180}
    ], ensure_ascii=False)

    print(f"Input Query: نمایش روند ماهانه")
    print(f"Data Sample: {data_sample}")
    print("\nOutput:")
    response = inference.visualization("نمایش روند ماهانه", data_sample)
    print(response)

    try:
        parsed = json.loads(response)
        if "chart_type" in parsed:
            print(f"\n✓ Chart type suggested: {parsed['chart_type']}")
        else:
            print("\n✗ No chart_type in response")
    except json.JSONDecodeError:
        print("\n✗ Invalid JSON")

    # Test 5: MDX Error Resolution
    print("\n" + "-" * 80)
    print("TEST 5: MDX Error Resolution")
    print("-" * 80)
    error_mdx = """SELECT [Measures].[FactFinancail DebitBalance] ON COLUMNS
FROM [ActualCPMDataCube]
WHERE ([DimLevel].[DimLevelSK]&[34])"""

    print(f"Error: Missing period before &")
    print(f"Original MDX:\n{error_mdx}")
    print("\nOutput:")
    response = inference.mdx_error_resolution("Member not found", error_mdx)
    print(response)

    try:
        parsed = json.loads(response)
        if "mdx_query" in parsed and ".&[" in parsed["mdx_query"]:
            print("\n✓ MDX corrected (period added)")
        else:
            print("\n✗ Error not resolved")
    except json.JSONDecodeError:
        print("\n✗ Invalid JSON")

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)


def interactive_mode(inference: MultiAgentInference):
    """Interactive mode for testing."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nAvailable agents:")
    print("  1. Query Understanding (qu)")
    print("  2. MDX Generation (mdx)")
    print("  3. Data Analysis (analysis)")
    print("  4. Visualization (viz)")
    print("  5. MDX Error Resolution (fix)")
    print("\nCommands:")
    print("  'exit' or 'quit' - Exit interactive mode")
    print("  'help' - Show this help")
    print("=" * 80)

    while True:
        print("\n" + "-" * 80)
        agent = input("Select agent (qu/mdx/analysis/viz/fix) or 'exit': ").strip().lower()

        if agent in ['exit', 'quit']:
            print("Exiting interactive mode.")
            break

        if agent == 'help':
            print("Available commands: qu, mdx, analysis, viz, fix, exit")
            continue

        if agent == 'qu':
            query = input("Enter query (Persian): ").strip()
            if query:
                print("\nGenerating response...")
                response = inference.query_understanding(query)
                print(f"\nResponse:\n{response}")

        elif agent == 'mdx':
            print("Enter Query Understanding JSON output (end with empty line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            qu_output = "\n".join(lines)
            if qu_output:
                print("\nGenerating MDX...")
                response = inference.mdx_generation(qu_output)
                print(f"\nResponse:\n{response}")

        elif agent == 'analysis':
            query = input("Enter query: ").strip()
            print("Enter query results JSON (end with empty line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            results = "\n".join(lines)
            if query and results:
                print("\nGenerating analysis...")
                response = inference.data_analysis(query, results)
                print(f"\nResponse:\n{response}")

        elif agent == 'viz':
            query = input("Enter query: ").strip()
            data = input("Enter data sample JSON: ").strip()
            if query and data:
                print("\nSuggesting chart...")
                response = inference.visualization(query, data)
                print(f"\nResponse:\n{response}")

        elif agent == 'fix':
            error = input("Enter error message: ").strip()
            print("Enter original MDX (end with empty line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            mdx = "\n".join(lines)
            if error and mdx:
                print("\nResolving error...")
                response = inference.mdx_error_resolution(error, mdx)
                print(f"\nResponse:\n{response}")

        else:
            print(f"Unknown agent: {agent}. Type 'help' for available commands.")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned multi-agent LLM")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned or merged model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming output")

    args = parser.parse_args()

    # Validate model path
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Initialize inference engine
    inference = MultiAgentInference(
        model_path=args.model_path,
        device=args.device,
        use_streaming=args.streaming,
    )

    if args.interactive:
        # Interactive mode
        interactive_mode(inference)
    else:
        # Run test cases
        run_test_cases(inference)


if __name__ == "__main__":
    main()