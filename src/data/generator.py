"""
Synthetic Data Generation Pipeline for Multi-Agent LLM Fine-Tuning
Generates training examples from agent specification documents.

Usage:
    python data_generation.py --output_dir ./data --num_samples 8000

Estimated runtime: 30-60 minutes on CPU
Output: train.jsonl (7200 samples), validation.jsonl (800 samples)
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import argparse


@dataclass
class AgentExample:
    """Structure for a single training example."""
    system_prompt: str
    user_input: str
    assistant_output: str
    agent_type: str


class SyntheticDataGenerator:
    """Generates synthetic training data from agent specifications."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.examples: List[AgentExample] = []

        # Persian fiscal years and common terms
        self.years = list(range(1400, 1405))
        self.months = list(range(1, 13))
        self.quarters = [1, 2, 3, 4]

        # Sample account names (from documents)
        self.accounts = [
            ("موجودی نقد نزد بانک ریالی", "1", "DimStandardLedgerAccountSK", "D", "B"),
            ("هزینه های اداری و عمومی", "42", "DimStandardGeneralAccountSK", "D", "A"),
            ("درآمد عملیاتی", "6", "DimStandardGeneralAccountSK", "C", "A"),
            ("سپرده حسن انجام کار", "27", "DimStandardLedgerAccountSK", "D", "B"),
            ("کل درآمد", "6", "DimStandardGeneralAccountSK", "C", "A"),
        ]

        # Sample organizations
        self.orgs = [
            ("شرکت احیا ریل ایرانیان", "34"),
            ("شرکت معادن منگنز ایران", "16"),
        ]

        # Measure mappings (from spec)
        self.measures = {
            ("D", "B"): "[Measures].[FactFinancail DebitBalance]",
            ("C", "B"): "[Measures].[FactFinancail CreditBalance]",
            ("D", "A"): "[Measures].[FactFinancail DebitValue]",
            ("C", "A"): "[Measures].[FactFinancail CreditValue]",
        }

    def generate_query_understanding_examples(self, n: int = 1500) -> None:
        """Generate Query Understanding agent examples."""
        system_prompt = """شما متخصص درک پرس‌وجوی مالی هستید. پرس‌وجوهای فارسی را به JSON ساختاری تبدیل کنید.
باید بعدها (dimensions)، زمان (time)، فیلترها (filters)، و معیارها (measures) را استخراج کنید.
خروجی باید JSON معتبر باشد."""

        templates = [
            # Basic retrieval
            "{account} شرکت {org} در سال {year} چقدر است؟",
            "مبلغ {account} {org} در {time_phrase} را نشان بده",
            "{account} در سال {year} برای {org} چند تومان بود؟",

            # Breakdown queries
            "{account} شرکت {org} در سال {year} به تفکیک {breakdown} چقدر است؟",
            "تفکیک {breakdown} {account} {org} در سال {year}",

            # Comparison queries
            "مقایسه {account} شرکت {org1} و {org2} در سال {year}",
            "{account} {org1} نسبت به {org2} در سال {year} به تفکیک {breakdown}",
        ]

        for _ in range(n):
            template = random.choice(templates)
            account_name, acc_id, acc_level, bt, n_type = random.choice(self.accounts)
            org_name, org_id = random.choice(self.orgs)
            year = random.choice(self.years)

            if "{breakdown}" in template:
                breakdown = random.choice(["ماه", "فصل", "نیمسال"])
                breakdown_by = {"ماه": "month", "فصل": "quarter", "نیمسال": "semiyear"}[breakdown]
            else:
                breakdown = ""
                breakdown_by = "none"

            if "{org1}" in template and "{org2}" in template:
                org1_name, org1_id = self.orgs[0]
                org2_name, org2_id = self.orgs[1]
                user_input = template.format(
                    account=account_name, org1=org1_name, org2=org2_name,
                    year=year, breakdown=breakdown
                )
                # Comparison output
                output = {
                    "question_type": "data_retrieval",
                    "dimensions": ["DimStandardAccount", "DimLevel", "DimDate"],
                    "time": {"type": "year", "year": year, "breakdown_by": breakdown_by},
                    "filters": [
                        {"dimension": "DimStandardAccount", "level": acc_level, "operator": "=", "id": acc_id,
                         "label": account_name}
                    ],
                    "measure": [self.measures[(bt, n_type)]],
                    "comparison": {
                        "dimension": "DimLevel",
                        "items": [
                            {"id": org1_id, "label": org1_name},
                            {"id": org2_id, "label": org2_name}
                        ]
                    },
                    "labels": [
                        {"text": account_name, "type": "account", "start": 0, "end": len(account_name)}
                    ],
                    "topic_shift": {"value": "false", "reason": "Same context", "anchor_turn": None},
                    "reformulated_question": user_input,
                    "currency": "IRR",
                    "scale": "unit"
                }
            else:
                user_input = template.format(
                    account=account_name, org=org_name, year=year,
                    time_phrase=f"سال {year}", breakdown=breakdown
                )
                # Standard output
                output = {
                    "question_type": "data_retrieval",
                    "dimensions": ["DimStandardAccount", "DimLevel", "DimDate"],
                    "time": {"type": "year", "year": year, "breakdown_by": breakdown_by},
                    "filters": [
                        {"dimension": "DimStandardAccount", "level": acc_level, "operator": "=", "id": acc_id,
                         "label": account_name},
                        {"dimension": "DimLevel", "level": "DimLevelSK", "operator": "=", "id": org_id,
                         "label": org_name}
                    ],
                    "measure": [self.measures[(bt, n_type)]],
                    "comparison": {"dimension": "", "items": []},
                    "labels": [
                        {"text": account_name, "type": "account", "start": 0, "end": len(account_name)},
                        {"text": org_name, "type": "org", "start": len(account_name) + 1,
                         "end": len(account_name) + len(org_name) + 1}
                    ],
                    "topic_shift": {"value": "false", "reason": "Initial query", "anchor_turn": None},
                    "reformulated_question": user_input,
                    "currency": "IRR",
                    "scale": "unit"
                }

            self.examples.append(AgentExample(
                system_prompt=system_prompt,
                user_input=user_input,
                assistant_output=json.dumps(output, ensure_ascii=False, indent=2),
                agent_type="query_understanding"
            ))

    def generate_mdx_generation_examples(self, n: int = 1500) -> None:
        """Generate MDX Generation agent examples."""
        system_prompt = """شما متخصص تولید پرس‌وجوی MDX هستید. از JSON ورودی، پرس‌وجوی MDX معتبر بسازید.
همیشه از فرمت [Dimension].[Level].&[ID] استفاده کنید.
خروجی باید JSON با کلید "mdx_query" باشد."""

        for _ in range(n):
            account_name, acc_id, acc_level, bt, n_type = random.choice(self.accounts)
            org_name, org_id = random.choice(self.orgs)
            year = random.choice(self.years)
            breakdown = random.choice(["none", "month", "quarter"])

            measure = self.measures[(bt, n_type)]

            # Input (Query Understanding output)
            input_json = {
                "question_type": "data_retrieval",
                "dimensions": ["DimStandardAccount", "DimLevel", "DimDate"],
                "time": {"type": "year", "year": year, "breakdown_by": breakdown},
                "filters": [
                    {"dimension": "DimStandardAccount", "level": acc_level, "operator": "=", "id": acc_id,
                     "label": account_name},
                    {"dimension": "DimLevel", "level": "DimLevelSK", "operator": "=", "id": org_id, "label": org_name}
                ],
                "measure": [measure],
                "currency": "IRR",
                "scale": "unit"
            }

            # Output MDX
            if breakdown == "none":
                mdx = f"""SELECT
                              {measure} ON COLUMNS
                            FROM [ActualCPMDataCube]
                            WHERE (
                              [DimDate].[FiscalYear].&[{year}],
                              [DimLevel].[DimLevelSK].&[{org_id}],
                              [DimStandardAccount].[{acc_level}].&[{acc_id}]
                            )"""
            else:
                breakdown_level = {"month": "FiscalMonth", "quarter": "FiscalQuarter"}[breakdown]
                mdx = f"""SELECT
                      {measure} ON COLUMNS,
                      [DimDate].[Calendar1FiscalFullDateHierarchy].[{breakdown_level}].Members ON ROWS
                    FROM [ActualCPMDataCube]
                    WHERE (
                      [DimStandardAccount].[{acc_level}].&[{acc_id}],
                      [DimLevel].[DimLevelSK].&[{org_id}],
                      [DimDate].[FiscalYear].&[{year}]
                    )"""

            output = {"mdx_query": mdx}

            self.examples.append(AgentExample(
                system_prompt=system_prompt,
                user_input=f"QUERY UNDERSTANDING OUTPUT:\n{json.dumps(input_json, ensure_ascii=False, indent=2)}",
                assistant_output=json.dumps(output, ensure_ascii=False, indent=2),
                agent_type="mdx_generation"
            ))

    def generate_data_analysis_examples(self, n: int = 1200) -> None:
        """Generate Data Analysis agent examples."""
        system_prompt = """شما متخصص تحلیل داده مالی هستید. نتایج پرس‌وجو را تحلیل کنید و بینش‌های معنادار استخراج کنید.
خروجی باید JSON با کلیدهای "insights" و "summary" به فارسی باشد."""

        for _ in range(n):
            account_name, acc_id, acc_level, bt, n_type = random.choice(self.accounts)
            org_name, org_id = random.choice(self.orgs)
            year = random.choice(self.years)

            # Generate fake query results (monthly data)
            base_value = random.randint(100_000_000, 5_000_000_000)
            trend = random.uniform(-0.05, 0.15)  # Monthly growth rate
            results = []
            for month in range(1, 13):
                value = int(base_value * (1 + trend) ** month + random.uniform(-0.1, 0.1) * base_value)
                results.append({
                    "period": f"{year}{month:02d}",
                    "value": value,
                    "formatted_value": f"{value:,} ریال"
                })

            user_input = f"""User Query: {account_name} {org_name} در سال {year} به تفکیک ماه
Query Results: {json.dumps(results, ensure_ascii=False)}"""

            # Generate insights
            avg_value = sum(r["value"] for r in results) // 12
            max_month = max(results, key=lambda x: x["value"])
            min_month = min(results, key=lambda x: x["value"])

            insights = [
                {
                    "type": "trend",
                    "description": f"{account_name} در طول سال {year} روند {'صعودی' if trend > 0 else 'نزولی'} داشته است.",
                    "significance": f"این روند نشان‌دهنده {'رشد' if trend > 0 else 'کاهش'} در عملکرد مالی است.",
                    "data_points": [f"ماه 1: {results[0]['formatted_value']}",
                                    f"ماه 12: {results[11]['formatted_value']}"]
                },
                {
                    "type": "statistic",
                    "description": f"میانگین ماهانه {account_name} برابر {avg_value:,} ریال است.",
                    "significance": "این مبلغ برای برنامه‌ریزی بودجه مفید است.",
                    "data_points": [f"میانگین: {avg_value:,} ریال"]
                },
                {
                    "type": "outlier",
                    "description": f"بیشترین مقدار در {max_month['period']} با {max_month['formatted_value']} ثبت شده است.",
                    "significance": "این افزایش نیاز به بررسی دارد.",
                    "data_points": [f"حداکثر: {max_month['formatted_value']}"]
                }
            ]

            output = {
                "insights": insights,
                "summary": f"تحلیل {account_name} شرکت {org_name} در سال {year} نشان می‌دهد که روند کلی {'مثبت' if trend > 0 else 'منفی'} بوده است."
            }

            self.examples.append(AgentExample(
                system_prompt=system_prompt,
                user_input=user_input,
                assistant_output=json.dumps(output, ensure_ascii=False, indent=2),
                agent_type="data_analysis"
            ))

    def generate_mdx_error_resolution_examples(self, n: int = 800) -> None:
        """Generate MDX Error Resolution examples."""
        system_prompt = """شما متخصص رفع خطای MDX هستید. پرس‌وجوهای معیوب را تصحیح کنید.
خروجی باید JSON با کلید "mdx_query" باشد."""

        common_errors = [
            # Missing period before &
            (
                """SELECT [Measures].[FactFinancail DebitBalance] ON COLUMNS
                    FROM [ActualCPMDataCube]
                    WHERE ([DimLevel].[DimLevelSK]&[34])""",
                                    """SELECT [Measures].[FactFinancail DebitBalance] ON COLUMNS
                    FROM [ActualCPMDataCube]
                    WHERE ([DimLevel].[DimLevelSK].&[34])"""
            ),
            # Missing dimension
            (
                """SELECT [Measures].[FactFinancail DebitBalance] ON COLUMNS
                    FROM [ActualCPMDataCube]
                    WHERE ([DimDate].[FiscalYear].&[1403])""",
                                    """SELECT [Measures].[FactFinancail DebitBalance] ON COLUMNS
                    FROM [ActualCPMDataCube]
                    WHERE (
                      [DimDate].[FiscalYear].&[1403],
                      [DimLevel].[DimLevelSK].&[34],
                      [DimStandardAccount].[DimStandardLedgerAccountSK].&[1]
                    )"""
            ),
        ]

        for _ in range(n):
            error_mdx, correct_mdx = random.choice(common_errors)
            error_msg = "The member was not found in the cube" if ".&" not in error_mdx else "Missing required dimension"

            user_input = f"""Error Message: {error_msg}
Original MDX:
{error_mdx}"""

            output = {"mdx_query": correct_mdx}

            self.examples.append(AgentExample(
                system_prompt=system_prompt,
                user_input=user_input,
                assistant_output=json.dumps(output, ensure_ascii=False, indent=2),
                agent_type="mdx_error_resolution"
            ))

    def generate_visualization_examples(self, n: int = 800) -> None:
        """Generate Visualization agent examples."""
        system_prompt = """شما متخصص انتخاب نمودار هستید. نوع نمودار مناسب را برای داده‌ها انتخاب کنید.
خروجی باید JSON با کلید "chart_type" باشد."""

        templates = [
            ("روند ماهانه را نشان بده", [{"month": i, "value": random.randint(1000, 5000)} for i in range(1, 13)],
             "line"),
            ("مقایسه بین دو شرکت", [{"org": "A", "value": 1000}, {"org": "B", "value": 1500}], "bar"),
            ("سهم هزینه‌ها", [{"category": "اداری", "percent": 30}, {"category": "عملیاتی", "percent": 70}], "pie"),
            ("تفکیک فصلی", [{"quarter": i, "value": random.randint(5000, 15000)} for i in range(1, 5)], "bar"),
        ]

        for _ in range(n):
            query, sample_data, chart_type = random.choice(templates)

            user_input = f"""User Query: {query}
Data Sample: {json.dumps(sample_data, ensure_ascii=False)}"""

            output = {"chart_type": chart_type}

            self.examples.append(AgentExample(
                system_prompt=system_prompt,
                user_input=user_input,
                assistant_output=json.dumps(output, ensure_ascii=False, indent=2),
                agent_type="visualization"
            ))

    def generate_all_examples(self, total_samples: int = 8000) -> None:
        """Generate all examples with balanced distribution."""
        print("Generating synthetic training data...")

        # Distribute samples across agents
        dist = {
            "query_understanding": int(total_samples * 0.25),  # 2000
            "mdx_generation": int(total_samples * 0.25),  # 2000
            "data_analysis": int(total_samples * 0.25),  # 2000
            "mdx_error_resolution": int(total_samples * 0.15),  # 1200
            "visualization": int(total_samples * 0.10),  # 800
        }

        print(f"  Query Understanding: {dist['query_understanding']} examples")
        self.generate_query_understanding_examples(dist["query_understanding"])

        print(f"  MDX Generation: {dist['mdx_generation']} examples")
        self.generate_mdx_generation_examples(dist["mdx_generation"])

        print(f"  Data Analysis: {dist['data_analysis']} examples")
        self.generate_data_analysis_examples(dist["data_analysis"])

        print(f"  MDX Error Resolution: {dist['mdx_error_resolution']} examples")
        self.generate_mdx_error_resolution_examples(dist["mdx_error_resolution"])

        print(f"  Visualization: {dist['visualization']} examples")
        self.generate_visualization_examples(dist["visualization"])

        print(f"\nTotal examples generated: {len(self.examples)}")

    def save_datasets(self, output_dir: str, train_split: float = 0.9) -> None:
        """Save train and validation datasets in chat format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Shuffle examples
        random.shuffle(self.examples)

        # Split
        split_idx = int(len(self.examples) * train_split)
        train_examples = self.examples[:split_idx]
        val_examples = self.examples[split_idx:]

        # Convert to chat format (Qwen2.5 template)
        def to_chat_format(ex: AgentExample) -> Dict[str, Any]:
            return {
                "messages": [
                    {"role": "system", "content": ex.system_prompt},
                    {"role": "user", "content": ex.user_input},
                    {"role": "assistant", "content": ex.assistant_output}
                ],
                "agent_type": ex.agent_type
            }

        # Save train
        train_path = output_path / "train.jsonl"
        with open(train_path, "w", encoding="utf-8") as f:
            for ex in train_examples:
                f.write(json.dumps(to_chat_format(ex), ensure_ascii=False) + "\n")

        # Save validation
        val_path = output_path / "validation.jsonl"
        with open(val_path, "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(to_chat_format(ex), ensure_ascii=False) + "\n")

        print(f"\n✓ Datasets saved:")
        print(f"  Train: {train_path} ({len(train_examples)} examples)")
        print(f"  Validation: {val_path} ({len(val_examples)} examples)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=8000, help="Total number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Data Generation Pipeline")
    print("=" * 60)

    generator = SyntheticDataGenerator(seed=args.seed)
    generator.generate_all_examples(total_samples=args.num_samples)
    generator.save_datasets(output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("✓ Data generation complete!")
    print("=" * 60)


# if __name__ == "__main__":
#     main()