"""
System Constants
Centralized constants used across the application
"""

from enum import Enum
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"


# ============================================================================
# MODEL CONSTANTS
# ============================================================================

class ModelType(str, Enum):
    """Supported model types."""
    QWEN = "Qwen/Qwen2.5-7B-Instruct"
    LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
    MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"


class MeasureType(str, Enum):
    """Financial measure types."""
    DEBIT_BALANCE = "[Measures].[FactFinancail DebitBalance]"
    CREDIT_BALANCE = "[Measures].[FactFinancail CreditBalance]"
    DEBIT_VALUE = "[Measures].[FactFinancail DebitValue]"
    CREDIT_VALUE = "[Measures].[FactFinancail CreditValue]"


# ============================================================================
# AGENT CONSTANTS
# ============================================================================

class AgentType(str, Enum):
    """Agent types in the system."""
    QUERY_UNDERSTANDING = "query_understanding"
    MDX_GENERATION = "mdx_generation"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    ERROR_RESOLUTION = "error_resolution"


# Agent system prompts
AGENT_SYSTEM_PROMPTS = {
    AgentType.QUERY_UNDERSTANDING: """شما متخصص درک پرس‌وجوی مالی هستید. پرس‌وجوهای فارسی را به JSON ساختاری تبدیل کنید.
باید بعدها (dimensions)، زمان (time)، فیلترها (filters)، و معیارها (measures) را استخراج کنید.
خروجی باید JSON معتبر باشد.""",

    AgentType.MDX_GENERATION: """شما متخصص تولید پرس‌وجوی MDX هستید. از JSON ورودی، پرس‌وجوی MDX معتبر بسازید.
همیشه از فرمت [Dimension].[Level].&[ID] استفاده کنید.
خروجی باید JSON با کلید "mdx_query" باشد.""",

    AgentType.DATA_ANALYSIS: """شما متخصص تحلیل داده مالی هستید. نتایج پرس‌وجو را تحلیل کنید و بینش‌های معنادار استخراج کنید.
خروجی باید JSON با کلیدهای "insights" و "summary" به فارسی باشد.""",

    AgentType.VISUALIZATION: """شما متخصص انتخاب نمودار هستید. نوع نمودار مناسب را برای داده‌ها انتخاب کنید.
خروجی باید JSON با کلید "chart_type" باشد.""",

    AgentType.ERROR_RESOLUTION: """شما متخصص رفع خطای MDX هستید. پرس‌وجوهای معیوب را تصحیح کنید.
خروجی باید JSON با کلید "mdx_query" باشد.""",
}


# ============================================================================
# DIMENSION CONSTANTS
# ============================================================================

class DimensionLevel(str, Enum):
    """MDX dimension levels."""
    # DimStandardAccount levels
    GROUP_ACCOUNT = "DimStandardGroupAccountSK"
    GENERAL_ACCOUNT = "DimStandardGeneralAccountSK"
    LEDGER_ACCOUNT = "DimStandardLedgerAccountSK"

    # DimLevel
    LEVEL_SK = "DimLevelSK"

    # DimDate levels
    FISCAL_YEAR = "FiscalYear"
    FISCAL_SEMIYEAR = "FiscalSemiYear"
    FISCAL_QUARTER = "FiscalQuarter"
    FISCAL_MONTH = "FiscalMonth"


class CubeNames(str, Enum):
    """OLAP cube names."""
    ACTUAL_CPM = "ActualCPMDataCube"


# ============================================================================
# TRAINING CONSTANTS
# ============================================================================

class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Default hyperparameters
DEFAULT_TRAINING_PARAMS = {
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lora_r": 128,
    "lora_alpha": 256,
}


# ============================================================================
# API CONSTANTS
# ============================================================================

class HTTPStatus(int, Enum):
    """Common HTTP status codes."""
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503


# API rate limits
RATE_LIMITS = {
    "default": "100/minute",
    "feedback": "1000/hour",
    "inference": "100/minute",
}


# ============================================================================
# FEEDBACK CONSTANTS
# ============================================================================

class FeedbackRating(int, Enum):
    """Feedback rating values."""
    VERY_BAD = 1
    BAD = 2
    NEUTRAL = 3
    GOOD = 4
    EXCELLENT = 5


MIN_FEEDBACK_QUALITY_SCORE = 0.7
MIN_FEEDBACK_SAMPLES_FOR_RETRAINING = 500
RETRAINING_INTERVAL_DAYS = 14

# ============================================================================
# PERSIAN CONSTANTS
# ============================================================================

PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹'
ARABIC_DIGITS = '0123456789'

PERSIAN_MONTHS = [
    "فروردین", "اردیبهشت", "خرداد", "تیر",
    "مرداد", "شهریور", "مهر", "آبان",
    "آذر", "دی", "بهمن", "اسفند"
]

PERSIAN_SEASONS = {
    "بهار": 1,  # Q1
    "تابستان": 2,  # Q2
    "پاییز": 3,  # Q3
    "زمستان": 4,  # Q4
}


# ============================================================================
# CHART TYPES
# ============================================================================

class ChartType(str, Enum):
    """Supported chart types."""
    DATALABEL = "datalabel"
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    RADAR = "radar"
    HEATMAP = "heatmap"
    BUBBLE = "bubble"
    FUNNEL = "funnel"
    GAUGE = "gauge"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    WATERFALL = "waterfall"
    BOXPLOT = "boxplot"


# ============================================================================
# MONITORING CONSTANTS
# ============================================================================

# Prometheus metric names
METRICS = {
    "requests_total": "api_requests_total",
    "request_latency": "api_request_latency_seconds",
    "tokens_generated": "api_tokens_generated",
    "errors_total": "api_errors_total",
    "model_score": "model_overall_score",
    "feedback_total": "feedback_total",
}

# Alert thresholds
ALERT_THRESHOLDS = {
    "latency_p95_seconds": 5.0,
    "error_rate_percent": 5.0,
    "gpu_memory_utilization_percent": 90.0,
}

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    "model_not_found": "Model not found in registry",
    "invalid_json": "Invalid JSON format in response",
    "mdx_syntax_error": "MDX query has syntax errors",
    "insufficient_feedback": "Not enough feedback samples for retraining",
    "training_failed": "Model training failed",
    "deployment_failed": "Model deployment failed",
    "rollback_failed": "Model rollback failed",
}

# ============================================================================
# VERSION INFO
# ============================================================================

PLATFORM_VERSION = "1.0.0"
API_VERSION = "v1"
MIN_PYTHON_VERSION = "3.10"

# if __name__ == "__main__":
#     # Test constants
#     print(f"Platform Version: {PLATFORM_VERSION}")
#     print(f"Project Root: {PROJECT_ROOT}")
#     print(f"Available Agents: {[a.value for a in AgentType]}")
#     print(f"Chart Types: {[c.value for c in ChartType]}")