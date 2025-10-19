import json
from typing import Dict, Any, Optional


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    return None


def format_number(num: float, scale: str = "unit", currency: str = "IRR") -> str:
    if scale == "thousand":
        num /= 1000
    elif scale == "million":
        num /= 1000000

    if currency == "IRT":  # Toman
        num /= 10
        suffix = "تومان"
    else:  # Rial
        suffix = "ریال"

    return f"{num:,.0f} {suffix}"