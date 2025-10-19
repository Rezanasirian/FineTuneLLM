import torch
from src.core.logger import get_logger

logger = get_logger("gpu_monitor")


def get_gpu_info():
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
    }


def log_gpu_usage():
    info = get_gpu_info()
    if info["available"]:
        logger.info(f"GPU: {info['memory_allocated_gb']:.2f}GB / {info['memory_total_gb']:.2f}GB")