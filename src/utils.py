from __future__ import annotations

from pathlib import Path
from typing import Mapping

import logging
import torch

logger = logging.getLogger(__name__)


def select_device(
    requested: str | torch.device | None,
    *,
    allow_cuda: bool = True,
    fallback: str = "cpu",
) -> torch.device:
    """根据可用性选择计算设备并在必要时降级。"""

    if isinstance(requested, torch.device):
        device = requested
    else:
        device_str = requested or "auto"
        device_str = str(device_str).lower()
        if device_str == "auto":
            if allow_cuda and torch.cuda.is_available():
                device_str = "cuda"
            else:
                device_str = fallback
        device = torch.device(device_str)

    if device.type.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("请求使用 CUDA 设备，但当前环境不可用，自动切换到 %s。", fallback)
        device = torch.device(fallback)
    return device


def format_config(config: Mapping[str, object]) -> str:
    """将配置映射格式化为 key=value 串，便于日志输出。"""

    def _stringify(value: object) -> str:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        return repr(value) if isinstance(value, str) and " " in value else str(value)

    items = []
    for key in sorted(config.keys()):
        items.append(f"{key}={_stringify(config[key])}")
    return ", ".join(items)
