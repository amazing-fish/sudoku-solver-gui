from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
import os
from pathlib import Path
import json
import time
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from .preprocess import PREPROCESS_VERSION, preprocess_cell_batch

logger = logging.getLogger(__name__)

_FONT_PATH = Path(
    os.environ.get(
        "SUDOKU_FONT_PATH",
        r"C:\\ProgramData\\kingsoft\\office6\\omath\\DejaVuMathTeXGyre.ttf",
    )
)


@dataclass
class SyntheticDigitConfig:
    samples_per_digit: int = 800
    image_size: int = 28
    angle_range: Tuple[float, float] = (-15, 15)
    noise_std: float = 0.05
    seed: int = 42
    max_grid_offset: int = 5
    line_width_range: Tuple[int, int] = (1, 3)
    digit_scale_range: Tuple[float, float] = (0.65, 0.95)
    digit_color_range: Tuple[int, int] = (0, 40)
    background_intensity_range: Tuple[int, int] = (215, 255)
    cell_intensity_range: Tuple[int, int] = (220, 255)
    blank_intensity_range: Tuple[int, int] = (185, 240)


class SyntheticDigitDataset(Dataset):
    """生成包含空白格的合成数独数字样本。"""

    def __init__(self, config: SyntheticDigitConfig | None = None) -> None:
        self.config = config or SyntheticDigitConfig()
        if not _FONT_PATH.exists():
            raise FileNotFoundError(f"未找到字体文件: {_FONT_PATH}")
        logger.info("加载字体文件: %s", _FONT_PATH)

        backend = self.config.preprocess_backend.lower()
        if backend not in {"cpu", "gpu"}:
            raise ValueError(f"不支持的预处理后端: {backend}")

        if self.config.synthesis_batch_size <= 0:
            raise ValueError("synthesis_batch_size 必须为正整数")
        if self.config.progress_interval <= 0:
            raise ValueError("progress_interval 必须大于 0")

        device_str = self.config.device
        if isinstance(device_str, str):
            device_str = device_str.lower()
        if device_str == "auto":
            device_str = "cuda" if (backend == "gpu" and torch.cuda.is_available()) else "cpu"

        self.preprocess_backend = backend
        self.preprocess_device = torch.device(device_str)

        if self.preprocess_backend == "gpu" and self.preprocess_device.type != "cuda":
            if torch.cuda.is_available():
                logger.info("GPU 预处理需要 CUDA，自动切换到可用设备 cuda:0")
                self.preprocess_device = torch.device("cuda")
            else:
                logger.warning("请求 GPU 预处理但当前环境不支持 CUDA，回退到 CPU 后端")
                self.preprocess_backend = "cpu"
                self.preprocess_device = torch.device("cpu")

        if self.preprocess_backend == "gpu" and not torch.cuda.is_available():
            logger.warning("CUDA 设备不可用，自动回退到 CPU 预处理后端")
            self.preprocess_backend = "cpu"
            self.preprocess_device = torch.device("cpu")

        if self.preprocess_backend == "gpu":
            logger.info(
                "预处理后端: GPU, 设备=%s, 批大小=%s", 
                self.preprocess_device,
                self.config.synthesis_batch_size,
            )
        else:
            logger.info("预处理后端: CPU, 批大小=%s", self.config.synthesis_batch_size)

        # 更新配置以反映实际运行参数，确保缓存键准确
        self.config.preprocess_backend = self.preprocess_backend
        self.config.device = str(self.preprocess_device)

        metadata = self._build_metadata()
        cache_key = self._build_cache_key(metadata)
        cache_path = _CACHE_DIR / f"synthetic_digits_{cache_key}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            self.images = payload["images"]
            self.labels = payload["labels"]
            logger.info(
                "从缓存加载合成数据集: 文件=%s, 样本数=%s",
                cache_path,
                self.labels.numel(),
            )
            return

        rng = np.random.default_rng(self.config.seed)
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []

        for digit in range(10):
            for _ in range(self.config.samples_per_digit):
                tensor = self._create_sample(digit, rng)
                self.images.append(tensor)
                self.labels.append(digit)

    def to_pil(self, index: int) -> Image.Image:
        array = self.images[index].squeeze(0).numpy()
        array = array * _STD + _MEAN
        array = np.clip(array, 0.0, 1.0)
        return Image.fromarray((array * 255).astype(np.uint8), mode="L")

    def render_ascii(self, index: int, levels: str = " .:-=+*#%@") -> str:
        array = self.images[index].squeeze(0).numpy()
        array = array * _STD + _MEAN
        array = np.clip(array, 0.0, 1.0)
        scaled = np.floor(array * (len(levels) - 1)).astype(int)
        lines = ["".join(levels[val] for val in row) for row in scaled]
        return "\n".join(lines)

    def _render_digit_layer(
        self, digit: int, rng: np.random.Generator, target_span: int
    ) -> Image.Image:
        base_side = max(target_span * 2, self.config.image_size)
        canvas = Image.new("L", (base_side, base_side), color=0)
        draw = ImageDraw.Draw(canvas)
        font_scale = rng.uniform(0.7, 1.1)
        font_size = max(12, int(target_span * font_scale))
        font = ImageFont.truetype(str(_FONT_PATH), font_size)
        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (canvas.width - text_w) / 2 - bbox[0]
        y = (canvas.height - text_h) / 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=255)
        angle = float(rng.uniform(*self.config.angle_range))
        rotated = canvas.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
        bbox = rotated.getbbox()
        if bbox is not None:
            rotated = rotated.crop(bbox)
        return rotated

    def _resize_with_aspect(self, image: Image.Image, target_span: int) -> Image.Image:
        width, height = image.size
        if max(width, height) == 0:
            return image
        scale = target_span / max(width, height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        return image.resize((new_w, new_h), Image.BICUBIC)

    def _create_sample(self, digit: int, rng: np.random.Generator) -> torch.Tensor:
        size = self.config.image_size
        bg_color = int(
            rng.integers(
                self.config.background_intensity_range[0],
                self.config.background_intensity_range[1] + 1,
            )
        )
        image = Image.new("L", (size, size), color=bg_color)
        draw = ImageDraw.Draw(image)

        max_offset = min(self.config.max_grid_offset, size // 4)
        for _ in range(10):
            pad_top = int(rng.integers(0, max_offset + 1))
            pad_bottom = int(rng.integers(0, max_offset + 1))
            pad_left = int(rng.integers(0, max_offset + 1))
            pad_right = int(rng.integers(0, max_offset + 1))
            line_width = int(
                rng.integers(self.config.line_width_range[0], self.config.line_width_range[1] + 1)
            )
            inner_width = size - pad_left - pad_right - 2 * line_width
            inner_height = size - pad_top - pad_bottom - 2 * line_width
            if inner_width > size * 0.4 and inner_height > size * 0.4:
                break
        else:
            pad_top = pad_bottom = pad_left = pad_right = max_offset // 2
            line_width = max(1, self.config.line_width_range[0])

        left = pad_left
        top = pad_top
        right = size - pad_right - 1
        bottom = size - pad_bottom - 1

        line_color = int(rng.integers(70, 140))
        draw.rectangle([left, top, right, bottom], outline=line_color, width=line_width)

        inner_left = left + line_width
        inner_top = top + line_width
        inner_right = right - line_width
        inner_bottom = bottom - line_width

        if inner_left >= inner_right or inner_top >= inner_bottom:
            inner_left = max(inner_left, 0)
            inner_top = max(inner_top, 0)
            inner_right = min(inner_right, size - 1)
            inner_bottom = min(inner_bottom, size - 1)

        if digit == 0:
            blank_intensity = int(
                rng.integers(
                    self.config.blank_intensity_range[0], self.config.blank_intensity_range[1] + 1
                )
            )
            draw.rectangle([inner_left, inner_top, inner_right, inner_bottom], fill=blank_intensity)
        else:
            cell_intensity = int(
                rng.integers(
                    self.config.cell_intensity_range[0], self.config.cell_intensity_range[1] + 1
                )
            )
            draw.rectangle([inner_left, inner_top, inner_right, inner_bottom], fill=cell_intensity)

            inner_width = max(1, inner_right - inner_left)
            inner_height = max(1, inner_bottom - inner_top)
            target_span = int(
                min(inner_width, inner_height)
                * rng.uniform(self.config.digit_scale_range[0], self.config.digit_scale_range[1])
            )
            target_span = max(8, target_span)
            digit_layer = self._render_digit_layer(digit, rng, target_span)
            digit_layer = self._resize_with_aspect(digit_layer, target_span)

            available_x = max(0, inner_width - digit_layer.size[0])
            available_y = max(0, inner_height - digit_layer.size[1])
            offset_x = int(rng.integers(0, available_x + 1)) if available_x > 0 else 0
            offset_y = int(rng.integers(0, available_y + 1)) if available_y > 0 else 0

            position_x = inner_left + offset_x
            position_y = inner_top + offset_y
            digit_color = int(
                rng.integers(self.config.digit_color_range[0], self.config.digit_color_range[1] + 1)
            )
            glyph = Image.new("L", digit_layer.size, color=digit_color)
            image.paste(glyph, (position_x, position_y), digit_layer)

        array = np.array(image, dtype=np.float32) / 255.0
        noise = rng.normal(loc=0.0, scale=self.config.noise_std, size=array.shape).astype(np.float32)
        array = np.clip(array + noise, 0.0, 1.0).astype(np.float32)
        array = (array - _MEAN) / _STD
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:  # type: ignore[override]
        image = self.images[index]
        label = self.labels[index]
        if isinstance(label, torch.Tensor):
            label = int(label.item())
        return image, label

    def _build_metadata(self) -> dict[str, object]:
        config_dict = asdict(self.config)
        config_dict["angle_range"] = list(config_dict["angle_range"])
        font_stat = _FONT_PATH.stat()
        return {
            "config": config_dict,
            "font_path": str(_FONT_PATH.resolve()),
            "font_mtime": font_stat.st_mtime,
            "font_size": font_stat.st_size,
            "preprocess_version": PREPROCESS_VERSION,
        }

    def _build_cache_key(self, metadata: dict[str, object]) -> str:
        serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()
