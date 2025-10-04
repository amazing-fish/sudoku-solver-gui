from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from .preprocess import preprocess_cell

logger = logging.getLogger(__name__)

_FONT_PATH = Path(
    os.environ.get(
        "SUDOKU_FONT_PATH",
        r"C:\\ProgramData\\kingsoft\\office6\\omath\\DejaVuMathTeXGyre.ttf",
    )
)


@dataclass
class SyntheticDigitConfig:
    samples_per_digit: int = 400
    include_blank: bool = True
    cell_size: int = 64
    angle_range: Tuple[float, float] = (-12, 12)
    noise_std: float = 12.0
    jitter_ratio: float = 0.12
    max_attempts: int = 6
    seed: int = 42


class SyntheticDigitDataset(Dataset):
    """Generate synthetic digit images similar to the Sudoku font."""

    def __init__(self, config: SyntheticDigitConfig | None = None) -> None:
        self.config = config or SyntheticDigitConfig()
        if not _FONT_PATH.exists():
            raise FileNotFoundError(f"未找到字体文件: {_FONT_PATH}")
        logger.info("加载字体文件: %s", _FONT_PATH)

        rng = np.random.default_rng(self.config.seed)
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []

        digits = list(range(1, 10))
        if self.config.include_blank:
            digits = [0] + digits

        for digit in digits:
            generated = 0
            while generated < self.config.samples_per_digit:
                tensor = self._create_sample(digit, rng)
                if tensor is None:
                    continue
                self.images.append(tensor)
                self.labels.append(digit)
                generated += 1

        logger.info(
            "合成数据集准备完成: 总样本数=%s, 类别数=%s, 包含空白=%s",
            len(self.labels),
            len(set(self.labels)),
            self.config.include_blank,
        )

    def _create_sample(
        self, digit: int, rng: np.random.Generator
    ) -> torch.Tensor | None:
        for _ in range(self.config.max_attempts):
            cell = self._render_cell(digit, rng)
            result = preprocess_cell(cell)

            if digit == 0:
                if not result.is_blank:
                    continue
            else:
                if result.is_blank:
                    continue

            tensor = torch.from_numpy(result.data).unsqueeze(0)
            return tensor

        return None

    def _render_cell(self, digit: int, rng: np.random.Generator) -> np.ndarray:
        cell_size = self.config.cell_size
        image = Image.new("L", (cell_size, cell_size), color=255)
        draw = ImageDraw.Draw(image)

        if digit != 0:
            font_min = int(cell_size * 0.45)
            font_max = int(cell_size * 0.75)
            font_size = int(rng.integers(low=font_min, high=max(font_min + 1, font_max)))
            font = ImageFont.truetype(str(_FONT_PATH), font_size)
            text = str(digit)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            jitter = int(cell_size * self.config.jitter_ratio)
            offset_x = int((cell_size - text_w) / 2 + rng.integers(-jitter, jitter + 1))
            offset_y = int((cell_size - text_h) / 2 + rng.integers(-jitter, jitter + 1))
            offset_x = int(np.clip(offset_x, 0, max(0, cell_size - text_w)))
            offset_y = int(np.clip(offset_y, 0, max(0, cell_size - text_h)))
            color = int(rng.integers(low=0, high=80))
            draw.text((offset_x, offset_y), text, font=font, fill=color)

        angle = float(rng.uniform(*self.config.angle_range))
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

        array = np.array(image, dtype=np.float32)
        if self.config.noise_std > 0:
            noise = rng.normal(loc=0.0, scale=self.config.noise_std, size=array.shape)
            array = np.clip(array + noise, 0.0, 255.0)

        return array.astype(np.uint8)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:  # type: ignore[override]
        return self.images[index], self.labels[index]
