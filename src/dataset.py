from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset

_MEAN = 0.1307
_STD = 0.3081
_FONT_PATH = Path("C:\ProgramData\kingsoft\office6\omath\DejaVuMathTeXGyre.ttf")


@dataclass
class SyntheticDigitConfig:
    samples_per_digit: int = 400
    image_size: int = 28
    angle_range: Tuple[float, float] = (-15, 15)
    noise_std: float = 0.05
    seed: int = 42


class SyntheticDigitDataset(Dataset):
    """Generate synthetic digit images similar to the Sudoku font."""

    def __init__(self, config: SyntheticDigitConfig | None = None) -> None:
        self.config = config or SyntheticDigitConfig()
        if not _FONT_PATH.exists():
            raise FileNotFoundError(f"未找到字体文件: {_FONT_PATH}")

        rng = np.random.default_rng(self.config.seed)
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []

        for digit in range(10):
            for _ in range(self.config.samples_per_digit):
                tensor = self._create_sample(digit, rng)
                self.images.append(tensor)
                self.labels.append(digit)

    def _create_sample(self, digit: int, rng: np.random.Generator) -> torch.Tensor:
        size = self.config.image_size
        canvas_size = size * 2
        image = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(image)
        font_size = int(rng.integers(low=size + 6, high=size + 14))
        font = ImageFont.truetype(str(_FONT_PATH), font_size)
        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        max_x = max(0, canvas_size - text_w)
        max_y = max(0, canvas_size - text_h)
        offset_x = int(rng.integers(0, max_x + 1))
        offset_y = int(rng.integers(0, max_y + 1))
        draw.text((offset_x, offset_y), text, font=font, fill=0)
        angle = float(rng.uniform(*self.config.angle_range))
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

        inverted = ImageOps.invert(image)
        bbox = inverted.getbbox()
        if bbox is not None:
            inverted = inverted.crop(bbox)
        inverted = inverted.resize((size, size), Image.BILINEAR)

        array = np.array(inverted, dtype=np.float32) / 255.0
        noise = rng.normal(loc=0.0, scale=self.config.noise_std, size=array.shape).astype(np.float32)
        array = np.clip(array + noise, 0.0, 1.0).astype(np.float32)
        array = (array - _MEAN) / _STD
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:  # type: ignore[override]
        return self.images[index], self.labels[index]
