from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
import os
from pathlib import Path
import json
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from .preprocess import PREPROCESS_VERSION, preprocess_cell

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


_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


class SyntheticDigitDataset(Dataset):
    """Generate synthetic digit images similar to the Sudoku font."""

    def __init__(self, config: SyntheticDigitConfig | None = None) -> None:
        self.config = config or SyntheticDigitConfig()
        if not _FONT_PATH.exists():
            raise FileNotFoundError(f"未找到字体文件: {_FONT_PATH}")
        logger.info("加载字体文件: %s", _FONT_PATH)

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
        digits = list(range(1, 10))
        if self.config.include_blank:
            digits = [0] + digits

        images: list[torch.Tensor] = []
        labels: list[int] = []
        total_target = self.config.samples_per_digit * len(digits)
        progress_step = max(1, total_target // 20)

        logger.info(
            "正在生成合成数据集: 目标样本数=%s, digit种类=%s, 缓存路径=%s",
            total_target,
            len(digits),
            cache_path,
        )

        produced = 0
        for digit in digits:
            generated = 0
            while generated < self.config.samples_per_digit:
                tensor = self._create_sample(digit, rng)
                if tensor is None:
                    continue
                images.append(tensor)
                labels.append(digit)
                generated += 1
                produced += 1
                if produced % progress_step == 0 or produced == total_target:
                    percent = produced / total_target * 100
                    logger.info(
                        "数据合成进度: %s/%s (%.1f%%)",
                        produced,
                        total_target,
                        percent,
                    )

        self.images = torch.stack(images, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.long)

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"images": self.images, "labels": self.labels, "metadata": metadata},
            cache_path,
        )
        logger.info("合成数据集准备完成并已缓存: 样本数=%s", self.labels.numel())

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
