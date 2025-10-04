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
from .utils import format_config, select_device

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
    seed: int = 42
    preprocess_backend: str = "cpu"
    device: str = "auto"
    synthesis_batch_size: int = 256
    progress_interval: float = 2.0


_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


class SyntheticDigitDataset(Dataset):
    """Generate synthetic digit images similar to the Sudoku font."""

    def __init__(self, config: SyntheticDigitConfig | None = None) -> None:
        self.config = config or SyntheticDigitConfig()
        if not _FONT_PATH.exists():
            raise FileNotFoundError(f"未找到字体文件: {_FONT_PATH}")
        logger.debug("加载字体文件: %s", _FONT_PATH)

        backend = self.config.preprocess_backend.lower()
        if backend not in {"cpu", "gpu"}:
            raise ValueError(f"不支持的预处理后端: {backend}")

        if self.config.synthesis_batch_size <= 0:
            raise ValueError("synthesis_batch_size 必须为正整数")
        if self.config.progress_interval <= 0:
            raise ValueError("progress_interval 必须大于 0")

        resolved_device = select_device(
            self.config.device,
            allow_cuda=(backend == "gpu"),
        )
        self.preprocess_backend = backend
        self.preprocess_device = resolved_device

        if self.preprocess_backend == "gpu" and self.preprocess_device.type != "cuda":
            logger.info("GPU 预处理所需的 CUDA 不可用，自动切换到 CPU 后端。")
            self.preprocess_backend = "cpu"
            self.preprocess_device = torch.device("cpu")

        # 更新配置以反映实际运行参数，确保缓存键准确
        self.config.preprocess_backend = self.preprocess_backend
        self.config.device = str(self.preprocess_device)

        logger.info(
            "初始化合成数据集: %s",
            format_config(
                {
                    "backend": self.preprocess_backend,
                    "device": self.preprocess_device,
                    "include_blank": self.config.include_blank,
                    "samples_per_digit": self.config.samples_per_digit,
                    "synthesis_batch": self.config.synthesis_batch_size,
                }
            ),
        )

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
        progress_step = max(1, total_target // 50)

        logger.info(
            "生成合成数据集: %s",
            format_config(
                {
                    "cache_path": cache_path,
                    "digits": len(digits),
                    "target_samples": total_target,
                }
            ),
        )

        start_time = time.perf_counter()
        last_progress = start_time

        produced = 0
        total_attempts = 0
        for digit in digits:
            generated = 0
            digit_attempts = 0
            pending: list[np.ndarray] = []
            logger.debug("开始生成 digit=%s 的样本……", digit)
            while generated < self.config.samples_per_digit:
                sample = self._render_cell(digit, rng)
                pending.append(sample)
                total_attempts += 1
                digit_attempts += 1

                if (
                    len(pending) >= self.config.synthesis_batch_size
                    or digit_attempts % self.config.synthesis_batch_size == 0
                ):
                    accepted = self._process_pending(
                        pending,
                        digit,
                        self.config.samples_per_digit - generated,
                        images,
                        labels,
                    )
                    generated += accepted
                    produced += accepted
                    pending.clear()

                now = time.perf_counter()
                should_log = False
                if produced > 0 and (
                    produced % progress_step == 0 or produced == total_target
                ):
                    should_log = True
                if now - last_progress >= self.config.progress_interval:
                    should_log = True

                if should_log:
                    percent = produced / total_target * 100 if total_target else 0.0
                    elapsed = now - start_time
                    rate = produced / elapsed if elapsed > 0 else 0.0
                    eta = (total_target - produced) / rate if rate > 0 else float("inf")
                    logger.debug(
                        "数据合成进度: %s/%s (%.1f%%), 耗时=%.1fs, 速度=%.1f样本/s, 预计剩余=%.1fs, 当前digit尝试=%s, 累计尝试=%s",
                        produced,
                        total_target,
                        percent,
                        elapsed,
                        rate,
                        eta,
                        digit_attempts,
                        total_attempts,
                    )
                    last_progress = now

            if pending:
                accepted = self._process_pending(
                    pending,
                    digit,
                    self.config.samples_per_digit - generated,
                    images,
                    labels,
                )
                generated += accepted
                produced += accepted
                pending.clear()

            if generated < self.config.samples_per_digit:
                logger.warning(
                    "digit=%s 的样本生成不足，期望=%s，实际=%s，请检查预处理阈值。",
                    digit,
                    self.config.samples_per_digit,
                    generated,
                )

        if not images:
            raise RuntimeError("未能生成任何合成样本，请检查预处理参数设置。")

        self.images = torch.stack(images, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.long)

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"images": self.images, "labels": self.labels, "metadata": metadata},
            cache_path,
        )
        total_elapsed = time.perf_counter() - start_time
        logger.info(
            "合成数据集准备完成并已缓存: 样本数=%s, 总耗时=%.1fs, 平均速度=%.1f样本/s",
            self.labels.numel(),
            total_elapsed,
            self.labels.numel() / total_elapsed if total_elapsed > 0 else 0.0,
        )

    def _process_pending(
        self,
        pending: list[np.ndarray],
        digit: int,
        remaining: int,
        images: list[torch.Tensor],
        labels: list[int],
    ) -> int:
        if not pending or remaining <= 0:
            pending.clear()
            return 0

        batch = np.stack(pending, axis=0)

        def _run_preprocess(current_backend: str) -> tuple[np.ndarray, np.ndarray]:
            if current_backend == "gpu":
                return preprocess_cell_batch(
                    batch,
                    backend="gpu",
                    device=self.preprocess_device,
                )
            return preprocess_cell_batch(batch, backend="cpu")

        data, blanks = _run_preprocess(self.preprocess_backend)

        mask = blanks if digit == 0 else ~blanks
        if not np.any(mask) and self.preprocess_backend == "gpu":
            logger.debug(
                "digit=%s 的 GPU 预处理批次未筛出有效样本，回退到 CPU 预处理以避免长时间重试。",
                digit,
            )
            data, blanks = _run_preprocess("cpu")
            mask = blanks if digit == 0 else ~blanks

        if not np.any(mask):
            pending.clear()
            return 0

        selected_indices = np.flatnonzero(mask)[:remaining]
        for idx in selected_indices:
            tensor = torch.from_numpy(data[idx]).unsqueeze(0)
            images.append(tensor)
            labels.append(digit)

        accepted = len(selected_indices)

        pending.clear()
        return accepted

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
        if digit != 0 and self.config.noise_std > 0:
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
