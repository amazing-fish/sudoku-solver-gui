"""生成用于训练数独识别模型的合成数据集。"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID_SIZE = 252
CELL_SIZE = GRID_SIZE // 9
DEFAULT_SAMPLES = 1000


@dataclass
class SampleMetadata:
    """记录数据集的基本信息。"""

    samples: int
    grid_size: int = GRID_SIZE
    cell_size: int = CELL_SIZE
    description: str = "synthetic sudoku board dataset"


def _load_font(size: int) -> ImageFont.ImageFont:
    """尝试加载常见字体，回退到默认字体。"""

    preferred_fonts: Iterable[str] = (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    )

    for font_path in preferred_fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue

    # 默认字体不支持调整大小，但可保证脚本可运行
    return ImageFont.load_default()


def _random_board(rng: random.Random) -> np.ndarray:
    """生成包含 0-9 的随机棋盘，0 表示空白。"""

    board = np.zeros((9, 9), dtype=np.uint8)
    for row in range(9):
        for col in range(9):
            if rng.random() < 0.55:  # 保留较多空格以贴近题面
                continue
            board[row, col] = rng.randint(1, 9)
    return board


def _draw_board(board: np.ndarray, rng: random.Random) -> np.ndarray:
    """将棋盘渲染为灰度图像。"""

    img = Image.new("L", (GRID_SIZE, GRID_SIZE), color=255)
    draw = ImageDraw.Draw(img)

    # 绘制网格线
    for i in range(10):
        line_width = 3 if i % 3 == 0 else 1
        offset = rng.randint(-1, 1)
        x = i * CELL_SIZE + offset
        y = i * CELL_SIZE + offset
        draw.line((x, 0, x, GRID_SIZE), fill=0, width=line_width)
        draw.line((0, y, GRID_SIZE, y), fill=0, width=line_width)

    for row in range(9):
        for col in range(9):
            value = int(board[row, col])
            if value == 0:
                continue

            font_size = rng.randint(math.floor(CELL_SIZE * 0.6), CELL_SIZE - 2)
            font = _load_font(font_size)
            text = str(value)

            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            base_x = col * CELL_SIZE + (CELL_SIZE - text_width) / 2
            base_y = row * CELL_SIZE + (CELL_SIZE - text_height) / 2

            jitter_x = rng.uniform(-CELL_SIZE * 0.1, CELL_SIZE * 0.1)
            jitter_y = rng.uniform(-CELL_SIZE * 0.1, CELL_SIZE * 0.1)

            draw.text(
                (base_x + jitter_x, base_y + jitter_y),
                text,
                fill=0,
                font=font,
            )

    array = np.array(img, dtype=np.uint8)
    return np.stack([array] * 3, axis=-1)


def generate_dataset(samples: int, rng: random.Random | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """批量生成图像与标签。"""

    rng = rng or random.Random()
    images = np.zeros((samples, GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    labels = np.zeros((samples, 9, 9), dtype=np.uint8)

    for idx in range(samples):
        board = _random_board(rng)
        image = _draw_board(board, rng)

        images[idx] = image
        labels[idx] = board

    return images, labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成合成数独识别数据集")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="生成的样本数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机数种子，便于复现",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/datasets/synthetic_sudoku.npz"),
        help="输出数据集文件路径",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="可选的 JSON 元数据输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    images, labels = generate_dataset(args.samples, rng)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, images=images, labels=labels)

    metadata = SampleMetadata(samples=args.samples)
    metadata_path = args.metadata or output_path.with_suffix(".json")
    Path(metadata_path).write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2))

    print(f"数据集已生成: {output_path} (samples={args.samples})")
    print(f"元数据: {metadata_path}")


if __name__ == "__main__":
    main()
