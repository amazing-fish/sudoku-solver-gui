from __future__ import annotations

from pathlib import Path
from typing import List
import logging

import cv2
import numpy as np
import torch

from .model import create_model
from .preprocess import preprocess_cell


logger = logging.getLogger(__name__)


def _load_cells(image_path: str | Path) -> List[np.ndarray | None]:
    logger.info("加载数独图片: %s", Path(image_path).resolve())
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法加载图片: {image_path}")

    height, width = image.shape
    cell_h = height // 9
    cell_w = width // 9

    cells: List[np.ndarray | None] = []
    for row in range(9):
        for col in range(9):
            y0 = row * cell_h
            y1 = (row + 1) * cell_h
            x0 = col * cell_w
            x1 = (col + 1) * cell_w
            margin_y = max(1, cell_h // 12)
            margin_x = max(1, cell_w // 12)
            cell = image[y0 + margin_y : y1 - margin_y, x0 + margin_x : x1 - margin_x]
            if cell.size == 0:
                cells.append(None)
                continue
            result = preprocess_cell(cell)
            cells.append(None if result.is_blank else result.data)
    logger.info("完成单元格切分，总单元格数量: %s，检测到待预测单元格: %s", len(cells), sum(cell is not None for cell in cells))
    return cells


def load_model(model_path: str | Path, device: str | torch.device | None = None) -> torch.nn.Module:
    if isinstance(device, torch.device):
        device_obj = device
    else:
        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(device_str).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("推理请求使用 CUDA 设备，但当前环境不可用，自动切换到 CPU。")
            device_str = "cpu"
        device_obj = torch.device(device_str)

    logger.info("加载模型: %s，目标设备: %s", Path(model_path).resolve(), device_obj)
    model = create_model()
    state_dict = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(state_dict)
    model.to(device_obj)
    model.eval()
    logger.info("模型加载完成，进入评估模式。")
    return model


def predict_sudoku(
    model_path: str | Path,
    image_path: str | Path,
    device: str | torch.device | None = None,
) -> List[List[int]]:
    if isinstance(device, torch.device):
        device_obj = device
    else:
        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(device_str).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("推理请求使用 CUDA 设备，但当前环境不可用，自动切换到 CPU。")
            device_str = "cpu"
        device_obj = torch.device(device_str)

    logger.info("推理阶段使用设备: %s", device_obj)
    model = load_model(model_path, device_obj)
    cells = _load_cells(image_path)

    sudoku_grid: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)]

    digit_batches: List[np.ndarray] = []
    positions: List[tuple[int, int]] = []

    for idx, cell in enumerate(cells):
        row, col = divmod(idx, 9)
        if cell is None:
            sudoku_grid[row][col] = 0
            continue
        digit_batches.append(cell)
        positions.append((row, col))

    if digit_batches:
        inputs = np.stack(digit_batches)
        inputs = torch.from_numpy(inputs).unsqueeze(1).to(device_obj)
        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        for (row, col), value in zip(positions, predictions):
            sudoku_grid[row][col] = int(value)
        logger.info("推理完成，填充数字数量: %s", len(positions))

    return sudoku_grid


def format_grid(grid: List[List[int]]) -> str:
    lines = []
    for row in grid:
        line = " ".join(str(num) if num != 0 else "." for num in row)
        lines.append(line)
    return "\n".join(lines)
