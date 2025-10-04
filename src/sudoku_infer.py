from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from .model import create_model

_MEAN = 0.1307
_STD = 0.3081


def _is_blank(binary_inv: np.ndarray) -> bool:
    """根据前景像素比例与包围盒判断是否为空白。"""

    non_zero = cv2.countNonZero(binary_inv)
    total = binary_inv.shape[0] * binary_inv.shape[1]
    if non_zero == 0:
        return True

    ratio = non_zero / total
    if ratio < 0.015:
        return True

    coords = cv2.findNonZero(binary_inv)
    if coords is None:
        return True

    x, y, w, h = cv2.boundingRect(coords)
    if w * h < 0.02 * total:
        return True

    return False


def _preprocess_cell(cell: np.ndarray) -> np.ndarray | None:
    """Convert a raw Sudoku cell to a normalized tensor-ready array."""

    blur = cv2.GaussianBlur(cell, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    if _is_blank(cleaned):
        return None

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = cleaned[y : y + h, x : x + w]
    side = max(w, h)
    padded = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - w) // 2
    y_offset = (side - h) // 2
    padded[y_offset : y_offset + h, x_offset : x_offset + w] = digit
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - _MEAN) / _STD
    return normalized


def _load_cells(image_path: str | Path) -> List[np.ndarray | None]:
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
            processed = _preprocess_cell(cell)
            cells.append(processed)
    return cells


def load_model(model_path: str | Path, device: str | torch.device | None = None) -> torch.nn.Module:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_sudoku(
    model_path: str | Path,
    image_path: str | Path,
    device: str | torch.device | None = None,
) -> List[List[int]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(model_path, device)
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
        inputs = torch.from_numpy(inputs).unsqueeze(1).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        for (row, col), value in zip(positions, predictions):
            sudoku_grid[row][col] = int(value)

    return sudoku_grid


def format_grid(grid: List[List[int]]) -> str:
    lines = []
    for row in grid:
        line = " ".join(str(num) if num != 0 else "." for num in row)
        lines.append(line)
    return "\n".join(lines)
