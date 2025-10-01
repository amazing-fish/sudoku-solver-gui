from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from image_processing.model import DEFAULT_MODEL_PATH, ensure_model_file
from image_processing.model.sudoku_resnet import SudokuResNet


def preprocess_image(image_path: Path | str) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    resized_image = cv2.resize(image, (252, 252))
    normalized_image = resized_image.astype("float32") / 255.0
    image_3ch = np.stack([normalized_image] * 3, axis=-1)
    return image_3ch


def predict_sudoku(model: SudokuResNet, image: np.ndarray) -> np.ndarray:
    image_tensor = np.expand_dims(image, axis=0)
    output = model(image_tensor)
    return output


def output_to_puzzle(output: np.ndarray) -> np.ndarray:
    predictions = np.squeeze(output, axis=0)
    puzzle = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            cell_probs = predictions[i, j]
            cell_value = int(np.argmax(cell_probs))
            if cell_value != 0:
                puzzle[i][j] = cell_value
    return puzzle


def _load_model(model_path: Path | str) -> SudokuResNet:
    model_path = ensure_model_file(model_path)

    # 占位模型无需实际加载权重，保持接口一致即可
    model = SudokuResNet()
    model.eval()
    return model


def recognize_sudoku_puzzle(image_path: Path | str, model_path: Path | str | None = None) -> np.ndarray:
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model = _load_model(model_path)
    image = preprocess_image(image_path)
    output = predict_sudoku(model, image)
    puzzle = output_to_puzzle(output)

    return puzzle


if __name__ == "__main__":
    image_path = Path("dataset/img.png")
    model_path = DEFAULT_MODEL_PATH
    result = recognize_sudoku_puzzle(image_path, model_path)
    print(result)
