from __future__ import annotations

from pathlib import Path

from src.sudoku_infer import format_grid, predict_sudoku
from src.train import train_model


def main() -> None:
    model_path = Path("models/digit_cnn.pt")
    image_path = Path("data/puzzle.png")

    print("开始训练数字识别模型……")
    train_model(model_path=model_path, epochs=3)

    print("开始进行数独图片识别……")
    grid = predict_sudoku(model_path=model_path, image_path=image_path)

    print("识别得到的数独棋盘：")
    print(format_grid(grid))


if __name__ == "__main__":
    main()
