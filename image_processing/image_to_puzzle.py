import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from image_processing.model.sudoku_resnet import SudokuResNet


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "sudoku_model.pth")


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    resized_image = cv2.resize(image, (252, 252))  # 调整图像尺寸
    normalized_image = resized_image.astype("float32") / 255.0  # 归一化像素值

    # ResNet 期望三通道输入，这里将灰度图复制到三个通道
    image_3ch = np.stack([normalized_image] * 3, axis=-1)
    return image_3ch


def predict_sudoku(model, image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    return output


def output_to_puzzle(output):
    predictions = output.squeeze(0).detach().cpu().numpy()
    puzzle = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            cell_probs = predictions[i, j]
            cell_value = int(np.argmax(cell_probs))
            if cell_value != 0:
                puzzle[i][j] = cell_value
    return puzzle


def _load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型文件: {model_path}\n"
            f"请将训练好的模型权重放置在此路径或在调用时提供 model_path 参数。"
        )

    model = SudokuResNet()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def recognize_sudoku_puzzle(image_path, model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model = _load_model(model_path)
    image = preprocess_image(image_path)
    output = predict_sudoku(model, image)
    puzzle = output_to_puzzle(output)

    return puzzle


if __name__ == "__main__":
    image_path = "dataset/img.png"
    model_path = DEFAULT_MODEL_PATH
    result = recognize_sudoku_puzzle(image_path, model_path)
    print(result)
