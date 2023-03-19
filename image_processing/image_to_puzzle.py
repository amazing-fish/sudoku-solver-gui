import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from model import SudokuCNN  # 假设你已经实现了一个名为SudokuCNN的卷积神经网络模型


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (252, 252))  # 调整图像尺寸
    normalized_image = resized_image / 255.0  # 归一化像素值
    return normalized_image


def predict_sudoku(model, image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_variable = Variable(image_tensor)
    output = model(image_variable)
    return output


def output_to_puzzle(output):
    puzzle = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            cell_probs = output[i * 9 + j].detach().numpy()
            cell_value = np.argmax(cell_probs)
            if cell_value != 0:
                puzzle[i][j] = cell_value
    return puzzle


def recognize_sudoku_puzzle(image_path, model_path):
    model = SudokuCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = preprocess_image(image_path)
    output = predict_sudoku(model, image)
    puzzle = output_to_puzzle(output)

    return puzzle


if __name__ == "__main__":
    image_path = "path/to/sudoku_image.jpg"
    model_path = "model/sudoku_model.pth"
    result = recognize_sudoku_puzzle(image_path, model_path)
    print(result)
