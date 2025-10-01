# 基于图像识别的数独求解器（带界面）

一个基于图像识别和卷积神经网络的数独求解器，具有图形用户界面。用户可以手动输入数独问题，也可以通过导入包含数独问题的图片来自动识别并求解。

## 功能

1. 手动输入数独题目。
2. 通过导入图片自动识别数独题目。
3. 验证输入的数独题目是否有效。
4. 求解数独问题。
5. 显示求解结果。
6. 重置输入和结果。

## 技术选型

1. Python 3
2. 图形用户界面：Tkinter
3. 数独求解算法：回溯法
4. 图像处理：OpenCV
5. 卷积神经网络：TensorFlow 或 PyTorch

## 项目结构

```
sudoku-solver-gui/
│
├─ sudoku/
│   ├─ solver.py           # 数独求解算法实现
│   └─ validator.py        # 数独输入验证实现
│
├─ gui/
│   └─ interface.py        # 图形用户界面实现
│
├─ image_processing/
│   ├─ image_to_puzzle.py  # 图像处理和数独识别实现
│   └─ model/              # 存放卷积神经网络模型
│
├─ main.py                 # 主程序入口
├─ requirements.txt        # 项目依赖列表
└─ README.md               # 项目文档
```

## 如何使用

### 安装依赖

在项目根目录下运行以下命令以安装所需的库：

```
pip install -r requirements.txt
```

### 运行程序

在项目根目录下运行以下命令以启动数独求解器：

```
python main.py
```

### 构建默认模型文件

首次运行或上传图片识别数独前，请确保已经生成模型占位文件。项目提供了
一个便捷脚本来创建轻量级占位模型（未训练，仅用于避免缺失文件导致的错
误）。

在项目根目录执行：

```
python -m image_processing.model.build_model
```

脚本会在 `image_processing/model/` 目录下生成 `sudoku_model.pth` 文件。若你
拥有真实的训练权重，可直接用其替换生成的占位文件；提交代码前请勿将真实
的 `.pth` 或 `.pt` 文件加入版本库。

> 💡 如果希望使用真实模型进行识别，请确保安装了 `torch` 与 `torchvision`
> （版本需与训练权重兼容），并将模型权重保存为 `sudoku_model.pth`/`pt`。
> 程序会优先尝试加载该文件；若加载失败或依赖缺失，则自动退回到占位模型
> 并给出日志提示。

### 使用说明

1. 在图形界面中手动输入数独题目，或点击 "Load Image" 按钮导入包含数独问题的图片。
2. 点击 "Solve" 按钮以求解数独问题。
3. 查看求解结果。
4. 点击 "Reset" 按钮以清空输入和结果。

## 学习文档

### 教程

1. Python基础：https://docs.python.org/3/tutorial/index.html
2. Tkinter教程：https://tkdocs.com/tutorial/index.html
3. 回溯法数独求解：https://www.geeksforgeeks.org/sudoku-backtracking-7/
4. OpenCV教程：https://docs.opencv.org/master/d9/df8/tutorial_root.html
5. TensorFlow教程：https://www.tensorflow.org/tutorials
6. PyTorch教程：https://pytorch.org/tutorials/

### 学习模块与顺序

1. 图形用户界面（Tkinter）
   - 窗口与布局
   - 控件（按钮、标签、文本框等）
   - 事件处理与绑定
2. 数独求解算法（回溯法）
   - 基本概念与原理
   - 算法实现与优化
3. 图像处理（OpenCV）
   - 读取、显示和保存图像
   - 图像预处理（缩放、灰度、二值化等）
   - 图像分割和提取
4. 卷积神经网络（TensorFlow 或 PyTorch）
   - 神经网络基本概念
   - 构建和训练卷积神经网络
   - 使用预训练模型进行图像识别
5. 集成和调试
   - 将各个模块组合到一起
   - 调试和优化程序性能
   - 测试程序功能

## 贡献指南

如果你对本项目感兴趣并希望对其进行改进，请遵循以下步骤：

1. Fork 本项目到你的 GitHub 账户。
2. 克隆你的 Fork 到本地。
3. 创建一个新的分支以进行更改。
4. 完成你的修改并提交更改。
5. 在 GitHub 上创建一个 Pull Request 以请求合并到主项目。

在提交 Pull Request 之前，请确保你的代码符合项目的代码风格，并在项目的各个模块中进行了充分的测试。

## 许可证

该项目采用 [MIT 许可证]()。在遵循许可证要求的前提下，你可以自由地使用、修改和分发本项目的代码。