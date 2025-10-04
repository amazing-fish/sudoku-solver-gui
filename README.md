# 数独手写数字端到端识别示例

该项目演示了一个从训练到推理的端到端流程。脚本会使用合成的字体数据训练卷积神经网络，然后识别提供的数独图片中的数字并将其转换为 9x9 数独棋盘格式。

## 环境准备

```bash
pip install -r requirements.txt
```

### 字体资源

训练与推理默认使用金山 WPS 中的 `DejaVuMathTeXGyre.ttf` 字体，路径为 `C:\ProgramData\kingsoft\office6\omath\DejaVuMathTeXGyre.ttf`。如需使用其他字体，可通过环境变量 `SUDOKU_FONT_PATH` 指定绝对路径。

## 运行

运行前请准备好待识别的数独图片（PNG 格式）。默认路径为 `data/puzzle.png`，可参考 `data/README.md` 了解图片准备要求。

执行以下命令将生成合成训练数据、训练模型并输出识别结果：

```bash
python main.py
```

训练与推理阶段会输出详细的 INFO 日志，包括命令行参数、实际使用的计算设备、数据集构建细节以及推理填充的数字数量，便于排查“选择了 CUDA 但实际运行在 CPU”等问题。

默认会训练 3 个周期，每次运行都会重新训练并在终端打印识别出来的数独棋盘内容。如果需要自定义训练与推理参数，可以使用以下可选参数：

```bash
python main.py \
  --image-path path/to/your_sudoku.png \
  --model-path models/digit_cnn.pt \
  --epochs 5 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --device cpu  # 或 cuda
```

若仅希望训练模型而暂时不进行数独识别，可添加 `--skip-inference` 参数。

## 训练与预处理一致性

训练阶段会合成带有空白单元格与 1-9 数字的数独格子图片，再通过与推理阶段完全一致的 `preprocess_cell` 流水线（含 CLAHE、自适应阈值、形态学开运算与居中裁剪）生成 28×28 归一化样本。这样模型在训练时即可看到与实际识别阶段一致的输入分布，同时学会区分空白格与数字，提高整体稳定性。

## 项目结构

```
.
├── data
│   └── README.md         # 数独图片准备说明
├── main.py               # 训练 + 推理脚本
├── requirements.txt      # 依赖声明
└── src
    ├── __init__.py
    ├── dataset.py        # 合成数独格子数据集（含空白格）
    ├── model.py          # 卷积神经网络结构
    ├── preprocess.py     # 格子预处理与空白判断
    ├── sudoku_infer.py   # 数独图片切分与推理逻辑
    └── train.py          # 训练流程
```
