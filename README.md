# 数独手写数字端到端识别示例

该项目演示了一个从训练到推理的端到端流程。脚本会使用合成的字体数据训练卷积神经网络，然后识别提供的数独图片中的数字并将其转换为 9x9 数独棋盘格式。

## 环境准备

```bash
pip install -r requirements.txt
```

## 运行

运行前请准备好待识别的数独图片（PNG 格式）。默认路径为 `data/puzzle.png`，可参考 `data/README.md` 了解图片准备要求。

执行以下命令将生成合成训练数据、训练模型并输出识别结果：

```bash
python main.py
```

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

## 项目结构

```
.
├── data
│   └── README.md         # 数独图片准备说明
├── main.py               # 训练 + 推理脚本
├── requirements.txt      # 依赖声明
└── src
    ├── __init__.py
    ├── dataset.py        # 合成手写数字数据集
    ├── model.py          # 卷积神经网络结构
    ├── sudoku_infer.py   # 数独图片切分与推理逻辑
    └── train.py          # 训练流程
```
