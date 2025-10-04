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

默认会训练 3 个周期，并在终端输出包含损失、准确率、F1 等多种指标的详细日志，同时自动保存表现最佳的模型权重。如果需要自定义训练与推理参数，可以使用以下可选参数：

```bash
python main.py \
  --image-path path/to/your_sudoku.png \
  --model-path models/digit_cnn.pt \
  --epochs 5 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --patience 15 \
  --min-delta 5e-5 \
  --target-metric f1_macro \
  --grad-clip-norm 1.0 \
  --no-amp \
  --no-save-last \
  --device cpu  # 或 cuda
  --synthetic-backend gpu \
  --synthetic-device cuda:0 \
  --synthetic-batch-size 512 \
  --synthetic-progress-interval 1.0
```

若仅希望训练模型而暂时不进行数独识别，可添加 `--skip-inference` 参数。

## 训练与预处理一致性

训练阶段会合成带有空白单元格与 1-9 数字的数独格子图片，再通过与推理阶段完全一致的预处理流水线生成 28×28 归一化样本。该流水线包含 CLAHE、自适应阈值、形态学开运算与居中裁剪，并可在 CPU 与 GPU 间自由切换：

- 默认使用 CPU 版 `preprocess_cell` 顺序处理；
- 指定 `--synthetic-backend gpu` 时会启用批量版 `preprocess_cell_batch`，在 GPU 上并行完成模糊、局部对比度增强、阈值化与形态学操作，再回落到 CPU 完成裁剪与归一化，以保证训练与推理输入保持一致。

生成的数据会自动缓存在项目根目录下的 `.cache` 文件夹中，后续运行在字体与预处理配置未变化的情况下会直接加载缓存，大幅缩短每次训练前的等待时间。如需重新生成数据，可删除对应缓存文件或清空 `.cache` 目录。

当需要观察合成过程时，可通过 `--synthetic-progress-interval` 控制日志刷新频率；日志会额外展示累计尝试次数与当前 digit 的尝试次数，便于诊断空白格生成等问题。对于拥有 GPU 的环境，可通过 `--synthetic-backend gpu --synthetic-device cuda:0` 启用 GPU 预处理，并用 `--synthetic-batch-size` 调整 GPU 上的批量处理规模，以提高吞吐量。
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
