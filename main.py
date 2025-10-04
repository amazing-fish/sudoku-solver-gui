from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.train import train_model

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查
    from src.sudoku_infer import format_grid, predict_sudoku


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练数字识别模型并识别数独图片")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/digit_cnn.pt"),
        help="训练后模型的保存路径",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path("data/puzzle.png"),
        help="待识别的数独图片路径",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练的迭代轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="训练批大小",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="训练使用的学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="优化器的权重衰减系数",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adamw", "adam", "sgd"],
        help="训练使用的优化器类型",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["onecycle", "cosine", "none"],
        help="学习率调度策略",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="交叉熵损失的标签平滑系数",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="梯度裁剪的最大范数，为 0 或负值表示不裁剪",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="提前停止的耐心轮数，默认为禁用",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="若大于 0 将启用参数指数滑动平均 (EMA)，数值推荐 0.99 左右",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="验证阶段使用的批大小，默认为与训练一致",
    )
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=None,
        help="最佳模型权重的保存路径，默认为在模型文件名后追加 _best",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="禁用 CUDA 自动混合精度训练",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练与推理使用的计算设备 (cpu/cuda)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="仅训练模型而跳过数独识别",
    )
    parser.add_argument(
        "--synthetic-backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="合成数据预处理后端，可选 cpu 或 gpu",
    )
    parser.add_argument(
        "--synthetic-device",
        type=str,
        default="auto",
        help="GPU 预处理设备标识，默认为 auto 自动选择",
    )
    parser.add_argument(
        "--synthetic-batch-size",
        type=int,
        default=256,
        help="合成数据预处理批大小",
    )
    parser.add_argument(
        "--synthetic-progress-interval",
        type=float,
        default=2.0,
        help="合成数据进度日志的时间间隔（秒）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    arg_summary = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    logger.info("命令行参数: %s", arg_summary)

    logger.info("开始训练数字识别模型……")
    train_model(
        model_path=model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler=None if args.scheduler == "none" else args.scheduler,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        eval_batch_size=args.eval_batch_size,
        best_model_path=args.best_model_path,
        use_amp=None if not args.disable_amp else False,
        ema_decay=args.ema_decay,
        device=args.device,
        synthetic_backend=args.synthetic_backend,
        synthetic_device=args.synthetic_device,
        synthetic_batch_size=args.synthetic_batch_size,
        synthetic_progress_interval=args.synthetic_progress_interval,
    )

    if args.skip_inference:
        logger.info("已根据参数跳过数独识别阶段。")
        return

    try:
        from src.sudoku_infer import format_grid, predict_sudoku
    except ModuleNotFoundError as exc:  # pragma: no cover - 运行时检查
        missing = exc.name or ""
        logger.error(
            "数独识别所需的依赖未安装，缺失模块: %s。请先执行 `pip install -r requirements.txt` 或使用 --skip-inference。",
            missing,
        )
        return

    logger.info("开始进行数独图片识别……")
    if not image_path.exists():
        logger.error(
            "未找到数独图片: %s，请放置 PNG 图片或使用 --image-path 指定。",
            image_path.resolve(),
        )
        return

    grid = predict_sudoku(
        model_path=model_path,
        image_path=image_path,
        device=args.device,
    )

    logger.info("识别得到的数独棋盘：\n%s", format_grid(grid))


if __name__ == "__main__":
    main()
