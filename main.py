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
        default=1e-4,
        help="AdamW 优化器的权重衰减系数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练与推理使用的计算设备 (cpu/cuda)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "reduce_on_plateau", "none"],
        help="学习率调度策略",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="早停耐心值（连续多少轮无提升后停止训练）",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="视为指标提升所需的最小增量",
    )
    parser.add_argument(
        "--target-metric",
        type=str,
        default="f1_macro",
        help="用于早停与保存最佳模型的评估指标名称",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="梯度裁剪的最大范数，设为 <=0 则关闭",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭自动混合精度训练",
    )
    parser.add_argument(
        "--no-save-last",
        action="store_true",
        help="仅保存最佳模型，不额外保存最后一轮模型",
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

    logger.info(
        "命令行参数: model_path=%s, image_path=%s, epochs=%s, batch_size=%s, learning_rate=%s, device=%s, skip_inference=%s, synthetic_backend=%s, synthetic_device=%s, synthetic_batch_size=%s, synthetic_progress_interval=%s",
        model_path,
        image_path,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.device,
        args.skip_inference,
        args.synthetic_backend,
        args.synthetic_device,
        args.synthetic_batch_size,
        args.synthetic_progress_interval,
    )

    logger.info("开始训练数字识别模型……")
    train_model(
        model_path=model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        target_metric=args.target_metric,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        use_amp=not args.no_amp,
        save_last=not args.no_save_last,
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
