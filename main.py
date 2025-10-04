from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.train import train_model
from src.utils import format_config

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

    logger.info(
        "命令行参数: %s",
        format_config(
            {
                "batch_size": args.batch_size,
                "device": args.device or "auto",
                "epochs": args.epochs,
                "image_path": image_path,
                "learning_rate": args.learning_rate,
                "model_path": model_path,
                "skip_inference": args.skip_inference,
                "synthetic_backend": args.synthetic_backend,
                "synthetic_batch_size": args.synthetic_batch_size,
                "synthetic_device": args.synthetic_device,
                "synthetic_progress_interval": args.synthetic_progress_interval,
            }
        ),
    )

    logger.info("开始训练数字识别模型……")
    train_model(
        model_path=model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        synthetic_backend=args.synthetic_backend,
        synthetic_device=(
            args.synthetic_device if args.synthetic_backend == "gpu" else None
        ),
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
