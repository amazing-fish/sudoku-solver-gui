"""生成数独识别模型占位文件的辅助脚本。"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from . import DEFAULT_MODEL_PATH

PLACEHOLDER_CONTENT: Dict[str, Any] = {
    "name": "sudoku-solver-gui-placeholder",
    "description": "占位模型文件，仅用于避免缺失权重时报错。",
    "version": 1,
}


def build_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """创建一个模型占位文件并返回生成的路径。"""

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(PLACEHOLDER_CONTENT)
    payload["generated_at"] = datetime.now(UTC).isoformat()

    model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成数独识别模型占位文件")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="模型文件输出路径 (默认为项目内的默认路径)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = build_model(args.output)
    print(f"模型文件已生成: {model_path}")


if __name__ == "__main__":
    main()
