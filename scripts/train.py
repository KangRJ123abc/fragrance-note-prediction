"""Explicitly launch the frozen trainer with a verified baseline model.

This release wrapper downloads the pinned baseline only after the caller supplies
training data and fresh output directories. It delegates to the original trainer
without changing optimization settings. Dependencies are the download helper and
the separate historical Unsloth training environment. It never runs on import.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from download_models import obtain_model
from model_files import REPOSITORY_ROOT, verify_files


# ========== Explicit training launch ==========

def validate_paths(arguments: argparse.Namespace) -> None:
    """Check private input splits and prevent overwriting training outputs.

    Args:
        arguments (argparse.Namespace): Parsed CLI paths supplied by the user.
    Returns:
        None: Returns only if inputs exist and output directories are new.
    Raises:
        ValueError: An input file is absent, outputs overlap, or output exists.
    """
    # 步骤1: Check inputs before downloading several gigabytes of weights.
    for source in (arguments.train_file, arguments.eval_file):
        if not source.is_file():
            raise ValueError(f"检查训练输入失败: 文件不存在, 上下文: path={source}")
    # 步骤2: Fresh separate outputs preserve previous checkpoints and merged files.
    output = arguments.output_dir.resolve()
    final = arguments.final_dir.resolve()
    if output == final or output.is_relative_to(final) or final.is_relative_to(output):
        raise ValueError("检查训练输出失败: 训练与合并目录必须分开, 上下文: overlapping outputs")
    for destination in (output, final):
        if destination.exists():
            raise ValueError(f"检查训练输出失败: 目录已存在, 上下文: path={destination}")


def main() -> None:
    """Launch training only for an explicit CLI request with complete input paths.

    Args:
        None: Data, output and optional baseline paths come from argparse.
    Returns:
        None: Waits for the delegated trainer to complete successfully.
    Raises:
        ValueError: Input, output or baseline verification fails.
        RuntimeError: Model download or training subprocess fails.
    """
    # 步骤1: An argument-free invocation shows usage and performs no work.
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-file", "eval-file", "output-dir", "final-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    arguments = parser.parse_args()
    validate_paths(arguments)
    directory = arguments.model_dir.resolve() if arguments.model_dir else obtain_model("base")
    if arguments.model_dir:
        verify_files(directory, REPOSITORY_ROOT / "configs/base_model_files.json")
    # 步骤2: Keep original training configuration and pass every path explicitly.
    command = [sys.executable, str(REPOSITORY_ROOT / "src/train_qwen35_4b_v2_chat_l3_fixed_unsloth.py"),
               "--model-dir", str(directory), "--train-file", str(arguments.train_file.resolve()),
               "--eval-file", str(arguments.eval_file.resolve()), "--output-dir", str(arguments.output_dir.resolve()),
               "--final-dir", str(arguments.final_dir.resolve())]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"启动训练失败: {error}, 上下文: trainer exit") from error


if __name__ == "__main__":
    # 步骤1: Example: python scripts/train.py --help
    main()
