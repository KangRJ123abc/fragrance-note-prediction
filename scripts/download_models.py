"""Download a pinned ModelScope model and check frozen file hashes.

Downloads occur only after an explicit command or an explicit inference/training
request. Configuration comes from configs/model_sources.json. Python standard
library suffices for --help and --dry-run; actual downloads require ModelScope.
This module does not load model weights or run training.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from model_files import REPOSITORY_ROOT, calculate_sha256, read_json_object, verify_files


# ========== Model source resolution ==========

def install_compatibility_overlays(directory: Path, settings: dict[str, Any]) -> None:
    """Apply only configured, hash-checked historical compatibility files.

    Args:
        directory (Path): Downloaded model cache supplied by obtain_model.
        settings (dict[str, Any]): Role configuration from configs/model_sources.json.
    Returns:
        None: Existing frozen files remain intact; recognized upstream files are replaced.
    Raises:
        ValueError: A source/target escapes its root or its hash is unrecognized.
        RuntimeError: Hashing, source reading, or copying fails.
    """
    # 步骤1: Permit only explicit public overlays, never an arbitrary model edit.
    for overlay in settings.get("compatibility_overlays", []):
        source = (REPOSITORY_ROOT / overlay["source"]).resolve()
        target = (directory / overlay["target"]).resolve()
        if not source.is_relative_to(REPOSITORY_ROOT) or not target.is_relative_to(directory.resolve()):
            raise ValueError("应用兼容配置失败: 路径越界, 上下文: configured overlay")
        if calculate_sha256(source) != overlay["sha256"]:
            raise ValueError("应用兼容配置失败: 发布副本哈希不一致, 上下文: overlay source")
        current = calculate_sha256(target)
        if current == overlay["sha256"]:
            continue
        if current != overlay["upstream_sha256"]:
            raise ValueError("应用兼容配置失败: 未识别的上游配置, 上下文: overlay target")
        # 步骤2: Replace only the recognized metadata file, never weights/vocabulary.
        try:
            shutil.copyfile(source, target)
        except OSError as error:
            raise RuntimeError(f"复制兼容配置失败: {error}, 上下文: file={target.name}") from error

def get_model_settings(role: str) -> dict[str, Any]:
    """Resolve a model role to its frozen download and validation settings.

    Args:
        role (str): CLI or wrapper selection, either base or finetuned.
    Returns:
        dict[str, Any]: Repository ID, revision, cache directory and manifest.
    Raises:
        ValueError: The supplied role is unsupported.
        RuntimeError: Configuration cannot be read.
    """
    # 步骤1: Validate role before using it as a configuration lookup key.
    if role not in {"base", "finetuned"}:
        raise ValueError(f"解析模型失败: 角色无效, 上下文: role={role}")
    return read_json_object(REPOSITORY_ROOT / "configs/model_sources.json")[role]


def obtain_model(role: str, destination: Path | None = None) -> Path:
    """Reuse a verified model or download its exact configured revision.

    Args:
        role (str): Model role supplied by a CLI or inference/training wrapper.
        destination (Path | None): Optional CLI cache override; None uses config.
    Returns:
        Path: Verified local model directory, containing the frozen artifacts.
    Raises:
        ValueError: Publication settings are incomplete or hashes differ.
        RuntimeError: Download dependency, network, or filesystem operation fails.
    """
    # 步骤1: Reuse only a directory whose frozen hashes pass completely.
    settings = get_model_settings(role)
    directory = (destination or REPOSITORY_ROOT / settings["cache_directory"]).resolve()
    manifest_path = REPOSITORY_ROOT / settings["hash_manifest"]
    if (directory / "model.safetensors.index.json").is_file():
        install_compatibility_overlays(directory, settings)
        verify_files(directory, manifest_path)
        return directory
    # 步骤2: Never silently download an unpinned author model or a different base.
    if not settings.get("repo_id") or not settings.get("revision"):
        raise ValueError(f"下载模型失败: 请填写仓库与固定revision或使用--model-dir, 上下文: role={role}")
    try:
        from modelscope import snapshot_download
        snapshot_download(model_id=settings["repo_id"], revision=settings["revision"], local_dir=str(directory))
    except Exception as error:
        raise RuntimeError(f"下载模型失败: 请检查ModelScope依赖、版本及网络, 上下文: role={role}") from error
    # 步骤3: Validate bytes rather than trusting only the remote repository label.
    install_compatibility_overlays(directory, settings)
    verify_files(directory, manifest_path)
    return directory


def main() -> None:
    """Expose model download with a network-free dry-run mode.

    Args:
        None: Arguments are supplied by the command-line caller through argparse.
    Returns:
        None: Prints configuration or a verified local model directory.
    Raises:
        RuntimeError: Configuration, filesystem, dependency, or download fails.
        ValueError: A model source is incomplete or fails validation.
    """
    # 步骤1: Require an explicit role so an empty command never downloads weights.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("base", "finetuned"), required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    # 步骤2: Dry-run inspects settings without importing ModelScope.
    if arguments.dry_run:
        print(json.dumps(get_model_settings(arguments.role), indent=2, ensure_ascii=False))
    else:
        print(obtain_model(arguments.role, arguments.destination))


if __name__ == "__main__":
    # 步骤1: Example: python scripts/download_models.py --role base --dry-run
    main()
