"""Download the published formal dataset at a fixed ModelScope revision.

The official SDK handles local ModelScope login when the service requires it.
Files are checked against frozen SHA-256 values without resplitting, rewriting,
or training. Help, planning and cached-file checks need only the standard library;
actual downloads additionally need ModelScope. No credentials are accepted by CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model_files import REPOSITORY_ROOT, calculate_sha256, read_json_object


# ========== Dataset configuration and cache verification ==========

def get_dataset_settings() -> dict[str, Any]:
    """Read the immutable dataset source and original split checksums.

    Args:
        None: The repository-local JSON file is the only configuration source.
    Returns:
        dict[str, Any]: Dataset ID, revision, cache location and split definitions.
    Raises:
        RuntimeError: Reading or decoding the configuration fails.
        ValueError: The configuration does not contain a JSON object.
    """
    # 步骤1: Keep configuration separate from downloader implementation.
    return read_json_object(REPOSITORY_ROOT / "configs/dataset_sources.json")


def split_paths(directory: Path, settings: dict[str, Any]) -> dict[str, Path]:
    """Resolve configured paths without permitting escape from the cache root.

    Args:
        directory (Path): Local cache root selected by the caller or configuration.
        settings (dict[str, Any]): Split definitions from get_dataset_settings.
    Returns:
        dict[str, Path]: Resolved train, validation and test file paths.
    Raises:
        ValueError: A configured file path resolves outside the cache root.
    """
    # 步骤1: Treat configuration paths as external input before any file operation.
    root = directory.resolve()
    paths = {}
    for name, definition in settings["files"].items():
        path = (root / definition["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"解析数据路径失败: 超出缓存目录, 上下文: split={name}")
        paths[name] = path
    return paths


def missing_splits(directory: Path, settings: dict[str, Any]) -> list[str]:
    """Validate existing data files and list absent splits without changing them.

    Args:
        directory (Path): Local dataset cache directory supplied by the workflow.
        settings (dict[str, Any]): Expected filenames and SHA-256 values.
    Returns:
        list[str]: Missing split names, empty when every frozen file is verified.
    Raises:
        ValueError: An existing file hash differs, rather than silently overwriting it.
        RuntimeError: Hashing an existing file fails.
    """
    # 步骤1: Preserve incorrect existing files for inspection; require a fresh cache.
    missing = []
    for name, path in split_paths(directory, settings).items():
        if not path.exists():
            missing.append(name)
        elif calculate_sha256(path) != settings["files"][name]["sha256"]:
            raise ValueError(f"校验数据失败: SHA-256不同，请使用新的缓存目录, 上下文: split={name}")
    return missing


# ========== Explicit download entry ==========

def download_snapshot(directory: Path, settings: dict[str, Any], missing: list[str]) -> None:
    """Use the official SDK to download only absent splits at the fixed revision.

    Args:
        directory (Path): Download destination selected by obtain_dataset.
        settings (dict[str, Any]): Published dataset ID and immutable revision.
        missing (list[str]): Verified absent split names from missing_splits.
    Returns:
        None: The SDK populates the selected cache using its local login state.
    Raises:
        RuntimeError: SDK availability, authentication, networking or writing fails.
    """
    # 步骤1: Delay SDK import so CPU-only configuration checks need no extra packages.
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
        dataset_snapshot_download(dataset_id=settings["repo_id"], revision=settings["revision"],
            local_dir=str(directory), allow_patterns=[settings["files"][name]["path"] for name in missing])
    except Exception as error:
        # 步骤2: Never claim success when the service requires local account login.
        raise RuntimeError(
            f"下载数据失败: 请检查ModelScope SDK、网络和本机登录状态, 上下文: dataset={settings['repo_id']}"
        ) from error


def obtain_dataset(destination: Path | None = None) -> dict[str, Path]:
    """Reuse verified frozen splits or explicitly download missing ones.

    Args:
        destination (Path | None): CLI cache override; None uses data/formal.
    Returns:
        dict[str, Path]: Verified local file paths for train, validation and test.
    Raises:
        ValueError: Existing bytes differ from the frozen experiment files.
        RuntimeError: Configuration, SDK, authentication or final verification fails.
    """
    # 步骤1: A complete verified cache needs neither credentials nor network access.
    settings = get_dataset_settings()
    directory = (destination or REPOSITORY_ROOT / settings["cache_directory"]).resolve()
    missing = missing_splits(directory, settings)
    if missing:
        download_snapshot(directory, settings, missing)
    # 步骤2: Verify the actual downloaded bytes, not just an SDK success response.
    remaining = missing_splits(directory, settings)
    if remaining:
        raise RuntimeError(f"校验下载结果失败: 文件未完整写入, 上下文: missing={remaining}")
    return split_paths(directory, settings)


def main() -> None:
    """Print a download plan or explicitly obtain the three formal split files.

    Args:
        None: CLI supplies an optional destination and --dry-run flag.
    Returns:
        None: Prints configured source information or verified local split paths.
    Raises:
        RuntimeError: Input configuration or requested downloading fails.
        ValueError: Cache bytes or configured target paths are invalid.
    """
    # 步骤1: Dry-run never imports ModelScope or initiates a file download.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.dry_run:
        print(json.dumps(get_dataset_settings(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps({name: str(path) for name, path in obtain_dataset(arguments.destination).items()}, indent=2))


if __name__ == "__main__":
    # 步骤1: Example: python scripts/download_data.py --dry-run
    main()
