"""Read release configuration and validate model artifacts without loading weights.

Shared helpers keep model provenance separate from inference. Verification streams
file bytes and rejects paths outside the requested model directory. Dependencies
are Python's standard library; no network or GPU operation occurs in this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# ------ Repository configuration ------
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


# ========== Configuration and file verification ==========

def read_json_object(file_path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object with explicit source context.

    Args:
        file_path (Path): Configuration or manifest path supplied by a caller.
    Returns:
        dict[str, Any]: Decoded mapping; keys depend on the selected file.
    Raises:
        RuntimeError: Reading or JSON decoding fails.
        ValueError: The document is not a JSON object.
    """
    # 步骤1: Preserve the path in every external-input error.
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取JSON失败: {error}, 上下文: path={file_path}") from error
    # 步骤2: Reject arrays before callers access configuration keys.
    if not isinstance(payload, dict):
        raise ValueError(f"校验JSON失败: 需要对象, 上下文: path={file_path}")
    return payload


def calculate_sha256(file_path: Path) -> str:
    """Stream one file into a SHA-256 digest without allocating weight tensors.

    Args:
        file_path (Path): Artifact path supplied by the manifest verifier.
    Returns:
        str: Lowercase 64-character hexadecimal digest.
    Raises:
        RuntimeError: Opening or reading the file fails.
    """
    # 步骤1: Stream large shards to keep verification independent of RAM size.
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError(f"计算文件哈希失败: {error}, 上下文: path={file_path}") from error
    return digest.hexdigest()


def verify_files(directory: Path, manifest_path: Path) -> int:
    """Validate every listed artifact and reject path traversal.

    Args:
        directory (Path): Artifact root supplied by CLI or model download code.
        manifest_path (Path): JSON object mapping relative paths to SHA-256 strings.
    Returns:
        int: Number of verified files, normally nine for frozen model manifests.
    Raises:
        ValueError: A manifest path escapes the root or a file hash differs.
        RuntimeError: Reading a manifest or listed artifact fails.
    """
    # 步骤1: Resolve every target before trusting externally supplied manifests.
    root = directory.resolve()
    manifest = read_json_object(manifest_path)
    for relative_name, expected_digest in manifest.items():
        target = (root / relative_name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"校验路径失败: 超出根目录, 上下文: file={relative_name}")
        actual_digest = calculate_sha256(target)
        # 步骤2: Fail closed rather than substitute an updated upstream artifact.
        if actual_digest != expected_digest:
            raise ValueError(f"校验模型失败: SHA-256不一致, 上下文: file={relative_name}")
    return len(manifest)


if __name__ == "__main__":
    # 步骤1: Print a non-mutating usage example instead of downloading anything.
    print("Example: python scripts/verify_files.py --root . --manifest SHA256SUMS.json")
