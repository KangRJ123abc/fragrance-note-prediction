"""Check dataset integration using tiny invented files and mocked SDK calls.

No formal split is read or downloaded, and no ModelScope login, model loading,
training or experiment rerun occurs. Tests cover pinned configuration, path
containment, cache reuse, corruption detection and SDK argument forwarding.
Dependencies are Python's standard library and the local downloader helpers.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Any

# ------ Local helpers; ModelScope imports are mocked or deferred ------
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
from download_data import download_snapshot, get_dataset_settings, obtain_dataset, split_paths


# ========== Invented fixtures ==========

def write_synthetic(path: Path, content: str) -> None:
    """Create an invented cache fixture in a tempfile directory.

    Args:
        path (Path): Temporary destination selected by a unit test.
        content (str): Invented tiny text supplied by the test.
    Returns:
        None: Creates parent directories and writes UTF-8 bytes.
    Raises:
        RuntimeError: Fixture directory creation or file writing fails.
    """
    # 步骤1: Test writes never touch formal dataset or project source files.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    except OSError as error:
        raise RuntimeError(f"写入测试样例失败: {error}, 上下文: path={path}") from error


def synthetic_settings() -> dict[str, Any]:
    """Return a one-file configuration for cache/download behavior tests.

    Args:
        None: All fixture values are invented inside this function.
    Returns:
        dict[str, Any]: Minimal synthetic dataset ID, revision and expected hash.
    Raises:
        RuntimeError: No normal runtime exception is expected.
    """
    # 步骤1: Use tiny invented bytes, not a truncated or copied formal sample.
    return {"repo_id": "synthetic/dataset", "revision": "a" * 40,
            "cache_directory": "synthetic-cache", "files": {"train": {
                "path": "data/train.jsonl", "sha256": hashlib.sha256(b"synthetic\n").hexdigest()}}}


# ========== CPU-only data integration tests ==========

class DataDownloadTests(unittest.TestCase):
    """Exercise publication configuration and download flow without real networking."""

    def test_published_dataset_is_pinned(self) -> None:
        """Check actual source metadata; no arguments/return; raises assertions or I/O errors."""
        # 步骤1: Map short public filenames to original frozen hash entries.
        settings = get_dataset_settings()
        try:
            hashes = json.loads((REPOSITORY_ROOT / "configs/dataset_files.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self.fail(f"读取校验元数据失败: {error}, 上下文: dataset_files.json")
        self.assertEqual(settings["repo_id"], "KangRJ123abc/Fragrance-Review-Note-Pyramids")
        self.assertEqual(settings["revision"], "8275415535d4ed2d4f7dfd8ad3ef964a8a3eeae4")
        self.assertEqual(set(settings["files"]), {"train", "validation", "test"})
        for definition in settings["files"].values():
            self.assertEqual(definition["sha256"], hashes[definition["original_name"]])

    def test_path_escape_rejected(self) -> None:
        """Reject traversal paths; no arguments/return; raises assertions."""
        # 步骤1: An external configuration cannot redirect writes outside the cache.
        settings = synthetic_settings()
        settings["files"]["train"]["path"] = "../outside"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                split_paths(Path(directory), settings)

    def test_valid_cache_avoids_network(self) -> None:
        """Reuse verified invented bytes; no arguments/return; raises assertions."""
        # 步骤1: A valid cache must not import the SDK or require authentication.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic(root / "data/train.jsonl", "synthetic\n")
            with patch("download_data.get_dataset_settings", return_value=synthetic_settings()):
                with patch("download_data.download_snapshot") as download:
                    paths = obtain_dataset(root)
                    download.assert_not_called()
            self.assertEqual(paths["train"], (root / "data/train.jsonl").resolve())

    def test_bad_cache_is_not_overwritten(self) -> None:
        """Reject incorrect invented data; no arguments/return; raises assertions."""
        # 步骤1: Do not destroy a mismatched existing file in an attempted repair.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic(root / "data/train.jsonl", "different synthetic bytes")
            with patch("download_data.get_dataset_settings", return_value=synthetic_settings()):
                with patch("download_data.download_snapshot") as download:
                    with self.assertRaises(ValueError):
                        obtain_dataset(root)
                    download.assert_not_called()

    def test_missing_file_is_verified_after_mock_download(self) -> None:
        """Validate newly created tiny fixture; no arguments/return; raises assertions."""
        # 步骤1: Mock output creation, not a real network or dataset operation.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("download_data.get_dataset_settings", return_value=synthetic_settings()):
                with patch("download_data.download_snapshot", side_effect=lambda destination, settings, missing:
                           write_synthetic(destination / "data/train.jsonl", "synthetic\n")) as download:
                    paths = obtain_dataset(root)
                    self.assertEqual(download.call_args.args[2], ["train"])
            self.assertTrue(paths["train"].is_file())

    def test_empty_sdk_result_is_not_success(self) -> None:
        """Reject a no-file mock response; no arguments/return; raises assertions."""
        # 步骤1: An SDK return without matching files cannot be reported as verified.
        with tempfile.TemporaryDirectory() as directory:
            with patch("download_data.get_dataset_settings", return_value=synthetic_settings()):
                with patch("download_data.download_snapshot"):
                    with self.assertRaises(RuntimeError):
                        obtain_dataset(Path(directory))

    def test_sdk_receives_fixed_revision_and_filter(self) -> None:
        """Check official SDK arguments; no arguments/return; raises assertions."""
        # 步骤1: Fake the entire optional import so no installed SDK can perform I/O.
        module = types.ModuleType("modelscope.hub.snapshot_download")
        module.dataset_snapshot_download = Mock()
        settings = synthetic_settings()
        modules = {"modelscope": types.ModuleType("modelscope"),
                   "modelscope.hub": types.ModuleType("modelscope.hub"),
                   "modelscope.hub.snapshot_download": module}
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, modules):
                download_snapshot(Path(directory), settings, ["train"])
            module.dataset_snapshot_download.assert_called_once_with(dataset_id="synthetic/dataset",
                revision="a" * 40, local_dir=directory, allow_patterns=["data/train.jsonl"])


if __name__ == "__main__":
    # 步骤1: Example: python -m unittest discover -s tests -v
    unittest.main()
