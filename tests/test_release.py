"""Check release helpers on CPU without downloads, model loading, or training.

Tests inspect syntax, prompt construction, JSON parsing, and hash-verifier failure
cases. Temporary fixtures are synthetic and isolated from formal experiment data.
Dependencies are Python's standard library plus NumPy for the original scorer.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

# ------ Local imports only; no optional GPU libraries are imported ------
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from model_files import calculate_sha256, read_json_object, verify_files
from download_models import get_model_settings, obtain_model
from predict import build_messages


# ========== Synthetic fixtures ==========

def write_fixture(path: Path, text: str) -> None:
    """Write a synthetic test fixture with contextual I/O errors.

    Args:
        path (Path): Isolated temporary destination supplied by a test.
        text (str): Invented content supplied by that test, never formal data.
    Returns:
        None: The requested fixture is written as UTF-8.
    Raises:
        RuntimeError: Writing fails.
    """
    # 步骤1: Keep test writes confined to tempfile-created directories.
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"写入测试样例失败: {error}, 上下文: path={path}") from error


# ========== CPU-only release tests ==========

class ReleaseTests(unittest.TestCase):
    """Exercise public helper invariants with no GPU or network dependencies."""

    def test_python_syntax(self) -> None:
        """Parse all release Python files; no arguments/return; raises syntax or I/O errors."""
        # 步骤1: Parse instead of importing modules that need CUDA packages.
        for path in REPOSITORY_ROOT.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, UnicodeError, SyntaxError) as error:
                self.fail(f"静态检查失败: {error}, 上下文: path={path}")

    def test_baseline_is_pinned(self) -> None:
        """Check baseline source configuration; no arguments/return; raises assertions."""
        # 步骤1: Distinguish original distribution from similarly named base models.
        settings = get_model_settings("base")
        self.assertEqual(settings["repo_id"], "unsloth/Qwen3___5-4B")
        self.assertEqual(len(settings["revision"]), 40)

    def test_unpublished_model_fails_without_network(self) -> None:
        """Reject missing publication ID; no arguments/return; raises assertions."""
        # 步骤1: An empty fresh directory cannot satisfy the cache or source checks.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                obtain_model("finetuned", Path(directory))

    def test_prompt_excludes_assistant(self) -> None:
        """Check frozen prompt assembly; no arguments/return; raises assertions."""
        # 步骤1: Never supply a gold assistant answer to the inference helper.
        messages = build_messages("Synthetic citrus review.")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Synthetic citrus review.", messages[1]["content"])
        self.assertIn("top_notes", messages[0]["content"])

    def test_empty_review_rejected(self) -> None:
        """Reject empty text; no arguments/return; raises assertions."""
        # 步骤1: Fail before download or inference on unusable external input.
        with self.assertRaises(ValueError):
            build_messages(" ")

    def test_verifier_accepts_exact_bytes(self) -> None:
        """Verify a synthetic file; no arguments/return; raises assertions or I/O errors."""
        # 步骤1: Use a tiny fixture, not gigabyte model weights, for this unit test.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "sample.txt", "synthetic")
            digest = calculate_sha256(root / "sample.txt")
            write_fixture(root / "manifest.json", json.dumps({"sample.txt": digest}))
            self.assertEqual(verify_files(root, root / "manifest.json"), 1)

    def test_verifier_rejects_changed_bytes(self) -> None:
        """Detect corrupted fixtures; no arguments/return; raises assertions or I/O errors."""
        # 步骤1: Never accept updated or corrupted model bytes under a frozen label.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "sample.txt", "synthetic")
            write_fixture(root / "manifest.json", json.dumps({"sample.txt": "0" * 64}))
            with self.assertRaises(ValueError):
                verify_files(root, root / "manifest.json")

    def test_verifier_rejects_path_traversal(self) -> None:
        """Reject external manifest targets; no arguments/return; raises assertions."""
        # 步骤1: A malicious manifest must not cause reads outside its root.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "manifest.json", json.dumps({"../outside": "0" * 64}))
            with self.assertRaises(ValueError):
                verify_files(root, root / "manifest.json")

    def test_missing_tier_preserves_recoverable_notes(self) -> None:
        """Check original parser semantics; no arguments/return; raises assertions."""
        # 步骤1: Schema failure is not equivalent to dropping all recoverable tiers.
        from eval_qwen35_4b_chat_l3_unsloth import parse_prediction
        parsed = parse_prediction('{"top_notes":["Bergamot"],"middle_notes":[]}')
        self.assertFalse(parsed.has_valid_schema)
        self.assertEqual(parsed.canonical_notes["top_notes"], ["bergamot"])
        self.assertEqual(parsed.canonical_notes["base_notes"], [])


if __name__ == "__main__":
    # 步骤1: Example: python -m unittest discover -s tests -v
    unittest.main()
