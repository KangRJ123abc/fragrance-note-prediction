"""Verify a release manifest using streaming SHA-256 checks.

The command is read-only and accepts either the code or model release folder.
It depends only on Python's standard library and the adjacent model_files helper.
It never executes models, downloads artifacts, or changes manifest contents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_files import verify_files


# ========== Read-only verification entry ==========

def main() -> None:
    """Parse artifact root and manifest paths and verify all listed files.

    Args:
        None: Root and manifest are supplied through command-line arguments.
    Returns:
        None: Prints the number of successfully verified artifacts.
    Raises:
        RuntimeError: Manifest or file reads fail.
        ValueError: Paths escape the root or file hashes differ.
    """
    # 步骤1: Require explicit paths to prevent checking the wrong release silently.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    # 步骤2: Reuse the same verifier as model downloads.
    verified_count = verify_files(arguments.root, arguments.manifest)
    print(f"PASS: {verified_count} files verified")


if __name__ == "__main__":
    # 步骤1: Example: python scripts/verify_files.py --root . --manifest SHA256SUMS.json
    main()
