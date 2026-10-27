"""Share safe I/O and original scoring primitives for release reconstructions.

These helpers do not load a model. They validate formal chat splits, preserve
zero-based physical source-line indexes, and reuse the archived domain parser.
Dependencies are the standard library and NumPy through the original scorer.
This is release-time code, not a recovered historical executor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from model_files import REPOSITORY_ROOT, calculate_sha256

# ------ Original parser, not a newly invented normalization rule ------
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from eval_qwen35_4b_chat_l3_unsloth import REQUIRED_NOTE_KEYS, calculate_set_scores, parse_prediction


# ========== Safe artifact I/O ==========

def read_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL objects and retain original physical line positions.

    Args:
        path (Path): Input JSONL supplied by the command-line caller.
    Returns:
        list[dict[str, Any]]: Records with an internal _physical_line_index field.
    Raises:
        RuntimeError: File access, encoding, or JSON decoding fails.
        ValueError: A nonblank line is not a JSON object.
    """
    # 步骤1: enumerate precedes blank-line removal, matching the original loader.
    records = []
    try:
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"读取记录失败: 需要对象, 上下文: path={path}, index={index}")
                record["_physical_line_index"] = index
                records.append(record)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取JSONL失败: {error}, 上下文: path={path}") from error
    return records


def write_json(path: Path, payload: Any) -> None:
    """Write a new JSON artifact without overwriting an existing file.

    Args:
        path (Path): New destination chosen by the reconstruction workflow.
        payload (Any): JSON-serializable result or provenance mapping.
    Returns:
        None: A UTF-8 result file is created.
    Raises:
        RuntimeError: Serialization, encoding, or filesystem operations fail.
    """
    # 步骤1: Exclusive creation preserves previous reconstruction results.
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"写入JSON失败: {error}, 上下文: path={path}") from error


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Persist JSONL records after removing internal loader metadata.

    Args:
        path (Path): New output path selected by the reconstruction workflow.
        records (list[dict[str, Any]]): Original or reconstructed generation rows.
    Returns:
        None: Writes records in supplied order, with no reference edits.
    Raises:
        RuntimeError: Serialization, encoding, or file creation fails.
    """
    # 步骤1: Loader-only indexes are not part of the original generation schema.
    try:
        with path.open("x", encoding="utf-8") as stream:
            for record in records:
                public = {key: value for key, value in record.items() if key != "_physical_line_index"}
                stream.write(json.dumps(public, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"写入JSONL失败: {error}, 上下文: path={path}") from error


def create_output_directory(path: Path) -> None:
    """Create one fresh result directory, rejecting existing destinations.

    Args:
        path (Path): Explicit result directory supplied by the CLI caller.
    Returns:
        None: Creates the target and any missing parent directories.
    Raises:
        RuntimeError: The path exists or directory creation fails.
    """
    # 步骤1: Never write reconstruction artifacts into the historical run itself.
    try:
        path.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise RuntimeError(f"创建输出目录失败: {error}, 上下文: path={path}") from error


# ========== Label and pairing validation ==========

def flatten_notes(notes: dict[str, list[str]]) -> set[str]:
    """Union the three normalized tiers for the original flat set metric.

    Args:
        notes (dict[str, list[str]]): Canonical notes produced by the original parser.
    Returns:
        set[str]: Unique terms across all tiers, without synonym merging.
    Raises:
        KeyError: A canonical required tier is absent.
    """
    # 步骤1: Cross-tier duplicates count only once in flat note-set metrics.
    return set().union(*(set(notes[key]) for key in REQUIRED_NOTE_KEYS))


def load_labels(path: Path, level: int | None = None) -> list[dict[str, Any]]:
    """Validate a chat split and extract references without altering prompts.

    Args:
        path (Path): Formal chat JSONL downloaded or otherwise selected by the caller.
        level (int | None): Optional metadata level filter; None retains all.
    Returns:
        list[dict[str, Any]]: Index, level, audit row ID, and normalized notes.
    Raises:
        RuntimeError: Input reading fails.
        ValueError: Messages, metadata level, or reference schema are malformed.
    """
    # 步骤1: Validate all source rows even when only Level 3 will be retained.
    labels = []
    for record in read_records(path):
        messages = record.get("messages", [])
        metadata = record.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as error:
                raise ValueError(f"读取元数据失败: {error}, 上下文: path={path}") from error
        if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
            raise ValueError(f"校验样本失败: messages不是对象列表, 上下文: path={path}")
        if not isinstance(metadata, dict) or [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"校验样本失败: metadata/messages格式无效, 上下文: path={path}")
        sample_level = metadata.get("scent_information_level")
        parsed = parse_prediction(messages[2]["content"])
        if sample_level not in (2, 3) or not parsed.has_valid_schema:
            raise ValueError(f"校验标签失败: level或参考结构无效, 上下文: path={path}")
        if level is None or sample_level == level:
            labels.append({"source_index": record["_physical_line_index"], "level": sample_level,
                           "row_id": metadata.get("row_id"), "notes": parsed.canonical_notes})
    if not labels:
        raise ValueError(f"校验标签失败: 筛选后无记录, 上下文: path={path}, level={level}")
    return labels


def align_predictions(labels: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[set[str]]:
    """Align final predictions to test references by exact source indexes.

    Args:
        labels (list[dict[str, Any]]): Validated test references from load_labels.
        predictions (list[dict[str, Any]]): Archived generation JSONL records.
    Returns:
        list[set[str]]: Flat normalized predictions in test-file order.
    Raises:
        ValueError: Indexes duplicate, sets differ, or references/audit IDs differ.
    """
    # 步骤1: Never assume record order alone proves correct pairing.
    by_index = {record["source_index"]: record for record in predictions}
    if len(by_index) != len(predictions) or set(by_index) != {row["source_index"] for row in labels}:
        raise ValueError("配对失败: 索引重复或集合不同, 上下文: test/predictions")
    aligned = []
    for row in labels:
        record = by_index[row["source_index"]]
        parsed_reference = parse_prediction(record["reference"])
        if parsed_reference.canonical_notes != row["notes"] or record.get("row_id") != row["row_id"]:
            raise ValueError(f"配对失败: 参考或row_id不同, 上下文: index={row['source_index']}")
        # 步骤2: Recover valid tiers even when the overall schema is invalid.
        aligned.append(flatten_notes(parse_prediction(record["raw_response"]).canonical_notes))
    return aligned


if __name__ == "__main__":
    # 步骤1: This module exposes helpers only, never performs a formal run.
    print("Release reconstruction helpers; see docs/RECONSTRUCTION.md")
