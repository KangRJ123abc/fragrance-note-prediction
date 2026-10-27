"""
Build the second-generation perfume review-to-note dataset.

Core purpose:
    This module converts the LLM-screened perfume review records into training,
    validation, test, and evaluation JSONL files for the note-pyramid prediction
    task.

Design:
    The builder reads ``screened_reviews.jsonl``, keeps only usable anonymized
    reviews, normalizes note labels, splits data by perfume name to reduce label
    leakage, and writes both legacy instruction/input/output files and modern
    chat-message files with an explicit system instruction.

External dependencies:
    Only Python standard-library modules are required. Input files are produced
    by ``src/deepseek_filter_reviews.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# ========== Path And Prompt Constants ==========

# ------ Project Paths ------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "outputs"
    / "deepseek_review_filter_check"
    / "20260713_222515"
    / "results"
    / "screened_reviews.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"

# ------ Output Naming ------
DEFAULT_OUTPUT_PREFIX = "qwen_finetune_notes_review_v2_strict"
SPLIT_NAMES = ("train", "val", "test")

# ------ Improved Task Instruction ------
SYSTEM_INSTRUCTION = """
You are a fragrance note-pyramid prediction assistant.
Given a perfume review, produce a structured estimate of the perfume's official-style top, middle, and base notes.
Use olfactory evidence in the review, including explicit note mentions, opening/heart/drydown descriptions, accords, materials, textures, and comparative scent descriptions.
When the review gives partial evidence, make conservative, best-supported inferences instead of listing every possible note.
Focus on the scent evidence in the review rather than product or brand names.
Return only a valid JSON object with exactly these keys: top_notes, middle_notes, base_notes.
Each value must be a list of English fragrance note terms.
""".strip()

USER_TASK_TEMPLATE = """
Review:
{review}

Predict the perfume note pyramid as JSON. Terms may be implied by the review rather than copied verbatim; keep them closely tied to the review evidence.
""".strip()

LEGACY_INSTRUCTION = """
Given a perfume review, produce a structured estimate of the perfume's official-style top notes, middle notes, and base notes.
Use olfactory evidence in the review, including explicit note mentions, opening/heart/drydown descriptions, accords, materials, textures, and comparative scent descriptions.
When the review gives partial evidence, make conservative, best-supported inferences instead of listing every possible note.
Focus on the scent evidence in the review rather than product or brand names. Do not output your reasoning process.
Output only a valid JSON object with exactly three keys: top_notes, middle_notes, and base_notes. Values must be lists of English fragrance note terms.
""".strip()

WORD_PATTERN = re.compile(r"\S+")
SPACE_PATTERN = re.compile(r"\s+")


# ========== Data Models ==========


@dataclass(frozen=True)
class DatasetExample:
    """Represent one cleaned review-note training example.

    Args:
        row_id (int): Original row id from the source Excel-derived dataset.
        perfume_name (str): Source perfume name used only for group splitting
            and metadata, not inserted into the model prompt.
        review (str): Anonymized review text used as model input.
        top_notes (list[str]): Normalized top-note labels.
        middle_notes (list[str]): Normalized middle-note labels.
        base_notes (list[str]): Normalized base-note labels.
        scent_information_level (int): LLM screening level from 0 to 3.
        source_url (str): Original source URL retained only as metadata.
        provider (str): Screening provider metadata, or ``unknown`` for older
            rows that predate provider-level metadata.

    Returns:
        DatasetExample: Immutable record consumed by output-format builders.

    Raises:
        ValueError: The dataclass itself does not validate fields; validation is
        performed by ``build_dataset_example`` before construction.
    """

    row_id: int
    perfume_name: str
    review: str
    top_notes: list[str]
    middle_notes: list[str]
    base_notes: list[str]
    scent_information_level: int
    source_url: str
    provider: str


# ========== Argument Parsing ==========


def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset construction.

    The defaults build the primary strict v2 dataset from the completed
    20260713_222515 filtering run. Ratios and thresholds remain configurable so
    full or more conservative variants can be regenerated without editing code.

    Args:
        None: ``argparse`` reads command-line arguments from ``sys.argv``.

    Returns:
        argparse.Namespace: Validated command-line values before semantic
        checks. Typical defaults create train/val/test files under ``data``.

    Raises:
        SystemExit: Raised by ``argparse`` when a command-line value is invalid.
    """

    # ---------- Parser Definition ----------

    # 步骤1: 暴露输入、输出、筛选阈值和切分比例，确保数据构造可复现。
    parser = argparse.ArgumentParser(description="Build v2 perfume review-to-note datasets.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH, help="screened_reviews.jsonl path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated files.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Generated file prefix.")
    parser.add_argument("--min-scent-level", type=int, default=2, help="Minimum scent_information_level to keep.")
    parser.add_argument("--min-anonymized-words", type=int, default=21, help="Minimum anonymized review word count.")
    parser.add_argument("--max-anonymized-words", type=int, default=512, help="Maximum anonymized review word count; use 0 to disable.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Perfume-level validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Perfume-level test split ratio.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for perfume-level split and row order.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing generated files.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate semantic constraints after command-line parsing.

    Args:
        args (argparse.Namespace): Parsed command-line values from
            ``parse_args``.

    Returns:
        None: The function raises on invalid values.

    Raises:
        ValueError: Thresholds or split ratios are outside meaningful ranges.
        FileNotFoundError: The source JSONL file does not exist.
    """

    # ---------- Source Validation ----------

    # 步骤1: 在读取前确认源文件存在，避免后续报错缺乏上下文。
    if not args.source.exists():
        raise FileNotFoundError(f"源文件不存在失败: path={args.source}, 上下文: source={args.source}")

    # ---------- Threshold Validation ----------

    # 步骤2: 阈值必须非负，scent level 的有效范围来自筛选脚本定义。
    if not 0 <= args.min_scent_level <= 3:
        raise ValueError(f"参数校验失败: min_scent_level必须在0到3之间, 上下文: value={args.min_scent_level}")
    if args.min_anonymized_words < 0:
        raise ValueError(f"参数校验失败: min_anonymized_words不能为负, 上下文: value={args.min_anonymized_words}")
    if args.max_anonymized_words < 0:
        raise ValueError(f"参数校验失败: max_anonymized_words不能为负, 上下文: value={args.max_anonymized_words}")
    if args.max_anonymized_words and args.max_anonymized_words < args.min_anonymized_words:
        raise ValueError(
            "参数校验失败: max_anonymized_words必须大于等于min_anonymized_words或设为0, "
            f"上下文: min={args.min_anonymized_words}, max={args.max_anonymized_words}"
        )

    # ---------- Split Ratio Validation ----------

    # 步骤3: validation/test 之和必须给 train 留出空间。
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError(f"参数校验失败: split ratio不能为负, 上下文: val={args.val_ratio}, test={args.test_ratio}")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError(f"参数校验失败: val_ratio+test_ratio必须小于1, 上下文: val={args.val_ratio}, test={args.test_ratio}")


# ========== JSON Utilities ==========


def read_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory.

    Args:
        jsonl_path (Path): File path produced by the review filtering script.

    Returns:
        list[dict[str, Any]]: Parsed JSON objects in source order. Empty lines
        are skipped.

    Raises:
        RuntimeError: File reading or JSON parsing fails.
    """

    # ---------- JSONL Reading ----------

    # 步骤1: 使用 utf-8-sig 兼容潜在 BOM，同时保持普通 UTF-8 可读。
    records: list[dict[str, Any]] = []
    try:
        with jsonl_path.open("r", encoding="utf-8-sig") as jsonl_file:
            for line_number, line_text in enumerate(jsonl_file, start=1):
                stripped_line = line_text.strip()
                if not stripped_line:
                    continue
                records.append(json.loads(stripped_line))
    except Exception as error:
        raise RuntimeError(f"读取JSONL失败: {error}, 上下文: path={jsonl_path}") from error
    return records


def write_jsonl(jsonl_path: Path, records: list[dict[str, Any]], should_overwrite: bool) -> None:
    """Write JSON-serializable records to a JSONL file.

    Args:
        jsonl_path (Path): Target output file path.
        records (list[dict[str, Any]]): Records to serialize one per line.
        should_overwrite (bool): Whether existing target files may be replaced.

    Returns:
        None: Data is written to disk as a side effect.

    Raises:
        FileExistsError: The file already exists and overwrite is disabled.
        RuntimeError: Directory creation or file writing fails.
    """

    # ---------- Overwrite Guard ----------

    # 步骤1: 默认不覆盖旧数据集，防止误删已经训练过的版本。
    if jsonl_path.exists() and not should_overwrite:
        raise FileExistsError(f"写入JSONL失败: 文件已存在, 上下文: path={jsonl_path}")

    # ---------- JSONL Writing ----------

    # 步骤2: 统一写普通 UTF-8 JSONL，避免旧文件中的 BOM 问题继续扩散。
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl_file:
            for record in records:
                jsonl_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as error:
        raise RuntimeError(f"写入JSONL失败: {error}, 上下文: path={jsonl_path}") from error


def write_json(json_path: Path, payload: dict[str, Any], should_overwrite: bool) -> None:
    """Write a JSON object to disk.

    Args:
        json_path (Path): Target path for the manifest or summary file.
        payload (dict[str, Any]): JSON-serializable object to write.
        should_overwrite (bool): Whether existing target files may be replaced.

    Returns:
        None: Data is written to disk as a side effect.

    Raises:
        FileExistsError: The file already exists and overwrite is disabled.
        RuntimeError: Directory creation or file writing fails.
    """

    # ---------- Overwrite Guard ----------

    # 步骤1: 与 JSONL 输出保持一致，默认保护已有 manifest。
    if json_path.exists() and not should_overwrite:
        raise FileExistsError(f"写入JSON失败: 文件已存在, 上下文: path={json_path}")

    # ---------- JSON Writing ----------

    # 步骤2: manifest 使用缩进格式，方便人工检查 prompt 和统计量。
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as error:
        raise RuntimeError(f"写入JSON失败: {error}, 上下文: path={json_path}") from error


# ========== Normalization And Filtering ==========


def normalize_space(text_value: Any) -> str:
    """Collapse whitespace in a text value.

    Args:
        text_value (Any): Raw value from a JSON record.

    Returns:
        str: Trimmed string with consecutive whitespace collapsed to one space.

    Raises:
        None: Non-string values are converted with ``str``.
    """

    # ---------- Whitespace Normalization ----------

    # 步骤1: 评论来自网页抓取，先压平换行和多空格，避免 prompt 里出现噪声。
    return SPACE_PATTERN.sub(" ", str(text_value or "")).strip()


def count_words(text_value: str) -> int:
    """Count whitespace-delimited words in text.

    Args:
        text_value (str): Normalized review or assistant output text.

    Returns:
        int: Number of non-whitespace spans. Typical values are 20 to 300.

    Raises:
        None: The regular expression handles empty strings.
    """

    # ---------- Word Counting ----------

    # 步骤1: 沿用筛选阶段的近似英文词数逻辑，保证阈值解释一致。
    return len(WORD_PATTERN.findall(text_value))


def normalize_note_term(note_value: Any) -> str:
    """Normalize a fragrance note label.

    Args:
        note_value (Any): Raw note term from the screened record.

    Returns:
        str: Lowercase English-style note term with normalized whitespace.

    Raises:
        None: Empty or non-string values become an empty string.
    """

    # ---------- Note Text Cleanup ----------

    # 步骤1: 输出标签统一小写，贴近旧数据集格式并减少同义大小写噪声。
    normalized_note = normalize_space(note_value).lower()
    return normalized_note


def normalize_note_list(note_values: Any) -> list[str]:
    """Normalize a list of fragrance note labels.

    Args:
        note_values (Any): Raw list-like note field from a screened record.

    Returns:
        list[str]: De-duplicated normalized terms in original order.

    Raises:
        ValueError: ``note_values`` is not a list.
    """

    # ---------- Type Validation ----------

    # 步骤1: note 字段必须是列表，否则说明上游结构异常。
    if not isinstance(note_values, list):
        raise ValueError(f"香调字段校验失败: 不是列表, 上下文: value={note_values}")

    # ---------- Order-Preserving Deduplication ----------

    # 步骤2: 保留原始金标准顺序，只去掉空项和重复项。
    normalized_notes: list[str] = []
    seen_notes: set[str] = set()
    for raw_note in note_values:
        normalized_note = normalize_note_term(raw_note)
        if normalized_note and normalized_note not in seen_notes:
            normalized_notes.append(normalized_note)
            seen_notes.add(normalized_note)
    return normalized_notes


def should_keep_record(
    raw_record: dict[str, Any],
    min_scent_level: int,
    min_anonymized_words: int,
    max_anonymized_words: int,
) -> bool:
    """Decide whether a screened record enters the generated dataset.

    Args:
        raw_record (dict[str, Any]): One parsed line from ``screened_reviews``.
        min_scent_level (int): Minimum accepted scent-information level.
        min_anonymized_words (int): Minimum accepted anonymized review length.
        max_anonymized_words (int): Maximum accepted anonymized review length.
            A value of 0 disables the upper bound.

    Returns:
        bool: ``True`` when the record passes all strict v2 filters.

    Raises:
        None: Missing fields simply cause the record to be rejected.
    """

    # ---------- LLM Screening Gate ----------

    # 步骤1: 只使用筛选阶段明确判定为可用的匿名评论。
    if raw_record.get("keep") is not True:
        return False

    # ---------- Information Density Gate ----------

    # 步骤2: level 和词数共同控制训练样本的信息密度。
    scent_information_level = int(raw_record.get("scent_information_level") or 0)
    anonymized_review = normalize_space(raw_record.get("anonymized_review"))
    anonymized_word_count = count_words(anonymized_review)
    if scent_information_level < min_scent_level:
        return False
    if anonymized_word_count < min_anonymized_words:
        return False
    return max_anonymized_words == 0 or anonymized_word_count <= max_anonymized_words


def build_dataset_example(raw_record: dict[str, Any]) -> DatasetExample:
    """Convert a screened JSON object into a normalized dataset example.

    Args:
        raw_record (dict[str, Any]): One record that already passed
        ``should_keep_record``.

    Returns:
        DatasetExample: Cleaned immutable example ready for splitting.

    Raises:
        ValueError: Required text or note fields are missing after
        normalization.
    """

    # ---------- Core Field Normalization ----------

    # 步骤1: 输入只使用匿名评论，避免模型通过品牌/香水名记忆答案。
    review_text = normalize_space(raw_record.get("anonymized_review"))
    perfume_name = normalize_space(raw_record.get("perfume_name"))
    if not review_text:
        raise ValueError(f"样本构造失败: anonymized_review为空, 上下文: row_id={raw_record.get('row_id')}")
    if not perfume_name:
        raise ValueError(f"样本构造失败: perfume_name为空, 上下文: row_id={raw_record.get('row_id')}")

    # ---------- Label Normalization ----------

    # 步骤2: 三段香调都必须非空，否则不能构成监督学习目标。
    top_notes = normalize_note_list(raw_record.get("top_notes"))
    middle_notes = normalize_note_list(raw_record.get("middle_notes"))
    base_notes = normalize_note_list(raw_record.get("base_notes"))
    if not top_notes or not middle_notes or not base_notes:
        raise ValueError(f"样本构造失败: 香调列表为空, 上下文: row_id={raw_record.get('row_id')}")

    # ---------- Example Construction ----------

    # 步骤3: 只把 source_url/provider 作为元数据，不进入模型 prompt。
    return DatasetExample(
        row_id=int(raw_record["row_id"]),
        perfume_name=perfume_name,
        review=review_text,
        top_notes=top_notes,
        middle_notes=middle_notes,
        base_notes=base_notes,
        scent_information_level=int(raw_record.get("scent_information_level") or 0),
        source_url=normalize_space(raw_record.get("source_url")),
        provider=normalize_space(raw_record.get("provider")) or "unknown",
    )


def build_examples(
    raw_records: list[dict[str, Any]],
    min_scent_level: int,
    min_anonymized_words: int,
    max_anonymized_words: int,
) -> list[DatasetExample]:
    """Build all dataset examples from screened records.

    Args:
        raw_records (list[dict[str, Any]]): Parsed source records.
        min_scent_level (int): Minimum accepted scent-information level.
        min_anonymized_words (int): Minimum accepted anonymized review length.
        max_anonymized_words (int): Maximum accepted anonymized review length.
            A value of 0 disables the upper bound.

    Returns:
        list[DatasetExample]: Valid examples that passed filters.

    Raises:
        RuntimeError: A record passes filtering but cannot be normalized.
    """

    # ---------- Filtering And Conversion ----------

    # 步骤1: 过滤在前、构造在后，让错误只暴露于理论上应可训练的记录。
    examples: list[DatasetExample] = []
    for raw_record in raw_records:
        if not should_keep_record(raw_record, min_scent_level, min_anonymized_words, max_anonymized_words):
            continue
        try:
            examples.append(build_dataset_example(raw_record))
        except Exception as error:
            raise RuntimeError(f"构造样本失败: {error}, 上下文: row_id={raw_record.get('row_id')}") from error
    return examples


# ========== Split Logic ==========


def split_perfume_names(perfume_names: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    """Assign perfume names to train, validation, and test splits.

    Args:
        perfume_names (list[str]): Unique perfume names from selected examples.
        val_ratio (float): Fraction of perfumes assigned to validation.
        test_ratio (float): Fraction of perfumes assigned to test.
        seed (int): Random seed for deterministic shuffling.

    Returns:
        dict[str, str]: Mapping from perfume name to split name.

    Raises:
        ValueError: Fewer than three perfumes are available for splitting.
    """

    # ---------- Input Validation ----------

    # 步骤1: group split 至少需要 train/val/test 各有一个香水。
    unique_perfume_names = sorted(set(perfume_names))
    if len(unique_perfume_names) < 3:
        raise ValueError(f"切分失败: 香水数量不足, 上下文: count={len(unique_perfume_names)}")

    # ---------- Deterministic Shuffle ----------

    # 步骤2: 按香水名整体切分，避免同一香水的同一标签同时出现在训练和测试中。
    random_generator = random.Random(seed)
    random_generator.shuffle(unique_perfume_names)

    # ---------- Split Boundary Calculation ----------

    # 步骤3: 用香水数量计算边界，而不是用评论数量，以控制标签泄漏。
    validation_count = max(1, int(round(len(unique_perfume_names) * val_ratio)))
    test_count = max(1, int(round(len(unique_perfume_names) * test_ratio)))
    train_count = len(unique_perfume_names) - validation_count - test_count
    if train_count <= 0:
        raise ValueError(f"切分失败: train为空, 上下文: total={len(unique_perfume_names)}")

    test_names = set(unique_perfume_names[:test_count])
    validation_names = set(unique_perfume_names[test_count : test_count + validation_count])
    return {
        perfume_name: ("test" if perfume_name in test_names else "val" if perfume_name in validation_names else "train")
        for perfume_name in unique_perfume_names
    }


def assign_examples_to_splits(examples: list[DatasetExample], split_by_perfume: dict[str, str], seed: int) -> dict[str, list[DatasetExample]]:
    """Group examples by their assigned split and shuffle each split.

    Args:
        examples (list[DatasetExample]): Selected normalized examples.
        split_by_perfume (dict[str, str]): Mapping produced by
            ``split_perfume_names``.
        seed (int): Random seed for row-order shuffling inside each split.

    Returns:
        dict[str, list[DatasetExample]]: Split name to examples.

    Raises:
        KeyError: An example's perfume name is missing from the split mapping.
    """

    # ---------- Split Assignment ----------

    # 步骤1: 根据 perfume-level mapping 分配样本，确保同香水不跨 split。
    split_examples: dict[str, list[DatasetExample]] = {split_name: [] for split_name in SPLIT_NAMES}
    for example in examples:
        split_name = split_by_perfume[example.perfume_name]
        split_examples[split_name].append(example)

    # ---------- Row Order Shuffle ----------

    # 步骤2: 每个 split 内部随机化行顺序，避免训练时按香水块连续出现。
    random_generator = random.Random(seed)
    for split_name in SPLIT_NAMES:
        random_generator.shuffle(split_examples[split_name])
    return split_examples


# ========== Output Record Builders ==========


def build_output_payload(example: DatasetExample) -> dict[str, list[str]]:
    """Build the assistant JSON payload for one example.

    Args:
        example (DatasetExample): Cleaned dataset example.

    Returns:
        dict[str, list[str]]: Note-pyramid object with three required keys.

    Raises:
        None: The example was already validated during construction.
    """

    # ---------- Label Payload ----------

    # 步骤1: 输出 schema 与旧数据集和评测脚本保持兼容。
    return {
        "top_notes": example.top_notes,
        "middle_notes": example.middle_notes,
        "base_notes": example.base_notes,
    }


def build_output_text(example: DatasetExample) -> str:
    """Serialize the assistant note-pyramid payload.

    Args:
        example (DatasetExample): Cleaned dataset example.

    Returns:
        str: Compact JSON string used as supervised target.

    Raises:
        TypeError: JSON serialization fails unexpectedly.
    """

    # ---------- Compact JSON Serialization ----------

    # 步骤1: 紧凑 JSON 减少 token 浪费，同时仍然是严格可解析对象。
    return json.dumps(build_output_payload(example), ensure_ascii=False, separators=(",", ":"))


def build_metadata(example: DatasetExample, split_name: str) -> dict[str, Any]:
    """Build non-prompt metadata for one output row.

    Args:
        example (DatasetExample): Cleaned dataset example.
        split_name (str): Split assigned to the example.

    Returns:
        dict[str, Any]: Metadata useful for audits and error analysis.

    Raises:
        None: All fields are copied from validated examples.
    """

    # ---------- Metadata Construction ----------

    # 步骤1: metadata 不进入 prompt，但能支撑后续抽查和溯源。
    return {
        "row_id": example.row_id,
        "perfume_name": example.perfume_name,
        "split": split_name,
        "scent_information_level": example.scent_information_level,
        "source_url": example.source_url,
        "screening_provider": example.provider,
    }


def build_iio_record(example: DatasetExample, split_name: str) -> dict[str, Any]:
    """Build a legacy instruction/input/output training row.

    Args:
        example (DatasetExample): Cleaned dataset example.
        split_name (str): Split assigned to the example.

    Returns:
        dict[str, Any]: JSONL row compatible with the current ``train.py``.

    Raises:
        TypeError: Output JSON serialization fails unexpectedly.
    """

    # ---------- Legacy Trainer Format ----------

    # 步骤1: 当前 train.py 读取 instruction/input/output，因此保留兼容格式。
    return {
        "instruction": LEGACY_INSTRUCTION,
        "input": example.review,
        "output": build_output_text(example),
        "metadata": build_metadata(example, split_name),
    }


def build_chat_record(example: DatasetExample, split_name: str) -> dict[str, Any]:
    """Build a chat-message training row with an explicit system instruction.

    Args:
        example (DatasetExample): Cleaned dataset example.
        split_name (str): Split assigned to the example.

    Returns:
        dict[str, Any]: JSONL row containing ``messages`` and ``metadata``.

    Raises:
        TypeError: Output JSON serialization fails unexpectedly.
    """

    # ---------- Chat Format ----------

    # 步骤1: chat 格式保存真正的 system/user/assistant 三段消息，便于新版训练脚本使用。
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": USER_TASK_TEMPLATE.format(review=example.review)},
            {"role": "assistant", "content": build_output_text(example)},
        ],
        "metadata": build_metadata(example, split_name),
    }


def build_eval_record(example: DatasetExample) -> dict[str, str]:
    """Build a custom evaluation row for EvalScope general QA scripts.

    Args:
        example (DatasetExample): Cleaned dataset example.

    Returns:
        dict[str, str]: Row with ``query`` and ``response`` fields.

    Raises:
        TypeError: Output JSON serialization fails unexpectedly.
    """

    # ---------- Evaluation Query Format ----------

    # 步骤1: query 使用改进后的任务说明，保证 baseline 和微调模型接收同一清晰边界。
    query_text = f"{LEGACY_INSTRUCTION}\n\nReview:\n{example.review}"
    return {"query": query_text, "response": build_output_text(example)}


def build_split_records(split_examples: list[DatasetExample], split_name: str) -> dict[str, list[dict[str, Any]]]:
    """Convert split examples into all supported output formats.

    Args:
        split_examples (list[DatasetExample]): Examples belonging to one split.
        split_name (str): Name of the split, such as ``train`` or ``test``.

    Returns:
        dict[str, list[dict[str, Any]]]: Records keyed by ``iio``, ``chat``,
        and ``eval``.

    Raises:
        TypeError: Any output JSON serialization fails unexpectedly.
    """

    # ---------- Multi-Format Conversion ----------

    # 步骤1: 同一批样本同时产出训练兼容格式、chat格式和评测格式，避免后续重复抽样。
    return {
        "iio": [build_iio_record(example, split_name) for example in split_examples],
        "chat": [build_chat_record(example, split_name) for example in split_examples],
        "eval": [build_eval_record(example) for example in split_examples],
    }


# ========== Summary And Persistence ==========


def summarize_split(examples: list[DatasetExample]) -> dict[str, Any]:
    """Summarize one dataset split.

    Args:
        examples (list[DatasetExample]): Examples assigned to the split.

    Returns:
        dict[str, Any]: Counts by row, perfume, scent level, and provider.

    Raises:
        None: Empty splits are summarized with zero counts.
    """

    # ---------- Split Statistics ----------

    # 步骤1: 汇总 split 的规模和来源分布，便于训练前判断是否均衡。
    perfume_names = {example.perfume_name for example in examples}
    scent_levels = Counter(str(example.scent_information_level) for example in examples)
    providers = Counter(example.provider for example in examples)
    return {
        "rows": len(examples),
        "perfumes": len(perfume_names),
        "scent_information_levels": dict(sorted(scent_levels.items())),
        "screening_providers": dict(providers.most_common()),
    }


def summarize_dataset(split_examples: dict[str, list[DatasetExample]], args: argparse.Namespace) -> dict[str, Any]:
    """Build a manifest summary for the generated dataset.

    Args:
        split_examples (dict[str, list[DatasetExample]]): Final split mapping.
        args (argparse.Namespace): Parsed and validated command-line arguments.

    Returns:
        dict[str, Any]: Manifest object written next to generated files.

    Raises:
        None: All values are derived from in-memory examples and args.
    """

    # ---------- Manifest Construction ----------

    # 步骤1: manifest 记录阈值、prompt 和 split 统计，支撑论文方法复现。
    total_rows = sum(len(examples) for examples in split_examples.values())
    total_perfumes = len({example.perfume_name for examples in split_examples.values() for example in examples})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.source),
        "output_prefix": args.output_prefix,
        "filters": {
            "keep": True,
            "min_scent_information_level": args.min_scent_level,
            "min_anonymized_words": args.min_anonymized_words,
            "max_anonymized_words": None if args.max_anonymized_words == 0 else args.max_anonymized_words,
        },
        "split_policy": {
            "unit": "perfume_name",
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "totals": {"rows": total_rows, "perfumes": total_perfumes},
        "splits": {split_name: summarize_split(split_examples[split_name]) for split_name in SPLIT_NAMES},
        "system_instruction": SYSTEM_INSTRUCTION,
        "legacy_instruction": LEGACY_INSTRUCTION,
    }


def output_paths(output_dir: Path, output_prefix: str) -> dict[str, Path]:
    """Create the complete set of output paths.

    Args:
        output_dir (Path): Directory for generated dataset files.
        output_prefix (str): Common file prefix.

    Returns:
        dict[str, Path]: Named output paths for JSONL files and manifest.

    Raises:
        None: Path objects are constructed without touching the filesystem.
    """

    # ---------- Path Construction ----------

    # 步骤1: 文件名明确标出格式和 split，减少训练时选错文件的风险。
    paths: dict[str, Path] = {"manifest": output_dir / f"{output_prefix}_manifest.json"}
    for split_name in SPLIT_NAMES:
        paths[f"{split_name}_iio"] = output_dir / f"{output_prefix}_{split_name}.jsonl"
        paths[f"{split_name}_chat"] = output_dir / f"{output_prefix}_chat_{split_name}.jsonl"
        paths[f"{split_name}_eval"] = output_dir / f"{output_prefix}_eval_{split_name}.jsonl"
    return paths


def ensure_outputs_can_be_written(paths: dict[str, Path], should_overwrite: bool) -> None:
    """Fail early if generated files would overwrite existing outputs.

    Args:
        paths (dict[str, Path]): Paths returned by ``output_paths``.
        should_overwrite (bool): Whether existing files are allowed.

    Returns:
        None: The function raises on conflicts.

    Raises:
        FileExistsError: At least one target exists and overwrite is disabled.
    """

    # ---------- Conflict Detection ----------

    # 步骤1: 在写任意文件前统一检查，避免生成一半后才失败。
    if should_overwrite:
        return
    existing_paths = [str(path) for path in paths.values() if path.exists()]
    if existing_paths:
        raise FileExistsError(f"输出文件已存在失败: 请使用--overwrite或更换prefix, 上下文: paths={existing_paths}")


def write_outputs(split_examples: dict[str, list[DatasetExample]], args: argparse.Namespace) -> dict[str, Path]:
    """Write all dataset files and the manifest.

    Args:
        split_examples (dict[str, list[DatasetExample]]): Final split mapping.
        args (argparse.Namespace): Parsed command-line arguments with output
        settings.

    Returns:
        dict[str, Path]: Paths that were written.

    Raises:
        FileExistsError: Output conflicts exist and overwrite is disabled.
        RuntimeError: Any file write fails.
    """

    # ---------- Output Path Preparation ----------

    # 步骤1: 先准备路径并检查冲突，保证输出过程是全有或全无。
    paths = output_paths(args.output_dir, args.output_prefix)
    ensure_outputs_can_be_written(paths, args.overwrite)

    # ---------- Split File Writing ----------

    # 步骤2: 每个 split 输出三种格式，分别服务训练、chat训练和评测。
    for split_name in SPLIT_NAMES:
        split_records = build_split_records(split_examples[split_name], split_name)
        write_jsonl(paths[f"{split_name}_iio"], split_records["iio"], args.overwrite)
        write_jsonl(paths[f"{split_name}_chat"], split_records["chat"], args.overwrite)
        write_jsonl(paths[f"{split_name}_eval"], split_records["eval"], args.overwrite)

    # ---------- Manifest Writing ----------

    # 步骤3: 最后写 manifest，说明前面所有数据文件已经成功生成。
    manifest = summarize_dataset(split_examples, args)
    write_json(paths["manifest"], manifest, args.overwrite)
    return paths


# ========== Main Workflow ==========


def main() -> None:
    """Run the complete dataset construction workflow.

    Args:
        None: Configuration comes from command-line arguments.

    Returns:
        None: Generated files are written under ``args.output_dir``.

    Raises:
        FileNotFoundError: Source file is missing.
        ValueError: Arguments or split setup are invalid.
        RuntimeError: JSON reading, normalization, or writing fails.
    """

    # ---------- Argument And Source Loading ----------

    # 步骤1: 验证参数后读取筛选结果，保证错误尽早暴露。
    args = parse_args()
    validate_args(args)
    raw_records = read_jsonl(args.source)

    # ---------- Example Selection ----------

    # 步骤2: 应用严格 v2 阈值，构造可训练样本。
    examples = build_examples(raw_records, args.min_scent_level, args.min_anonymized_words, args.max_anonymized_words)
    if not examples:
        raise ValueError("数据集构造失败: 筛选后没有样本, 上下文: 请降低min_scent_level或min_anonymized_words")

    # ---------- Perfume-Level Split ----------

    # 步骤3: 按 perfume_name 分组切分，降低同标签记忆导致的评测泄漏。
    perfume_names = [example.perfume_name for example in examples]
    split_by_perfume = split_perfume_names(perfume_names, args.val_ratio, args.test_ratio, args.seed)
    split_examples = assign_examples_to_splits(examples, split_by_perfume, args.seed)

    # ---------- Persistence ----------

    # 步骤4: 写出数据文件和 manifest，并在控制台给出最重要的规模信息。
    paths = write_outputs(split_examples, args)
    manifest = summarize_dataset(split_examples, args)
    print(json.dumps({"written": {key: str(value) for key, value in paths.items()}, "summary": manifest["totals"], "splits": manifest["splits"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
