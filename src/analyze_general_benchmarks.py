"""
Analyze paired Base and finetuned outputs from the formal general benchmarks.

The module joins EvalScope prediction and review records by dataset, subset file,
and sample index. It reproduces official accuracies, computes paired significance
tests, audits invalid answers and length truncation, and writes a clean-pair
sensitivity analysis without changing the official all-sample results.

External dependencies: NumPy and SciPy. Input files follow the EvalScope 1.4.2
directory layout produced by ``src/eval_benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.stats import binomtest


# ========== Constants and data models ==========

# ------ Formal benchmark configuration ------

DATASET_NAMES: tuple[str, ...] = (
    "ceval",
    "cmmlu",
    "mmlu",
    "gsm8k",
    "hellaswag",
    "truthful_qa",
)
ROLE_NAMES: tuple[str, str] = ("base", "finetuned")
MULTIPLE_CHOICE_DATASETS: frozenset[str] = frozenset(
    {"ceval", "cmmlu", "mmlu", "hellaswag", "truthful_qa"}
)
LENGTH_STOP_REASONS: frozenset[str] = frozenset({"length", "max_tokens"})
OPTION_LABEL_PATTERN = re.compile(r"(?m)^\s*([A-Z])(?:[\.、:：\)])\s+")
CHINESE_FINAL_ANSWER_PATTERN = re.compile(r"答案\s*[:：]\s*([A-Z])\s*[。.]?\s*$")
ENGLISH_FINAL_ANSWER_PATTERN = re.compile(r"ANSWER\s*:\s*([A-Z])\s*[.]?\s*$", re.I)
BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\s*\{[^{}]+\}\s*[。.]?\s*$")


@dataclass(frozen=True)
class SampleRecord:
    """Represent one scored EvalScope prediction with its generation diagnostics.

    The record combines one line from a prediction JSONL file and the matching
    line from its review JSONL file. All fields are normalized to plain Python
    values so that paired comparisons and JSON serialization remain deterministic.

    Attributes:
        dataset_name: Benchmark identifier supplied by the run directory.
        subset_name: Source JSONL filename stem identifying the benchmark subset.
        sample_index: EvalScope sample index written by the evaluator.
        review_input: Human-readable prompt stored in the review record.
        target: Reference answer supplied by EvalScope.
        extracted_prediction: Answer extracted by the benchmark evaluator.
        raw_output: Complete generated model text.
        is_correct: Whether EvalScope assigned accuracy 1 to this sample.
        is_valid_answer: Whether the extracted answer is nonempty and admissible.
        has_expected_answer_marker: Whether output ends with the requested marker.
        is_length_truncated: Whether generation stopped at a token limit.
        has_generation_error: Whether prediction generation reported an error.
        stop_reason: vLLM/EvalScope completion stop reason.
        input_tokens: Number of prompt tokens reported by the inference server.
        output_tokens: Number of generated tokens reported by the inference server.
    """

    dataset_name: str
    subset_name: str
    sample_index: int
    review_input: str
    target: str
    extracted_prediction: str
    raw_output: str
    is_correct: bool
    is_valid_answer: bool
    has_expected_answer_marker: bool
    is_length_truncated: bool
    has_generation_error: bool
    stop_reason: str
    input_tokens: int
    output_tokens: int

    @property
    def sample_key(self) -> tuple[str, str, int]:
        """Return the stable key used to pair Base and finetuned records.

        The key deliberately excludes model name and line position because
        EvalScope may write review rows in a different order from predictions.

        Returns:
            tuple[str, str, int]: Dataset, subset stem, and EvalScope index.

        Raises:
            No exception is expected because all fields are validated at load time.
        """

        return (self.dataset_name, self.subset_name, self.sample_index)


@dataclass(frozen=True)
class PairedSample:
    """Hold matched Base and finetuned records for one benchmark question.

    Args:
        base_record: Base-model record loaded from the formal EvalScope output.
        finetuned_record: Finetuned-model record with the same stable sample key.

    Returns:
        PairedSample: Immutable paired sample used by all statistical analyses.

    Raises:
        ValueError: Raised when keys, prompts, review inputs, or targets differ.
    """

    base_record: SampleRecord
    finetuned_record: SampleRecord

    def __post_init__(self) -> None:
        """Validate that both model records describe exactly the same question.

        Values originate from the two role directories in the same formal run.
        Equality checks prevent accidental comparisons across shuffled records or
        differing prompts.

        Returns:
            None: Validation succeeds without modifying the immutable object.

        Raises:
            ValueError: Raised when any pairing invariant is violated.
        """

        # 步骤1: 检查稳定键，避免把同一子集中的不同题目配在一起。
        if self.base_record.sample_key != self.finetuned_record.sample_key:
            raise ValueError(
                "配对样本键不一致, 上下文: "
                f"base={self.base_record.sample_key}, "
                f"finetuned={self.finetuned_record.sample_key}"
            )

        # 步骤2: 检查渲染后的评分输入与参考答案完全一致。
        compared_fields = ("review_input", "target")
        for field_name in compared_fields:
            if getattr(self.base_record, field_name) != getattr(
                self.finetuned_record, field_name
            ):
                raise ValueError(
                    f"配对样本字段不一致: {field_name}, "
                    f"上下文: key={self.base_record.sample_key}"
                )

    @property
    def is_clean_pair(self) -> bool:
        """Return whether neither side is invalid, truncated, or errored.

        This property defines the post-hoc sensitivity subset. It must never be
        used to replace the official all-sample benchmark score.

        Returns:
            bool: True only when both generations pass all technical checks.

        Raises:
            No exception is expected because the component flags are booleans.
        """

        return all(
            (
                self.base_record.is_valid_answer,
                self.finetuned_record.is_valid_answer,
                not self.base_record.is_length_truncated,
                not self.finetuned_record.is_length_truncated,
                not self.base_record.has_generation_error,
                not self.finetuned_record.has_generation_error,
            )
        )


# ========== Command-line interface ==========


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the benchmark analysis.

    Defaults are suitable for a formal run directory containing the ``evalscope``
    and ``reports`` subdirectories. The caller may override output location,
    bootstrap count, and random seed for reproducibility checks.

    Returns:
        argparse.ArgumentParser: Configured parser consumed by ``main``.

    Raises:
        No exception is expected while constructing static argument definitions.
    """

    parser = argparse.ArgumentParser(
        description="Analyze paired Base/finetuned general benchmark outputs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Formal general benchmark run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <run-dir>/analysis.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=20_000,
        help="Paired bootstrap repetitions used for delta-accuracy intervals.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used by paired bootstrap resampling.",
    )
    return parser


def validate_arguments(arguments: argparse.Namespace) -> tuple[Path, Path]:
    """Validate CLI arguments and resolve input/output directories.

    Args:
        arguments: Namespace produced by ``build_argument_parser`` from user CLI
            input.

    Returns:
        tuple[Path, Path]: Resolved formal run directory and analysis directory.

    Raises:
        ValueError: Raised for a missing run directory or nonpositive bootstrap
            repetition count.
    """

    # 步骤1: 解析绝对路径并确认正式运行目录存在。
    run_directory = arguments.run_dir.expanduser().resolve()
    if not run_directory.is_dir():
        raise ValueError(f"运行目录不存在, 上下文: run_dir={run_directory}")

    # 步骤2: 验证统计参数，防止生成没有意义的置信区间。
    if arguments.bootstrap_repetitions <= 0:
        raise ValueError(
            "bootstrap 次数必须为正整数, 上下文: "
            f"bootstrap_repetitions={arguments.bootstrap_repetitions}"
        )

    # 步骤3: 输出目录默认置于原始结果内部，但不覆盖任何原始文件。
    output_directory = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else run_directory / "analysis"
    )
    return run_directory, output_directory


# ========== EvalScope file loading ==========


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into a list of objects.

    Args:
        file_path: JSONL path discovered under the formal EvalScope run directory.

    Returns:
        list[dict[str, Any]]: Parsed rows in their source order.

    Raises:
        RuntimeError: Raised when file I/O or JSON parsing fails, with line context.
        ValueError: Raised when a parsed JSON value is not an object.
    """

    parsed_rows: list[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue
                parsed_value = json.loads(stripped_line)
                if not isinstance(parsed_value, dict):
                    raise ValueError(
                        "JSONL 行不是对象, 上下文: "
                        f"file={file_path}, line={line_number}"
                    )
                parsed_rows.append(parsed_value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"读取 JSONL 失败: {error}, 上下文: file={file_path}"
        ) from error
    return parsed_rows


def discover_jsonl_map(root_directory: Path) -> dict[str, Path]:
    """Discover one JSONL file per subset below an EvalScope artifact directory.

    Args:
        root_directory: Predictions or reviews directory for one role and dataset,
            produced by EvalScope.

    Returns:
        dict[str, Path]: Mapping from filename to its unique absolute path.

    Raises:
        ValueError: Raised when no files exist or duplicate filenames are found.
        RuntimeError: Raised when recursive directory enumeration fails.
    """

    try:
        discovered_paths = sorted(root_directory.rglob("*.jsonl"))
    except OSError as error:
        raise RuntimeError(
            f"枚举 JSONL 文件失败: {error}, 上下文: root={root_directory}"
        ) from error
    if not discovered_paths:
        raise ValueError(f"未发现 JSONL 文件, 上下文: root={root_directory}")

    # 步骤1: 使用文件名对齐 predictions 与 reviews，并拒绝静默覆盖。
    path_by_filename: dict[str, Path] = {}
    for discovered_path in discovered_paths:
        if discovered_path.name in path_by_filename:
            raise ValueError(
                "发现重复子集文件名, 上下文: "
                f"filename={discovered_path.name}, root={root_directory}"
            )
        path_by_filename[discovered_path.name] = discovered_path
    return path_by_filename


def extract_prediction_fields(
    prediction_row: dict[str, Any], file_path: Path
) -> tuple[str, str, int, int, bool]:
    """Extract generation text, stop reason, token usage, and error status.

    Args:
        prediction_row: One object read from an EvalScope prediction JSONL file.
        file_path: Source file path used only to enrich validation errors.

    Returns:
        tuple[str, str, int, int, bool]: Raw output, stop reason, input token count,
        output token count, and generation-error flag.

    Raises:
        ValueError: Raised when required prediction structures have invalid types.
    """

    # 步骤1: 读取服务器返回结构，缺失 completion 视为生成错误而非跳过样本。
    model_output = prediction_row.get("model_output")
    if not isinstance(model_output, dict):
        return "", "", 0, 0, True
    choices = model_output.get("choices")
    has_generation_error = bool(model_output.get("error")) or not isinstance(choices, list)
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(first_choice, dict):
        raise ValueError(
            f"prediction choice 类型无效, 上下文: file={file_path}"
        )

    # 步骤2: 抽取文本、停止原因与 token 用量，缺失值统一为安全默认值。
    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        message = {}
        has_generation_error = True
    raw_output = str(message.get("content") or "")
    stop_reason = str(first_choice.get("stop_reason") or "")
    usage = model_output.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = _coerce_nonnegative_integer(usage.get("input_tokens"))
    output_tokens = _coerce_nonnegative_integer(usage.get("output_tokens"))
    if not raw_output:
        has_generation_error = True
    return (
        raw_output,
        stop_reason,
        input_tokens,
        output_tokens,
        has_generation_error,
    )


def _coerce_nonnegative_integer(raw_value: Any) -> int:
    """Convert a token-count value to a nonnegative integer.

    Args:
        raw_value: Usage value returned by the OpenAI-compatible inference API.

    Returns:
        int: Parsed nonnegative count, or zero when the value is absent.

    Raises:
        ValueError: Raised when a present value is nonnumeric or negative.
    """

    if raw_value is None:
        return 0
    try:
        integer_value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"token 数量无法转换为整数: value={raw_value}") from error
    if integer_value < 0:
        raise ValueError(f"token 数量不能为负数: value={raw_value}")
    return integer_value


def extract_review_fields(
    review_row: dict[str, Any], file_path: Path, dataset_name: str
) -> tuple[str, str, str, bool]:
    """Extract prompt text, target, evaluator answer, and correctness from a review.

    Args:
        review_row: One object read from an EvalScope review JSONL file.
        file_path: Source file path included in failure messages.
        dataset_name: Benchmark identifier used to select its official score key.

    Returns:
        tuple[str, str, str, bool]: Review input, target, extracted prediction, and
        binary correctness assigned by EvalScope.

    Raises:
        ValueError: Raised when the nested sample score or accuracy is unavailable.
    """

    # 步骤1: 沿 EvalScope 1.4.2 的固定评分结构读取答案字段。
    try:
        score_object = review_row["sample_score"]["score"]
        score_values = score_object["value"]
        accuracy_key = "multi_choice_acc" if dataset_name == "truthful_qa" else "acc"
        accuracy_value = float(score_values[accuracy_key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"review 评分字段无效: {error}, 上下文: file={file_path}"
        ) from error
    if accuracy_value not in {0.0, 1.0}:
        raise ValueError(
            "review acc 不是二值分数, 上下文: "
            f"file={file_path}, acc={accuracy_value}"
        )

    # 步骤2: 字符串字段统一规范化，但不改写答案内容。
    review_input = str(review_row.get("input") or "")
    target = str(review_row.get("target") or "").strip()
    extracted_prediction = str(score_object.get("extracted_prediction") or "").strip()
    return review_input, target, extracted_prediction, accuracy_value == 1.0


def parse_allowed_option_labels(review_input: str) -> frozenset[str]:
    """Parse admissible multiple-choice labels from an EvalScope review prompt.

    Args:
        review_input: Human-readable prompt emitted in a review JSONL record.

    Returns:
        frozenset[str]: Uppercase option labels found at line starts, typically
        ``{"A", "B", "C", "D"}``.

    Raises:
        No exception is expected because regular-expression matching is total.
    """

    return frozenset(OPTION_LABEL_PATTERN.findall(review_input))


def is_valid_extracted_answer(
    dataset_name: str, extracted_prediction: str, review_input: str
) -> bool:
    """Determine whether an extracted answer is present and admissible.

    Args:
        dataset_name: Benchmark identifier supplied by the run directory.
        extracted_prediction: Evaluator-extracted answer from the review record.
        review_input: Prompt text used to recover legal multiple-choice labels.

    Returns:
        bool: True for a nonempty GSM8K answer or an allowed MCQ option label.

    Raises:
        ValueError: Raised for a dataset outside the formal six-benchmark suite.
    """

    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"未知数据集, 上下文: dataset={dataset_name}")
    normalized_prediction = extracted_prediction.strip().upper()
    if not normalized_prediction:
        return False
    if dataset_name == "gsm8k":
        return True

    # 步骤1: 对选择题要求抽取结果是单一合法选项，阻止截断公式中的 Q 等误判。
    allowed_labels = parse_allowed_option_labels(review_input)
    if not allowed_labels:
        allowed_labels = frozenset({"A", "B", "C", "D"})
    return len(normalized_prediction) == 1 and normalized_prediction in allowed_labels


def has_expected_answer_marker(dataset_name: str, raw_output: str) -> bool:
    """Check whether generation ends with the answer marker requested by prompts.

    Args:
        dataset_name: Benchmark identifier from the formal run.
        raw_output: Complete generated text returned by vLLM.

    Returns:
        bool: True when the final output contains the dataset's expected marker.

    Raises:
        ValueError: Raised for a dataset outside the configured formal suite.
    """

    normalized_output = raw_output.strip()
    if dataset_name in {"ceval", "cmmlu"}:
        return bool(CHINESE_FINAL_ANSWER_PATTERN.search(normalized_output))
    if dataset_name in {"mmlu", "hellaswag", "truthful_qa"}:
        return bool(ENGLISH_FINAL_ANSWER_PATTERN.search(normalized_output))
    if dataset_name == "gsm8k":
        return bool(BOXED_ANSWER_PATTERN.search(normalized_output))
    raise ValueError(f"未知数据集, 上下文: dataset={dataset_name}")


def build_sample_record(
    dataset_name: str,
    subset_name: str,
    prediction_row: dict[str, Any],
    review_row: dict[str, Any],
    prediction_file: Path,
    review_file: Path,
) -> SampleRecord:
    """Build one normalized sample from matching prediction and review rows.

    Args:
        dataset_name: Benchmark directory name supplied by the formal run.
        subset_name: Shared source filename stem for the two JSONL files.
        prediction_row: Prediction object indexed by EvalScope sample id.
        review_row: Review object with the same EvalScope sample id.
        prediction_file: Prediction path included in validation errors.
        review_file: Review path included in validation errors.

    Returns:
        SampleRecord: Validated generation and scoring record.

    Raises:
        ValueError: Raised for mismatched indices or malformed row structures.
    """

    # 步骤1: 验证两种产物确实指向同一 EvalScope 样本。
    prediction_index = _extract_sample_index(prediction_row, prediction_file)
    review_index = _extract_sample_index(review_row, review_file)
    if prediction_index != review_index:
        raise ValueError(
            "prediction/review index 不一致, 上下文: "
            f"prediction={prediction_index}, review={review_index}"
        )

    # 步骤2: 分别解析生成与评分字段，再派生技术诊断标志。
    generation_fields = extract_prediction_fields(prediction_row, prediction_file)
    review_fields = extract_review_fields(review_row, review_file, dataset_name)
    raw_output, stop_reason, input_tokens, output_tokens, has_error = generation_fields
    review_input, target, extracted_prediction, is_correct = review_fields
    return SampleRecord(
        dataset_name=dataset_name,
        subset_name=subset_name,
        sample_index=prediction_index,
        review_input=review_input,
        target=target,
        extracted_prediction=extracted_prediction,
        raw_output=raw_output,
        is_correct=is_correct,
        is_valid_answer=is_valid_extracted_answer(
            dataset_name, extracted_prediction, review_input
        ),
        has_expected_answer_marker=has_expected_answer_marker(dataset_name, raw_output),
        is_length_truncated=stop_reason.lower() in LENGTH_STOP_REASONS,
        has_generation_error=has_error,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _extract_sample_index(row: dict[str, Any], file_path: Path) -> int:
    """Extract and validate an EvalScope integer sample index.

    Args:
        row: Prediction or review row loaded from a JSONL file.
        file_path: Source file path used to provide error context.

    Returns:
        int: Nonnegative EvalScope sample index.

    Raises:
        ValueError: Raised when index is absent, nonintegral, or negative.
    """

    try:
        sample_index = int(row["index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"提取样本 index 失败: {error}, 上下文: file={file_path}"
        ) from error
    if sample_index < 0:
        raise ValueError(
            f"样本 index 不能为负数, 上下文: file={file_path}, index={sample_index}"
        )
    return sample_index


def index_rows_by_sample(
    rows: Iterable[dict[str, Any]], file_path: Path
) -> dict[int, dict[str, Any]]:
    """Index JSONL rows by EvalScope sample id and reject duplicates.

    Args:
        rows: Parsed JSONL objects produced by ``read_jsonl``.
        file_path: Source file path included in duplicate-key errors.

    Returns:
        dict[int, dict[str, Any]]: Mapping from index to source row.

    Raises:
        ValueError: Raised when a row has an invalid or duplicate sample index.
    """

    indexed_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_index = _extract_sample_index(row, file_path)
        if sample_index in indexed_rows:
            raise ValueError(
                "JSONL 中存在重复 index, 上下文: "
                f"file={file_path}, index={sample_index}"
            )
        indexed_rows[sample_index] = row
    return indexed_rows


def load_role_dataset_records(
    run_directory: Path, role_name: str, dataset_name: str
) -> dict[tuple[str, str, int], SampleRecord]:
    """Load all normalized records for one model role and benchmark.

    Args:
        run_directory: Formal benchmark run root supplied by the CLI.
        role_name: ``base`` or ``finetuned`` EvalScope role directory.
        dataset_name: One of the six configured benchmark names.

    Returns:
        dict[tuple[str, str, int], SampleRecord]: Records keyed by stable sample id.

    Raises:
        ValueError: Raised for file-set, row-index, or stable-key mismatches.
        RuntimeError: Raised when input JSONL files cannot be read.
    """

    artifact_root = run_directory / "evalscope" / role_name / dataset_name
    prediction_files = discover_jsonl_map(artifact_root / "predictions")
    review_files = discover_jsonl_map(artifact_root / "reviews")
    if prediction_files.keys() != review_files.keys():
        raise ValueError(
            "predictions/reviews 子集文件集合不一致, 上下文: "
            f"role={role_name}, dataset={dataset_name}"
        )

    # 步骤1: 每个子集内部按 index 配对，不能假定两类 JSONL 的行顺序一致。
    records_by_key: dict[tuple[str, str, int], SampleRecord] = {}
    for filename in sorted(prediction_files):
        subset_records = _load_subset_records(
            dataset_name,
            Path(filename).stem,
            prediction_files[filename],
            review_files[filename],
        )
        for sample_record in subset_records:
            if sample_record.sample_key in records_by_key:
                raise ValueError(
                    "跨子集出现重复稳定键, 上下文: "
                    f"role={role_name}, key={sample_record.sample_key}"
                )
            records_by_key[sample_record.sample_key] = sample_record
    return records_by_key


def _load_subset_records(
    dataset_name: str,
    subset_name: str,
    prediction_file: Path,
    review_file: Path,
) -> list[SampleRecord]:
    """Load and pair prediction/review rows for one benchmark subset file.

    Args:
        dataset_name: Benchmark identifier supplied by the parent directory.
        subset_name: Filename stem identifying the benchmark subject/subset.
        prediction_file: EvalScope prediction JSONL path.
        review_file: Matching EvalScope review JSONL path.

    Returns:
        list[SampleRecord]: Records sorted by integer sample index.

    Raises:
        ValueError: Raised when prediction and review index sets differ.
        RuntimeError: Raised when either source JSONL file cannot be read.
    """

    prediction_rows = index_rows_by_sample(read_jsonl(prediction_file), prediction_file)
    review_rows = index_rows_by_sample(read_jsonl(review_file), review_file)
    if prediction_rows.keys() != review_rows.keys():
        raise ValueError(
            "prediction/review index 集合不一致, 上下文: "
            f"prediction_file={prediction_file}, review_file={review_file}"
        )

    # 步骤1: 使用排序后的 index 生成确定性输出，便于重复运行后逐字核对。
    return [
        build_sample_record(
            dataset_name,
            subset_name,
            prediction_rows[sample_index],
            review_rows[sample_index],
            prediction_file,
            review_file,
        )
        for sample_index in sorted(prediction_rows)
    ]


def pair_dataset_records(
    base_records: dict[tuple[str, str, int], SampleRecord],
    finetuned_records: dict[tuple[str, str, int], SampleRecord],
) -> list[PairedSample]:
    """Pair all Base and finetuned records for one dataset by stable key.

    Args:
        base_records: Base records loaded by ``load_role_dataset_records``.
        finetuned_records: Finetuned records loaded from the same formal run.

    Returns:
        list[PairedSample]: Deterministically ordered, fully validated pairs.

    Raises:
        ValueError: Raised when either role contains missing or extra sample keys.
    """

    if base_records.keys() != finetuned_records.keys():
        base_only_keys = sorted(base_records.keys() - finetuned_records.keys())[:5]
        finetuned_only_keys = sorted(finetuned_records.keys() - base_records.keys())[:5]
        raise ValueError(
            "Base/Fine 样本键集合不一致, 上下文: "
            f"base_only={base_only_keys}, finetuned_only={finetuned_only_keys}"
        )

    # 步骤1: 构造对象时会再次核验提示、评分输入与参考答案。
    return [
        PairedSample(base_records[sample_key], finetuned_records[sample_key])
        for sample_key in sorted(base_records)
    ]


# ========== Statistical analysis ==========


def calculate_exact_mcnemar_pvalue(finetuned_only: int, base_only: int) -> float:
    """Compute a two-sided exact McNemar p-value from discordant pair counts.

    Args:
        finetuned_only: Pairs correct only for the finetuned model.
        base_only: Pairs correct only for the Base model.

    Returns:
        float: Exact two-sided binomial p-value in the closed interval [0, 1].

    Raises:
        ValueError: Raised when either count is negative.
    """

    if finetuned_only < 0 or base_only < 0:
        raise ValueError(
            "McNemar 计数不能为负数, 上下文: "
            f"finetuned_only={finetuned_only}, base_only={base_only}"
        )
    discordant_count = finetuned_only + base_only
    if discordant_count == 0:
        return 1.0
    return float(
        binomtest(
            k=finetuned_only,
            n=discordant_count,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def calculate_paired_bootstrap_interval(
    paired_differences: np.ndarray,
    repetition_count: int,
    random_seed: int,
    batch_size: int = 500,
) -> tuple[float, float]:
    """Estimate a percentile interval for paired accuracy difference.

    Args:
        paired_differences: One-dimensional Fine-minus-Base correctness array built
            from paired samples.
        repetition_count: Number of bootstrap replicates supplied by the CLI.
        random_seed: Deterministic seed supplied by the CLI.
        batch_size: Replicates generated per batch to bound memory usage.

    Returns:
        tuple[float, float]: 2.5th and 97.5th percentile limits.

    Raises:
        ValueError: Raised for empty/non-1D data or nonpositive configuration values.
    """

    if paired_differences.ndim != 1 or paired_differences.size == 0:
        raise ValueError(
            "bootstrap 输入必须是一维非空数组, 上下文: "
            f"shape={paired_differences.shape}"
        )
    if repetition_count <= 0 or batch_size <= 0:
        raise ValueError(
            "bootstrap 参数必须为正数, 上下文: "
            f"repetitions={repetition_count}, batch_size={batch_size}"
        )

    # 步骤1: 分批生成索引，避免一次创建 repetitions × samples 的大矩阵。
    random_generator = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(repetition_count, dtype=np.float64)
    for start_index in range(0, repetition_count, batch_size):
        end_index = min(start_index + batch_size, repetition_count)
        sampled_indices = random_generator.integers(
            0,
            paired_differences.size,
            size=(end_index - start_index, paired_differences.size),
            dtype=np.int32,
        )
        bootstrap_means[start_index:end_index] = paired_differences[
            sampled_indices
        ].mean(axis=1)
    lower_limit, upper_limit = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower_limit), float(upper_limit)


def summarize_paired_accuracy(
    paired_samples: Sequence[PairedSample],
    bootstrap_repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    """Summarize paired accuracy, discordance, significance, and uncertainty.

    Args:
        paired_samples: Matched records for one dataset or sensitivity subset.
        bootstrap_repetitions: Bootstrap count supplied by the CLI.
        random_seed: Dataset-specific deterministic seed derived by ``main``.

    Returns:
        dict[str, Any]: Counts, accuracies, delta, confidence interval, and exact
        McNemar p-value.

    Raises:
        ValueError: Raised when the paired sample sequence is empty.
    """

    if not paired_samples:
        raise ValueError("无法汇总空的配对样本序列")

    # 步骤1: 用布尔数组直接构建 2×2 配对表和逐样本差值。
    base_correct = np.fromiter(
        (pair.base_record.is_correct for pair in paired_samples), dtype=np.int8
    )
    finetuned_correct = np.fromiter(
        (pair.finetuned_record.is_correct for pair in paired_samples), dtype=np.int8
    )
    both_correct = int(np.sum((base_correct == 1) & (finetuned_correct == 1)))
    finetuned_only = int(np.sum((base_correct == 0) & (finetuned_correct == 1)))
    base_only = int(np.sum((base_correct == 1) & (finetuned_correct == 0)))
    both_wrong = int(np.sum((base_correct == 0) & (finetuned_correct == 0)))
    paired_differences = finetuned_correct - base_correct
    confidence_interval = calculate_paired_bootstrap_interval(
        paired_differences,
        bootstrap_repetitions,
        random_seed,
    )

    # 步骤2: 保留精确计数和未四舍五入浮点值，展示格式留给报告层处理。
    sample_count = len(paired_samples)
    base_correct_count = int(base_correct.sum())
    finetuned_correct_count = int(finetuned_correct.sum())
    return {
        "sample_count": sample_count,
        "base_correct_count": base_correct_count,
        "finetuned_correct_count": finetuned_correct_count,
        "base_accuracy": base_correct_count / sample_count,
        "finetuned_accuracy": finetuned_correct_count / sample_count,
        "accuracy_delta": float(paired_differences.mean()),
        "bootstrap_ci_95": list(confidence_interval),
        "both_correct": both_correct,
        "finetuned_only_correct": finetuned_only,
        "base_only_correct": base_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_pvalue": calculate_exact_mcnemar_pvalue(
            finetuned_only, base_only
        ),
    }


def apply_holm_correction(
    pvalues_by_dataset: dict[str, float]
) -> dict[str, float]:
    """Apply Holm's step-down family-wise correction across benchmark p-values.

    Args:
        pvalues_by_dataset: Raw p-values keyed by formal benchmark name.

    Returns:
        dict[str, float]: Holm-adjusted p-values in the original key space.

    Raises:
        ValueError: Raised for an empty mapping or values outside [0, 1].
    """

    if not pvalues_by_dataset:
        raise ValueError("Holm 校正至少需要一个 p 值")
    if any(not 0.0 <= value <= 1.0 for value in pvalues_by_dataset.values()):
        raise ValueError(f"p 值超出 [0, 1], 上下文: {pvalues_by_dataset}")

    # 步骤1: 按原始 p 值升序计算，并用累计最大值保持调整值单调。
    sorted_items = sorted(pvalues_by_dataset.items(), key=lambda item: item[1])
    family_size = len(sorted_items)
    adjusted_values: dict[str, float] = {}
    previous_adjusted = 0.0
    for rank_index, (dataset_name, raw_pvalue) in enumerate(sorted_items):
        candidate_value = min(1.0, (family_size - rank_index) * raw_pvalue)
        adjusted_value = max(previous_adjusted, candidate_value)
        adjusted_values[dataset_name] = adjusted_value
        previous_adjusted = adjusted_value
    return adjusted_values


def summarize_model_diagnostics(
    paired_samples: Sequence[PairedSample], role_name: str
) -> dict[str, Any]:
    """Aggregate answer validity, truncation, error, marker, and token statistics.

    Args:
        paired_samples: Matched records for one formal benchmark.
        role_name: ``base`` or ``finetuned``, selecting one side of every pair.

    Returns:
        dict[str, Any]: Diagnostic counts, rates, and token distribution summaries.

    Raises:
        ValueError: Raised for an unknown role or empty sample sequence.
    """

    if role_name not in ROLE_NAMES:
        raise ValueError(f"未知模型角色, 上下文: role={role_name}")
    if not paired_samples:
        raise ValueError("无法诊断空的配对样本序列")

    # 步骤1: 选取模型侧记录并计算技术失败标志。
    records = [
        pair.base_record if role_name == "base" else pair.finetuned_record
        for pair in paired_samples
    ]
    sample_count = len(records)
    invalid_count = sum(not record.is_valid_answer for record in records)
    truncated_count = sum(record.is_length_truncated for record in records)
    error_count = sum(record.has_generation_error for record in records)
    marker_count = sum(record.has_expected_answer_marker for record in records)
    input_tokens = np.asarray([record.input_tokens for record in records], dtype=float)
    output_tokens = np.asarray([record.output_tokens for record in records], dtype=float)
    return {
        "sample_count": sample_count,
        "invalid_answer_count": invalid_count,
        "invalid_answer_rate": invalid_count / sample_count,
        "length_truncated_count": truncated_count,
        "length_truncated_rate": truncated_count / sample_count,
        "generation_error_count": error_count,
        "generation_error_rate": error_count / sample_count,
        "expected_answer_marker_count": marker_count,
        "expected_answer_marker_rate": marker_count / sample_count,
        "input_tokens": summarize_numeric_distribution(input_tokens),
        "output_tokens": summarize_numeric_distribution(output_tokens),
    }


def summarize_numeric_distribution(values: np.ndarray) -> dict[str, float]:
    """Summarize one nonempty numeric vector for JSON reporting.

    Args:
        values: One-dimensional token-count array created from sample records.

    Returns:
        dict[str, float]: Mean, median, 95th percentile, and maximum.

    Raises:
        ValueError: Raised when values are empty or not one-dimensional.
    """

    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"数值分布必须是一维非空数组, 上下文: shape={values.shape}")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def summarize_pair_statuses(paired_samples: Sequence[PairedSample]) -> dict[str, Any]:
    """Decompose paired truncation, invalidity, and discordant correctness status.

    Args:
        paired_samples: Matched Base/finetuned records for one benchmark.

    Returns:
        dict[str, Any]: Joint status counts and technical causes among discordant
        correctness outcomes.

    Raises:
        ValueError: Raised when the sample sequence is empty.
    """

    if not paired_samples:
        raise ValueError("无法分解空的配对样本序列")

    # 步骤1: 分别统计截断与无效答案在两侧的联合分布。
    truncation_status = _joint_boolean_status(
        paired_samples,
        lambda pair: pair.base_record.is_length_truncated,
        lambda pair: pair.finetuned_record.is_length_truncated,
    )
    invalid_status = _joint_boolean_status(
        paired_samples,
        lambda pair: not pair.base_record.is_valid_answer,
        lambda pair: not pair.finetuned_record.is_valid_answer,
    )

    # 步骤2: 检查胜负样本中有多少可由对手侧技术失败直接解释。
    finetuned_only_pairs = [
        pair
        for pair in paired_samples
        if pair.finetuned_record.is_correct and not pair.base_record.is_correct
    ]
    base_only_pairs = [
        pair
        for pair in paired_samples
        if pair.base_record.is_correct and not pair.finetuned_record.is_correct
    ]
    return {
        "truncation_joint_counts": truncation_status,
        "invalid_answer_joint_counts": invalid_status,
        "finetuned_only_correct_technical_context": _summarize_losing_side_issues(
            finetuned_only_pairs, losing_role="base"
        ),
        "base_only_correct_technical_context": _summarize_losing_side_issues(
            base_only_pairs, losing_role="finetuned"
        ),
    }


def _joint_boolean_status(
    paired_samples: Sequence[PairedSample],
    base_flag_getter: Callable[[PairedSample], bool],
    finetuned_flag_getter: Callable[[PairedSample], bool],
) -> dict[str, int]:
    """Count four joint states for corresponding Base and finetuned flags.

    Args:
        paired_samples: Matched sample sequence from one benchmark.
        base_flag_getter: Callable extracting a boolean from a paired sample.
        finetuned_flag_getter: Callable extracting the corresponding Fine flag.

    Returns:
        dict[str, int]: Neither, Base-only, Fine-only, and both flag counts.

    Raises:
        No exception is expected when getters return boolean-compatible values.
    """

    counts = {"neither": 0, "base_only": 0, "finetuned_only": 0, "both": 0}
    for paired_sample in paired_samples:
        base_flag = bool(base_flag_getter(paired_sample))
        finetuned_flag = bool(finetuned_flag_getter(paired_sample))
        if base_flag and finetuned_flag:
            counts["both"] += 1
        elif base_flag:
            counts["base_only"] += 1
        elif finetuned_flag:
            counts["finetuned_only"] += 1
        else:
            counts["neither"] += 1
    return counts


def _summarize_losing_side_issues(
    discordant_pairs: Sequence[PairedSample], losing_role: str
) -> dict[str, int]:
    """Count technical issues on the losing side of correctness-discordant pairs.

    Args:
        discordant_pairs: Pairs where exactly one model answered correctly.
        losing_role: Role name of the incorrect model in these pairs.

    Returns:
        dict[str, int]: Total discordances and losing-side issue counts.

    Raises:
        ValueError: Raised when ``losing_role`` is not a configured role.
    """

    if losing_role not in ROLE_NAMES:
        raise ValueError(f"未知失败侧角色, 上下文: losing_role={losing_role}")
    losing_records = [
        pair.base_record if losing_role == "base" else pair.finetuned_record
        for pair in discordant_pairs
    ]
    return {
        "discordant_count": len(losing_records),
        "losing_side_truncated": sum(
            record.is_length_truncated for record in losing_records
        ),
        "losing_side_invalid": sum(not record.is_valid_answer for record in losing_records),
        "losing_side_generation_error": sum(
            record.has_generation_error for record in losing_records
        ),
        "losing_side_any_technical_issue": sum(
            record.is_length_truncated
            or not record.is_valid_answer
            or record.has_generation_error
            for record in losing_records
        ),
    }


def build_per_sample_diagnostic(pair: PairedSample) -> dict[str, Any]:
    """Convert one pair into a compact row for downstream case-level auditing.

    Args:
        pair: Validated Base/finetuned sample pair.

    Returns:
        dict[str, Any]: Stable identifiers, correctness, technical flags, token use,
        extracted answers, and raw outputs for both models.

    Raises:
        No exception is expected after pair validation.
    """

    return {
        "dataset": pair.base_record.dataset_name,
        "subset": pair.base_record.subset_name,
        "index": pair.base_record.sample_index,
        "target": pair.base_record.target,
        "base": _record_diagnostic_payload(pair.base_record),
        "finetuned": _record_diagnostic_payload(pair.finetuned_record),
        "is_clean_pair": pair.is_clean_pair,
        "outcome": _classify_pair_outcome(pair),
    }


def _record_diagnostic_payload(record: SampleRecord) -> dict[str, Any]:
    """Select serializable model-side fields for per-sample diagnostics.

    Args:
        record: Normalized record loaded from one model role.

    Returns:
        dict[str, Any]: Correctness, extracted answer, output, stop, and token fields.

    Raises:
        No exception is expected because the record is already normalized.
    """

    return {
        "is_correct": record.is_correct,
        "extracted_prediction": record.extracted_prediction,
        "raw_output": record.raw_output,
        "is_valid_answer": record.is_valid_answer,
        "has_expected_answer_marker": record.has_expected_answer_marker,
        "is_length_truncated": record.is_length_truncated,
        "has_generation_error": record.has_generation_error,
        "stop_reason": record.stop_reason,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
    }


def _classify_pair_outcome(pair: PairedSample) -> str:
    """Classify one pair into the four cells of the paired correctness table.

    Args:
        pair: Validated Base/finetuned sample pair.

    Returns:
        str: ``both_correct``, ``finetuned_only_correct``, ``base_only_correct``, or
        ``both_wrong``.

    Raises:
        No exception is expected because correctness fields are booleans.
    """

    if pair.base_record.is_correct and pair.finetuned_record.is_correct:
        return "both_correct"
    if pair.finetuned_record.is_correct:
        return "finetuned_only_correct"
    if pair.base_record.is_correct:
        return "base_only_correct"
    return "both_wrong"


# ========== Official report validation ==========


def load_official_report_score(
    run_directory: Path, role_name: str, dataset_name: str
) -> tuple[float, Path]:
    """Load the unique official EvalScope report score for a role and dataset.

    Args:
        run_directory: Formal benchmark run root supplied by the CLI.
        role_name: ``base`` or ``finetuned`` report directory name.
        dataset_name: Benchmark report filename stem.

    Returns:
        tuple[float, Path]: Official rounded score and source report path.

    Raises:
        ValueError: Raised when report discovery is ambiguous or score is invalid.
        RuntimeError: Raised when the JSON report cannot be opened or parsed.
    """

    report_root = (
        run_directory / "evalscope" / role_name / dataset_name / "reports"
    )
    try:
        matching_paths = sorted(report_root.rglob(f"{dataset_name}.json"))
    except OSError as error:
        raise RuntimeError(
            f"枚举官方报告失败: {error}, 上下文: root={report_root}"
        ) from error
    if len(matching_paths) != 1:
        raise ValueError(
            "官方报告数量不是 1, 上下文: "
            f"role={role_name}, dataset={dataset_name}, paths={matching_paths}"
        )
    report_path = matching_paths[0]
    try:
        report_object = json.loads(report_path.read_text(encoding="utf-8"))
        score_value = float(report_object["score"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"读取官方报告分数失败: {error}, 上下文: report={report_path}"
        ) from error
    return score_value, report_path


def validate_recomputed_accuracy(
    recomputed_accuracy: float,
    official_score: float,
    role_name: str,
    dataset_name: str,
    report_path: Path,
) -> None:
    """Verify that row-level correctness reproduces the rounded official score.

    Args:
        recomputed_accuracy: Exact correct-count ratio from review JSONL rows.
        official_score: Rounded score loaded from the EvalScope report.
        role_name: Model role included in any mismatch message.
        dataset_name: Benchmark name included in any mismatch message.
        report_path: Official report source path included for auditability.

    Returns:
        None: Validation succeeds without producing a value.

    Raises:
        ValueError: Raised when the score differs beyond four-decimal rounding.
    """

    if not math.isclose(round(recomputed_accuracy, 4), official_score, abs_tol=1e-8):
        raise ValueError(
            "逐样本准确率无法复现官方报告, 上下文: "
            f"role={role_name}, dataset={dataset_name}, "
            f"recomputed={recomputed_accuracy}, official={official_score}, "
            f"report={report_path}"
        )


# ========== Output writing ==========


def write_json(output_path: Path, payload: Any) -> None:
    """Write one UTF-8, indented JSON artifact atomically enough for this workflow.

    Args:
        output_path: Destination path under the new analysis directory.
        payload: JSON-serializable analysis object created by this module.

    Returns:
        None: The completed file is available at ``output_path``.

    Raises:
        RuntimeError: Raised when directory creation, serialization, or writing fails.
    """

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)
        output_path.write_text(serialized_payload + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"写入 JSON 失败: {error}, 上下文: output={output_path}"
        ) from error


def write_jsonl(output_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write iterable analysis rows to a UTF-8 JSONL artifact.

    Args:
        output_path: Destination path under the analysis directory.
        rows: Serializable dictionaries generated from paired samples.

    Returns:
        None: All rows are persisted at ``output_path``.

    Raises:
        RuntimeError: Raised when directory creation, serialization, or writing fails.
    """

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"写入 JSONL 失败: {error}, 上下文: output={output_path}"
        ) from error


def write_markdown_report(output_path: Path, analysis: dict[str, Any]) -> None:
    """Render the structured analysis into a concise Markdown audit report.

    Args:
        output_path: Destination report path under the analysis directory.
        analysis: Complete analysis object assembled by ``run_analysis``.

    Returns:
        None: Human-readable report is written to ``output_path``.

    Raises:
        RuntimeError: Raised when report formatting or file writing fails.
    """

    try:
        report_text = render_markdown_report(analysis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"写入 Markdown 报告失败: {error}, 上下文: output={output_path}"
        ) from error


def render_markdown_report(analysis: dict[str, Any]) -> str:
    """Build a Markdown report from structured benchmark analysis results.

    Args:
        analysis: Complete result mapping returned by ``run_analysis``.

    Returns:
        str: Markdown containing formal scores, tests, diagnostics, and sensitivity.

    Raises:
        KeyError: Raised when required analysis sections are missing.
    """

    lines = [
        "# General Benchmark Paired Analysis",
        "",
        "Official accuracy keeps every evaluated sample. The clean-pair table is a ",
        "post-hoc sensitivity analysis only and does not replace formal scores.",
        "",
        "## Paired significance",
        "",
        "| Dataset | n | Base | Fine | Delta | 95% paired bootstrap CI | Fine-only | Base-only | Exact McNemar p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in DATASET_NAMES:
        result = analysis["datasets"][dataset_name]["official_paired_accuracy"]
        interval = result["bootstrap_ci_95"]
        lines.append(
            f"| {dataset_name} | {result['sample_count']:,} | "
            f"{_format_percent(result['base_accuracy'])} | "
            f"{_format_percent(result['finetuned_accuracy'])} | "
            f"{_format_signed_percent(result['accuracy_delta'])} | "
            f"[{_format_signed_percent(interval[0])}, {_format_signed_percent(interval[1])}] | "
            f"{result['finetuned_only_correct']:,} | {result['base_only_correct']:,} | "
            f"{_format_pvalue(result['mcnemar_exact_pvalue'])} | "
            f"{_format_pvalue(result['holm_adjusted_pvalue'])} |"
        )

    lines.extend(_render_diagnostic_tables(analysis))
    lines.extend(_render_sensitivity_table(analysis))
    lines.extend(_render_interpretation(analysis))
    return "\n".join(lines).rstrip() + "\n"


def _render_diagnostic_tables(analysis: dict[str, Any]) -> list[str]:
    """Render model-side technical diagnostics as Markdown rows.

    Args:
        analysis: Complete result mapping returned by ``run_analysis``.

    Returns:
        list[str]: Markdown lines for invalid, truncation, error, marker, and tokens.

    Raises:
        KeyError: Raised when required per-dataset diagnostic fields are absent.
    """

    lines = [
        "",
        "## Generation diagnostics",
        "",
        "An invalid MCQ answer is empty or not one of the option labels parsed from the prompt. GSM8K is invalid only when extraction is empty. Marker rate checks whether output ends with the requested answer marker.",
        "",
        "| Dataset | Model | Invalid | Truncated | Gen. error | Expected marker | Output tokens mean / p95 / max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in DATASET_NAMES:
        for role_name in ROLE_NAMES:
            diagnostics = analysis["datasets"][dataset_name]["diagnostics"][role_name]
            token_summary = diagnostics["output_tokens"]
            lines.append(
                f"| {dataset_name} | {role_name} | "
                f"{diagnostics['invalid_answer_count']:,} ({_format_percent(diagnostics['invalid_answer_rate'])}) | "
                f"{diagnostics['length_truncated_count']:,} ({_format_percent(diagnostics['length_truncated_rate'])}) | "
                f"{diagnostics['generation_error_count']:,} ({_format_percent(diagnostics['generation_error_rate'])}) | "
                f"{diagnostics['expected_answer_marker_count']:,} ({_format_percent(diagnostics['expected_answer_marker_rate'])}) | "
                f"{token_summary['mean']:.1f} / {token_summary['p95']:.1f} / {token_summary['max']:.0f} |"
            )
    return lines


def _render_sensitivity_table(analysis: dict[str, Any]) -> list[str]:
    """Render the nonofficial clean-pair sensitivity analysis table.

    Args:
        analysis: Complete result mapping returned by ``run_analysis``.

    Returns:
        list[str]: Markdown lines for retained counts and paired clean deltas.

    Raises:
        KeyError: Raised when a dataset lacks sensitivity statistics.
    """

    lines = [
        "",
        "## Clean-pair sensitivity analysis",
        "",
        "Pairs are retained only when both outputs are non-truncated, generation-error-free, and have valid extracted answers.",
        "",
        "| Dataset | Clean n | Retained | Base | Fine | Delta | 95% paired bootstrap CI | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in DATASET_NAMES:
        dataset_result = analysis["datasets"][dataset_name]
        sensitivity = dataset_result["clean_pair_sensitivity"]
        interval = sensitivity["bootstrap_ci_95"]
        retained_rate = sensitivity["sample_count"] / dataset_result[
            "official_paired_accuracy"
        ]["sample_count"]
        lines.append(
            f"| {dataset_name} | {sensitivity['sample_count']:,} | "
            f"{_format_percent(retained_rate)} | {_format_percent(sensitivity['base_accuracy'])} | "
            f"{_format_percent(sensitivity['finetuned_accuracy'])} | "
            f"{_format_signed_percent(sensitivity['accuracy_delta'])} | "
            f"[{_format_signed_percent(interval[0])}, {_format_signed_percent(interval[1])}] | "
            f"{_format_pvalue(sensitivity['mcnemar_exact_pvalue'])} |"
        )
    return lines


def _render_interpretation(analysis: dict[str, Any]) -> list[str]:
    """Render restrained automatic interpretation of multiplicity-adjusted results.

    Args:
        analysis: Complete result mapping returned by ``run_analysis``.

    Returns:
        list[str]: Markdown lines naming positive, negative, and nonsignificant sets.

    Raises:
        KeyError: Raised when paired results are incomplete.
    """

    positive_names: list[str] = []
    negative_names: list[str] = []
    nonsignificant_names: list[str] = []
    for dataset_name in DATASET_NAMES:
        result = analysis["datasets"][dataset_name]["official_paired_accuracy"]
        if result["holm_adjusted_pvalue"] >= 0.05:
            nonsignificant_names.append(dataset_name)
        elif result["accuracy_delta"] > 0:
            positive_names.append(dataset_name)
        else:
            negative_names.append(dataset_name)
    return [
        "",
        "## Statistical reading",
        "",
        f"- Holm-significant positive changes: {_format_name_list(positive_names)}.",
        f"- Holm-significant negative changes: {_format_name_list(negative_names)}.",
        f"- Not significant after Holm correction: {_format_name_list(nonsignificant_names)}.",
        "- Statistical significance describes paired score differences; it does not by itself establish practical importance or explain the mechanism.",
        "- Technical-failure exclusions are reported only as sensitivity checks. Formal benchmark claims should use the all-sample table.",
    ]


def _format_percent(value: float) -> str:
    """Format a proportion as a percentage with two decimal places.

    Args:
        value: Proportion generated by the analysis, usually in [0, 1].

    Returns:
        str: Percentage text such as ``74.37%``.

    Raises:
        No exception is expected for finite numeric input.
    """

    return f"{100.0 * value:.2f}%"


def _format_signed_percent(value: float) -> str:
    """Format a proportion as an explicitly signed percentage-point change.

    Args:
        value: Fine-minus-Base difference generated by paired analysis.

    Returns:
        str: Signed percentage-point text such as ``+2.67 pp``.

    Raises:
        No exception is expected for finite numeric input.
    """

    return f"{100.0 * value:+.2f} pp"


def _format_pvalue(value: float) -> str:
    """Format a p-value without hiding very small nonzero values.

    Args:
        value: Exact or adjusted p-value in [0, 1].

    Returns:
        str: Fixed notation for ordinary values and scientific notation below 1e-4.

    Raises:
        No exception is expected for validated p-values.
    """

    return f"{value:.3e}" if value < 0.0001 else f"{value:.4f}"


def _format_name_list(names: Sequence[str]) -> str:
    """Format dataset names for a human-readable interpretation sentence.

    Args:
        names: Dataset names classified by adjusted significance and delta direction.

    Returns:
        str: Comma-separated names or ``none`` for an empty category.

    Raises:
        No exception is expected for a sequence of strings.
    """

    return ", ".join(names) if names else "none"


# ========== Analysis orchestration ==========


def analyze_one_dataset(
    run_directory: Path,
    dataset_name: str,
    bootstrap_repetitions: int,
    random_seed: int,
) -> tuple[dict[str, Any], list[PairedSample]]:
    """Load, validate, and analyze one formal benchmark dataset.

    Args:
        run_directory: Formal benchmark run root supplied by the CLI.
        dataset_name: One configured benchmark identifier.
        bootstrap_repetitions: Number of paired bootstrap replicates.
        random_seed: Dataset-specific deterministic bootstrap seed.

    Returns:
        tuple[dict[str, Any], list[PairedSample]]: Dataset summary and validated
        paired records retained for per-sample output.

    Raises:
        ValueError: Raised for pairing, report-reproduction, or empty-clean-set issues.
        RuntimeError: Raised for source file I/O or JSON parsing failures.
    """

    # 步骤1: 独立加载两侧记录，并按稳定键和实际提示严格配对。
    base_records = load_role_dataset_records(run_directory, "base", dataset_name)
    finetuned_records = load_role_dataset_records(
        run_directory, "finetuned", dataset_name
    )
    paired_samples = pair_dataset_records(base_records, finetuned_records)
    official_summary = summarize_paired_accuracy(
        paired_samples, bootstrap_repetitions, random_seed
    )

    # 步骤2: 用官方报告反查逐样本重算结果，防止分析了错误目录。
    official_report_paths: dict[str, str] = {}
    for role_name in ROLE_NAMES:
        report_score, report_path = load_official_report_score(
            run_directory, role_name, dataset_name
        )
        recomputed_accuracy = official_summary[f"{role_name}_accuracy"]
        validate_recomputed_accuracy(
            recomputed_accuracy, report_score, role_name, dataset_name, report_path
        )
        official_report_paths[role_name] = str(report_path)

    # 步骤3: 技术诊断和 clean subset 只用于解释与稳健性检查。
    clean_pairs = [pair for pair in paired_samples if pair.is_clean_pair]
    if not clean_pairs:
        raise ValueError(f"clean-pair 子集为空, 上下文: dataset={dataset_name}")
    clean_summary = summarize_paired_accuracy(
        clean_pairs, bootstrap_repetitions, random_seed + 10_000
    )
    dataset_summary = {
        "official_paired_accuracy": official_summary,
        "official_report_paths": official_report_paths,
        "diagnostics": {
            role_name: summarize_model_diagnostics(paired_samples, role_name)
            for role_name in ROLE_NAMES
        },
        "pair_statuses": summarize_pair_statuses(paired_samples),
        "clean_pair_sensitivity": clean_summary,
    }
    return dataset_summary, paired_samples


def run_analysis(
    run_directory: Path,
    output_directory: Path,
    bootstrap_repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    """Run all six paired analyses and write structured and human-readable outputs.

    Args:
        run_directory: Formal benchmark run root validated from CLI input.
        output_directory: New analysis artifact directory.
        bootstrap_repetitions: Number of paired bootstrap replicates per table row.
        random_seed: Global deterministic seed supplied by the CLI.

    Returns:
        dict[str, Any]: Complete analysis object also written to JSON.

    Raises:
        ValueError: Raised when input artifacts violate pairing or score invariants.
        RuntimeError: Raised when source or output file operations fail.
    """

    # 步骤1: 逐数据集分析并保留全部配对记录用于案例级产物。
    dataset_results: dict[str, dict[str, Any]] = {}
    all_paired_samples: list[PairedSample] = []
    for dataset_index, dataset_name in enumerate(DATASET_NAMES):
        dataset_result, paired_samples = analyze_one_dataset(
            run_directory,
            dataset_name,
            bootstrap_repetitions,
            random_seed + dataset_index,
        )
        dataset_results[dataset_name] = dataset_result
        all_paired_samples.extend(paired_samples)

    # 步骤2: 对六次正式 McNemar 检验统一进行 Holm 多重比较校正。
    raw_pvalues = {
        dataset_name: dataset_results[dataset_name]["official_paired_accuracy"][
            "mcnemar_exact_pvalue"
        ]
        for dataset_name in DATASET_NAMES
    }
    adjusted_pvalues = apply_holm_correction(raw_pvalues)
    for dataset_name, adjusted_pvalue in adjusted_pvalues.items():
        paired_result = dataset_results[dataset_name]["official_paired_accuracy"]
        paired_result["holm_adjusted_pvalue"] = adjusted_pvalue
        paired_result["is_significant_after_holm_0_05"] = adjusted_pvalue < 0.05

    # 步骤3: 写入汇总、全部样本和胜负分歧样本，原始 EvalScope 文件不改动。
    analysis = {
        "analysis_protocol": {
            "pair_key": ["dataset", "subset_filename_stem", "index"],
            "official_scope": "all samples",
            "sensitivity_scope": "both outputs valid, non-truncated, and error-free",
            "mcnemar_test": "two-sided exact binomial test on discordant pairs",
            "multiplicity_correction": "Holm correction across six datasets",
            "bootstrap_method": "paired percentile bootstrap",
            "bootstrap_repetitions": bootstrap_repetitions,
            "random_seed": random_seed,
        },
        "run_directory": str(run_directory),
        "total_paired_samples": len(all_paired_samples),
        "datasets": dataset_results,
    }
    diagnostic_rows = [
        build_per_sample_diagnostic(paired_sample)
        for paired_sample in all_paired_samples
    ]
    discordant_rows = [
        row
        for row in diagnostic_rows
        if row["outcome"] in {"finetuned_only_correct", "base_only_correct"}
    ]
    write_json(output_directory / "general_benchmark_analysis.json", analysis)
    write_jsonl(output_directory / "per_sample_diagnostics.jsonl", diagnostic_rows)
    write_jsonl(output_directory / "discordant_cases.jsonl", discordant_rows)
    write_markdown_report(
        output_directory / "GENERAL_BENCHMARK_ANALYSIS.md", analysis
    )
    return analysis


def main() -> None:
    """Parse CLI input and execute the formal paired benchmark analysis.

    The function is the module's command-line entry point. Paths and statistical
    settings come from user arguments; all analysis artifacts are written beneath
    a separate output directory.

    Returns:
        None: A compact completion summary is printed after files are written.

    Raises:
        ValueError: Propagated for invalid arguments or inconsistent input artifacts.
        RuntimeError: Propagated for input/output failures with detailed context.
    """

    # 步骤1: 验证命令参数并运行六数据集的统一分析协议。
    parser = build_argument_parser()
    arguments = parser.parse_args()
    run_directory, output_directory = validate_arguments(arguments)
    analysis = run_analysis(
        run_directory,
        output_directory,
        arguments.bootstrap_repetitions,
        arguments.random_seed,
    )

    # 步骤2: 控制台只输出稳定进度摘要，详细统计保存在报告中。
    print(f"Analysis complete: {output_directory}")
    print(f"Paired samples: {analysis['total_paired_samples']}")
    for dataset_name in DATASET_NAMES:
        result = analysis["datasets"][dataset_name]["official_paired_accuracy"]
        print(
            f"{dataset_name}: base={result['base_accuracy']:.4f}, "
            f"finetuned={result['finetuned_accuracy']:.4f}, "
            f"delta={result['accuracy_delta']:+.4f}, "
            f"holm_p={result['holm_adjusted_pvalue']:.6g}"
        )


if __name__ == "__main__":
    # 用法示例：python src/analyze_general_benchmarks.py --run-dir <formal-run-dir>
    main()
