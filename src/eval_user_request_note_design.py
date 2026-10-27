"""
Evaluate Base and finetuned models on user-request-to-note-pyramid design.

The module supports separate single-model generation phases for a 24 GB GPU,
then pairs outputs for schema, explicit-constraint, vocabulary, diversity, and
high-frequency-term diagnostics. The release excludes canceled questionnaire
creation and reports only researcher-led exploratory analysis.

External dependencies for generation are vLLM, Transformers, and PyTorch. The
analysis phase uses only the Python standard library and local JSONL artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence


# ========== Configuration ==========

# ------ Reproducible pilot protocol ------

RANDOM_SEED = 3407
DEFAULT_MAX_MODEL_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_GPU_MEMORY_UTILIZATION = 0.82
REQUIRED_NOTE_KEYS: tuple[str, str, str] = (
    "top_notes",
    "middle_notes",
    "base_notes",
)
MODEL_LABELS = {
    "base": "Qwen3.5-4B-Base",
    "finetuned": "Qwen3.5-4B-Finetuned-Step700",
}
HIGH_FREQUENCY_TERMS: tuple[str, str, str] = ("bergamot", "musk", "amber")
NOTE_CONSTRAINT_ALIASES: dict[str, tuple[str, ...]] = {
    "cedar": ("cedarwood",),
    "iris": ("orris", "orris root"),
    "smoke": ("smoky", "smoky notes"),
}
CJK_CHARACTER_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ASCII_LETTER_PATTERN = re.compile(r"[A-Za-z]")

DESIGN_SYSTEM_PROMPT = """You are a fragrance note-pyramid design assistant.
Translate the user's scent brief into an original, concept-level fragrance note architecture that can guide later formula development.
Interpret the requested atmosphere, use context, emotional tone, olfactory preferences, exclusions, and desired evolution from opening to drydown.
Choose a coherent and selective set of notes rather than listing every plausible material.
Treat explicit constraints as binding.
Return only a valid JSON object with exactly these keys: top_notes, middle_notes, base_notes.
Each value must be a list of 1 to 5 English fragrance note terms."""

VISIBLE_PLAN_SYSTEM_PROMPT = """You are a fragrance note-pyramid design assistant.
Translate the user's scent brief into an original, concept-level fragrance note architecture that can guide later formula development.
Interpret the requested atmosphere, use context, emotional tone, olfactory preferences, exclusions, and desired evolution from opening to drydown.
Choose a coherent and selective set of notes rather than listing every plausible material.
Treat explicit constraints as binding.
First write a concise planning paragraph in English of no more than 100 words. Use it to identify the brief's central idea, binding constraints, and opening-to-drydown progression; do not enumerate many alternative note lists.
Then write FINAL_JSON: followed by one valid JSON object with exactly these keys: top_notes, middle_notes, base_notes.
Each value must be a list of 1 to 5 English fragrance note terms. Do not write anything after the JSON object."""
REASONING_MODES: tuple[str, str, str] = (
    "direct",
    "native_thinking",
    "visible_plan",
)


@dataclass(frozen=True)
class ParsedPrediction:
    """Represent strict JSON parsing and normalized fragrance-note tiers.

    Attributes:
        parsed_json: Strictly parsed JSON object, or ``None`` on parse failure.
        canonical_notes: Three normalized note lists, empty when unavailable.
        is_strict_json: Whether the complete response is one JSON object.
        is_recoverable_json: Whether a final JSON object can be recovered after a preamble.
        has_exact_keys: Whether the object contains exactly the required three keys.
        has_valid_schema: Whether all required values are nonempty string lists.
        has_valid_tier_sizes: Whether every tier contains between one and five notes.
        preamble_text: Text appearing before the recovered final JSON object.
        trailing_text: Non-whitespace text appearing after the recovered JSON object.
    """

    parsed_json: dict[str, Any] | None
    canonical_notes: dict[str, list[str]]
    is_strict_json: bool
    is_recoverable_json: bool
    has_exact_keys: bool
    has_valid_schema: bool
    has_valid_tier_sizes: bool
    preamble_text: str
    trailing_text: str


# ========== Command-line interface ==========


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI options for model generation and paired analysis.

    Returns:
        argparse.ArgumentParser: Configured parser consumed by ``main``.

    Raises:
        No exception is expected while constructing static argument definitions.
    """

    project_root = Path(__file__).resolve().parents[1]
    default_run_name = datetime.now().strftime("%Y%m%d_%H%M%S_user_request_pilot")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "analyze"), required=True)
    parser.add_argument("--model", choices=("base", "finetuned"))
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--requests-file", type=Path)
    parser.add_argument("--training-file", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--finetuned-model", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=project_root / "src/outputs/user_request_pilot" / default_run_name,
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-model-length", type=int, default=DEFAULT_MAX_MODEL_LENGTH)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Legacy alias for --reasoning-mode native_thinking.",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=REASONING_MODES,
        default="direct",
        help="Use direct JSON, native Qwen thinking, or a bounded visible planning paragraph.",
    )
    parser.add_argument(
        "--comparison-run-dir",
        type=Path,
        default=None,
        help="Optional prior no-thinking run used for same-request anchor comparison.",
    )
    return parser


def resolve_and_validate_arguments(arguments: argparse.Namespace) -> argparse.Namespace:
    """Resolve project-relative defaults and validate requested pilot settings.

    Args:
        arguments: Raw namespace produced from user-provided CLI arguments.

    Returns:
        argparse.Namespace: Namespace with resolved dataset, model, and output paths.

    Raises:
        ValueError: Raised for missing model role or invalid numeric configuration.
        FileNotFoundError: Raised when a required request, training, or model path is absent.
    """

    # 步骤1: 根据可覆盖的项目根目录补齐全部默认路径。
    project_root = arguments.project_root.expanduser().resolve()
    arguments.project_root = project_root
    arguments.requests_file = (
        arguments.requests_file or project_root / "data/user_request_note_design_pilot_v1.jsonl"
    ).resolve()
    arguments.training_file = (
        arguments.training_file
        or project_root / "data/qwen_finetune_notes_review_v2_strict_chat_train.jsonl"
    ).resolve()
    arguments.base_model = (
        arguments.base_model or project_root / "model/Qwen3.5-4b/unsloth/Qwen3.5-4B"
    ).resolve()
    arguments.finetuned_model = (
        arguments.finetuned_model
        or project_root / "output/finetuned_qwen35_4b_v2_chat_l3_fixed_unsloth_merged"
    ).resolve()
    arguments.run_dir = arguments.run_dir.expanduser().resolve()
    arguments.comparison_run_dir = (
        arguments.comparison_run_dir.expanduser().resolve()
        if arguments.comparison_run_dir is not None
        else None
    )

    # 步骤2: 将旧 thinking 开关规范化为单一推理模式，避免提示与模板不一致。
    if arguments.enable_thinking and arguments.reasoning_mode != "direct":
        raise ValueError(
            "推理模式参数冲突, 上下文: "
            f"enable_thinking={arguments.enable_thinking}, "
            f"reasoning_mode={arguments.reasoning_mode}"
        )
    if arguments.enable_thinking:
        arguments.reasoning_mode = "native_thinking"
    arguments.enable_thinking = arguments.reasoning_mode == "native_thinking"

    # 步骤3: 拒绝会破坏生成或显存配置的参数。
    if arguments.phase == "generate" and arguments.model is None:
        raise ValueError(
            f"生成阶段必须指定模型角色, 上下文: phase={arguments.phase}"
        )
    if arguments.max_new_tokens <= 0 or arguments.max_model_length <= 0:
        raise ValueError(
            "token 上限必须为正数, 上下文: "
            f"max_new_tokens={arguments.max_new_tokens}, "
            f"max_model_length={arguments.max_model_length}"
        )
    if not 0.1 <= arguments.gpu_memory_utilization <= 0.95:
        raise ValueError(
            "GPU 显存利用率超出范围, 上下文: "
            f"gpu_memory_utilization={arguments.gpu_memory_utilization}"
        )

    # 步骤4: 按阶段检查真正需要读取的外部文件。
    required_paths = [arguments.requests_file]
    if arguments.phase == "generate":
        selected_model_path = (
            arguments.base_model
            if arguments.model == "base"
            else arguments.finetuned_model
        )
        required_paths.append(selected_model_path)
    else:
        required_paths.append(arguments.training_file)
        required_paths.extend(
            arguments.run_dir / "generations" / f"{role_name}.jsonl"
            for role_name in MODEL_LABELS
        )
        if arguments.comparison_run_dir is not None:
            required_paths.extend(
                arguments.comparison_run_dir
                / "generations"
                / f"{role_name}.jsonl"
                for role_name in MODEL_LABELS
            )
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"必要输入不存在, 上下文: path={required_path}, phase={arguments.phase}"
            )
    return arguments


# ========== File and request handling ==========


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file and validate that every nonempty line is an object.

    Args:
        file_path: Existing input or generation path selected by the caller.

    Returns:
        list[dict[str, Any]]: Parsed objects in unchanged source order.

    Raises:
        RuntimeError: Raised when file I/O or JSON parsing fails with line context.
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
                        f"JSONL 行不是对象, line={line_number}, value={type(parsed_value)}"
                    )
                parsed_rows.append(parsed_value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"读取 JSONL 失败: {error}, 上下文: file={file_path}"
        ) from error
    return parsed_rows


def write_json(file_path: Path, payload: Any) -> None:
    """Write one indented UTF-8 JSON artifact.

    Args:
        file_path: Destination beneath the pilot run directory.
        payload: JSON-serializable configuration or analysis object.

    Returns:
        None: The completed file is available at ``file_path``.

    Raises:
        RuntimeError: Raised when directory creation, serialization, or writing fails.
    """

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)
        file_path.write_text(serialized_payload + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"写入 JSON 失败: {error}, 上下文: file={file_path}"
        ) from error


def write_jsonl(file_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries as deterministic UTF-8 JSONL rows.

    Args:
        file_path: Destination beneath the pilot run directory.
        rows: Serializable row iterable produced by generation or analysis.

    Returns:
        None: All rows are persisted at ``file_path``.

    Raises:
        RuntimeError: Raised when directory creation, serialization, or writing fails.
    """

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"写入 JSONL 失败: {error}, 上下文: file={file_path}"
        ) from error


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate a lowercase SHA-256 digest for an external input or artifact.

    Args:
        file_path: Existing file whose exact contents should be recorded.

    Returns:
        str: Sixty-four-character lowercase hexadecimal digest.

    Raises:
        RuntimeError: Raised when the file cannot be opened or read completely.
    """

    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as input_file:
            for data_chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(data_chunk)
    except OSError as error:
        raise RuntimeError(
            f"计算文件哈希失败: {error}, 上下文: file={file_path}"
        ) from error
    return digest.hexdigest()


def hash_text(text_value: str) -> str:
    """Calculate a SHA-256 digest for a fixed prompt string.

    Args:
        text_value: System prompt or other protocol text defined by this module.

    Returns:
        str: Lowercase SHA-256 digest of the UTF-8 encoded string.

    Raises:
        No exception is expected for a Python string.
    """

    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def load_and_validate_requests(file_path: Path) -> list[dict[str, Any]]:
    """Load simulated user briefs and validate constraint metadata.

    Args:
        file_path: Pilot request JSONL supplied by CLI or project default.

    Returns:
        list[dict[str, Any]]: Ordered request rows ready for generation.

    Raises:
        ValueError: Raised for duplicate ids, missing text, unknown language, or
            malformed tier/constraint fields.
        RuntimeError: Raised when the source JSONL cannot be read.
    """

    request_rows = read_jsonl(file_path)
    if not request_rows:
        raise ValueError(f"请求文件为空, 上下文: file={file_path}")
    seen_request_ids: set[str] = set()
    for request_row in request_rows:
        _validate_request_row(request_row, seen_request_ids)
        seen_request_ids.add(request_row["request_id"])
    return request_rows


def _validate_request_row(
    request_row: dict[str, Any], seen_request_ids: set[str]
) -> None:
    """Validate one simulated user request and its objective-check metadata.

    Args:
        request_row: Object loaded from the request-bank JSONL file.
        seen_request_ids: Ids validated earlier in the same source file.

    Returns:
        None: Validation succeeds without modifying the row.

    Raises:
        ValueError: Raised when required values or constraint shapes are invalid.
    """

    # 步骤1: 验证问卷和配对依赖的稳定标识与文本字段。
    request_id = request_row.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError(f"request_id 无效, 上下文: row={request_row}")
    if request_id in seen_request_ids:
        raise ValueError(f"request_id 重复, 上下文: request_id={request_id}")
    if request_row.get("language") not in {"zh", "en"}:
        raise ValueError(f"language 无效, 上下文: request_id={request_id}")
    if not isinstance(request_row.get("user_request"), str) or not request_row[
        "user_request"
    ].strip():
        raise ValueError(f"user_request 无效, 上下文: request_id={request_id}")

    # 步骤2: 验证 tier 约束只引用正式输出键，且值类型可自动审计。
    required_by_tier = request_row.get("required_by_tier")
    exact_tier_counts = request_row.get("exact_tier_counts")
    forbidden_anywhere = request_row.get("forbidden_anywhere")
    if not isinstance(required_by_tier, dict) or not isinstance(exact_tier_counts, dict):
        raise ValueError(f"层级约束必须是对象, 上下文: request_id={request_id}")
    if not isinstance(forbidden_anywhere, list) or not all(
        isinstance(note_name, str) and note_name.strip()
        for note_name in forbidden_anywhere
    ):
        raise ValueError(f"禁用香调列表无效, 上下文: request_id={request_id}")
    for tier_name, note_names in required_by_tier.items():
        if tier_name not in REQUIRED_NOTE_KEYS or not isinstance(note_names, list):
            raise ValueError(
                f"必选层级约束无效, 上下文: request_id={request_id}, tier={tier_name}"
            )
    for tier_name, tier_count in exact_tier_counts.items():
        if tier_name not in REQUIRED_NOTE_KEYS or not isinstance(tier_count, int):
            raise ValueError(
                f"层级数量约束无效, 上下文: request_id={request_id}, tier={tier_name}"
            )


# ========== JSON prediction parsing ==========


def normalize_note_term(note_term: str) -> str:
    """Normalize a fragrance-note term for exact constraint and vocabulary checks.

    Args:
        note_term: Model-generated or metadata-provided English fragrance term.

    Returns:
        str: Lowercase whitespace-collapsed term without surrounding punctuation.

    Raises:
        No exception is expected for a validated string input.
    """

    return " ".join(note_term.strip().lower().strip(".,;:").split())


def note_matches_constraint(generated_note: str, constraint_note: str) -> bool:
    """Match a generated term to a user constraint at phrase or alias level.

    Args:
        generated_note: Normalized or raw model-generated fragrance-note term.
        constraint_note: Normalized or raw required/forbidden term from request metadata.

    Returns:
        bool: True for exact, whole-phrase, or explicitly registered alias matches.

    Raises:
        No exception is expected for string inputs.
    """

    normalized_generated = normalize_note_term(generated_note)
    normalized_constraint = normalize_note_term(constraint_note)
    candidate_constraints = (
        normalized_constraint,
        *NOTE_CONSTRAINT_ALIASES.get(normalized_constraint, ()),
    )
    for candidate_constraint in candidate_constraints:
        if normalized_generated == candidate_constraint:
            return True
        phrase_pattern = re.compile(
            rf"(?<![a-z]){re.escape(candidate_constraint)}(?![a-z])"
        )
        if phrase_pattern.search(normalized_generated):
            return True
    return False


def is_english_note_term(note_term: str) -> bool:
    """Check whether a generated note term follows the requested English output language.

    Args:
        note_term: Model-generated fragrance-note string from any output tier.

    Returns:
        bool: True when the term has an ASCII letter and contains no CJK character.

    Raises:
        No exception is expected for a validated string input.
    """

    return bool(ASCII_LETTER_PATTERN.search(note_term)) and not bool(
        CJK_CHARACTER_PATTERN.search(note_term)
    )


def parse_prediction(raw_response: str) -> ParsedPrediction:
    """Parse a strict response or recover the final JSON after a thinking preamble.

    Args:
        raw_response: Text decoded from one model generation candidate.

    Returns:
        ParsedPrediction: Parsing, preamble, schema, tier-size, and normalized-note state.

    Raises:
        No exception is propagated; malformed output is represented by false flags.
    """

    # 步骤1: 优先接受严格 JSON；thinking 模式则恢复响应中最后一个三键对象。
    parsed_value, is_strict_json, object_start, object_end = _recover_final_json_object(
        raw_response
    )
    if parsed_value is None:
        return _empty_parsed_prediction()

    # 步骤2: 独立检查键集合、字符串列表和每层 1 至 5 项限制。
    has_exact_keys = set(parsed_value) == set(REQUIRED_NOTE_KEYS)
    has_valid_schema = has_exact_keys and all(
        isinstance(parsed_value.get(tier_name), list)
        and bool(parsed_value[tier_name])
        and all(
            isinstance(note_term, str) and bool(note_term.strip())
            for note_term in parsed_value[tier_name]
        )
        for tier_name in REQUIRED_NOTE_KEYS
    )
    canonical_notes = {
        tier_name: [
            normalize_note_term(note_term)
            for note_term in parsed_value.get(tier_name, [])
            if isinstance(note_term, str) and note_term.strip()
        ]
        for tier_name in REQUIRED_NOTE_KEYS
    }
    has_valid_tier_sizes = has_valid_schema and all(
        1 <= len(canonical_notes[tier_name]) <= 5
        for tier_name in REQUIRED_NOTE_KEYS
    )
    return ParsedPrediction(
        parsed_json=parsed_value,
        canonical_notes=canonical_notes,
        is_strict_json=is_strict_json,
        is_recoverable_json=True,
        has_exact_keys=has_exact_keys,
        has_valid_schema=has_valid_schema,
        has_valid_tier_sizes=has_valid_tier_sizes,
        preamble_text=raw_response[:object_start].strip(),
        trailing_text=raw_response[object_end:].strip(),
    )


def _recover_final_json_object(
    raw_response: str,
) -> tuple[dict[str, Any] | None, bool, int, int]:
    """Recover the last required-key JSON object from a possibly reasoned response.

    Args:
        raw_response: Complete decoded generation, potentially containing thinking text.

    Returns:
        tuple[dict[str, Any] | None, bool, int, int]: Parsed object, strictness flag,
        and character offsets delimiting the selected object in the raw response.

    Raises:
        No exception is propagated; malformed candidates are skipped.
    """

    stripped_response = raw_response.strip()
    try:
        strict_value = json.loads(stripped_response)
    except (json.JSONDecodeError, TypeError):
        strict_value = None
    if isinstance(strict_value, dict):
        leading_characters = len(raw_response) - len(raw_response.lstrip())
        return (
            strict_value,
            True,
            leading_characters,
            leading_characters + len(stripped_response),
        )

    # 步骤1: 从每个左花括号尝试 raw_decode，只保留包含正式三键的对象。
    decoder = json.JSONDecoder()
    candidates: list[tuple[dict[str, Any], int, int]] = []
    for character_index, character_value in enumerate(raw_response):
        if character_value != "{":
            continue
        try:
            candidate_value, relative_end = decoder.raw_decode(
                raw_response[character_index:]
            )
        except json.JSONDecodeError:
            continue
        if isinstance(candidate_value, dict) and set(REQUIRED_NOTE_KEYS).issubset(
            candidate_value
        ):
            candidates.append(
                (candidate_value, character_index, character_index + relative_end)
            )
    if not candidates:
        return None, False, 0, 0
    selected_value, selected_start, selected_end = candidates[-1]
    return selected_value, False, selected_start, selected_end


def _empty_parsed_prediction() -> ParsedPrediction:
    """Create the canonical parse result used for malformed model responses.

    Returns:
        ParsedPrediction: False flags and empty lists for all required tiers.

    Raises:
        No exception is expected while constructing fixed empty values.
    """

    return ParsedPrediction(
        parsed_json=None,
        canonical_notes={tier_name: [] for tier_name in REQUIRED_NOTE_KEYS},
        is_strict_json=False,
        is_recoverable_json=False,
        has_exact_keys=False,
        has_valid_schema=False,
        has_valid_tier_sizes=False,
        preamble_text="",
        trailing_text="",
    )


# ========== Model generation ==========


def select_design_system_prompt(reasoning_mode: str) -> str:
    """Select the shared system instruction for the requested reasoning protocol.

    Args:
        reasoning_mode: Validated CLI mode supplied by the experiment operator.

    Returns:
        str: Direct/native prompt or the bounded visible-planning prompt.

    Raises:
        ValueError: Raised when a caller bypasses CLI validation with an unknown mode.
    """

    # 步骤1: 原生 thinking 只改变聊天模板，消息正文与直接回答保持一致。
    if reasoning_mode in {"direct", "native_thinking"}:
        return DESIGN_SYSTEM_PROMPT
    if reasoning_mode == "visible_plan":
        return VISIBLE_PLAN_SYSTEM_PROMPT
    raise ValueError(f"未知推理模式, 上下文: reasoning_mode={reasoning_mode}")


def build_prompt_token_ids(
    tokenizer: Any,
    request_rows: Sequence[dict[str, Any]],
    is_thinking_enabled: bool,
    system_prompt: str,
) -> list[list[int]]:
    """Render the shared application prompt and tokenize every user request.

    Args:
        tokenizer: Hugging Face tokenizer loaded from the selected model directory.
        request_rows: Validated request bank in fixed source order.
        is_thinking_enabled: CLI-selected Qwen chat-template thinking mode.
        system_prompt: Reasoning-mode-specific instruction shared by both models.

    Returns:
        list[list[int]]: Prompt-token sequences aligned one-to-one with requests.

    Raises:
        RuntimeError: Raised when chat-template rendering or tokenization fails.
    """

    prompt_token_ids: list[list[int]] = []
    try:
        for request_row in request_rows:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_row["user_request"]},
            ]
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=is_thinking_enabled,
            )
            prompt_token_ids.append(
                tokenizer.encode(rendered_prompt, add_special_tokens=False)
            )
    except Exception as error:
        raise RuntimeError(
            f"构造模型提示失败: {error}, 上下文: requests={len(request_rows)}"
        ) from error
    return prompt_token_ids


def collect_stop_token_ids(tokenizer: Any) -> list[int]:
    """Collect unique Qwen end-of-text and end-of-message token ids.

    Args:
        tokenizer: Tokenizer used for prompt encoding and output decoding.

    Returns:
        list[int]: Sorted nonnegative token ids accepted by vLLM SamplingParams.

    Raises:
        RuntimeError: Raised when tokenizer token conversion unexpectedly fails.
    """

    try:
        candidate_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]
    except Exception as error:
        raise RuntimeError(
            f"解析停止 token 失败: {error}, 上下文: tokenizer={type(tokenizer).__name__}"
        ) from error
    return sorted(
        {
            int(token_id)
            for token_id in candidate_ids
            if isinstance(token_id, int) and token_id >= 0
        }
    )


def generate_model_records(
    arguments: argparse.Namespace, request_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Load one selected model and generate auditable outputs for all requests.

    Args:
        arguments: Validated CLI namespace with model paths and decoding settings.
        request_rows: Ordered simulated user requests loaded from the pilot bank.

    Returns:
        list[dict[str, Any]]: Generation records aligned to the request source order.

    Raises:
        RuntimeError: Raised for dependency, tokenizer, model, generation, or decode failures.
        ValueError: Raised when prompt plus output budget exceeds model context length.
    """

    # 步骤1: 确保 FlashInfer 子进程能够找到当前环境的编译工具。
    environment_binary_directory = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    if environment_binary_directory not in current_path.split(os.pathsep):
        os.environ["PATH"] = environment_binary_directory + os.pathsep + current_path

    # 步骤2: 仅在生成阶段导入 GPU 依赖，分析阶段保持轻量。
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError(
            f"导入推理依赖失败: {error}, 上下文: python={sys.executable}"
        ) from error
    model_path = (
        arguments.base_model
        if arguments.model == "base"
        else arguments.finetuned_model
    )
    model_label = MODEL_LABELS[arguments.model]

    # 步骤3: 用每个模型自己的同源 tokenizer 渲染完全相同的消息内容。
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as error:
        raise RuntimeError(
            f"加载 tokenizer 失败: {error}, 上下文: model_path={model_path}"
        ) from error
    system_prompt = select_design_system_prompt(arguments.reasoning_mode)
    prompt_token_ids = build_prompt_token_ids(
        tokenizer,
        request_rows,
        arguments.enable_thinking,
        system_prompt,
    )
    maximum_prompt_length = max(len(token_ids) for token_ids in prompt_token_ids)
    if maximum_prompt_length + arguments.max_new_tokens > arguments.max_model_length:
        raise ValueError(
            "上下文长度不足, 上下文: "
            f"prompt_max={maximum_prompt_length}, output_max={arguments.max_new_tokens}, "
            f"model_max={arguments.max_model_length}"
        )

    # 步骤4: 固定贪心解码，初步试验只比较权重差异而不引入采样方差。
    try:
        inference_engine = LLM(
            model=str(model_path),
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=arguments.gpu_memory_utilization,
            max_model_len=arguments.max_model_length,
            seed=arguments.seed,
            disable_log_stats=True,
        )
        sampling_parameters = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=arguments.max_new_tokens,
            seed=arguments.seed,
            stop_token_ids=collect_stop_token_ids(tokenizer),
        )
        token_prompts = [
            {"prompt_token_ids": token_ids} for token_ids in prompt_token_ids
        ]
        generation_start = time.perf_counter()
        model_outputs = inference_engine.generate(
            token_prompts, sampling_parameters, use_tqdm=True
        )
        elapsed_seconds = time.perf_counter() - generation_start
    except Exception as error:
        raise RuntimeError(
            f"vLLM 生成失败: {error}, 上下文: model={model_label}, path={model_path}"
        ) from error

    # 步骤5: 保存原始文本、解析状态和 token 诊断，供后续盲化与审计复用。
    generation_records = _decode_generation_records(
        tokenizer,
        request_rows,
        model_outputs,
        arguments.model,
        model_label,
        model_path,
        maximum_prompt_length,
        arguments.enable_thinking,
        arguments.reasoning_mode,
        system_prompt,
    )
    print(
        f"Generated {len(generation_records)} requests for {model_label} in "
        f"{elapsed_seconds:.2f}s."
    )
    del inference_engine
    gc.collect()
    return generation_records


def _decode_generation_records(
    tokenizer: Any,
    request_rows: Sequence[dict[str, Any]],
    model_outputs: Sequence[Any],
    model_role: str,
    model_label: str,
    model_path: Path,
    maximum_prompt_length: int,
    is_thinking_enabled: bool,
    reasoning_mode: str,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Decode vLLM candidates and combine them with request metadata.

    Args:
        tokenizer: Tokenizer used by the selected inference model.
        request_rows: Ordered simulated user request rows.
        model_outputs: vLLM request outputs aligned with the request rows.
        model_role: Stable ``base`` or ``finetuned`` role.
        model_label: Human-readable model label stored for audit only.
        model_path: Exact server model directory used for inference.
        maximum_prompt_length: Largest input length in this generation phase.
        is_thinking_enabled: Chat-template thinking mode selected for this run.
        reasoning_mode: Auditable direct, native-thinking, or visible-plan mode.
        system_prompt: Exact system instruction rendered for both model roles.

    Returns:
        list[dict[str, Any]]: Fully auditable generation rows.

    Raises:
        ValueError: Raised when vLLM returns a different number of outputs.
    """

    if len(request_rows) != len(model_outputs):
        raise ValueError(
            "生成输出数量不一致, 上下文: "
            f"requests={len(request_rows)}, outputs={len(model_outputs)}"
        )
    records: list[dict[str, Any]] = []
    for request_row, request_output in zip(request_rows, model_outputs, strict=True):
        candidate_output = request_output.outputs[0] if request_output.outputs else None
        output_token_ids = list(candidate_output.token_ids) if candidate_output else []
        raw_response = tokenizer.decode(
            output_token_ids, skip_special_tokens=True
        ).strip()
        parsed_prediction = parse_prediction(raw_response)
        records.append(
            _create_generation_record(
                request_row,
                model_role,
                model_label,
                model_path,
                raw_response,
                parsed_prediction,
                output_token_ids,
                str(candidate_output.finish_reason)
                if candidate_output
                else "missing_output",
                maximum_prompt_length,
                is_thinking_enabled,
                reasoning_mode,
                system_prompt,
            )
        )
    return records


def _create_generation_record(
    request_row: dict[str, Any],
    model_role: str,
    model_label: str,
    model_path: Path,
    raw_response: str,
    parsed_prediction: ParsedPrediction,
    output_token_ids: Sequence[int],
    finish_reason: str,
    maximum_prompt_length: int,
    is_thinking_enabled: bool,
    reasoning_mode: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Build one serializable generation row with parsing and protocol metadata.

    Args:
        request_row: Source user request and objective-constraint metadata.
        model_role: Stable role used to pair generation files.
        model_label: Human-readable model identifier.
        model_path: Exact model directory used by vLLM.
        raw_response: Complete decoded model output.
        parsed_prediction: Strict parse result from ``parse_prediction``.
        output_token_ids: Generated token ids from vLLM.
        finish_reason: vLLM completion reason or ``missing_output``.
        maximum_prompt_length: Largest prompt length in the generation batch.
        is_thinking_enabled: Chat-template mode selected for this generation run.
        reasoning_mode: Auditable direct, native-thinking, or visible-plan mode.
        system_prompt: Exact system instruction used to render the prompt.

    Returns:
        dict[str, Any]: Auditable row written to ``generations/<role>.jsonl``.

    Raises:
        No exception is expected for validated source and generation values.
    """

    return {
        "request_id": request_row["request_id"],
        "language": request_row["language"],
        "category": request_row["category"],
        "user_request": request_row["user_request"],
        "required_by_tier": request_row["required_by_tier"],
        "forbidden_anywhere": request_row["forbidden_anywhere"],
        "exact_tier_counts": request_row["exact_tier_counts"],
        "model_role": model_role,
        "model_label": model_label,
        "model_path": str(model_path),
        "system_prompt_sha256": hash_text(system_prompt),
        "enable_thinking": is_thinking_enabled,
        "reasoning_mode": reasoning_mode,
        "raw_response": raw_response,
        "parsed_json": parsed_prediction.parsed_json,
        "canonical_notes": parsed_prediction.canonical_notes,
        "is_strict_json": parsed_prediction.is_strict_json,
        "is_recoverable_json": parsed_prediction.is_recoverable_json,
        "has_exact_keys": parsed_prediction.has_exact_keys,
        "has_valid_schema": parsed_prediction.has_valid_schema,
        "has_valid_tier_sizes": parsed_prediction.has_valid_tier_sizes,
        "preamble_text": parsed_prediction.preamble_text,
        "trailing_text": parsed_prediction.trailing_text,
        "has_preamble": bool(parsed_prediction.preamble_text),
        "preamble_characters": len(parsed_prediction.preamble_text),
        "finish_reason": finish_reason,
        "output_tokens": len(output_token_ids),
        "maximum_prompt_tokens_in_batch": maximum_prompt_length,
    }


def run_generation_phase(
    arguments: argparse.Namespace, request_rows: Sequence[dict[str, Any]]
) -> None:
    """Generate one model side and persist its records and reproducibility manifest.

    Args:
        arguments: Validated CLI namespace selecting Base or finetuned generation.
        request_rows: Validated pilot requests in fixed order.

    Returns:
        None: Generation JSONL and model-specific manifest are written to the run.

    Raises:
        RuntimeError: Propagated from generation or output file operations.
    """

    generation_records = generate_model_records(arguments, request_rows)
    output_path = arguments.run_dir / "generations" / f"{arguments.model}.jsonl"
    write_jsonl(output_path, generation_records)
    selected_model_path = (
        arguments.base_model
        if arguments.model == "base"
        else arguments.finetuned_model
    )
    system_prompt = select_design_system_prompt(arguments.reasoning_mode)
    manifest = {
        "phase": "generate",
        "model_role": arguments.model,
        "model_label": MODEL_LABELS[arguments.model],
        "model_path": str(selected_model_path),
        "requests_file": str(arguments.requests_file),
        "requests_sha256": calculate_file_sha256(arguments.requests_file),
        "request_count": len(request_rows),
        "system_prompt": system_prompt,
        "system_prompt_sha256": hash_text(system_prompt),
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": arguments.max_new_tokens,
            "max_model_length": arguments.max_model_length,
            "enable_thinking": arguments.enable_thinking,
            "reasoning_mode": arguments.reasoning_mode,
            "seed": arguments.seed,
        },
        "generation_sha256": calculate_file_sha256(output_path),
    }
    write_json(
        arguments.run_dir / "manifests" / f"generation_{arguments.model}.json",
        manifest,
    )


# ========== Constraint and diversity analysis ==========


def load_level3_training_vocabulary(training_file: Path) -> set[str]:
    """Load normalized Level-3 reference note terms from the formal train split.

    Args:
        training_file: Strict-chat training JSONL selected by CLI or project default.

    Returns:
        set[str]: Distinct normalized note terms used by Level-3 assistant labels.

    Raises:
        RuntimeError: Raised when the training JSONL cannot be read or parsed.
        ValueError: Raised when no Level-3 vocabulary can be recovered.
    """

    vocabulary: set[str] = set()
    for training_row in read_jsonl(training_file):
        metadata = training_row.get("metadata", {})
        if not isinstance(metadata, dict) or metadata.get("scent_information_level") != 3:
            continue
        assistant_content = _extract_assistant_content(training_row, training_file)
        try:
            assistant_object = json.loads(assistant_content)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"解析训练标签失败: {error}, 上下文: file={training_file}"
            ) from error
        for tier_name in REQUIRED_NOTE_KEYS:
            for note_term in assistant_object.get(tier_name, []):
                if isinstance(note_term, str) and note_term.strip():
                    vocabulary.add(normalize_note_term(note_term))
    if not vocabulary:
        raise ValueError(f"Level-3 训练词表为空, 上下文: file={training_file}")
    return vocabulary


def _extract_assistant_content(training_row: dict[str, Any], file_path: Path) -> str:
    """Extract the unique assistant message content from one strict-chat row.

    Args:
        training_row: Parsed training JSONL object containing a messages list.
        file_path: Source path included in malformed-row errors.

    Returns:
        str: Assistant JSON string used to recover reference note terms.

    Raises:
        ValueError: Raised when the row does not contain exactly one assistant message.
    """

    messages = training_row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"训练行缺少 messages, 上下文: file={file_path}")
    assistant_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(assistant_messages) != 1 or not isinstance(
        assistant_messages[0].get("content"), str
    ):
        raise ValueError(f"训练行 assistant 消息无效, 上下文: file={file_path}")
    return assistant_messages[0]["content"]


def pair_generation_records(
    request_rows: Sequence[dict[str, Any]],
    base_records: Sequence[dict[str, Any]],
    finetuned_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair generation rows by request id and verify immutable source fields.

    Args:
        request_rows: Canonical request bank loaded from the source JSONL.
        base_records: Base generation rows loaded from the run directory.
        finetuned_records: Finetuned generation rows from the same run.

    Returns:
        list[dict[str, Any]]: Canonical request plus Base/Fine records in source order.

    Raises:
        ValueError: Raised for duplicate, missing, extra, or source-mismatched records.
    """

    request_by_id = _index_unique_rows(request_rows, "request_id", "requests")
    base_by_id = _index_unique_rows(base_records, "request_id", "base generations")
    finetuned_by_id = _index_unique_rows(
        finetuned_records, "request_id", "finetuned generations"
    )
    expected_ids = set(request_by_id)
    if set(base_by_id) != expected_ids or set(finetuned_by_id) != expected_ids:
        raise ValueError(
            "生成记录 request_id 集合不一致, 上下文: "
            f"requests={len(expected_ids)}, base={len(base_by_id)}, "
            f"finetuned={len(finetuned_by_id)}"
        )

    # 步骤1: 逐条核验两侧真实请求文本和约束元数据没有漂移。
    paired_rows: list[dict[str, Any]] = []
    immutable_fields = (
        "language",
        "category",
        "user_request",
        "required_by_tier",
        "forbidden_anywhere",
        "exact_tier_counts",
    )
    for request_row in request_rows:
        request_id = request_row["request_id"]
        base_record = base_by_id[request_id]
        finetuned_record = finetuned_by_id[request_id]
        for field_name in immutable_fields:
            if base_record.get(field_name) != request_row.get(field_name) or finetuned_record.get(
                field_name
            ) != request_row.get(field_name):
                raise ValueError(
                    f"配对源字段不一致: {field_name}, 上下文: request_id={request_id}"
                )
        paired_rows.append(
            {"request": request_row, "base": base_record, "finetuned": finetuned_record}
        )
    return paired_rows


def _index_unique_rows(
    rows: Sequence[dict[str, Any]], key_name: str, source_name: str
) -> dict[str, dict[str, Any]]:
    """Index rows by a required unique string field.

    Args:
        rows: Parsed source rows to index.
        key_name: Field carrying the stable unique key.
        source_name: Human-readable source label included in errors.

    Returns:
        dict[str, dict[str, Any]]: Mapping from stable key to source row.

    Raises:
        ValueError: Raised for missing, nonstring, or duplicate keys.
    """

    indexed_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        key_value = row.get(key_name)
        if not isinstance(key_value, str) or not key_value:
            raise ValueError(f"唯一键无效, 上下文: source={source_name}, row={row}")
        if key_value in indexed_rows:
            raise ValueError(
                f"唯一键重复, 上下文: source={source_name}, key={key_value}"
            )
        indexed_rows[key_value] = row
    return indexed_rows


def calculate_record_metrics(
    request_row: dict[str, Any],
    generation_record: dict[str, Any],
    training_vocabulary: set[str],
) -> dict[str, Any]:
    """Calculate objective diagnostics for one request/model generation.

    Args:
        request_row: Canonical request and explicit constraint metadata.
        generation_record: Parsed generation row from Base or finetuned model.
        training_vocabulary: Level-3 reference-term set loaded from training data.

    Returns:
        dict[str, Any]: Schema, constraint, vocabulary, duplication, and bias metrics.

    Raises:
        ValueError: Raised when stored canonical note tiers have invalid types.
    """

    canonical_notes = generation_record.get("canonical_notes")
    if not isinstance(canonical_notes, dict):
        raise ValueError(
            f"生成记录缺少 canonical_notes, 上下文: request_id={request_row['request_id']}"
        )
    all_notes = [
        note_term
        for tier_name in REQUIRED_NOTE_KEYS
        for note_term in canonical_notes.get(tier_name, [])
    ]
    required_result = _evaluate_required_notes(request_row, canonical_notes)
    forbidden_result = _evaluate_forbidden_notes(request_row, all_notes)
    count_result = _evaluate_exact_tier_counts(request_row, canonical_notes)
    english_note_count = sum(is_english_note_term(note_term) for note_term in all_notes)
    has_all_english_terms = bool(all_notes) and english_note_count == len(all_notes)
    has_objective_constraints = bool(
        request_row["required_by_tier"]
        or request_row["forbidden_anywhere"]
        or request_row["exact_tier_counts"]
    )
    objective_pass = (
        generation_record.get("has_valid_tier_sizes") is True
        and has_all_english_terms
        and required_result["required_notes_met"] == required_result["required_notes_total"]
        and forbidden_result["forbidden_violations"] == 0
        and count_result["tier_count_rules_met"] == count_result["tier_count_rules_total"]
    )
    return {
        "request_id": request_row["request_id"],
        "language": request_row["language"],
        "category": request_row["category"],
        "is_strict_json": generation_record.get("is_strict_json") is True,
        "is_recoverable_json": generation_record.get("is_recoverable_json", generation_record.get("is_strict_json")) is True,
        "has_valid_schema": generation_record.get("has_valid_schema") is True,
        "has_valid_tier_sizes": generation_record.get("has_valid_tier_sizes") is True,
        "has_preamble": generation_record.get("has_preamble") is True,
        "preamble_characters": int(generation_record.get("preamble_characters", 0)),
        "has_trailing_text": bool(generation_record.get("trailing_text", "")),
        "finish_reason": generation_record.get("finish_reason"),
        "output_tokens": int(generation_record.get("output_tokens", 0)),
        "total_note_count": len(all_notes),
        "unique_note_count": len(set(all_notes)),
        "has_cross_tier_duplicates": len(set(all_notes)) != len(all_notes),
        "english_note_terms": english_note_count,
        "english_note_terms_total": len(all_notes),
        "has_all_english_terms": has_all_english_terms,
        "training_vocabulary_hits": sum(
            note_term in training_vocabulary for note_term in all_notes
        ),
        "training_vocabulary_total": len(all_notes),
        "high_frequency_terms_present": [
            note_term for note_term in HIGH_FREQUENCY_TERMS if note_term in all_notes
        ],
        "has_objective_constraints": has_objective_constraints,
        "passes_all_objective_constraints": has_objective_constraints and objective_pass,
        **required_result,
        **forbidden_result,
        **count_result,
    }


def _evaluate_required_notes(
    request_row: dict[str, Any], canonical_notes: dict[str, list[str]]
) -> dict[str, Any]:
    """Check tier-specific required notes using normalized exact terms.

    Args:
        request_row: Request metadata containing ``required_by_tier``.
        canonical_notes: Model notes normalized by output tier.

    Returns:
        dict[str, Any]: Required counts and detailed missing tier-note pairs.

    Raises:
        No exception is expected for validated request and generation structures.
    """

    missing_notes: list[dict[str, str]] = []
    required_total = 0
    for tier_name, required_notes in request_row["required_by_tier"].items():
        generated_tier = canonical_notes.get(tier_name, [])
        for required_note in required_notes:
            required_total += 1
            normalized_required = normalize_note_term(required_note)
            if not any(
                note_matches_constraint(generated_note, normalized_required)
                for generated_note in generated_tier
            ):
                missing_notes.append({"tier": tier_name, "note": normalized_required})
    return {
        "required_notes_total": required_total,
        "required_notes_met": required_total - len(missing_notes),
        "missing_required_notes": missing_notes,
    }


def _evaluate_forbidden_notes(
    request_row: dict[str, Any], all_notes: Sequence[str]
) -> dict[str, Any]:
    """Check whether explicitly forbidden exact note terms appear anywhere.

    Args:
        request_row: Request metadata containing ``forbidden_anywhere``.
        all_notes: Flattened normalized model output terms from all three tiers.

    Returns:
        dict[str, Any]: Forbidden-term count and sorted violations.

    Raises:
        No exception is expected for validated request metadata.
    """

    forbidden_terms = [
        normalize_note_term(note_term) for note_term in request_row["forbidden_anywhere"]
    ]
    match_details = [
        {"constraint": forbidden_term, "generated": generated_note}
        for forbidden_term in forbidden_terms
        for generated_note in all_notes
        if note_matches_constraint(generated_note, forbidden_term)
    ]
    violations = sorted({match_detail["constraint"] for match_detail in match_details})
    return {
        "forbidden_terms_total": len(forbidden_terms),
        "forbidden_violations": len(violations),
        "violated_forbidden_notes": violations,
        "forbidden_match_details": match_details,
    }


def _evaluate_exact_tier_counts(
    request_row: dict[str, Any], canonical_notes: dict[str, list[str]]
) -> dict[str, Any]:
    """Check explicit per-tier output-count rules.

    Args:
        request_row: Request metadata containing ``exact_tier_counts``.
        canonical_notes: Model notes normalized by output tier.

    Returns:
        dict[str, Any]: Rule totals, met totals, and detailed mismatches.

    Raises:
        No exception is expected for validated request metadata.
    """

    mismatches: list[dict[str, int | str]] = []
    count_rules = request_row["exact_tier_counts"]
    for tier_name, expected_count in count_rules.items():
        actual_count = len(canonical_notes.get(tier_name, []))
        if actual_count != expected_count:
            mismatches.append(
                {"tier": tier_name, "expected": expected_count, "actual": actual_count}
            )
    return {
        "tier_count_rules_total": len(count_rules),
        "tier_count_rules_met": len(count_rules) - len(mismatches),
        "tier_count_mismatches": mismatches,
    }


def aggregate_model_metrics(
    model_role: str,
    generation_records: Sequence[dict[str, Any]],
    record_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate schema, constraints, terminology, diversity, and bias for one model.

    Args:
        model_role: Stable model role used in report tables.
        generation_records: Generation rows aligned with ``record_metrics``.
        record_metrics: Per-request objective diagnostics for the selected model.

    Returns:
        dict[str, Any]: Overall rates, counts, vocabulary diversity, and language splits.

    Raises:
        ValueError: Raised for empty or differently sized input sequences.
    """

    if not generation_records or len(generation_records) != len(record_metrics):
        raise ValueError(
            "聚合输入为空或长度不一致, 上下文: "
            f"role={model_role}, generations={len(generation_records)}, "
            f"metrics={len(record_metrics)}"
        )
    return {
        "model_role": model_role,
        "overall": _aggregate_metric_subset(generation_records, record_metrics),
        "by_language": {
            language_name: _aggregate_metric_subset(
                [
                    record
                    for record in generation_records
                    if record["language"] == language_name
                ],
                [metric for metric in record_metrics if metric["language"] == language_name],
            )
            for language_name in ("zh", "en")
        },
    }


def _aggregate_metric_subset(
    generation_records: Sequence[dict[str, Any]],
    record_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one model's diagnostics over an overall or language subset.

    Args:
        generation_records: Generation rows for the selected subset.
        record_metrics: Per-request metrics for exactly the same subset.

    Returns:
        dict[str, Any]: Counts and rates used by JSON and Markdown reports.

    Raises:
        ValueError: Raised when subset inputs are empty or differently sized.
    """

    if not record_metrics or len(generation_records) != len(record_metrics):
        raise ValueError(
            "指标子集为空或长度不一致, 上下文: "
            f"generations={len(generation_records)}, metrics={len(record_metrics)}"
        )
    sample_count = len(record_metrics)
    constrained_metrics = [
        metric for metric in record_metrics if metric["has_objective_constraints"]
    ]
    all_notes = [
        note_term
        for record in generation_records
        for tier_name in REQUIRED_NOTE_KEYS
        for note_term in record["canonical_notes"].get(tier_name, [])
    ]
    required_total = sum(metric["required_notes_total"] for metric in record_metrics)
    forbidden_total = sum(metric["forbidden_terms_total"] for metric in record_metrics)
    count_rule_total = sum(metric["tier_count_rules_total"] for metric in record_metrics)
    unique_pyramid_count = len(
        {
            _pyramid_signature(record["canonical_notes"])
            for record in generation_records
        }
    )
    return {
        "sample_count": sample_count,
        "strict_json_count": sum(metric["is_strict_json"] for metric in record_metrics),
        "strict_json_rate": _safe_ratio(
            sum(metric["is_strict_json"] for metric in record_metrics), sample_count
        ),
        "recoverable_json_count": sum(
            metric["is_recoverable_json"] for metric in record_metrics
        ),
        "recoverable_json_rate": _safe_ratio(
            sum(metric["is_recoverable_json"] for metric in record_metrics), sample_count
        ),
        "valid_schema_count": sum(
            metric["has_valid_schema"] for metric in record_metrics
        ),
        "valid_schema_rate": _safe_ratio(
            sum(metric["has_valid_schema"] for metric in record_metrics), sample_count
        ),
        "valid_tier_sizes_count": sum(
            metric["has_valid_tier_sizes"] for metric in record_metrics
        ),
        "valid_tier_sizes_rate": _safe_ratio(
            sum(metric["has_valid_tier_sizes"] for metric in record_metrics), sample_count
        ),
        "all_english_terms_count": sum(
            metric["has_all_english_terms"] for metric in record_metrics
        ),
        "all_english_terms_rate": _safe_ratio(
            sum(metric["has_all_english_terms"] for metric in record_metrics), sample_count
        ),
        "english_note_term_rate": _safe_ratio(
            sum(metric["english_note_terms"] for metric in record_metrics),
            sum(metric["english_note_terms_total"] for metric in record_metrics),
        ),
        "objective_constrained_request_count": len(constrained_metrics),
        "all_objective_constraints_pass_count": sum(
            metric["passes_all_objective_constraints"] for metric in constrained_metrics
        ),
        "all_objective_constraints_pass_rate": _safe_ratio(
            sum(
                metric["passes_all_objective_constraints"]
                for metric in constrained_metrics
            ),
            len(constrained_metrics),
        ),
        "required_note_recall": _safe_ratio(
            sum(metric["required_notes_met"] for metric in record_metrics),
            required_total,
        ),
        "forbidden_term_compliance": _safe_ratio(
            forbidden_total
            - sum(metric["forbidden_violations"] for metric in record_metrics),
            forbidden_total,
        ),
        "tier_count_rule_compliance": _safe_ratio(
            sum(metric["tier_count_rules_met"] for metric in record_metrics),
            count_rule_total,
        ),
        "mean_notes_per_pyramid": sum(
            metric["total_note_count"] for metric in record_metrics
        )
        / sample_count,
        "cross_tier_duplicate_request_count": sum(
            metric["has_cross_tier_duplicates"] for metric in record_metrics
        ),
        "training_vocabulary_alignment": _safe_ratio(
            sum(metric["training_vocabulary_hits"] for metric in record_metrics),
            sum(metric["training_vocabulary_total"] for metric in record_metrics),
        ),
        "distinct_output_terms": len(set(all_notes)),
        "unique_pyramid_count": unique_pyramid_count,
        "unique_pyramid_rate": unique_pyramid_count / sample_count,
        "mean_between_request_jaccard": _mean_pairwise_jaccard(
            [
                {
                    note_term
                    for tier_name in REQUIRED_NOTE_KEYS
                    for note_term in record["canonical_notes"].get(tier_name, [])
                }
                for record in generation_records
            ]
        ),
        "high_frequency_term_request_coverage": {
            note_term: sum(
                note_term in metric["high_frequency_terms_present"]
                for metric in record_metrics
            )
            / sample_count
            for note_term in HIGH_FREQUENCY_TERMS
        },
        "high_frequency_term_family_request_coverage": {
            note_term: sum(
                any(note_matches_constraint(output_note, note_term) for output_note in all_notes)
                for all_notes in (
                    [
                        output_note
                        for tier_name in REQUIRED_NOTE_KEYS
                        for output_note in record["canonical_notes"].get(tier_name, [])
                    ]
                    for record in generation_records
                )
            )
            / sample_count
            for note_term in HIGH_FREQUENCY_TERMS
        },
        "most_common_terms": Counter(all_notes).most_common(12),
        "mean_output_tokens": sum(
            metric["output_tokens"] for metric in record_metrics
        )
        / sample_count,
        "preamble_count": sum(metric["has_preamble"] for metric in record_metrics),
        "mean_preamble_characters": sum(
            metric["preamble_characters"] for metric in record_metrics
        )
        / sample_count,
        "trailing_text_count": sum(
            metric["has_trailing_text"] for metric in record_metrics
        ),
        "length_finish_count": sum(
            str(metric["finish_reason"]).lower() in {"length", "max_tokens"}
            for metric in record_metrics
        ),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """Divide counts while representing an unavailable denominator as ``None``.

    Args:
        numerator: Successful or compliant item count.
        denominator: Total applicable item count.

    Returns:
        float | None: Ratio when denominator is positive, otherwise ``None``.

    Raises:
        ValueError: Raised when either count is negative.
    """

    if numerator < 0 or denominator < 0:
        raise ValueError(
            f"比例计数不能为负, 上下文: numerator={numerator}, denominator={denominator}"
        )
    return numerator / denominator if denominator else None


def _pyramid_signature(canonical_notes: dict[str, list[str]]) -> str:
    """Serialize normalized tiers into a stable exact-pyramid signature.

    Args:
        canonical_notes: Three normalized note lists from a generation record.

    Returns:
        str: Compact JSON preserving tier assignment and note order.

    Raises:
        No exception is expected for normalized dictionary values.
    """

    return json.dumps(
        {tier_name: canonical_notes.get(tier_name, []) for tier_name in REQUIRED_NOTE_KEYS},
        ensure_ascii=False,
        sort_keys=True,
    )


def calculate_jaccard(first_terms: set[str], second_terms: set[str]) -> float:
    """Calculate Jaccard similarity between two flattened fragrance-note sets.

    Args:
        first_terms: Unique normalized terms from one pyramid.
        second_terms: Unique normalized terms from another pyramid.

    Returns:
        float: Intersection divided by union, or 1.0 when both sets are empty.

    Raises:
        No exception is expected for set inputs.
    """

    union_terms = first_terms | second_terms
    if not union_terms:
        return 1.0
    return len(first_terms & second_terms) / len(union_terms)


def _mean_pairwise_jaccard(note_sets: Sequence[set[str]]) -> float:
    """Calculate mean Jaccard similarity across every pair of different requests.

    Args:
        note_sets: One flattened normalized note set per user request.

    Returns:
        float: Mean cross-request similarity, or 0.0 for fewer than two requests.

    Raises:
        No exception is expected for a finite sequence of sets.
    """

    similarities = [
        calculate_jaccard(first_set, second_set)
        for first_set, second_set in combinations(note_sets, 2)
    ]
    return sum(similarities) / len(similarities) if similarities else 0.0


def build_paired_analysis_rows(
    paired_rows: Sequence[dict[str, Any]],
    metrics_by_role: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build detailed Base/Fine comparison rows for each simulated request.

    Args:
        paired_rows: Canonical request rows paired with both generation records.
        metrics_by_role: Per-request metric mappings for Base and finetuned roles.

    Returns:
        list[dict[str, Any]]: Detailed rows containing both outputs and overlap score.

    Raises:
        KeyError: Raised when a paired request lacks per-role metric records.
    """

    analysis_rows: list[dict[str, Any]] = []
    for paired_row in paired_rows:
        request_row = paired_row["request"]
        request_id = request_row["request_id"]
        base_record = paired_row["base"]
        finetuned_record = paired_row["finetuned"]
        base_terms = _flatten_note_set(base_record)
        finetuned_terms = _flatten_note_set(finetuned_record)
        analysis_rows.append(
            {
                **request_row,
                "base": {
                    "raw_response": base_record["raw_response"],
                    "preamble_text": base_record.get("preamble_text", ""),
                    "parsed_json": base_record["parsed_json"],
                    "metrics": metrics_by_role["base"][request_id],
                },
                "finetuned": {
                    "raw_response": finetuned_record["raw_response"],
                    "preamble_text": finetuned_record.get("preamble_text", ""),
                    "parsed_json": finetuned_record["parsed_json"],
                    "metrics": metrics_by_role["finetuned"][request_id],
                },
                "between_model_note_jaccard": calculate_jaccard(
                    base_terms, finetuned_terms
                ),
                "outputs_exactly_equal": base_record["canonical_notes"]
                == finetuned_record["canonical_notes"],
            }
        )
    return analysis_rows


def _flatten_note_set(generation_record: dict[str, Any]) -> set[str]:
    """Flatten one generation record's normalized tiers into a unique term set.

    Args:
        generation_record: Base or finetuned record with ``canonical_notes``.

    Returns:
        set[str]: Unique normalized terms across top, middle, and base.

    Raises:
        No exception is expected after generation-record validation.
    """

    return {
        note_term
        for tier_name in REQUIRED_NOTE_KEYS
        for note_term in generation_record["canonical_notes"].get(tier_name, [])
    }


def build_mode_comparison(
    comparison_run_directory: Path,
    request_rows: Sequence[dict[str, Any]],
    current_paired_rows: Sequence[dict[str, Any]],
    training_vocabulary: set[str],
) -> dict[str, Any]:
    """Compare thinking outputs with a prior run on exact shared request anchors.

    Args:
        comparison_run_directory: Prior no-thinking run selected through CLI.
        request_rows: Current request bank containing anchors and new requests.
        current_paired_rows: Current Base/Fine generation pairs.
        training_vocabulary: Level-3 normalized note set used by both analyses.

    Returns:
        dict[str, Any]: Shared ids plus per-model old/new aggregates and per-anchor rows.

    Raises:
        ValueError: Raised for absent anchors or changed anchor request text.
        RuntimeError: Raised when prior generation JSONL files cannot be read.
    """

    # 步骤1: 读取旧运行并找到 Base/Fine 和当前请求共同拥有的稳定 id。
    previous_records_by_role = {
        role_name: _index_unique_rows(
            read_jsonl(
                comparison_run_directory / "generations" / f"{role_name}.jsonl"
            ),
            "request_id",
            f"comparison {role_name}",
        )
        for role_name in MODEL_LABELS
    }
    current_rows_by_id = {
        paired_row["request"]["request_id"]: paired_row
        for paired_row in current_paired_rows
    }
    shared_ids = [
        request_row["request_id"]
        for request_row in request_rows
        if request_row["request_id"] in previous_records_by_role["base"]
        and request_row["request_id"] in previous_records_by_role["finetuned"]
    ]
    if not shared_ids:
        raise ValueError(
            f"thinking 对照没有共享请求, 上下文: run={comparison_run_directory}"
        )

    # 步骤2: 锚点文本必须逐字一致，确保唯一实验差异是 thinking/输出预算协议。
    request_by_id = {request_row["request_id"]: request_row for request_row in request_rows}
    for request_id in shared_ids:
        expected_text = request_by_id[request_id]["user_request"]
        for role_name in MODEL_LABELS:
            if previous_records_by_role[role_name][request_id].get(
                "user_request"
            ) != expected_text:
                raise ValueError(
                    "thinking 锚点文本发生变化, 上下文: "
                    f"request_id={request_id}, role={role_name}"
                )

    # 步骤3: 对每个模型分别计算模式前后聚合变化与逐锚点答案相似度。
    model_comparisons = {
        role_name: _compare_one_model_mode(
            role_name,
            shared_ids,
            request_by_id,
            previous_records_by_role[role_name],
            current_rows_by_id,
            training_vocabulary,
        )
        for role_name in MODEL_LABELS
    }
    return {
        "comparison_run_directory": str(comparison_run_directory),
        "anchor_count": len(shared_ids),
        "anchor_request_ids": shared_ids,
        "models": model_comparisons,
    }


def _compare_one_model_mode(
    role_name: str,
    shared_ids: Sequence[str],
    request_by_id: dict[str, dict[str, Any]],
    previous_records_by_id: dict[str, dict[str, Any]],
    current_rows_by_id: dict[str, dict[str, Any]],
    training_vocabulary: set[str],
) -> dict[str, Any]:
    """Compare no-thinking and thinking records for one model over shared anchors.

    Args:
        role_name: Stable Base or finetuned role.
        shared_ids: Ordered ids common to old and new runs.
        request_by_id: Current canonical request metadata keyed by id.
        previous_records_by_id: Prior no-thinking generation rows keyed by id.
        current_rows_by_id: Current paired rows keyed by request id.
        training_vocabulary: Shared Level-3 normalized reference-term vocabulary.

    Returns:
        dict[str, Any]: Old/new aggregates, stability statistics, and anchor details.

    Raises:
        KeyError: Raised when a shared id is unexpectedly absent from current pairs.
    """

    previous_records = [previous_records_by_id[request_id] for request_id in shared_ids]
    current_records = [
        current_rows_by_id[request_id][role_name] for request_id in shared_ids
    ]
    anchor_requests = [request_by_id[request_id] for request_id in shared_ids]
    previous_metrics = [
        calculate_record_metrics(request_row, record, training_vocabulary)
        for request_row, record in zip(anchor_requests, previous_records, strict=True)
    ]
    current_metrics = [
        calculate_record_metrics(request_row, record, training_vocabulary)
        for request_row, record in zip(anchor_requests, current_records, strict=True)
    ]
    anchor_rows = [
        _build_mode_anchor_row(request_row, previous_record, current_record, old_metric, new_metric)
        for request_row, previous_record, current_record, old_metric, new_metric in zip(
            anchor_requests,
            previous_records,
            current_records,
            previous_metrics,
            current_metrics,
            strict=True,
        )
    ]
    return {
        "no_thinking": _aggregate_metric_subset(previous_records, previous_metrics),
        "thinking": _aggregate_metric_subset(current_records, current_metrics),
        "exactly_equal_pyramids": sum(row["outputs_exactly_equal"] for row in anchor_rows),
        "mean_same_request_jaccard": sum(
            row["same_request_note_jaccard"] for row in anchor_rows
        )
        / len(anchor_rows),
        "anchors": anchor_rows,
    }


def _build_mode_anchor_row(
    request_row: dict[str, Any],
    previous_record: dict[str, Any],
    current_record: dict[str, Any],
    previous_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build one same-model no-thinking/thinking comparison row.

    Args:
        request_row: Canonical anchor request metadata.
        previous_record: Prior no-thinking model generation.
        current_record: Current thinking-enabled model generation.
        previous_metrics: Objective diagnostics for the prior response.
        current_metrics: Objective diagnostics for the thinking response.

    Returns:
        dict[str, Any]: Outputs, metrics, exact equality, and note-set similarity.

    Raises:
        No exception is expected for previously paired records.
    """

    return {
        "request_id": request_row["request_id"],
        "language": request_row["language"],
        "user_request": request_row["user_request"],
        "no_thinking": {
            "parsed_json": previous_record.get("parsed_json"),
            "metrics": previous_metrics,
        },
        "thinking": {
            "preamble_text": current_record.get("preamble_text", ""),
            "parsed_json": current_record.get("parsed_json"),
            "metrics": current_metrics,
        },
        "outputs_exactly_equal": previous_record.get("canonical_notes")
        == current_record.get("canonical_notes"),
        "same_request_note_jaccard": calculate_jaccard(
            _flatten_note_set(previous_record), _flatten_note_set(current_record)
        ),
    }


# ========== Automatic report rendering ==========








# ========== Report rendering and orchestration ==========


def render_automatic_report(analysis: dict[str, Any]) -> str:
    """Render structured pilot diagnostics as a concise Markdown report.

    Args:
        analysis: Complete analysis object assembled by ``run_analysis_phase``.

    Returns:
        str: Markdown report containing model diagnostics and request-level constraints.

    Raises:
        KeyError: Raised when required aggregate or paired fields are absent.
    """

    lines = [
        "# User Request Note-Design Pilot",
        "",
        "This is an application-transfer pilot from review-based note prediction to user-brief-based note architecture. Subjective quality has no synthetic gold answer; the automatic checks cover structure, explicit constraints, terminology, diversity, and known high-frequency terms.",
        "",
        "## Automatic diagnostics",
        "",
        "| Model | Valid JSON/schema | All-English pyramids | Full explicit constraints | Required-note recall | Forbidden-term compliance | Training-vocab alignment | Distinct terms | Mean notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role_name in ("base", "finetuned"):
        metrics = analysis["models"][role_name]["overall"]
        lines.append(
            f"| {role_name} | {_format_rate(metrics['valid_schema_rate'])} | "
            f"{_format_rate(metrics['all_english_terms_rate'])} | "
            f"{_format_rate(metrics['all_objective_constraints_pass_rate'])} | "
            f"{_format_rate(metrics['required_note_recall'])} | "
            f"{_format_rate(metrics['forbidden_term_compliance'])} | "
            f"{_format_rate(metrics['training_vocabulary_alignment'])} | "
            f"{metrics['distinct_output_terms']} | {metrics['mean_notes_per_pyramid']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Reasoning and output behavior",
            "",
            "| Model | Strict whole-response JSON | Recoverable final JSON | Preamble present | Mean preamble chars | Mean output tokens | Length stops |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for role_name in ("base", "finetuned"):
        metrics = analysis["models"][role_name]["overall"]
        lines.append(
            f"| {role_name} | {_format_rate(metrics['strict_json_rate'])} | "
            f"{_format_rate(metrics['recoverable_json_rate'])} | "
            f"{metrics['preamble_count']}/{metrics['sample_count']} | "
            f"{metrics['mean_preamble_characters']:.1f} | "
            f"{metrics['mean_output_tokens']:.1f} | {metrics['length_finish_count']} |"
        )
    lines.extend(
        [
            "",
            "## Diversity and prior-sensitive terms",
            "",
            "| Model | Unique pyramids | Mean cross-request Jaccard | Bergamot family | Musk family | Amber family |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for role_name in ("base", "finetuned"):
        metrics = analysis["models"][role_name]["overall"]
        coverage = metrics["high_frequency_term_family_request_coverage"]
        lines.append(
            f"| {role_name} | {metrics['unique_pyramid_count']}/{metrics['sample_count']} | "
            f"{metrics['mean_between_request_jaccard']:.4f} | "
            f"{_format_rate(coverage['bergamot'])} | {_format_rate(coverage['musk'])} | "
            f"{_format_rate(coverage['amber'])} |"
        )
    lines.extend(
        [
            "",
            "## Explicit-constraint cases",
            "",
            "| Request | Base pass | Fine pass | Base missing / violations | Fine missing / violations |",
            "|---|---:|---:|---|---|",
        ]
    )
    for paired_row in analysis["paired_rows"]:
        base_metrics = paired_row["base"]["metrics"]
        fine_metrics = paired_row["finetuned"]["metrics"]
        if not base_metrics["has_objective_constraints"]:
            continue
        lines.append(
            f"| {paired_row['request_id']} | "
            f"{_format_bool(base_metrics['passes_all_objective_constraints'])} | "
            f"{_format_bool(fine_metrics['passes_all_objective_constraints'])} | "
            f"{_format_constraint_issues(base_metrics)} | "
            f"{_format_constraint_issues(fine_metrics)} |"
        )
    if "mode_comparison" in analysis:
        lines.extend(_render_mode_comparison(analysis["mode_comparison"]))
    lines.extend(
        [
            "",
            "## Pair summary",
            "",
            f"- Mean Base/Fine note-set Jaccard: {analysis['pair_summary']['mean_between_model_note_jaccard']:.4f}",
            f"- Exactly identical pyramids: {analysis['pair_summary']['exactly_equal_pyramids']}/{analysis['request_count']}",
            "- Researcher-led exploratory analysis only; no questionnaire or human-preference result is included.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_mode_comparison(mode_comparison: dict[str, Any]) -> list[str]:
    """Render direct-answer versus reasoned-answer anchor diagnostics as Markdown.

    Args:
        mode_comparison: Shared-anchor analysis returned by ``build_mode_comparison``.

    Returns:
        list[str]: Markdown heading, protocol note, and one aggregate row per model.

    Raises:
        KeyError: Raised when required model comparison fields are absent.
    """

    # 步骤1: 使用协议清单中的真实模式名，避免把受控前置分析误称为原生 thinking。
    current_mode = mode_comparison.get("current_reasoning_mode", "reasoned")
    display_mode = current_mode.replace("_", " ")
    lines = [
        "",
        f"## Direct versus {display_mode} anchors",
        "",
        f"The comparison uses {mode_comparison['anchor_count']} exact shared requests. The reasoned run also has a different instruction and a larger output budget, so this is a protocol comparison rather than a pure single-factor causal estimate.",
        "",
        f"| Model | Exact same | Same-request Jaccard | Constraint pass direct -> {display_mode} | Forbidden compliance direct -> {display_mode} | Vocab alignment direct -> {display_mode} | Cross-request Jaccard direct -> {display_mode} | Preambles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role_name in ("base", "finetuned"):
        model_result = mode_comparison["models"][role_name]
        old_metrics = model_result["no_thinking"]
        thinking_metrics = model_result["thinking"]
        lines.append(
            f"| {role_name} | {model_result['exactly_equal_pyramids']}/{mode_comparison['anchor_count']} | "
            f"{model_result['mean_same_request_jaccard']:.4f} | "
            f"{_format_rate(old_metrics['all_objective_constraints_pass_rate'])} -> "
            f"{_format_rate(thinking_metrics['all_objective_constraints_pass_rate'])} | "
            f"{_format_rate(old_metrics['forbidden_term_compliance'])} -> "
            f"{_format_rate(thinking_metrics['forbidden_term_compliance'])} | "
            f"{_format_rate(old_metrics['training_vocabulary_alignment'])} -> "
            f"{_format_rate(thinking_metrics['training_vocabulary_alignment'])} | "
            f"{old_metrics['mean_between_request_jaccard']:.4f} -> "
            f"{thinking_metrics['mean_between_request_jaccard']:.4f} | "
            f"{thinking_metrics['preamble_count']}/{mode_comparison['anchor_count']} |"
        )
    return lines


def _format_rate(rate_value: float | None) -> str:
    """Format an optional proportion as percentage text.

    Args:
        rate_value: Ratio from aggregate metrics, or ``None`` when not applicable.

    Returns:
        str: Percentage with one decimal place or ``n/a``.

    Raises:
        No exception is expected for optional finite numeric input.
    """

    return "n/a" if rate_value is None else f"{100.0 * rate_value:.1f}%"


def _format_bool(flag_value: bool) -> str:
    """Format a boolean pass flag for a compact Markdown table.

    Args:
        flag_value: Objective constraint pass state from a per-record metric.

    Returns:
        str: ``yes`` for true and ``no`` for false.

    Raises:
        No exception is expected for boolean-compatible input.
    """

    return "yes" if flag_value else "no"


def _format_constraint_issues(record_metrics: dict[str, Any]) -> str:
    """Summarize missing, forbidden, and count-rule issues for one Markdown cell.

    Args:
        record_metrics: Per-request objective metrics from ``calculate_record_metrics``.

    Returns:
        str: Semicolon-separated issue summary or ``none``.

    Raises:
        KeyError: Raised when required metric fields are absent.
    """

    issue_parts: list[str] = []
    if not record_metrics["has_all_english_terms"]:
        issue_parts.append("non-English note terms")
    if record_metrics["missing_required_notes"]:
        missing_text = ", ".join(
            f"{item['tier']}:{item['note']}"
            for item in record_metrics["missing_required_notes"]
        )
        issue_parts.append(f"missing {missing_text}")
    if record_metrics["violated_forbidden_notes"]:
        issue_parts.append(
            "forbidden " + ", ".join(record_metrics["violated_forbidden_notes"])
        )
    if record_metrics["tier_count_mismatches"]:
        count_text = ", ".join(
            f"{item['tier']}:{item['actual']}!={item['expected']}"
            for item in record_metrics["tier_count_mismatches"]
        )
        issue_parts.append(f"counts {count_text}")
    return "; ".join(issue_parts) if issue_parts else "none"


def run_analysis_phase(
    arguments: argparse.Namespace, request_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Pair generations and compute researcher-led exploratory diagnostics.

    Args:
        arguments: Validated analysis CLI namespace with run and training paths.
        request_rows: Canonical validated pilot request bank.

    Returns:
        dict[str, Any]: Complete structured analysis also persisted to disk.

    Raises:
        RuntimeError: Raised for input/output failures.
        ValueError: Raised for pairing, vocabulary, or generation invariants.
    """

    # 步骤1: 读取两模型输出，验证请求配对并建立训练标签词表。
    generation_records = {
        role_name: read_jsonl(
            arguments.run_dir / "generations" / f"{role_name}.jsonl"
        )
        for role_name in MODEL_LABELS
    }
    paired_rows = pair_generation_records(
        request_rows,
        generation_records["base"],
        generation_records["finetuned"],
    )
    training_vocabulary = load_level3_training_vocabulary(arguments.training_file)

    # 步骤2: 计算逐请求与聚合指标，但不以启发式指标替代真人偏好。
    metrics_by_role: dict[str, dict[str, dict[str, Any]]] = {}
    model_summaries: dict[str, dict[str, Any]] = {}
    for role_name in MODEL_LABELS:
        ordered_role_records = [paired_row[role_name] for paired_row in paired_rows]
        role_metrics = [
            calculate_record_metrics(request_row, generation_record, training_vocabulary)
            for request_row, generation_record in zip(
                request_rows, ordered_role_records, strict=True
            )
        ]
        metrics_by_role[role_name] = {
            metric["request_id"]: metric for metric in role_metrics
        }
        model_summaries[role_name] = aggregate_model_metrics(
            role_name, ordered_role_records, role_metrics
        )
    paired_analysis_rows = build_paired_analysis_rows(paired_rows, metrics_by_role)
    between_model_similarities = [
        row["between_model_note_jaccard"] for row in paired_analysis_rows
    ]
    system_prompt = select_design_system_prompt(arguments.reasoning_mode)
    analysis = {
        "protocol": {
            "task": "user scent brief to concept-level note pyramid",
            "system_prompt": system_prompt,
            "system_prompt_sha256": hash_text(system_prompt),
            "requests_file": str(arguments.requests_file),
            "requests_sha256": calculate_file_sha256(arguments.requests_file),
            "training_vocabulary_file": str(arguments.training_file),
            "training_vocabulary_size": len(training_vocabulary),
            "enable_thinking": arguments.enable_thinking,
            "reasoning_mode": arguments.reasoning_mode,
            "max_new_tokens": arguments.max_new_tokens,
            "max_model_length": arguments.max_model_length,
            "subjective_winner_policy": "researcher-led exploratory analysis; no preference winner",
        },
        "request_count": len(request_rows),
        "models": model_summaries,
        "pair_summary": {
            "mean_between_model_note_jaccard": sum(between_model_similarities)
            / len(between_model_similarities),
            "exactly_equal_pyramids": sum(
                row["outputs_exactly_equal"] for row in paired_analysis_rows
            ),
        },
        "paired_rows": paired_analysis_rows,
    }

    # 步骤3: 若提供上一轮目录，对共享锚点执行同模型 thinking 前后比较。
    if arguments.comparison_run_dir is not None:
        mode_comparison = build_mode_comparison(
            arguments.comparison_run_dir,
            request_rows,
            paired_rows,
            training_vocabulary,
        )
        mode_comparison["current_reasoning_mode"] = arguments.reasoning_mode
        analysis["mode_comparison"] = mode_comparison

    # 步骤4: Persist structured diagnostics without creating canceled survey files.
    analysis_directory = arguments.run_dir / "analysis"
    write_json(analysis_directory / "automatic_analysis.json", analysis)
    write_jsonl(analysis_directory / "paired_outputs.jsonl", paired_analysis_rows)
    report_path = analysis_directory / "AUTOMATIC_PILOT_REPORT.md"
    try:
        report_path.write_text(render_automatic_report(analysis), encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"写入自动分析报告失败: {error}, 上下文: file={report_path}"
        ) from error
    return analysis


def main() -> None:
    """Parse CLI arguments and execute one generation side or paired analysis.

    Returns:
        None: Completion summary is printed after required artifacts are written.

    Raises:
        ValueError: Propagated for invalid parameters, requests, or pair invariants.
        RuntimeError: Propagated for model, parsing, or file-operation failures.
        FileNotFoundError: Propagated when required project artifacts are absent.
    """

    parser = build_argument_parser()
    arguments = resolve_and_validate_arguments(parser.parse_args())
    request_rows = load_and_validate_requests(arguments.requests_file)
    if arguments.phase == "generate":
        run_generation_phase(arguments, request_rows)
        print(
            f"Generation complete: role={arguments.model}, requests={len(request_rows)}, "
            f"run_dir={arguments.run_dir}"
        )
        return
    analysis = run_analysis_phase(arguments, request_rows)
    print(
        f"Analysis complete: requests={analysis['request_count']}, "
        f"run_dir={arguments.run_dir}"
    )
    for role_name in ("base", "finetuned"):
        model_metrics = analysis["models"][role_name]["overall"]
        print(
            f"{role_name}: schema={model_metrics['valid_schema_rate']:.3f}, "
            f"constraints={model_metrics['all_objective_constraints_pass_rate']:.3f}, "
            f"vocab={model_metrics['training_vocabulary_alignment']:.3f}, "
            f"terms={model_metrics['distinct_output_terms']}"
        )


if __name__ == "__main__":
    # 用法示例：先分别 generate --model base/finetuned，再用 analyze 汇总。
    main()
