"""
Evaluate the corrected Unsloth Qwen3.5-4B model against its base model.

The module performs generation and metric computation as separate phases so a
single 24 GB GPU only hosts one model at a time. Both models receive the exact
same system and user messages from the chat-format test set. The evaluation
stores auditable generations, paired predictions, schema diagnostics,
note-set metrics, BLEU/ROUGE scores, and Level-2/Level-3 summaries. External
dependencies are vLLM, Transformers, NumPy, and the local test/model files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# ========== Configuration ==========

# ------ Reproducible evaluation defaults ------
RANDOM_SEED = 3407
DEFAULT_MAX_MODEL_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_GPU_MEMORY_UTILIZATION = 0.82
REQUIRED_NOTE_KEYS = ("top_notes", "middle_notes", "base_notes")
MODEL_LABELS = {
    "base": "Qwen3.5-4B-Base",
    "finetuned": "Qwen3.5-4B-Finetuned-Unsloth",
}


@dataclass(frozen=True)
class ParsedPrediction:
    """Represent strict/recoverable JSON parsing and normalized note fields.

    Attributes:
        parsed_json (dict[str, Any] | None): Recovered JSON object, or ``None``.
        canonical_notes (dict[str, list[str]]): Normalized required-key values.
        is_strict_json (bool): Whether the complete raw output is one JSON object.
        is_recoverable_json (bool): Whether a leading JSON object can be recovered.
        has_required_keys (bool): Whether all three required keys are present.
        has_exact_keys (bool): Whether no required key is missing and no extra key exists.
        has_valid_schema (bool): Whether exact keys map only to lists of strings.
    """

    parsed_json: dict[str, Any] | None
    canonical_notes: dict[str, list[str]]
    is_strict_json: bool
    is_recoverable_json: bool
    has_required_keys: bool
    has_exact_keys: bool
    has_valid_schema: bool


# ========== Command-Line Arguments ==========


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line options for generation and metric phases.

    The caller selects exactly one phase. Generation also requires a model role,
    while the metrics phase consumes the two previously written generation files.

    Returns:
        argparse.ArgumentParser: Fully configured parser consumed by ``main``.

    Raises:
        RuntimeError: This function does not normally raise an exception.
    """

    # 步骤1: Derive portable defaults from the server-side script location.
    project_root = Path(__file__).resolve().parent.parent
    default_run_name = datetime.now().strftime("%Y%m%d_%H%M%S_qwen35_4b_chat_l3_unsloth")

    # 步骤2: Expose only parameters that affect reproducibility or resource use.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "metrics"), required=True)
    parser.add_argument("--model", choices=("base", "finetuned"))
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--finetuned-model", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=project_root / "src/outputs/evaluate_rerun_v2" / default_run_name,
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--level", choices=("all", "2", "3"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-model-length", type=int, default=DEFAULT_MAX_MODEL_LENGTH)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def resolve_and_validate_arguments(arguments: argparse.Namespace) -> argparse.Namespace:
    """Resolve default paths and reject incomplete or unsafe argument combinations.

    Args:
        arguments (argparse.Namespace): Raw values produced by ``build_argument_parser``.

    Returns:
        argparse.Namespace: The same namespace with project-relative paths resolved.

    Raises:
        ValueError: If numeric limits are invalid or generation lacks ``--model``.
        FileNotFoundError: If the selected dataset or model directory is absent.
    """

    # 步骤1: Fill paths after the project root can be overridden by the caller.
    project_root = arguments.project_root.resolve()
    arguments.test_file = arguments.test_file or project_root / "data/qwen_finetune_notes_review_v2_strict_chat_test.jsonl"
    arguments.base_model = arguments.base_model or project_root / "model/Qwen3.5-4b/unsloth/Qwen3.5-4B"
    arguments.finetuned_model = arguments.finetuned_model or project_root / "output/finetuned_qwen35_4b_v2_chat_l3_fixed_unsloth_merged"
    arguments.run_dir = arguments.run_dir.resolve()

    # 步骤2: Reject settings that would make sampling or vLLM allocation invalid.
    if arguments.phase == "generate" and arguments.model is None:
        raise ValueError(f"Argument validation failed: --model is required for generation, context: phase={arguments.phase}")
    if arguments.max_samples < 0 or arguments.max_new_tokens <= 0 or arguments.max_model_length <= 0:
        raise ValueError(f"Argument validation failed: invalid token/sample limit, context: {vars(arguments)}")
    if not 0.1 <= arguments.gpu_memory_utilization <= 0.95:
        raise ValueError(f"Argument validation failed: GPU utilization out of range, context: value={arguments.gpu_memory_utilization}")

    # 步骤3: Validate only model paths needed by the requested phase.
    required_paths = [arguments.test_file]
    if arguments.phase == "generate":
        required_paths.append(arguments.base_model if arguments.model == "base" else arguments.finetuned_model)
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Input validation failed: path does not exist, context: path={required_path}")
    return arguments


# ========== Dataset and JSON Handling ==========


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 digest for an external evaluation input.

    Args:
        file_path (Path): Existing file selected through CLI or a project default.

    Returns:
        str: Lowercase 64-character hexadecimal digest.

    Raises:
        RuntimeError: If the file cannot be opened or read completely.
    """

    # 步骤1: Stream the file so dataset size does not determine memory usage.
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as input_handle:
            for data_chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                digest.update(data_chunk)
    except OSError as exception:
        raise RuntimeError(f"SHA-256 calculation failed: {exception}, context: file_path={file_path}") from exception
    return digest.hexdigest()


def load_test_samples(test_file: Path, level: str, maximum_samples: int) -> list[dict[str, Any]]:
    """Load chat test rows while preserving source indexes and optional level filtering.

    Args:
        test_file (Path): Chat-format JSONL dataset supplied through CLI.
        level (str): ``all``, ``2``, or ``3`` selected by the evaluation caller.
        maximum_samples (int): Zero for all matching rows, otherwise a leading subset.

    Returns:
        list[dict[str, Any]]: Validated rows with an added ``source_index`` field.

    Raises:
        RuntimeError: If file reading or JSON decoding fails.
        ValueError: If a row lacks system/user/assistant messages or level metadata.
    """

    # 步骤1: Parse line by line so malformed input reports the exact source line.
    selected_samples: list[dict[str, Any]] = []
    try:
        with test_file.open("r", encoding="utf-8-sig") as input_handle:
            for source_index, raw_line in enumerate(input_handle):
                if not raw_line.strip():
                    continue
                sample = json.loads(raw_line)
                sample_level = int(sample.get("metadata", {}).get("scent_information_level", -1))
                if level != "all" and sample_level != int(level):
                    continue
                validate_chat_sample(sample, source_index)
                sample["source_index"] = source_index
                selected_samples.append(sample)
                if maximum_samples and len(selected_samples) >= maximum_samples:
                    break
    except (OSError, json.JSONDecodeError) as exception:
        raise RuntimeError(f"Test dataset loading failed: {exception}, context: test_file={test_file}") from exception
    if not selected_samples:
        raise ValueError(f"Test dataset selection failed: no matching rows, context: level={level}, maximum_samples={maximum_samples}")
    return selected_samples


def validate_chat_sample(sample: dict[str, Any], source_index: int) -> None:
    """Validate the three-message structure required for fair generation.

    Args:
        sample (dict[str, Any]): One decoded JSONL row produced by ``load_test_samples``.
        source_index (int): Zero-based line index from the original test file.

    Returns:
        None: Successful validation leaves the row unchanged.

    Raises:
        ValueError: If message roles/content or scent-level metadata are invalid.
    """

    # 步骤1: Require the exact role order used by dataset construction and training.
    messages = sample.get("messages")
    expected_roles = ["system", "user", "assistant"]
    actual_roles = [message.get("role") for message in messages] if isinstance(messages, list) else []
    if actual_roles != expected_roles:
        raise ValueError(f"Chat row validation failed: unexpected roles, context: source_index={source_index}, roles={actual_roles}")

    # 步骤2: Reject empty messages and unknown evidence levels before inference.
    if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
        raise ValueError(f"Chat row validation failed: empty message content, context: source_index={source_index}")
    sample_level = sample.get("metadata", {}).get("scent_information_level")
    if sample_level not in (2, 3):
        raise ValueError(f"Chat row validation failed: unsupported level, context: source_index={source_index}, level={sample_level}")


def normalize_note_term(note_term: Any) -> str:
    """Normalize one predicted or reference fragrance-note term.

    Args:
        note_term (Any): Value obtained from a note list in decoded model JSON.

    Returns:
        str: Lowercase, whitespace-collapsed term; empty for blank values.

    Raises:
        RuntimeError: This pure normalization function does not normally raise.
    """

    # 步骤1: Collapse formatting variation without performing synonym conflation.
    normalized_term = re.sub(r"\s+", " ", str(note_term)).strip().lower()
    return normalized_term


def strip_thinking_block(raw_text: str) -> str:
    """Remove an optional Qwen thinking block before recoverable JSON parsing.

    Args:
        raw_text (str): Decoded model response produced by vLLM.

    Returns:
        str: Trimmed response with complete ``<think>`` blocks removed.

    Raises:
        RuntimeError: This regular-expression substitution does not normally raise.
    """

    # 步骤1: Thinking is disabled, but defensive removal keeps parser diagnostics stable.
    return re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()


def parse_prediction(raw_text: str) -> ParsedPrediction:
    """Parse strict JSON first and then recover the first embedded JSON object.

    Args:
        raw_text (str): Complete decoded response returned for one test prompt.

    Returns:
        ParsedPrediction: Parsing flags and normalized note lists for metric code.

    Raises:
        RuntimeError: Unexpected non-JSON parser failures are propagated to expose bugs.
    """

    # 步骤1: Measure strict compliance using the complete response after trimming only.
    strict_object: dict[str, Any] | None = None
    try:
        strict_candidate = json.loads(raw_text.strip())
        strict_object = strict_candidate if isinstance(strict_candidate, dict) else None
    except json.JSONDecodeError:
        strict_object = None

    # 步骤2: Recover one balanced JSON object without greedy regular expressions.
    parsed_object = strict_object
    if parsed_object is None:
        recoverable_text = strip_thinking_block(raw_text)
        object_start = recoverable_text.find("{")
        if object_start >= 0:
            try:
                recovered_candidate, _ = json.JSONDecoder().raw_decode(recoverable_text[object_start:])
                parsed_object = recovered_candidate if isinstance(recovered_candidate, dict) else None
            except json.JSONDecodeError:
                parsed_object = None

    # 步骤3: Separate key presence from the stricter list-of-string schema check.
    object_keys = set(parsed_object.keys()) if parsed_object else set()
    required_key_set = set(REQUIRED_NOTE_KEYS)
    has_required_keys = required_key_set.issubset(object_keys)
    has_exact_keys = object_keys == required_key_set
    has_valid_schema = has_exact_keys and all(
        isinstance(parsed_object.get(key), list)
        and all(isinstance(note_term, str) for note_term in parsed_object[key])
        for key in REQUIRED_NOTE_KEYS
    ) if parsed_object else False

    # 步骤4: Canonicalize usable list values while retaining schema failure flags.
    canonical_notes: dict[str, list[str]] = {}
    for required_key in REQUIRED_NOTE_KEYS:
        raw_notes = parsed_object.get(required_key, []) if parsed_object else []
        normalized_notes = [normalize_note_term(note_term) for note_term in raw_notes] if isinstance(raw_notes, list) else []
        canonical_notes[required_key] = [note_term for note_term in normalized_notes if note_term]
    return ParsedPrediction(
        parsed_json=parsed_object,
        canonical_notes=canonical_notes,
        is_strict_json=strict_object is not None,
        is_recoverable_json=parsed_object is not None,
        has_required_keys=has_required_keys,
        has_exact_keys=has_exact_keys,
        has_valid_schema=has_valid_schema,
    )


# ========== Generation ==========


def build_prompt_token_ids(tokenizer: Any, samples: Sequence[dict[str, Any]]) -> list[list[int]]:
    """Render exact dataset messages and tokenize them with each model's tokenizer.

    Args:
        tokenizer (Any): Hugging Face tokenizer loaded from the selected model path.
        samples (Sequence[dict[str, Any]]): Validated chat rows from ``load_test_samples``.

    Returns:
        list[list[int]]: One prompt-token sequence per sample in unchanged order.

    Raises:
        RuntimeError: If chat-template rendering or tokenization fails.
    """

    # 步骤1: Exclude the reference assistant turn so no label reaches generation.
    prompt_token_ids: list[list[int]] = []
    try:
        for sample in samples:
            prompt_messages = sample["messages"][:2]
            rendered_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_token_ids.append(tokenizer.encode(rendered_prompt, add_special_tokens=False))
    except Exception as exception:
        raise RuntimeError(f"Prompt tokenization failed: {exception}, context: sample_count={len(samples)}") from exception
    return prompt_token_ids


def collect_stop_token_ids(tokenizer: Any) -> list[int]:
    """Collect Qwen end-of-text and end-of-message IDs for deterministic stopping.

    Args:
        tokenizer (Any): Model tokenizer used for prompt encoding and output decoding.

    Returns:
        list[int]: Unique, non-negative stop token IDs accepted by vLLM.

    Raises:
        RuntimeError: If tokenizer conversion unexpectedly fails.
    """

    # 步骤1: Include both generation-config EOS variants used by Qwen3.5 chat.
    try:
        candidate_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]
    except Exception as exception:
        raise RuntimeError(f"Stop-token resolution failed: {exception}, context: tokenizer={type(tokenizer).__name__}") from exception
    return sorted({int(token_id) for token_id in candidate_ids if isinstance(token_id, int) and token_id >= 0})


def generate_model_records(arguments: argparse.Namespace, samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load one model in vLLM and generate auditable records for all selected rows.

    Args:
        arguments (argparse.Namespace): Validated CLI values including model role and paths.
        samples (Sequence[dict[str, Any]]): Ordered chat test rows selected for this run.

    Returns:
        list[dict[str, Any]]: Generation records aligned one-to-one with ``samples``.

    Raises:
        RuntimeError: If tokenizer/model loading, vLLM generation, or decoding fails.
    """

    # 步骤1: Expose environment-local build tools to FlashInfer child processes.
    environment_binary_directory = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    if environment_binary_directory not in current_path.split(os.pathsep):
        os.environ["PATH"] = environment_binary_directory + os.pathsep + current_path

    # 步骤2: Import GPU-heavy dependencies only during the generation phase.
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exception:
        raise RuntimeError(f"Inference dependency import failed: {exception}, context: python={sys.executable}") from exception

    # 步骤3: Resolve the selected model and prepare exact prompt token IDs.
    model_path = arguments.base_model if arguments.model == "base" else arguments.finetuned_model
    model_label = MODEL_LABELS[arguments.model]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exception:
        raise RuntimeError(f"Tokenizer loading failed: {exception}, context: model_path={model_path}") from exception
    prompt_token_ids = build_prompt_token_ids(tokenizer, samples)
    maximum_prompt_length = max(len(token_ids) for token_ids in prompt_token_ids)
    if maximum_prompt_length + arguments.max_new_tokens > arguments.max_model_length:
        raise ValueError(
            f"Prompt length validation failed: context window too small, context: maximum_prompt_length={maximum_prompt_length}, "
            f"max_new_tokens={arguments.max_new_tokens}, max_model_length={arguments.max_model_length}"
        )

    # 步骤4: Load one model only; separate CLI invocations release the full GPU afterward.
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
        token_prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]
        generation_start = time.perf_counter()
        model_outputs = inference_engine.generate(token_prompts, sampling_parameters, use_tqdm=True)
        generation_elapsed = time.perf_counter() - generation_start
    except Exception as exception:
        raise RuntimeError(f"vLLM generation failed: {exception}, context: model_label={model_label}, model_path={model_path}") from exception

    # 步骤5: Decode with the same tokenizer used for prompt IDs and preserve diagnostics.
    generation_records: list[dict[str, Any]] = []
    for sample, request_output in zip(samples, model_outputs, strict=True):
        candidate_output = request_output.outputs[0] if request_output.outputs else None
        output_token_ids = list(candidate_output.token_ids) if candidate_output else []
        raw_response = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()
        parsed_prediction = parse_prediction(raw_response)
        generation_records.append(
            create_generation_record(
                sample=sample,
                model_label=model_label,
                model_path=model_path,
                raw_response=raw_response,
                parsed_prediction=parsed_prediction,
                output_token_ids=output_token_ids,
                finish_reason=str(candidate_output.finish_reason) if candidate_output else "missing_output",
                maximum_new_tokens=arguments.max_new_tokens,
            )
        )
    print(
        f"Generated {len(generation_records)} rows for {model_label} in {generation_elapsed:.2f}s "
        f"({generation_elapsed / len(generation_records):.4f}s/sample)."
    )

    # 步骤6: Explicit cleanup reduces the chance of a lingering vLLM worker on failure paths.
    del inference_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generation_records


def create_generation_record(
    sample: dict[str, Any],
    model_label: str,
    model_path: Path,
    raw_response: str,
    parsed_prediction: ParsedPrediction,
    output_token_ids: Sequence[int],
    finish_reason: str,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    """Create one serializable generation record with complete audit fields.

    Args:
        sample (dict[str, Any]): Source test row containing messages and metadata.
        model_label (str): Stable display label selected from ``MODEL_LABELS``.
        model_path (Path): Local model directory used by the inference engine.
        raw_response (str): Decoded model output excluding the prompt.
        parsed_prediction (ParsedPrediction): Structured parser result for the output.
        output_token_ids (Sequence[int]): Generated token IDs returned by vLLM.
        finish_reason (str): vLLM completion reason such as ``stop`` or ``length``.
        maximum_new_tokens (int): CLI generation ceiling used to detect truncation.

    Returns:
        dict[str, Any]: JSON-serializable record written to ``generations``.

    Raises:
        RuntimeError: This record-construction function does not normally raise.
    """

    # 步骤1: Keep prompts and references together for later pairwise or judge audits.
    metadata = sample["metadata"]
    return {
        "source_index": sample["source_index"],
        "row_id": metadata.get("row_id"),
        "perfume_name": metadata.get("perfume_name"),
        "scent_information_level": metadata.get("scent_information_level"),
        "model_name": model_label,
        "model_path": str(model_path),
        "messages": sample["messages"][:2],
        "reference": sample["messages"][2]["content"],
        "raw_response": raw_response,
        "parsed_json": parsed_prediction.parsed_json,
        "canonical_notes": parsed_prediction.canonical_notes,
        "strict_json": parsed_prediction.is_strict_json,
        "recoverable_json": parsed_prediction.is_recoverable_json,
        "required_keys": parsed_prediction.has_required_keys,
        "exact_keys": parsed_prediction.has_exact_keys,
        "valid_schema": parsed_prediction.has_valid_schema,
        "output_token_count": len(output_token_ids),
        "finish_reason": finish_reason,
        "hit_max_new_tokens": len(output_token_ids) >= maximum_new_tokens or finish_reason == "length",
    }


# ========== Metric Primitives ==========


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two metric values while defining zero-denominator results as zero.

    Args:
        numerator (float): Match or overlap count produced by metric code.
        denominator (float): Candidate/reference count that may be zero.

    Returns:
        float: Quotient when possible, otherwise ``0.0``.

    Raises:
        RuntimeError: This arithmetic helper does not normally raise.
    """

    # 步骤1: Empty predictions should not create NaN values in JSON summaries.
    return float(numerator / denominator) if denominator else 0.0


def calculate_set_scores(reference_terms: set[str], predicted_terms: set[str]) -> dict[str, float]:
    """Calculate precision, recall, F1, and Jaccard for two note-term sets.

    Args:
        reference_terms (set[str]): Canonical reference notes from one or more tiers.
        predicted_terms (set[str]): Canonical predicted notes from matching tiers.

    Returns:
        dict[str, float]: Four bounded overlap metrics in the range ``[0, 1]``.

    Raises:
        RuntimeError: This pure set operation does not normally raise.
    """

    # 步骤1: Count unique-term overlap so duplicate predictions cannot inflate scores.
    overlap_count = len(reference_terms & predicted_terms)
    precision = safe_divide(overlap_count, len(predicted_terms))
    recall = safe_divide(overlap_count, len(reference_terms))
    f1_score = safe_divide(2.0 * precision * recall, precision + recall)
    jaccard = safe_divide(overlap_count, len(reference_terms | predicted_terms))
    return {"precision": precision, "recall": recall, "f1": f1_score, "jaccard": jaccard}


def tokenize_canonical_json(canonical_notes: dict[str, list[str]]) -> list[str]:
    """Tokenize canonical note JSON for reproducible BLEU and ROUGE metrics.

    Args:
        canonical_notes (dict[str, list[str]]): Required-key note lists from parser output.

    Returns:
        list[str]: Lowercase words and JSON punctuation in deterministic key order.

    Raises:
        RuntimeError: Serialization should not fail for normalized strings.
    """

    # 步骤1: Re-serialize in fixed key order before lexical token extraction.
    ordered_notes = {key: canonical_notes.get(key, []) for key in REQUIRED_NOTE_KEYS}
    canonical_text = json.dumps(ordered_notes, ensure_ascii=False, separators=(",", ":"))
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[{}\[\]:,]", canonical_text.lower())


def create_ngrams(tokens: Sequence[str], ngram_order: int) -> Counter[tuple[str, ...]]:
    """Create counted n-grams for clipped BLEU/ROUGE overlap.

    Args:
        tokens (Sequence[str]): Canonical token sequence from one JSON object.
        ngram_order (int): Positive n-gram order requested by the metric caller.

    Returns:
        Counter[tuple[str, ...]]: Multiplicity-preserving n-gram counts.

    Raises:
        ValueError: If ``ngram_order`` is not positive.
    """

    # 步骤1: Validate the order because negative slicing would silently mis-score data.
    if ngram_order <= 0:
        raise ValueError(f"N-gram construction failed: order must be positive, context: ngram_order={ngram_order}")
    return Counter(tuple(tokens[index:index + ngram_order]) for index in range(len(tokens) - ngram_order + 1))


def calculate_bleu_precisions(reference_tokens: Sequence[str], predicted_tokens: Sequence[str]) -> dict[str, float]:
    """Calculate clipped BLEU-style precision for orders one through four.

    Args:
        reference_tokens (Sequence[str]): Canonical reference JSON tokens.
        predicted_tokens (Sequence[str]): Canonical prediction JSON tokens.

    Returns:
        dict[str, float]: ``BLEU-1`` through ``BLEU-4`` clipped precisions.

    Raises:
        RuntimeError: N-gram construction errors are propagated to expose bad code paths.
    """

    # 步骤1: Use counted intersections rather than set overlap to preserve multiplicity.
    bleu_scores: dict[str, float] = {}
    for ngram_order in range(1, 5):
        reference_ngrams = create_ngrams(reference_tokens, ngram_order)
        predicted_ngrams = create_ngrams(predicted_tokens, ngram_order)
        match_count = sum((reference_ngrams & predicted_ngrams).values())
        bleu_scores[f"BLEU-{ngram_order}"] = safe_divide(match_count, sum(predicted_ngrams.values()))
    return bleu_scores


def calculate_rouge_scores(reference_tokens: Sequence[str], predicted_tokens: Sequence[str]) -> dict[str, float]:
    """Calculate ROUGE-1/2 F1 and ROUGE-L F1 for canonical JSON tokens.

    Args:
        reference_tokens (Sequence[str]): Canonical reference JSON tokens.
        predicted_tokens (Sequence[str]): Canonical prediction JSON tokens.

    Returns:
        dict[str, float]: Three F1-style ROUGE metrics in ``[0, 1]``.

    Raises:
        RuntimeError: N-gram or LCS failures are propagated as implementation defects.
    """

    # 步骤1: Compute counted unigram and bigram F1 values.
    rouge_scores: dict[str, float] = {}
    for ngram_order in (1, 2):
        reference_ngrams = create_ngrams(reference_tokens, ngram_order)
        predicted_ngrams = create_ngrams(predicted_tokens, ngram_order)
        match_count = sum((reference_ngrams & predicted_ngrams).values())
        precision = safe_divide(match_count, sum(predicted_ngrams.values()))
        recall = safe_divide(match_count, sum(reference_ngrams.values()))
        rouge_scores[f"ROUGE-{ngram_order}-F"] = safe_divide(2.0 * precision * recall, precision + recall)

    # 步骤2: Compute LCS with rolling rows to bound memory for long outputs.
    previous_row = [0] * (len(predicted_tokens) + 1)
    for reference_token in reference_tokens:
        current_row = [0]
        for predicted_index, predicted_token in enumerate(predicted_tokens, start=1):
            if reference_token == predicted_token:
                current_row.append(previous_row[predicted_index - 1] + 1)
            else:
                current_row.append(max(previous_row[predicted_index], current_row[-1]))
        previous_row = current_row
    lcs_length = previous_row[-1] if previous_row else 0
    lcs_precision = safe_divide(lcs_length, len(predicted_tokens))
    lcs_recall = safe_divide(lcs_length, len(reference_tokens))
    rouge_scores["ROUGE-L-F"] = safe_divide(2.0 * lcs_precision * lcs_recall, lcs_precision + lcs_recall)
    return rouge_scores


def calculate_record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Calculate all reference-alignment metrics for one generation record.

    Args:
        record (dict[str, Any]): Generation record loaded from a JSONL artifact.

    Returns:
        dict[str, Any]: Flat/tier set scores, text metrics, and count diagnostics.

    Raises:
        ValueError: If the stored reference answer is not recoverable JSON.
    """

    # 步骤1: Parse the held-out reference independently from model output parsing.
    reference_prediction = parse_prediction(record["reference"])
    if not reference_prediction.is_recoverable_json:
        raise ValueError(f"Reference parsing failed: invalid held-out label, context: source_index={record['source_index']}")
    predicted_notes = record.get("canonical_notes", {key: [] for key in REQUIRED_NOTE_KEYS})
    reference_notes = reference_prediction.canonical_notes

    # 步骤2: Score tier-agnostic overlap and each pyramid layer separately.
    flat_reference = {term for key in REQUIRED_NOTE_KEYS for term in reference_notes[key]}
    flat_prediction = {term for key in REQUIRED_NOTE_KEYS for term in predicted_notes.get(key, [])}
    flat_scores = calculate_set_scores(flat_reference, flat_prediction)
    tier_scores = {
        key: calculate_set_scores(set(reference_notes[key]), set(predicted_notes.get(key, [])))
        for key in REQUIRED_NOTE_KEYS
    }
    tier_macro_f1 = float(np.mean([tier_scores[key]["f1"] for key in REQUIRED_NOTE_KEYS]))

    # 步骤3: Add lexical metrics for continuity with earlier paper tables.
    reference_tokens = tokenize_canonical_json(reference_notes)
    predicted_tokens = tokenize_canonical_json(predicted_notes)
    text_metrics = calculate_bleu_precisions(reference_tokens, predicted_tokens)
    text_metrics.update(calculate_rouge_scores(reference_tokens, predicted_tokens))

    # 步骤4: Count list length and cross-tier duplication as collapse diagnostics.
    predicted_sequence = [term for key in REQUIRED_NOTE_KEYS for term in predicted_notes.get(key, [])]
    metric_record: dict[str, Any] = {
        "source_index": record["source_index"],
        "row_id": record.get("row_id"),
        "scent_information_level": record["scent_information_level"],
        "flat": flat_scores,
        "tier": tier_scores,
        "tier_macro_f1": tier_macro_f1,
        "exact_flat_set": flat_reference == flat_prediction,
        "exact_tiered_set": all(set(reference_notes[key]) == set(predicted_notes.get(key, [])) for key in REQUIRED_NOTE_KEYS),
        "predicted_note_count": len(predicted_sequence),
        "duplicate_note_count": len(predicted_sequence) - len(set(predicted_sequence)),
    }
    metric_record.update(text_metrics)
    return metric_record


# ========== Aggregation and Artifacts ==========


def mean_metric(metric_records: Sequence[dict[str, Any]], extractor: Any) -> float:
    """Average one numeric field extracted from per-sample metric records.

    Args:
        metric_records (Sequence[dict[str, Any]]): Non-empty records for one group.
        extractor (Any): Callable selecting and converting the desired value.

    Returns:
        float: NumPy mean represented as a native Python float.

    Raises:
        ValueError: If the caller provides an empty record sequence.
    """

    # 步骤1: Empty groups indicate a selection/configuration bug and should not become NaN.
    if not metric_records:
        raise ValueError("Metric aggregation failed: group contains no records, context: metric_records=0")
    return float(np.mean([float(extractor(metric_record)) for metric_record in metric_records]))


def summarize_group(records: Sequence[dict[str, Any]], metric_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate compliance, overlap, lexical, and diversity metrics for one group.

    Args:
        records (Sequence[dict[str, Any]]): Generation records in the selected level group.
        metric_records (Sequence[dict[str, Any]]): Aligned per-sample metric dictionaries.

    Returns:
        dict[str, Any]: JSON-serializable macro-average summary.

    Raises:
        ValueError: If record lists are empty or misaligned.
    """

    # 步骤1: Validate alignment before averaging any values.
    if not records or len(records) != len(metric_records):
        raise ValueError(f"Metric aggregation failed: record mismatch, context: records={len(records)}, metrics={len(metric_records)}")

    # 步骤2: Collect normalized vocabulary and output-level diagnostics.
    all_predicted_terms: list[str] = []
    for record in records:
        for required_key in REQUIRED_NOTE_KEYS:
            all_predicted_terms.extend(record.get("canonical_notes", {}).get(required_key, []))
    top_terms = Counter(all_predicted_terms).most_common(20)

    # 步骤3: Report schema, set-overlap, tier, and length diagnostics together.
    summary: dict[str, Any] = {
        "samples": len(records),
        "strict_json_rate": float(np.mean([record["strict_json"] for record in records])),
        "recoverable_json_rate": float(np.mean([record["recoverable_json"] for record in records])),
        "required_keys_rate": float(np.mean([record["required_keys"] for record in records])),
        "exact_keys_rate": float(np.mean([record["exact_keys"] for record in records])),
        "valid_schema_rate": float(np.mean([record["valid_schema"] for record in records])),
        "precision": mean_metric(metric_records, lambda item: item["flat"]["precision"]),
        "recall": mean_metric(metric_records, lambda item: item["flat"]["recall"]),
        "f1": mean_metric(metric_records, lambda item: item["flat"]["f1"]),
        "jaccard": mean_metric(metric_records, lambda item: item["flat"]["jaccard"]),
        "tier_macro_f1": mean_metric(metric_records, lambda item: item["tier_macro_f1"]),
        "exact_flat_set_rate": mean_metric(metric_records, lambda item: item["exact_flat_set"]),
        "exact_tiered_set_rate": mean_metric(metric_records, lambda item: item["exact_tiered_set"]),
        "average_note_count": mean_metric(metric_records, lambda item: item["predicted_note_count"]),
        "average_duplicate_notes": mean_metric(metric_records, lambda item: item["duplicate_note_count"]),
        "hit_max_new_tokens_rate": float(np.mean([record["hit_max_new_tokens"] for record in records])),
        "average_output_tokens": float(np.mean([record["output_token_count"] for record in records])),
        "unique_output_rate": safe_divide(len({record["raw_response"] for record in records}), len(records)),
        "distinct_terms": len(set(all_predicted_terms)),
        "top_terms": top_terms,
    }

    # 步骤4: Preserve the earlier BLEU/ROUGE reporting surface.
    for metric_name in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1-F", "ROUGE-2-F", "ROUGE-L-F"):
        summary[metric_name] = mean_metric(metric_records, lambda item, name=metric_name: item[name])
    for required_key in REQUIRED_NOTE_KEYS:
        summary[f"{required_key}_f1"] = mean_metric(metric_records, lambda item, key=required_key: item["tier"][key]["f1"])
    return summary


def build_model_summary(records: Sequence[dict[str, Any]], metric_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build overall and evidence-level summaries for one evaluated model.

    Args:
        records (Sequence[dict[str, Any]]): All generation records for one model.
        metric_records (Sequence[dict[str, Any]]): Metrics aligned by source index.

    Returns:
        dict[str, Any]: ``overall``, ``level_2``, and ``level_3`` summaries when present.

    Raises:
        ValueError: If input lists are misaligned or empty.
    """

    # 步骤1: Aggregate the complete selected test set.
    model_summary = {"overall": summarize_group(records, metric_records)}

    # 步骤2: Add level-specific groups without inventing empty statistics.
    for evidence_level in (2, 3):
        grouped_pairs = [
            (record, metric_record)
            for record, metric_record in zip(records, metric_records, strict=True)
            if int(record["scent_information_level"]) == evidence_level
        ]
        if grouped_pairs:
            grouped_records = [pair[0] for pair in grouped_pairs]
            grouped_metrics = [pair[1] for pair in grouped_pairs]
            model_summary[f"level_{evidence_level}"] = summarize_group(grouped_records, grouped_metrics)
    return model_summary


def calculate_summary_deltas(base_summary: dict[str, Any], finetuned_summary: dict[str, Any]) -> dict[str, Any]:
    """Subtract base numeric metrics from finetuned metrics for each shared group.

    Args:
        base_summary (dict[str, Any]): Grouped summary for the base model.
        finetuned_summary (dict[str, Any]): Grouped summary for the tuned model.

    Returns:
        dict[str, Any]: Signed finetuned-minus-base deltas for shared scalar fields.

    Raises:
        RuntimeError: This deterministic dictionary traversal does not normally raise.
    """

    # 步骤1: Compare only shared numeric fields; counts and term lists remain in source summaries.
    delta_summary: dict[str, Any] = {}
    for group_name in sorted(set(base_summary) & set(finetuned_summary)):
        group_deltas: dict[str, float] = {}
        for metric_name, base_value in base_summary[group_name].items():
            finetuned_value = finetuned_summary[group_name].get(metric_name)
            if isinstance(base_value, (int, float)) and isinstance(finetuned_value, (int, float)) and metric_name != "samples":
                group_deltas[metric_name] = float(finetuned_value - base_value)
        delta_summary[group_name] = group_deltas
    return delta_summary


def build_case_audit(
    base_records: Sequence[dict[str, Any]],
    finetuned_records: Sequence[dict[str, Any]],
    base_metrics: Sequence[dict[str, Any]],
    finetuned_metrics: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Select the largest flat-F1 improvements and regressions for qualitative review.

    Args:
        base_records (Sequence[dict[str, Any]]): Base generations ordered by source index.
        finetuned_records (Sequence[dict[str, Any]]): Tuned generations in matching order.
        base_metrics (Sequence[dict[str, Any]]): Per-sample base alignment metrics.
        finetuned_metrics (Sequence[dict[str, Any]]): Per-sample tuned alignment metrics.

    Returns:
        dict[str, list[dict[str, Any]]]: Ten strongest improvements and regressions.

    Raises:
        ValueError: If the four aligned sequences have different lengths.
    """

    # 步骤1: Pair every prediction with its reference and signed F1 changes.
    sequence_lengths = {len(base_records), len(finetuned_records), len(base_metrics), len(finetuned_metrics)}
    if len(sequence_lengths) != 1:
        raise ValueError(f"Case audit failed: input lengths differ, context: lengths={sequence_lengths}")
    audit_rows: list[dict[str, Any]] = []
    for base_record, tuned_record, base_metric, tuned_metric in zip(
        base_records,
        finetuned_records,
        base_metrics,
        finetuned_metrics,
        strict=True,
    ):
        audit_rows.append(
            {
                "source_index": base_record["source_index"],
                "row_id": base_record.get("row_id"),
                "perfume_name": base_record.get("perfume_name"),
                "scent_information_level": base_record["scent_information_level"],
                "review": base_record["messages"][1]["content"],
                "reference": base_record["reference"],
                "base_response": base_record["raw_response"],
                "finetuned_response": tuned_record["raw_response"],
                "base_f1": base_metric["flat"]["f1"],
                "finetuned_f1": tuned_metric["flat"]["f1"],
                "f1_delta": tuned_metric["flat"]["f1"] - base_metric["flat"]["f1"],
                "tier_macro_f1_delta": tuned_metric["tier_macro_f1"] - base_metric["tier_macro_f1"],
            }
        )

    # 步骤2: Sort once and slice both tails for deterministic manual inspection.
    ordered_rows = sorted(audit_rows, key=lambda item: (item["f1_delta"], item["tier_macro_f1_delta"], -item["source_index"]))
    return {
        "largest_regressions": ordered_rows[:10],
        "largest_improvements": list(reversed(ordered_rows[-10:])),
    }


# ========== File Input and Output ==========


def write_json_file(output_path: Path, payload: Any) -> None:
    """Write one UTF-8 JSON artifact with stable indentation.

    Args:
        output_path (Path): Destination beneath the selected run directory.
        payload (Any): JSON-serializable configuration, summary, or audit object.

    Returns:
        None: Successful completion writes and closes the destination file.

    Raises:
        RuntimeError: If directory creation, serialization, or writing fails.
    """

    # 步骤1: Create only the required parent and include path context on failures.
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, TypeError) as exception:
        raise RuntimeError(f"JSON artifact writing failed: {exception}, context: output_path={output_path}") from exception


def write_jsonl_file(output_path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dictionaries as UTF-8 JSON Lines.

    Args:
        output_path (Path): Destination beneath the selected run directory.
        records (Iterable[dict[str, Any]]): Ordered serializable records from generation or metrics.

    Returns:
        None: Successful completion writes every record followed by a newline.

    Raises:
        RuntimeError: If directory creation, serialization, or writing fails.
    """

    # 步骤1: Stream records so full evaluation artifacts need not be duplicated in memory.
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_handle:
            for record in records:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, TypeError) as exception:
        raise RuntimeError(f"JSONL artifact writing failed: {exception}, context: output_path={output_path}") from exception


def read_jsonl_file(input_path: Path) -> list[dict[str, Any]]:
    """Read one evaluation JSONL artifact into an ordered list.

    Args:
        input_path (Path): Existing generation artifact produced by this module.

    Returns:
        list[dict[str, Any]]: Decoded non-empty records in source order.

    Raises:
        RuntimeError: If reading or JSON decoding fails.
        ValueError: If the artifact contains no records.
    """

    # 步骤1: Decode with explicit UTF-8 and preserve source order for pairing.
    decoded_records: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as input_handle:
            for raw_line in input_handle:
                if raw_line.strip():
                    decoded_records.append(json.loads(raw_line))
    except (OSError, json.JSONDecodeError) as exception:
        raise RuntimeError(f"JSONL artifact reading failed: {exception}, context: input_path={input_path}") from exception
    if not decoded_records:
        raise ValueError(f"JSONL artifact validation failed: no records, context: input_path={input_path}")
    return decoded_records


def update_run_configuration(arguments: argparse.Namespace, samples: Sequence[dict[str, Any]]) -> None:
    """Create or update the run configuration with immutable evaluation inputs.

    Args:
        arguments (argparse.Namespace): Validated paths, selection, and generation settings.
        samples (Sequence[dict[str, Any]]): Rows selected for the current generation phase.

    Returns:
        None: The run's ``config.json`` is created or updated.

    Raises:
        RuntimeError: If an existing configuration cannot be read or the update cannot be written.
    """

    # 步骤1: Preserve information written by the first model generation.
    config_path = arguments.run_dir / "config.json"
    existing_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            raise RuntimeError(f"Run configuration reading failed: {exception}, context: config_path={config_path}") from exception

    # 步骤2: Record exact files, hashes, message policy, selection, and decoder settings.
    level_counts = Counter(int(sample["metadata"]["scent_information_level"]) for sample in samples)
    existing_config.update(
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()),
            "python": sys.executable,
            "test_file": str(arguments.test_file),
            "test_file_sha256": calculate_file_sha256(arguments.test_file),
            "base_model": str(arguments.base_model),
            "finetuned_model": str(arguments.finetuned_model),
            "selection": {
                "level": arguments.level,
                "max_samples": arguments.max_samples,
                "selected_samples": len(samples),
                "level_counts": dict(sorted(level_counts.items())),
            },
            "prompt_policy": "exact_system_and_user_messages_from_chat_test_v1",
            "reference_excluded_from_generation": True,
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": arguments.max_new_tokens,
                "max_model_length": arguments.max_model_length,
                "enable_thinking": False,
                "seed": arguments.seed,
                "dtype": "bfloat16",
                "gpu_memory_utilization": arguments.gpu_memory_utilization,
                "explicit_stop_tokens": ["<|endoftext|>", "<|im_end|>"],
            },
        }
    )
    generated_models = set(existing_config.get("generated_models", []))
    if arguments.model:
        generated_models.add(arguments.model)
    existing_config["generated_models"] = sorted(generated_models)
    write_json_file(config_path, existing_config)


def run_generation_phase(arguments: argparse.Namespace) -> None:
    """Run one model generation phase and save its complete JSONL artifact.

    Args:
        arguments (argparse.Namespace): Validated generation arguments from ``main``.

    Returns:
        None: Generation and configuration artifacts are written under ``run_dir``.

    Raises:
        RuntimeError: Propagated from dataset loading, vLLM generation, or file output.
    """

    # 步骤1: Select identical held-out rows for each independently launched model.
    samples = load_test_samples(arguments.test_file, arguments.level, arguments.max_samples)
    update_run_configuration(arguments, samples)

    # 步骤2: Generate and persist before process exit releases the vLLM engine.
    generation_records = generate_model_records(arguments, samples)
    generation_path = arguments.run_dir / "generations" / f"{arguments.model}.jsonl"
    write_jsonl_file(generation_path, generation_records)
    print(f"Saved generation artifact: {generation_path}")


def run_metrics_phase(arguments: argparse.Namespace) -> None:
    """Pair model generations, compute metrics, and write summaries and case audits.

    Args:
        arguments (argparse.Namespace): Validated metric-phase paths from ``main``.

    Returns:
        None: Paired predictions and all metric artifacts are written under ``run_dir``.

    Raises:
        RuntimeError: Propagated from artifact reading or writing.
        ValueError: If baseline and finetuned source indexes are not identical.
    """

    # 步骤1: Read independently generated files and enforce exact sample alignment.
    base_records = read_jsonl_file(arguments.run_dir / "generations/base.jsonl")
    finetuned_records = read_jsonl_file(arguments.run_dir / "generations/finetuned.jsonl")
    base_indexes = [record["source_index"] for record in base_records]
    finetuned_indexes = [record["source_index"] for record in finetuned_records]
    if base_indexes != finetuned_indexes:
        raise ValueError(
            f"Generation pairing failed: source indexes differ, context: base_count={len(base_indexes)}, "
            f"finetuned_count={len(finetuned_indexes)}"
        )

    # 步骤2: Calculate per-sample metrics independently for both models.
    base_metrics = [calculate_record_metrics(record) for record in base_records]
    finetuned_metrics = [calculate_record_metrics(record) for record in finetuned_records]
    metric_rows = [
        {
            "source_index": base_record["source_index"],
            "row_id": base_record.get("row_id"),
            "scent_information_level": base_record["scent_information_level"],
            "base": base_metric,
            "finetuned": finetuned_metric,
        }
        for base_record, base_metric, finetuned_metric in zip(base_records, base_metrics, finetuned_metrics, strict=True)
    ]

    # 步骤3: Build paired predictions for future reference-free/reference-guided judging.
    paired_records = [
        {
            "source_index": base_record["source_index"],
            "row_id": base_record.get("row_id"),
            "perfume_name": base_record.get("perfume_name"),
            "scent_information_level": base_record["scent_information_level"],
            "messages": base_record["messages"],
            "reference": base_record["reference"],
            "base_raw": base_record["raw_response"],
            "base_parsed": base_record["parsed_json"],
            "base_valid_json": base_record["recoverable_json"],
            "finetuned_raw": tuned_record["raw_response"],
            "finetuned_parsed": tuned_record["parsed_json"],
            "finetuned_valid_json": tuned_record["recoverable_json"],
        }
        for base_record, tuned_record in zip(base_records, finetuned_records, strict=True)
    ]

    # 步骤4: Aggregate overall and evidence-level summaries and signed deltas.
    base_summary = build_model_summary(base_records, base_metrics)
    finetuned_summary = build_model_summary(finetuned_records, finetuned_metrics)
    final_summary = {
        "num_samples": len(base_records),
        "models": {
            MODEL_LABELS["base"]: base_summary,
            MODEL_LABELS["finetuned"]: finetuned_summary,
        },
        "finetuned_minus_base": calculate_summary_deltas(base_summary, finetuned_summary),
    }
    case_audit = build_case_audit(base_records, finetuned_records, base_metrics, finetuned_metrics)

    # 步骤5: Write every layer needed for independent recomputation and manual review.
    write_jsonl_file(arguments.run_dir / "generations/paired_predictions.jsonl", paired_records)
    write_jsonl_file(arguments.run_dir / "metrics/per_sample_metrics.jsonl", metric_rows)
    write_json_file(arguments.run_dir / "metrics/automatic_metrics.json", final_summary)
    write_json_file(arguments.run_dir / "metrics/case_audit.json", case_audit)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


# ========== Entrypoint ==========


def main() -> None:
    """Execute one reproducible generation or metric-computation phase.

    Command-line arguments come from the current process. Generation writes one
    model file per invocation; metrics requires both files in the same run directory.

    Returns:
        None: Successful completion prints artifact locations or the final summary.

    Raises:
        RuntimeError: Wraps workflow failures with phase and run-directory context.
    """

    # 步骤1: Parse and validate all external paths before model allocation.
    parser = build_argument_parser()
    arguments = resolve_and_validate_arguments(parser.parse_args())

    # 步骤2: Dispatch exactly one phase so GPU processes have a clear lifecycle.
    try:
        if arguments.phase == "generate":
            run_generation_phase(arguments)
        else:
            run_metrics_phase(arguments)
    except Exception as exception:
        raise RuntimeError(
            f"Evaluation workflow failed: {exception}, context: phase={arguments.phase}, model={arguments.model}, run_dir={arguments.run_dir}"
        ) from exception


if __name__ == "__main__":
    main()
