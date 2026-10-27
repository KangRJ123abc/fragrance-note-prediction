"""Reconstruct the documented length-only JSON fallback without claiming replay.

Default execution only prints a plan. GPU generation requires --execute, an
explicit seed and a verified local model directory. Original records are never
overwritten. Historical schema/API/seed details were not recovered; this code
records its own choices. CPU planning uses standard library and the original
NumPy-backed parser; explicit generation additionally requires vLLM/Transformers.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from model_files import REPOSITORY_ROOT, verify_files
from reconstruction_common import (
    REQUIRED_NOTE_KEYS, calculate_sha256, create_output_directory,
    parse_prediction, read_records, write_json, write_records,
)

# ------ Documented settings and explicitly reconstructed choices ------
MAXIMUM_ITEMS_PER_TIER = 12
MAXIMUM_NEW_TOKENS = 512
RECONSTRUCTION_STATUS = "release_reconstruction_v1_not_original_executor"


# ========== Pure planning and record transformations ==========

def build_schema() -> dict[str, Any]:
    """Build an explicit three-key schema matching the documented tier bound.

    Args:
        None: Required key names and maximum items are local protocol constants.
    Returns:
        dict[str, Any]: JSON Schema with three required string-array properties.
    Raises:
        RuntimeError: No normal runtime exception is expected.
    """
    # 步骤1: Do not invent unrecorded minimum counts or minimum string lengths.
    properties = {tier: {"type": "array", "items": {"type": "string"},
                         "maxItems": MAXIMUM_ITEMS_PER_TIER} for tier in REQUIRED_NOTE_KEYS}
    return {"type": "object", "properties": properties,
            "required": list(REQUIRED_NOTE_KEYS), "additionalProperties": False}


def select_length_failures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select initial length failures without retrying normally stopped bad JSON.

    Args:
        records (list[dict[str, Any]]): Original first-pass generation rows.
    Returns:
        list[dict[str, Any]]: Selected records in unchanged original order.
    Raises:
        ValueError: Indexes duplicate, rows were already repaired, or the cap differs.
    """
    # 步骤1: Reject repeat repair and duplicate source indexes before generation.
    indexes = [row["source_index"] for row in records]
    if len(indexes) != len(set(indexes)) or any("repair_metadata" in row for row in records):
        raise ValueError("选择回退失败: 索引重复或输入已修复, 上下文: first-pass generations")
    selected = [row for row in records if row.get("finish_reason") == "length"]
    for row in selected:
        # 步骤2: Bind this reconstruction to the documented 256-token first pass.
        if row.get("output_token_count") != 256:
            raise ValueError(f"选择回退失败: 首轮上限不是256, 上下文: index={row['source_index']}")
        messages = row.get("messages", [])
        if [item.get("role") for item in messages] != ["system", "user"]:
            raise ValueError(f"选择回退失败: 输入必须仅含system/user, 上下文: index={row['source_index']}")
    return selected


def deduplicate_response(raw_response: str) -> tuple[str, int]:
    """Remove casefold-identical strings per tier while preserving the first form.

    Args:
        raw_response (str): Raw decoded fallback response supplied by the engine.
    Returns:
        tuple[str, int]: Processed JSON text and removed count; invalid text unchanged.
    Raises:
        RuntimeError: No normal runtime exception is expected for model text.
    """
    # 步骤1: Preserve invalid responses rather than repairing missing content by hand.
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return raw_response, 0
    if not parse_prediction(raw_response).has_valid_schema:
        return raw_response, 0
    removed = 0
    for tier in REQUIRED_NOTE_KEYS:
        seen = set()
        retained = []
        # 步骤2: Exact casefold equality does not merge synonyms or cross-tier terms.
        for term in payload[tier]:
            if term.casefold() in seen:
                removed += 1
            else:
                retained.append(term)
                seen.add(term.casefold())
        payload[tier] = retained
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), removed


def update_record(original: dict[str, Any], raw_response: str, tokens: int,
                  finish_reason: str, seed: int) -> dict[str, Any]:
    """Create an auditable repaired copy while preserving input and reference data.

    Args:
        original (dict[str, Any]): One selected first-pass archived generation row.
        raw_response (str): New raw fallback text supplied by an inference caller.
        tokens (int): Engine output-token count before post-processing.
        finish_reason (str): Engine finish reason, including unsuccessful truncation.
        seed (int): Explicit new seed chosen for this reconstruction.
    Returns:
        dict[str, Any]: Independent record with parser flags and reconstruction metadata.
    Raises:
        ValueError: Output token count is negative.
    """
    # 步骤1: Keep originals intact and preserve raw engine text before deduplication.
    if tokens < 0:
        raise ValueError(f"更新记录失败: token数为负, 上下文: tokens={tokens}")
    response, removed = deduplicate_response(raw_response)
    parsed = parse_prediction(response)
    record = copy.deepcopy(original)
    record.update({"raw_response": response, "parsed_json": parsed.parsed_json,
                   "canonical_notes": parsed.canonical_notes, "strict_json": parsed.is_strict_json,
                   "recoverable_json": parsed.is_recoverable_json, "required_keys": parsed.has_required_keys,
                   "exact_keys": parsed.has_exact_keys, "valid_schema": parsed.has_valid_schema,
                   "output_token_count": tokens, "finish_reason": finish_reason,
                   "hit_max_new_tokens": tokens >= MAXIMUM_NEW_TOKENS or finish_reason == "length"})
    # 步骤2: Label this as newly reconstructed output, never historical repair24.
    record["repair_metadata"] = {"policy": "length_failure_structured_json_fallback_v1",
        "implementation_status": RECONSTRUCTION_STATUS, "raw_model_response": raw_response,
        "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.05, "frequency_penalty": 0.10,
        "max_new_tokens": MAXIMUM_NEW_TOKENS, "maximum_items_per_tier": MAXIMUM_ITEMS_PER_TIER,
        "exact_duplicate_items_removed": removed, "reconstruction_seed": seed}
    return record


def merge_records(originals: list[dict[str, Any]], repaired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace selected records by source index without changing other records.

    Args:
        originals (list[dict[str, Any]]): Complete ordered first-pass generation rows.
        repaired (list[dict[str, Any]]): Exactly one reconstructed row per selected index.
    Returns:
        list[dict[str, Any]]: Complete ordered rows; unselected rows are unchanged.
    Raises:
        ValueError: Repaired indexes duplicate, differ from selection, or labels changed.
    """
    # 步骤1: Require complete coverage, not silent skipping of failed retry outputs.
    expected = {row["source_index"]: row for row in select_length_failures(originals)}
    replacements = {row["source_index"]: row for row in repaired}
    if len(replacements) != len(repaired) or set(replacements) != set(expected):
        raise ValueError("合并回退失败: 替换索引不完整或重复, 上下文: reconstructed records")
    for index, record in replacements.items():
        # 步骤2: Neither prompt nor reference may be replaced during technical fallback.
        for field in ("messages", "reference", "row_id", "scent_information_level", "model_name", "model_path"):
            if record.get(field) != expected[index].get(field):
                raise ValueError(f"合并回退失败: 源字段改变, 上下文: index={index}, field={field}")
    return [replacements.get(row["source_index"], row) for row in originals]


# ========== Explicit optional GPU execution ==========

def generate_repaired_records(selected: list[dict[str, Any]], model_directory: Path,
                              seed: int) -> list[dict[str, Any]]:
    """Run the documented fallback only after explicit CLI execution consent.

    Args:
        selected (list[dict[str, Any]]): Validated length failures with two input turns.
        model_directory (Path): Local model files verified by the execution entry.
        seed (int): Explicit reconstruction seed, not claimed as recovered history.
    Returns:
        list[dict[str, Any]]: One auditable new generation record per selected input.
    Raises:
        RuntimeError: Dependencies, model access, prompt rendering, or inference fail.
    """
    # 步骤1: GPU imports stay inside this explicit execution-only function.
    try:
        if importlib.metadata.version("vllm") != "0.24.0":
            raise ValueError("This reconstruction targets vLLM 0.24.0; do not silently change its API/backend.")
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from eval_qwen35_4b_chat_l3_unsloth import collect_stop_token_ids
        tokenizer = AutoTokenizer.from_pretrained(str(model_directory), trust_remote_code=False)
        prompts = [tokenizer.apply_chat_template(row["messages"], tokenize=False,
                   add_generation_prompt=True, enable_thinking=False) for row in selected]
        parameters = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512,
            repetition_penalty=1.05, frequency_penalty=0.10, seed=seed,
            stop_token_ids=collect_stop_token_ids(tokenizer),
            structured_outputs=StructuredOutputsParams(json=build_schema()))
        engine = LLM(model=str(model_directory), dtype="bfloat16", max_model_len=4096,
                     gpu_memory_utilization=0.82, seed=seed, trust_remote_code=False)
        outputs = engine.generate(prompts, parameters, use_tqdm=True)
        return decode_outputs(selected, outputs, tokenizer, seed)
    except Exception as error:
        raise RuntimeError(f"重建回退生成失败: {error}, 上下文: selected={len(selected)}") from error


def decode_outputs(selected: list[dict[str, Any]], outputs: list[Any], tokenizer: Any,
                   seed: int) -> list[dict[str, Any]]:
    """Decode engine outputs including failures, preserving one row per input.

    Args:
        selected (list[dict[str, Any]]): Original selected rows in engine prompt order.
        outputs (list[Any]): vLLM request outputs from an explicit inference call.
        tokenizer (Any): The tokenizer used for model prompts and decoding.
        seed (int): Reconstruction seed supplied by the CLI caller.
    Returns:
        list[dict[str, Any]]: Updated records, including empty failed completions.
    Raises:
        ValueError: Engine output count differs from selected input count.
        RuntimeError: Token decoding fails.
    """
    # 步骤1: Keep failed completions as empty outputs instead of removing samples.
    if len(outputs) != len(selected):
        raise ValueError("解码回退失败: 输出数量不同, 上下文: selected/outputs")
    records = []
    try:
        for original, output in zip(selected, outputs, strict=True):
            candidate = output.outputs[0] if output.outputs else None
            tokens = list(candidate.token_ids) if candidate else []
            raw_response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            finish = str(candidate.finish_reason) if candidate else "missing_output"
            records.append(update_record(original, raw_response, len(tokens), finish, seed))
    except Exception as error:
        raise RuntimeError(f"解码回退失败: {error}, 上下文: reconstructed outputs") from error
    return records


def build_parser() -> argparse.ArgumentParser:
    """Expose a planning-default CLI with separate opt-in model execution.

    Args:
        None: Actual input values are provided by the command-line caller.
    Returns:
        argparse.ArgumentParser: Parser requiring input, new output and model role.
    Raises:
        RuntimeError: No normal runtime exception is expected.
    """
    # 步骤1: No model download or load can occur without --execute and explicit paths.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("base", "finetuned"), default="finetuned")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--execute", action="store_true")
    return parser


def execute_and_save(arguments: argparse.Namespace, originals: list[dict[str, Any]],
                     selected: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    """Verify a supplied model and save reconstruction outputs in a new run.

    Args:
        arguments (argparse.Namespace): Explicit execute request with seed/model path.
        originals (list[dict[str, Any]]): Complete original first-pass rows.
        selected (list[dict[str, Any]]): Planned length failures selected from originals.
        plan (dict[str, Any]): Provenance and schema generated before execution.
    Returns:
        None: Writes new config, original-failure, repaired and merged JSONL files.
    Raises:
        ValueError: Seed/model path are missing, model identity differs, or hashes fail.
        RuntimeError: Inference or fresh output creation fails.
    """
    # 步骤1: The original seed is unknown; require a conscious new choice.
    if arguments.seed is None or arguments.model_dir is None:
        raise ValueError("执行回退失败: 必须显式给出--seed和--model-dir, 上下文: reconstructed execution")
    labels = {row.get("model_name", "") for row in originals}
    if len(labels) != 1 or any(("Finetuned" in label) != (arguments.role == "finetuned") for label in labels):
        raise ValueError("执行回退失败: role与输入模型身份不同, 上下文: original model labels")
    verify_files(arguments.model_dir, REPOSITORY_ROOT / f"configs/{arguments.role}_model_files.json")
    repaired = generate_repaired_records(selected, arguments.model_dir, arguments.seed) if selected else []
    for record in repaired:
        record["repair_metadata"]["reconstruction_model_directory"] = str(arguments.model_dir.resolve())
    merged = merge_records(originals, repaired)
    # 步骤2: Preserve original and new text separately; do not alter historical results.
    create_output_directory(arguments.output_dir)
    generation_directory = arguments.output_dir / "generations"
    create_output_directory(generation_directory)
    write_json(arguments.output_dir / "config.json", {**plan, "execution_performed": True,
               "execution_model_directory": str(arguments.model_dir.resolve()),
               "valid_repaired_count": sum(row["valid_schema"] for row in repaired)})
    write_records(generation_directory / f"{arguments.role}_original_length_failures.jsonl", selected)
    write_records(generation_directory / f"{arguments.role}_repaired_length_failures.jsonl", repaired)
    write_records(generation_directory / f"{arguments.role}.jsonl", merged)


def main() -> None:
    """Print a plan by default; execute only when the caller explicitly opts in.

    Args:
        None: CLI supplies original generation path and reconstruction choices.
    Returns:
        None: Prints a plan or writes a separate reconstructed run.
    Raises:
        ValueError: Count, output safety, or execution settings are invalid.
        RuntimeError: Input reading, hashing, or explicit generation fails.
    """
    # 步骤1: Check that the result cannot replace the original generation folder.
    arguments = build_parser().parse_args()
    source_directory = arguments.input_generations.resolve().parent
    protected = source_directory.parent if source_directory.name == "generations" else source_directory
    if arguments.output_dir.exists() or arguments.output_dir.resolve().is_relative_to(protected):
        raise ValueError("规划回退失败: 输出必须是原运行目录以外的新目录, 上下文: output-dir")
    originals = read_records(arguments.input_generations)
    selected = select_length_failures(originals)
    if arguments.expected_count is not None and len(selected) != arguments.expected_count:
        raise ValueError(f"规划回退失败: 数量不同, 上下文: actual={len(selected)}")
    # 步骤2: Planning never loads a tokenizer, model, or inference engine.
    plan = {"status": RECONSTRUCTION_STATUS, "execution_performed": False,
            "input_sha256": calculate_sha256(arguments.input_generations),
            "selected_source_indexes": [row["source_index"] for row in selected],
            "selected_count": len(selected), "schema": build_schema(), "seed": arguments.seed,
            "historical_seed_unknown": True, "structured_output_api": "StructuredOutputsParams(json=...)",
            "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.05,
            "frequency_penalty": 0.10, "max_new_tokens": 512}
    if arguments.execute:
        execute_and_save(arguments, originals, selected, plan)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 步骤1: Example: python scripts/reconstruct_length_fallback.py --help
    main()
