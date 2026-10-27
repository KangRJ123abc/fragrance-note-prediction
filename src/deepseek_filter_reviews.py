"""
Batch-screen and anonymize perfume reviews with DeepSeek.

Default workflow:
1. Read data/全部香水评论数据_合并.xlsx.
2. Rule-filter rows with complete top/middle/base notes and review length >= 20 words.
3. De-duplicate exact same comments within each perfume.
4. Select at most 100 reviews per perfume with deterministic even spacing.
5. Send batches of 15 rows to DeepSeek for usability screening and anonymization.
6. Save JSONL results with checkpoint-friendly append semantics.

Run from project root:
    python src/deepseek_filter_reviews.py

Resume an interrupted run:
    python src/deepseek_filter_reviews.py --run-dir src/outputs/deepseek_review_filter/<run_id>

The script reads the configured API key environment variable. It never logs the
full key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "全部香水评论数据_合并.xlsx"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "src" / "outputs" / "deepseek_review_filter"

EXPECTED_COLUMNS = ["序号", "目标URL", "香水名", "前调", "中调", "后调", "额外HTML", "评论"]

JSONL_WRITE_LOCK = threading.Lock()

# ========== Provider Presets ==========

# ------ OpenAI-Compatible Provider Defaults ------
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "ark": {
        "api_key_env": "ARK_API_KEY",
        "api_mode": "responses",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "deepseek-v4-flash-260425",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_mode": "chat",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
    },
}

NOTE_NOISE_PATTERN = re.compile(
    r"vote for ingredients|rate only the most important ingredients|main accords|perfume pyramid",
    re.IGNORECASE,
)

REJECTION_REASONS = {
    "valid",
    "too_short",
    "no_scent_information",
    "subjective_only",
    "price_or_performance_only",
    "bottle_or_packaging_only",
    "off_topic",
    "comparison_only_without_scent",
    "garbled_or_scraping_error",
    "non_english_or_unusable",
    "missing_notes",
    "dirty_note_fields",
    "identity_leak_unfixable",
}

REJECTION_REASON_ALIASES = {
    "performance_only": "price_or_performance_only",
}

PROMPT_VERSION = "review_screen_anonymize_v1"

SYSTEM_PROMPT = """
You are a strict data cleaning assistant for a perfume review dataset.
You must evaluate whether each review is useful for supervised fine-tuning.
You must anonymize perfume identity leaks when possible.
Return only valid JSON. Do not return markdown or commentary.
""".strip()

USER_PROMPT_TEMPLATE = """
Task:
For each item, decide whether the row can be used to train a model that infers top_notes, middle_notes, and base_notes from a perfume review.

Important boundaries:
- Do NOT complete or modify top_notes, middle_notes, or base_notes.
- Do NOT invent missing fragrance information.
- Use the provided note fields only to check row format and context.
- Your main job is review usability screening and anonymization.

Keep a row only if:
- The review contains concrete scent, olfactory, material, note, accord, texture, opening, drydown, or wear-experience information.
- The review is not only price, bottle, compliment, longevity, projection, purchase intent, rating, or generic preference.
- The row is not garbled, scraping noise, spam, or unusable.
- Product/brand/perfume identity leaks can be removed without destroying the scent description.

Identity anonymization rules:
- Remove or generalize exact perfume names, brand names, flankers, and direct comparisons that identify a specific product.
- Preserve scent descriptions, note terms, temporal development, qualitative impressions, and wearer experience.
- If identity leakage cannot be fixed without removing nearly all useful scent information, set keep=false and rejection_reason="identity_leak_unfixable".
- Do not make the review more positive, more negative, longer, or more specific than the original.

Return exactly one JSON object with this schema:
{
  "items": [
    {
      "row_id": 123,
      "format_ok": true,
      "comment_usable": true,
      "keep": true,
      "rejection_reason": "valid",
      "scent_information_level": 3,
      "has_identity_leak": false,
      "anonymized_review": "...",
      "removed_entities": [],
      "notes_issue": null
    }
  ]
}

Field rules:
- row_id must match the input row_id.
- format_ok is false if note fields are missing, dirty, or structurally unusable.
- comment_usable is false if the review is not useful for scent inference.
- keep must be true only when both format_ok and comment_usable are true.
- rejection_reason must be one of:
  valid, too_short, no_scent_information, subjective_only, price_or_performance_only,
  bottle_or_packaging_only, off_topic, comparison_only_without_scent,
  garbled_or_scraping_error, non_english_or_unusable, missing_notes,
  dirty_note_fields, identity_leak_unfixable.
- scent_information_level is 0, 1, 2, or 3:
  0 = no useful scent information;
  1 = weak or vague scent information;
  2 = usable scent information;
  3 = rich scent information.
- anonymized_review must be empty when keep=false.
- notes_issue is null if note fields are usable; otherwise a short string.

Input items:
__ITEMS_JSON__
""".strip()


@dataclass
class Candidate:
    row_id: int
    source_index: Any
    source_url: str
    perfume_name: str
    top_notes: list[str]
    middle_notes: list[str]
    base_notes: list[str]
    review: str
    review_word_count: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the review filtering script.

    This function only defines command-line surface area. Provider-specific
    defaults are filled later by ``apply_provider_preset`` so that users can
    switch between Ark and direct DeepSeek access without pasting long commands.

    Args:
        None: Arguments are read from ``sys.argv`` by ``argparse``.

    Returns:
        argparse.Namespace: Parsed configuration. Provider-dependent fields may
        be ``None`` until ``apply_provider_preset`` is called.

    Raises:
        SystemExit: Raised by ``argparse`` when command-line validation fails.
    """

    parser = argparse.ArgumentParser(description="Filter and anonymize perfume reviews with an OpenAI-compatible LLM provider.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input Excel file.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for new runs.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing run directory to resume.")
    parser.add_argument("--rebuild-candidates", action="store_true", help="Rebuild candidates even if candidates.jsonl exists.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build candidate files; do not call API.")
    parser.add_argument("--min-words", type=int, default=20, help="Minimum review word count before LLM screening.")
    parser.add_argument("--cap-per-perfume", type=int, default=100, help="Maximum candidate rows per perfume.")
    parser.add_argument("--batch-size", type=int, default=15, help="Rows per provider request.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent provider batch workers.")
    parser.add_argument("--selection-mode", choices=["even", "first", "random"], default="even")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit-batches", type=int, default=None, help="Process only N pending batches for smoke tests.")
    parser.add_argument("--provider", choices=sorted(PROVIDER_PRESETS), default="ark", help="Provider preset. Default uses Volcengine Ark.")
    parser.add_argument("--api-key", default=None, help="API key. Prefer the variable named by --api-key-env.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable that stores the API key. Defaults come from --provider.")
    parser.add_argument("--api-mode", choices=["chat", "responses"], default=None, help="OpenAI-compatible endpoint mode. Defaults come from --provider.")
    parser.add_argument("--base-url", default=None, help="Provider base URL. Defaults come from --provider.")
    parser.add_argument("--model", default=None, help="Model name. Defaults come from --provider.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--progress-interval", type=float, default=30.0, help="Seconds between heartbeat progress logs.")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--no-response-format", action="store_true", help="Disable JSON response_format.")
    parser.add_argument("--split-on-failure", action="store_true", default=True)
    return parser.parse_args()


def apply_provider_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing provider configuration from a named provider preset.

    The command line intentionally keeps ``--api-key-env``, ``--api-mode``,
    ``--base-url``, and ``--model`` optional. This function supplies those
    values from ``PROVIDER_PRESETS`` unless the user explicitly overrides them.

    Args:
        args (argparse.Namespace): Parsed command-line arguments produced by
            ``parse_args``. The namespace must contain a ``provider`` attribute.

    Returns:
        argparse.Namespace: The same namespace with provider-dependent fields
        populated. Typical values are ``ARK_API_KEY`` and ``responses`` for Ark.

    Raises:
        ValueError: ``provider`` is not known, or a required provider field is
        still empty after preset application.
    """

    # ---------- Provider Preset Lookup ----------

    # 步骤1: 先根据 provider 找默认值，避免用户每次粘贴长 base_url/model 参数。
    provider_preset = PROVIDER_PRESETS.get(args.provider)
    if provider_preset is None:
        raise ValueError(f"unsupported provider={args.provider}")

    # ---------- Conditional Default Assignment ----------

    # 步骤2: 只填补用户没有显式传入的字段，保留命令行覆盖能力。
    if args.api_key_env is None:
        args.api_key_env = provider_preset["api_key_env"]
    if args.api_mode is None:
        args.api_mode = provider_preset["api_mode"]
    if args.base_url is None:
        args.base_url = provider_preset["base_url"]
    if args.model is None:
        args.model = provider_preset["model"]

    # ---------- Required Field Validation ----------

    # 步骤3: 在真正发起网络请求前报出清晰错误，避免跑到一半才发现配置缺失。
    missing_field_names = [
        field_name
        for field_name in ("api_key_env", "api_mode", "base_url", "model")
        if not getattr(args, field_name)
    ]
    if missing_field_names:
        raise ValueError(f"provider preset is incomplete: provider={args.provider}, missing={missing_field_names}")

    return args


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = args.run_dir
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    run_dir = args.output_root / now_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("deepseek_filter_reviews")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def masked(value: str | None) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "<masked>"
    return f"{value[:3]}...{value[-4:]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file.

    This function serializes a single dictionary and appends it as one line. It uses
    a process-local lock so concurrent worker threads cannot interleave writes to
    failure or result files.

    Args:
        path (Path): Target JSONL path produced by the screening pipeline.
        record (dict[str, Any]): One structured result record produced by the pipeline.

    Returns:
        None: The record is appended to disk.

    Raises:
        RuntimeError: The underlying file operation or JSON serialization fails.
    """

    try:
        # ---------- ??1: ???????????????????? ----------
        serialized_record = json.dumps(record, ensure_ascii=False) + "\n"

        # ---------- ??2: ??????????????? worker ?? JSONL ----------
        path.parent.mkdir(parents=True, exist_ok=True)
        with JSONL_WRITE_LOCK:
            with path.open("a", encoding="utf-8") as file:
                file.write(serialized_record)
                file.flush()
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"??JSONL??: {error}, ???: path={path}") from error


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def clean_cell(value: Any) -> str:
    return "" if value is None else str(value).strip()


def split_clean_notes(value: str) -> list[str]:
    notes: list[str] = []
    for raw_part in re.split(r",|/|;", value or ""):
        term = raw_part.strip()
        if not term:
            continue
        if NOTE_NOISE_PATTERN.search(term):
            continue
        notes.append(term)
    return notes


def even_sample(items: list[Candidate], cap: int) -> list[Candidate]:
    if cap <= 0 or len(items) <= cap:
        return items
    if cap == 1:
        return [items[0]]
    indices = [round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)]
    sampled: list[Candidate] = []
    used: set[int] = set()
    for index in indices:
        if index in used:
            continue
        sampled.append(items[index])
        used.add(index)
    cursor = 0
    while len(sampled) < cap and cursor < len(items):
        if cursor not in used:
            sampled.append(items[cursor])
            used.add(cursor)
        cursor += 1
    return sampled


def select_candidates(items: list[Candidate], cap: int, mode: str, seed: int) -> list[Candidate]:
    if cap <= 0 or len(items) <= cap:
        return items
    if mode == "first":
        return items[:cap]
    if mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(items, cap), key=lambda item: item.row_id)
    return even_sample(items, cap)


def validate_headers(headers: list[str]) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in headers]
    if missing:
        raise ValueError(f"Input workbook missing required columns: {missing}. Found columns: {headers}")


def prepare_candidates(args: argparse.Namespace, run_dir: Path, logger: logging.Logger) -> Path:
    candidate_path = run_dir / "prepared" / "candidates.jsonl"
    summary_path = run_dir / "prepared" / "prefilter_summary.json"
    if candidate_path.exists() and not args.rebuild_candidates:
        logger.info("Using existing candidates: %s", candidate_path)
        return candidate_path

    logger.info("Preparing candidates from: %s", args.input)
    workbook = openpyxl.load_workbook(args.input, read_only=True, data_only=True)
    worksheet = workbook[workbook.sheetnames[0]]
    headers = [worksheet.cell(1, column).value for column in range(1, worksheet.max_column + 1)]
    headers = [str(header).strip() if header is not None else "" for header in headers]
    validate_headers(headers)
    column_index = {header: index for index, header in enumerate(headers)}

    per_perfume: dict[str, list[Candidate]] = defaultdict(list)
    seen_comment_by_perfume: set[tuple[str, str]] = set()
    counters: Counter[str] = Counter()

    for excel_row_id, row in enumerate(worksheet.iter_rows(min_row=2, values_only=True), start=2):
        counters["rows_total"] += 1
        source_index = row[column_index["序号"]]
        source_url = clean_cell(row[column_index["目标URL"]])
        perfume_name = clean_cell(row[column_index["香水名"]])
        top_raw = clean_cell(row[column_index["前调"]])
        middle_raw = clean_cell(row[column_index["中调"]])
        base_raw = clean_cell(row[column_index["后调"]])
        review = clean_cell(row[column_index["评论"]])

        if not (top_raw and middle_raw and base_raw):
            counters["drop_missing_notes"] += 1
            continue

        top_notes = split_clean_notes(top_raw)
        middle_notes = split_clean_notes(middle_raw)
        base_notes = split_clean_notes(base_raw)
        if not (top_notes and middle_notes and base_notes):
            counters["drop_dirty_or_empty_cleaned_notes"] += 1
            continue

        review_words = word_count(review)
        if review_words < args.min_words:
            counters["drop_short_review"] += 1
            continue

        normalized_review = normalize_text(review)
        duplicate_key = (perfume_name, normalized_review)
        if duplicate_key in seen_comment_by_perfume:
            counters["drop_duplicate_same_perfume_review"] += 1
            continue
        seen_comment_by_perfume.add(duplicate_key)

        per_perfume[perfume_name].append(
            Candidate(
                row_id=excel_row_id,
                source_index=source_index,
                source_url=source_url,
                perfume_name=perfume_name,
                top_notes=top_notes,
                middle_notes=middle_notes,
                base_notes=base_notes,
                review=review,
                review_word_count=review_words,
            )
        )
        counters["candidate_before_cap"] += 1

    selected: list[Candidate] = []
    perfume_counts_before = {name: len(items) for name, items in per_perfume.items()}
    for perfume_name in sorted(per_perfume):
        items = per_perfume[perfume_name]
        selected.extend(select_candidates(items, args.cap_per_perfume, args.selection_mode, args.seed))
    selected.sort(key=lambda item: item.row_id)
    counters["candidate_after_cap"] = len(selected)

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("w", encoding="utf-8") as file:
        for item in selected:
            file.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(args.input),
        "min_words": args.min_words,
        "cap_per_perfume": args.cap_per_perfume,
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "counters": dict(counters),
        "perfumes_before_cap": len(per_perfume),
        "perfumes_after_cap": len({item.perfume_name for item in selected}),
        "top_perfumes_before_cap": sorted(
            perfume_counts_before.items(), key=lambda kv: kv[1], reverse=True
        )[:30],
    }
    write_json(summary_path, summary)
    logger.info("Prepared %s candidates at %s", len(selected), candidate_path)
    logger.info("Prefilter counters: %s", dict(counters))
    return candidate_path


def load_candidates(path: Path) -> list[Candidate]:
    candidates = []
    for record in read_jsonl(path):
        candidates.append(Candidate(**record))
    return candidates


def build_messages(batch: list[Candidate]) -> list[dict[str, str]]:
    items = [
        {
            "row_id": item.row_id,
            "perfume_name": item.perfume_name,
            "top_notes": item.top_notes,
            "middle_notes": item.middle_notes,
            "base_notes": item.base_notes,
            "review": item.review,
        }
        for item in batch
    ]
    user_prompt = USER_PROMPT_TEMPLATE.replace(
        "__ITEMS_JSON__", json.dumps(items, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response content")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("response does not contain a JSON object")


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def validate_item(raw: dict[str, Any], expected_ids: set[int]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"item is not an object: {raw!r}")
    row_id = int(raw.get("row_id"))
    if row_id not in expected_ids:
        raise ValueError(f"unexpected row_id={row_id}, expected one of {sorted(expected_ids)}")

    rejection_reason = str(raw.get("rejection_reason", "")).strip()
    rejection_reason = REJECTION_REASON_ALIASES.get(rejection_reason, rejection_reason)
    if rejection_reason not in REJECTION_REASONS:
        raise ValueError(f"invalid rejection_reason for row_id={row_id}: {rejection_reason!r}")

    level = int(raw.get("scent_information_level", 0))
    if level < 0 or level > 3:
        raise ValueError(f"invalid scent_information_level for row_id={row_id}: {level}")

    removed_entities = raw.get("removed_entities", [])
    if not isinstance(removed_entities, list):
        removed_entities = [str(removed_entities)]

    normalized = {
        "row_id": row_id,
        "format_ok": normalize_bool(raw.get("format_ok", False)),
        "comment_usable": normalize_bool(raw.get("comment_usable", False)),
        "keep": normalize_bool(raw.get("keep", False)),
        "rejection_reason": rejection_reason,
        "scent_information_level": level,
        "has_identity_leak": normalize_bool(raw.get("has_identity_leak", False)),
        "anonymized_review": str(raw.get("anonymized_review", "") or ""),
        "removed_entities": [str(entity) for entity in removed_entities],
        "notes_issue": raw.get("notes_issue"),
    }

    if normalized["keep"] and normalized["rejection_reason"] != "valid":
        raise ValueError(f"row_id={row_id} has keep=true but rejection_reason={rejection_reason}")
    if not normalized["keep"] and normalized["rejection_reason"] == "valid":
        raise ValueError(f"row_id={row_id} has keep=false but rejection_reason=valid")
    if normalized["keep"] and not normalized["anonymized_review"].strip():
        raise ValueError(f"row_id={row_id} has keep=true but empty anonymized_review")
    if not normalized["keep"]:
        normalized["anonymized_review"] = ""
    return normalized


def parse_response(content: str, expected_ids: set[int]) -> list[dict[str, Any]]:
    obj = extract_json_object(content)
    items = obj.get("items")
    if not isinstance(items, list):
        raise ValueError("response JSON missing items list")
    normalized_items = [validate_item(item, expected_ids) for item in items]
    returned_ids = {item["row_id"] for item in normalized_items}
    if returned_ids != expected_ids:
        missing = sorted(expected_ids - returned_ids)
        extra = sorted(returned_ids - expected_ids)
        raise ValueError(f"row_id mismatch: missing={missing}, extra={extra}")
    return sorted(normalized_items, key=lambda item: item["row_id"])


def make_client(args: argparse.Namespace) -> OpenAI:
    """Create an OpenAI SDK client from command-line configuration.

    The API key is read from ``args.api_key`` only when explicitly supplied.
    Normal runs should rely on the environment variable named by
    ``args.api_key_env`` so the command line does not expose or require the key.

    Args:
        args (argparse.Namespace): Parsed and provider-expanded configuration.
            ``api_key_env``, ``base_url``, and ``timeout`` are used here.

    Returns:
        OpenAI: Configured SDK client. Typical values target either Ark
        Responses API or direct DeepSeek Chat Completions.

    Raises:
        EnvironmentError: The selected API key environment variable is missing.
    """

    # ---------- API Key Resolution ----------

    # 步骤1: 优先使用显式传入的 key；常规运行则从 provider 对应环境变量读取。
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"{args.api_key_env} is not set. Set it in the environment or pass --api-key.")

    # ---------- Client Construction ----------

    # 步骤2: 复用 OpenAI SDK 的兼容客户端，让 Ark 和 DeepSeek 共享后续调用逻辑。
    return OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)


def extract_response_text(response: Any) -> str:
    """Extract plain text from an OpenAI Responses API object.

    The OpenAI SDK exposes ``output_text`` for normal Responses API calls. Some
    compatible gateways may instead return nested ``output`` message objects, so
    this function keeps a small fallback parser for text fragments.

    Args:
        response (Any): Response object returned by ``client.responses.create``.

    Returns:
        str: Concatenated response text. Typical value is a JSON object string.

    Raises:
        ValueError: No text content can be found in the response object.
    """

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    text_parts: list[str] = []
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            content_text = getattr(content_item, "text", None)
            if content_text:
                text_parts.append(str(content_text))
    if text_parts:
        return "".join(text_parts)
    raise ValueError("responses API returned no text content")


def call_deepseek_with_chat(client: OpenAI, args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
    """Call an OpenAI-compatible Chat Completions endpoint.

    Args:
        client (OpenAI): Configured OpenAI SDK client produced by ``make_client``.
        args (argparse.Namespace): Parsed command-line configuration.
        messages (list[dict[str, str]]): System and user messages built for one
            review-screening batch.

    Returns:
        str: Assistant message content, expected to be a JSON object string.

    Raises:
        Exception: Propagates SDK, network, or gateway errors to the retry layer.
    """

    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if not args.no_response_format:
        kwargs["response_format"] = {"type": "json_object"}
    if args.thinking == "enabled":
        kwargs["reasoning_effort"] = args.reasoning_effort
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def call_deepseek_with_responses(client: OpenAI, args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
    """Call an OpenAI-compatible Responses API endpoint.

    This mode is intended for gateways such as Volcengine Ark, whose DeepSeek
    examples use ``client.responses.create``. The filtering task does not need
    web-search tools, so only the conversation input and optional JSON response
    format are sent.

    Args:
        client (OpenAI): Configured OpenAI SDK client produced by ``make_client``.
        args (argparse.Namespace): Parsed command-line configuration.
        messages (list[dict[str, str]]): System and user messages built for one
            review-screening batch.

    Returns:
        str: Model response text, expected to be a JSON object string.

    Raises:
        Exception: Propagates SDK, network, gateway, or empty-response errors to
            the retry layer.
    """

    kwargs: dict[str, Any] = {
        "model": args.model,
        "input": messages,
        "temperature": args.temperature,
        "max_output_tokens": args.max_tokens,
    }
    if not args.no_response_format:
        kwargs["text"] = {"format": {"type": "json_object"}}
    response = client.responses.create(**kwargs)
    return extract_response_text(response)


def call_deepseek(client: OpenAI, args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
    """Dispatch one batch request to the selected OpenAI-compatible API mode.

    Args:
        client (OpenAI): Configured OpenAI SDK client produced by ``make_client``.
        args (argparse.Namespace): Parsed command-line configuration containing
            ``api_mode``.
        messages (list[dict[str, str]]): System and user messages for one batch.

    Returns:
        str: Raw model text to be parsed as JSON.

    Raises:
        ValueError: ``api_mode`` is not recognized.
        Exception: Propagates provider call errors to the retry layer.
    """

    if args.api_mode == "chat":
        return call_deepseek_with_chat(client, args, messages)
    if args.api_mode == "responses":
        return call_deepseek_with_responses(client, args, messages)
    raise ValueError(f"unsupported api_mode={args.api_mode}")


def merge_candidate_result(candidate: Candidate, result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge one candidate row with one validated model screening result.

    The merged record is the canonical row written to ``screened_reviews.jsonl``.
    Provider metadata is included so mixed-provider resume runs can still be
    audited at row level.

    Args:
        candidate (Candidate): Original row selected from the Excel dataset.
        result (dict[str, Any]): Validated model output for the same ``row_id``.
        args (argparse.Namespace): Parsed and provider-expanded configuration
            containing ``provider``, ``api_mode``, and ``model``.

    Returns:
        dict[str, Any]: Serializable JSONL record combining source data,
        screening labels, anonymized review text, and provider metadata.

    Raises:
        KeyError: Required keys are missing from ``result``.
    """

    # ---------- Source Row Snapshot ----------

    # 步骤1: 先保留候选行原始字段，保证后续筛选结果可追溯到 Excel 来源。
    record = asdict(candidate)

    # ---------- Screening And Provider Metadata ----------

    # 步骤2: 再叠加模型筛选结果和供应商信息，便于混合续跑时逐条审计。
    record.update(
        {
            "prompt_version": PROMPT_VERSION,
            "provider": args.provider,
            "api_mode": args.api_mode,
            "model": args.model,
            "format_ok": result["format_ok"],
            "comment_usable": result["comment_usable"],
            "keep": result["keep"],
            "rejection_reason": result["rejection_reason"],
            "scent_information_level": result["scent_information_level"],
            "has_identity_leak": result["has_identity_leak"],
            "anonymized_review": result["anonymized_review"],
            "removed_entities": result["removed_entities"],
            "notes_issue": result["notes_issue"],
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return record


def save_result(record: dict[str, Any], run_dir: Path) -> None:
    append_jsonl(run_dir / "results" / "screened_reviews.jsonl", record)
    if record["keep"]:
        append_jsonl(run_dir / "results" / "kept_reviews.jsonl", record)
    else:
        append_jsonl(run_dir / "results" / "rejected_reviews.jsonl", record)


def batch_records(records: list[Candidate], batch_size: int) -> list[list[Candidate]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def update_checkpoint(run_dir: Path, state: dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "state" / "checkpoint.json", state)


def process_batch_once(
    client: OpenAI,
    args: argparse.Namespace,
    batch: list[Candidate],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    expected_ids = {item.row_id for item in batch}
    messages = build_messages(batch)
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            content = call_deepseek(client, args, messages)
            return parse_response(content, expected_ids)
        except Exception as error:  # noqa: BLE001 - keep batch processing resilient.
            last_error = error
            sleep_seconds = args.retry_sleep * (2 ** (attempt - 1)) + random.random()
            logger.warning(
                "Batch row_ids=%s failed attempt %s/%s: %s; sleeping %.1fs",
                sorted(expected_ids),
                attempt,
                args.max_retries,
                error,
                sleep_seconds,
            )
            if attempt < args.max_retries:
                time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def record_failed_batch(run_dir: Path, batch: list[Candidate], error: Exception) -> None:
    append_jsonl(
        run_dir / "results" / "failed_batches.jsonl",
        {
            "failed_at": datetime.now().isoformat(timespec="seconds"),
            "row_ids": [item.row_id for item in batch],
            "error": str(error),
            "items": [asdict(item) for item in batch],
        },
    )


def process_batch_resilient(
    client: OpenAI,
    args: argparse.Namespace,
    run_dir: Path,
    batch: list[Candidate],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    try:
        return process_batch_once(client, args, batch, logger)
    except Exception as error:  # noqa: BLE001
        if args.split_on_failure and len(batch) > 1:
            midpoint = len(batch) // 2
            logger.warning(
                "Splitting failed batch row_ids=%s into %s and %s",
                [item.row_id for item in batch],
                [item.row_id for item in batch[:midpoint]],
                [item.row_id for item in batch[midpoint:]],
            )
            return process_batch_resilient(client, args, run_dir, batch[:midpoint], logger) + process_batch_resilient(
                client, args, run_dir, batch[midpoint:], logger
            )
        logger.error("Batch row_ids=%s failed permanently: %s", [item.row_id for item in batch], error)
        record_failed_batch(run_dir, batch, error)
        return []


def process_batch_for_worker(
    args: argparse.Namespace,
    run_dir: Path,
    batch_index: int,
    batch: list[Candidate],
    logger: logging.Logger,
) -> tuple[int, list[Candidate], list[dict[str, Any]]]:
    """Process one DeepSeek batch inside a worker thread.

    The worker creates its own OpenAI-compatible client, calls the existing
    resilient batch processor, and returns the parsed item results to the main
    thread. Result files are still written by the main thread, while permanent
    failed batches are protected by the JSONL write lock.

    Args:
        args (argparse.Namespace): Parsed command-line configuration from
            ``parse_args``.
        run_dir (Path): Current run directory where results and state are stored.
        batch_index (int): One-based batch position assigned by the main thread.
        batch (list[Candidate]): Candidate rows to send in one DeepSeek request.
        logger (logging.Logger): Shared logger configured for the current run.

    Returns:
        tuple[int, list[Candidate], list[dict[str, Any]]]: Batch index, original
        batch, and parsed DeepSeek screening results. The result list is empty
        when all retry and split attempts fail.

    Raises:
        EnvironmentError: The DeepSeek API key is missing when the worker creates
        its client.
    """

    # ---------- ??1: ???????????????? HTTP ?? ----------
    worker_client = make_client(args)

    # ---------- ??2: ??????????????? batch ----------
    results = process_batch_resilient(worker_client, args, run_dir, batch, logger)
    return batch_index, batch, results


def save_results_and_update_state(
    run_dir: Path,
    state: dict[str, Any],
    candidate_by_id: dict[int, Candidate],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Persist parsed batch results and update in-memory counters.

    This function keeps result persistence in one place for both sequential and
    concurrent execution modes. It maps each parsed row back to the original
    candidate before appending to the screened, kept, or rejected JSONL files.

    Args:
        run_dir (Path): Current run directory where JSONL outputs are written.
        state (dict[str, Any]): Mutable checkpoint state created in ``main``.
        candidate_by_id (dict[int, Candidate]): Mapping from Excel row id to the
            source candidate object.
        results (list[dict[str, Any]]): Parsed and validated provider results.
        args (argparse.Namespace): Parsed and provider-expanded configuration
            used to add per-row provider metadata.

    Returns:
        None: Results are written to disk and counters are updated in ``state``.

    Raises:
        KeyError: A result references a row id that is not present in candidates.
        RuntimeError: Writing any JSONL output fails.
    """

    # ---------- ??1: ?????????????? ----------
    for result in results:
        candidate = candidate_by_id[result["row_id"]]
        record = merge_candidate_result(candidate, result, args)

        # ---------- ??2: ?????? kept/rejected ???? ----------
        save_result(record, run_dir)
        state["processed_this_run"] += 1
        if record["keep"]:
            state["kept_this_run"] += 1
        else:
            state["rejected_this_run"] += 1


def log_progress_snapshot(
    logger: logging.Logger,
    state: dict[str, Any],
    completed_batch_count: int,
    total_batch_count: int,
    pending_batch_count: int,
    started_monotonic_seconds: float,
) -> None:
    """Log a heartbeat snapshot for long concurrent runs.

    The snapshot is intentionally independent of warnings. It tells the user
    that the process is still alive even when no batch has completed recently
    because many provider requests are sleeping, retrying, or waiting for HTTP
    responses.

    Args:
        logger (logging.Logger): Run logger configured by ``setup_logging``.
        state (dict[str, Any]): Mutable checkpoint state with processed, kept,
            rejected, and failed counters.
        completed_batch_count (int): Number of futures that have completed in
            the current process.
        total_batch_count (int): Total number of planned batches this run.
        pending_batch_count (int): Number of futures still waiting to complete.
        started_monotonic_seconds (float): ``time.monotonic`` value captured at
            the beginning of batch processing.

    Returns:
        None: The function only writes one INFO log record.

    Raises:
        KeyError: Required keys are missing from ``state``.
    """

    # ---------- Progress Metric Calculation ----------

    # 步骤1: 使用 monotonic 计算速度，避免系统时间调整影响吞吐估算。
    elapsed_minutes = max((time.monotonic() - started_monotonic_seconds) / 60.0, 0.001)
    processed_this_run = int(state["processed_this_run"])
    total_processed_rows = int(state["processed_before_start"]) + processed_this_run
    rows_per_minute = processed_this_run / elapsed_minutes

    # ---------- Heartbeat Log ----------

    # 步骤2: 打印稳定的 INFO 心跳，帮助区分“没有完成项”和“日志级别被过滤”。
    logger.info(
        "Progress heartbeat: completed_batches=%s/%s pending_batches=%s processed_this_run=%s total_processed=%s "
        "kept=%s rejected=%s failed_batches=%s speed=%.1f rows/min",
        completed_batch_count,
        total_batch_count,
        pending_batch_count,
        processed_this_run,
        total_processed_rows,
        state["kept_this_run"],
        state["rejected_this_run"],
        state["failed_batches_seen_this_run"],
        rows_per_minute,
    )


def load_processed_ids(run_dir: Path) -> set[int]:
    processed = set()
    for record in read_jsonl(run_dir / "results" / "screened_reviews.jsonl"):
        if "row_id" in record:
            processed.add(int(record["row_id"]))
    return processed


def write_run_config(args: argparse.Namespace, run_dir: Path) -> None:
    """Write the resolved run configuration to ``config.json``.

    The saved file records provider, endpoint mode, model, and a masked key so
    that interrupted runs are auditable without leaking credentials.

    Args:
        args (argparse.Namespace): Parsed and provider-expanded configuration.
        run_dir (Path): Existing run directory where ``config.json`` is written.

    Returns:
        None: The function writes a JSON file as its side effect.

    Raises:
        OSError: File writing fails because the target directory or file is not
        writable.
    """

    # ---------- Serializable Run Metadata ----------

    # 步骤1: 记录 provider 解析后的最终值，方便恢复和排查跨供应商差异。
    config = {
        "created_or_resumed_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "input": str(args.input),
        "provider": args.provider,
        "api_mode": args.api_mode,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "api_key": masked(args.api_key or os.environ.get(args.api_key_env)),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "response_format": not args.no_response_format,
        "prompt_version": PROMPT_VERSION,
        "min_words": args.min_words,
        "cap_per_perfume": args.cap_per_perfume,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "progress_interval": args.progress_interval,
    }

    # ---------- Persist Configuration ----------

    # 步骤2: 统一通过 write_json 输出，保持现有 UTF-8/缩进格式。
    write_json(run_dir / "config.json", config)


def summarize_results(run_dir: Path) -> dict[str, Any]:
    records = read_jsonl(run_dir / "results" / "screened_reviews.jsonl")
    reasons = Counter(str(record.get("rejection_reason")) for record in records)
    levels = Counter(str(record.get("scent_information_level")) for record in records)
    keep_count = sum(1 for record in records if record.get("keep") is True)
    rejected_count = len(records) - keep_count
    failed_batches = read_jsonl(run_dir / "results" / "failed_batches.jsonl")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "processed_rows": len(records),
        "kept_rows": keep_count,
        "rejected_rows": rejected_count,
        "keep_rate": keep_count / len(records) if records else 0.0,
        "rejection_reasons": dict(reasons),
        "scent_information_levels": dict(levels),
        "failed_batches": len(failed_batches),
        "failed_row_count": sum(len(batch.get("row_ids", [])) for batch in failed_batches),
    }


def main() -> None:
    """Run the end-to-end review filtering workflow.

    The workflow prepares candidate rows, resumes already processed JSONL
    outputs, dispatches batches through the selected provider, and writes a
    final summary when all pending rows are done.

    Args:
        None: Runtime behavior comes from command-line arguments and environment
        variables such as ``ARK_API_KEY``.

    Returns:
        None: Results are written under the selected run directory.

    Raises:
        ValueError: Command-line values such as ``workers`` are invalid.
        EnvironmentError: The selected provider key is missing when API calls
        are required.
        Exception: Network, parsing, or file I/O errors may propagate after
        retry handling.
    """

    # ---------- Argument And Provider Setup ----------

    # 步骤1: 先解析命令行，再套用 provider 预设，让一行命令即可切换供应商。
    args = apply_provider_preset(parse_args())
    run_dir = ensure_run_dir(args)
    logger = setup_logging(run_dir)
    write_run_config(args, run_dir)

    logger.info("Run directory: %s", run_dir)
    logger.info(
        "Provider=%s model=%s api_mode=%s base_url=%s api_key_env=%s api_key=%s",
        args.provider,
        args.model,
        args.api_mode,
        args.base_url,
        args.api_key_env,
        masked(args.api_key or os.environ.get(args.api_key_env)),
    )

    candidate_path = prepare_candidates(args, run_dir, logger)
    candidates = load_candidates(candidate_path)
    logger.info("Loaded candidates: %s", len(candidates))
    if args.prepare_only:
        logger.info("Prepare-only mode enabled. Stop before API calls.")
        return

    processed_ids = load_processed_ids(run_dir)
    pending = [candidate for candidate in candidates if candidate.row_id not in processed_ids]
    logger.info("Already processed: %s; pending: %s", len(processed_ids), len(pending))
    if not pending:
        summary = summarize_results(run_dir)
        write_json(run_dir / "results" / "summary.json", summary)
        logger.info("No pending rows. Summary: %s", summary)
        return

    if args.workers < 1:
        raise ValueError(f"workers must be >= 1, got {args.workers}")
    if args.progress_interval <= 0:
        raise ValueError(f"progress_interval must be > 0, got {args.progress_interval}")

    client = make_client(args)
    batches = batch_records(pending, args.batch_size)
    if args.limit_batches is not None:
        batches = batches[: args.limit_batches]
        logger.info("Limit-batches active: processing only %s batches", len(batches))

    state = {
        "candidate_rows": len(candidates),
        "processed_before_start": len(processed_ids),
        "planned_pending_rows_this_run": sum(len(batch) for batch in batches),
        "planned_batches_this_run": len(batches),
        "workers": args.workers,
        "processed_this_run": 0,
        "kept_this_run": 0,
        "rejected_this_run": 0,
        "failed_batches_seen_this_run": 0,
    }
    update_checkpoint(run_dir, state)

    candidate_by_id = {candidate.row_id: candidate for candidate in candidates}
    if args.workers == 1:
        for batch_index, batch in enumerate(batches, start=1):
            row_ids = [item.row_id for item in batch]
            logger.info("Processing batch %s/%s row_ids=%s", batch_index, len(batches), row_ids)
            results = process_batch_resilient(client, args, run_dir, batch, logger)
            if not results:
                state["failed_batches_seen_this_run"] += 1
                update_checkpoint(run_dir, state)
                continue
            save_results_and_update_state(run_dir, state, candidate_by_id, results, args)
            update_checkpoint(run_dir, state)
    else:
        logger.info("Parallel processing enabled: workers=%s batches=%s", args.workers, len(batches))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_batch = {
                executor.submit(process_batch_for_worker, args, run_dir, batch_index, batch, logger): (batch_index, batch)
                for batch_index, batch in enumerate(batches, start=1)
            }
            pending_futures = set(future_to_batch)
            completed_count = 0
            processing_started_seconds = time.monotonic()
            next_progress_seconds = processing_started_seconds + args.progress_interval
            while pending_futures:
                done_futures, pending_futures = wait(
                    pending_futures,
                    timeout=args.progress_interval,
                    return_when=FIRST_COMPLETED,
                )
                if not done_futures:
                    log_progress_snapshot(
                        logger,
                        state,
                        completed_count,
                        len(batches),
                        len(pending_futures),
                        processing_started_seconds,
                    )
                    next_progress_seconds = time.monotonic() + args.progress_interval
                    continue

                for future in done_futures:
                    completed_count += 1
                    batch_index, batch = future_to_batch[future]
                    row_ids = [item.row_id for item in batch]
                    try:
                        _, _, results = future.result()
                    except Exception as error:  # noqa: BLE001 - record worker-level failures and continue.
                        logger.error("Worker failed batch %s/%s row_ids=%s: %s", batch_index, len(batches), row_ids, error)
                        record_failed_batch(run_dir, batch, error)
                        results = []
                    logger.info(
                        "Completed batch %s/%s finished=%s/%s row_ids=%s",
                        batch_index,
                        len(batches),
                        completed_count,
                        len(batches),
                        row_ids,
                    )
                    if not results:
                        state["failed_batches_seen_this_run"] += 1
                        update_checkpoint(run_dir, state)
                        continue
                    save_results_and_update_state(run_dir, state, candidate_by_id, results, args)
                    update_checkpoint(run_dir, state)

                if time.monotonic() >= next_progress_seconds:
                    log_progress_snapshot(
                        logger,
                        state,
                        completed_count,
                        len(batches),
                        len(pending_futures),
                        processing_started_seconds,
                    )
                    next_progress_seconds = time.monotonic() + args.progress_interval

    summary = summarize_results(run_dir)
    write_json(run_dir / "results" / "summary.json", summary)
    logger.info("Run complete. Summary: %s", summary)


if __name__ == "__main__":
    # 步骤1: Historical preprocessing is retained for inspection, not paid reruns.
    raise SystemExit("Historical screening entry is disabled in this frozen release; no API requests were sent.")
