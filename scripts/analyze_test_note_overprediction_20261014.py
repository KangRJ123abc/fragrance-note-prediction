"""Audit note overprediction against the same frozen test references.

This descriptive post-processing uses only archived JSONL predictions and labels.
It verifies alignment and source hashes, reproduces frozen aggregate metrics,
and reports review-weighted and perfume-equal label statistics. No inference,
API, training, synonym merging, or hypothesis testing is performed. Dependencies
are exclusively Python's standard library. Results go to a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

# ------ Frozen inputs and descriptive scope ------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIRECTORY = PROJECT_ROOT / "src/outputs/evaluate_rerun_v2/20260715_qwen35_4b_chat_l3_unsloth_full_repair24"
OUTPUT_DIRECTORY = PROJECT_ROOT / "src/outputs/statistical_postprocessing/20261014_test_note_overprediction"
NOTE_KEYS = ("top_notes", "middle_notes", "base_notes")
FOCUS_TERMS = ("bergamot", "musk", "amber")
MODEL_KEYS = {"Base": "Qwen3.5-4B-Base", "Finetuned": "Qwen3.5-4B-Finetuned-Unsloth"}


# ========== Read-only input and validation ==========

def read_json(path: Path, is_jsonl: bool = False) -> Any:
    """Read a JSON object or JSONL records with explicit failure context.

    Args:
        path (Path): Frozen local source, supplied by the analysis workflow.
        is_jsonl (bool): Whether each nonempty line is a separate JSON object.
    Returns:
        Any: A decoded JSON value or list of decoded records.
    Raises:
        RuntimeError: A filesystem, Unicode, or JSON decoding error occurs.
    """
    # 步骤1: Keep failures visible instead of silently omitting malformed rows.
    try:
        with path.open(encoding="utf-8") as handle:
            if is_jsonl:
                return [json.loads(line) for line in handle if line.strip()]
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取统计输入失败: {error}, 上下文: path={path}") from error


def source_digest(path: Path) -> str:
    """Calculate a source SHA-256 without modifying the source.

    Args:
        path (Path): Source artifact supplied by the workflow.
    Returns:
        str: A 64-character lowercase hexadecimal digest.
    Raises:
        RuntimeError: Source bytes cannot be read.
    """
    # 步骤1: Chunked reads avoid dependence on source-file size.
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1048576), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as error:
        raise RuntimeError(f"计算输入哈希失败: {error}, 上下文: path={path}") from error


def normalized_notes(answer: dict[str, Any]) -> dict[str, list[str]]:
    """Normalize stored note lists using the frozen scorer's string convention.

    Args:
        answer (dict[str, Any]): Parsed reference or canonical prediction.
    Returns:
        dict[str, list[str]]: Three lowercase whitespace-collapsed lists. Missing
            or non-list tiers become empty; cross-tier repetition is retained.
    Raises:
        ValueError: The supplied answer is not a mapping.
    """
    # 步骤1: Validate external data before applying the scoring convention.
    if not isinstance(answer, dict):
        raise ValueError(f"规范化香调失败: 对象不是字典, 上下文: answer={answer!r}")
    output = {}
    for key in NOTE_KEYS:
        raw_values = answer.get(key, [])
        values = [" ".join(str(value).split()).lower() for value in raw_values] if isinstance(raw_values, list) else []
        output[key] = [value for value in values if value]
    return output


def flatten_notes(notes: dict[str, list[str]]) -> set[str]:
    """Combine tiers into an exact-string set for tier-agnostic comparison.

    Args:
        notes (dict[str, list[str]]): Normalized tier lists from the parser.
    Returns:
        set[str]: Unique note strings, including an empty set for empty output.
    Raises:
        KeyError: An internal normalized object is missing a tier.
    """
    # 步骤1: Count a term only once per review even when it spans several tiers.
    return set().union(*(set(notes[key]) for key in NOTE_KEYS))


def align_sample(test: dict[str, Any], predictions: dict[str, dict[str, Any]], index: int) -> dict[str, Any]:
    """Validate one held-out sample against both model generation records.

    Args:
        test (dict[str, Any]): One record from the frozen test dataset.
        predictions (dict[str, dict[str, Any]]): Base/Fine records keyed by model.
        index (int): Expected zero-based source index supplied by the caller.
    Returns:
        dict[str, Any]: Validated labels, normalized predictions, and metadata.
    Raises:
        AssertionError: Identity, level, input, or reference alignment fails.
    """
    # 步骤1: Verify identity and all scoring inputs, rather than trusting row order.
    metadata = test["metadata"]
    reference = normalized_notes(json.loads(test["messages"][-1]["content"]))
    output = {"row_id": metadata["row_id"], "perfume": metadata["perfume_name"],
              "level": metadata["scent_information_level"], "reference": reference, "models": {}}
    for model, record in predictions.items():
        assert record["source_index"] == index, f"对齐失败: source_index, 上下文: index={index}, model={model}"
        assert record["row_id"] == metadata["row_id"], f"对齐失败: row_id, 上下文: index={index}, model={model}"
        assert record["perfume_name"] == metadata["perfume_name"], f"对齐失败: perfume, 上下文: index={index}"
        assert record["scent_information_level"] == output["level"], f"对齐失败: level, 上下文: index={index}"
        assert record["messages"] == test["messages"][:2], f"对齐失败: prompt, 上下文: index={index}, model={model}"
        assert normalized_notes(json.loads(record["reference"])) == reference, f"对齐失败: reference, 上下文: index={index}"
        output["models"][model] = normalized_notes(record["canonical_notes"])
    return output


def load_samples(paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Load and align all frozen samples, preserving invalid-schema rows.

    Args:
        paths (dict[str, Path]): Test, Base, and Finetuned artifact paths.
    Returns:
        list[dict[str, Any]]: Exactly 3,289 aligned samples covering 52 perfumes.
    Raises:
        AssertionError: Counts, unique identities, or label consistency fail.
        RuntimeError: Any input cannot be loaded.
    """
    # 步骤1: Retain the complete test denominator and enforce unique identities.
    sources = {name: read_json(path, True) for name, path in paths.items() if name in ("test", "Base", "Finetuned")}
    assert all(len(rows) == 3289 for rows in sources.values()), "样本核查失败: 预期每侧3289条"
    samples = []
    for index, test in enumerate(sources["test"]):
        predictions = {model: sources[model][index] for model in MODEL_KEYS}
        samples.append(align_sample(test, predictions, index))
    assert len({row["row_id"] for row in samples}) == 3289, "样本核查失败: 重复row_id"
    assert Counter(row["level"] for row in samples) == {2: 1832, 3: 1457}, "样本核查失败: 等级数量不符"
    perfume_references = {}
    for row in samples:
        previous = perfume_references.setdefault(row["perfume"], row["reference"])
        assert previous == row["reference"], f"标签核查失败: 同款参考不一致, 上下文: perfume={row['perfume']}"
    assert len(perfume_references) == 52, "样本核查失败: 预期52款香水"
    return samples


# ========== Descriptive statistics ==========

def set_scores(reference: set[str], predicted: set[str]) -> dict[str, float]:
    """Calculate per-review exact-set overlap scores for frozen-metric verification.

    Args:
        reference (set[str]): Normalized held-out reference terms.
        predicted (set[str]): Normalized archived output terms.
    Returns:
        dict[str, float]: Precision, recall, and F1 in [0, 1].
    Raises:
        RuntimeError: This pure set arithmetic does not normally raise.
    """
    # 步骤1: Use empty-prediction zero scores as in the formal domain scorer.
    matches = len(reference & predicted)
    return {"precision": matches / len(predicted) if predicted else 0.0,
            "recall": matches / len(reference) if reference else 0.0,
            "f1": 2 * matches / (len(predicted) + len(reference)) if predicted or reference else 0.0}


def aggregate_model(samples: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Summarize output size and error counts within one descriptive subgroup.

    Args:
        samples (list[dict[str, Any]]): Nonempty aligned subgroup from the caller.
        model (str): Base or Finetuned, supplied by the analysis loop.
    Returns:
        dict[str, Any]: Mean list/set counts, score reproductions, and occurrence totals.
    Raises:
        ValueError: The supplied subgroup is empty.
    """
    # 步骤1: Keep list counts distinct from cross-tier-deduplicated set counts.
    if not samples:
        raise ValueError(f"模型汇总失败: 空子集, 上下文: model={model}")
    lengths = []; metrics = []; label_occurrences = Counter(); totals = Counter()
    for row in samples:
        reference = flatten_notes(row["reference"]); predicted = flatten_notes(row["models"][model])
        count = sum(len(row["models"][model][key]) for key in NOTE_KEYS)
        lengths.append({"list_count": count, "unique_count": len(predicted), "reference_count": len(reference),
                        "false_positive_count": len(predicted - reference), "false_negative_count": len(reference - predicted)})
        metrics.append(set_scores(reference, predicted)); label_occurrences.update(predicted)
        totals.update({"tp": len(predicted & reference), "fp": len(predicted - reference), "fn": len(reference - predicted)})
    output = {"mean_" + key: mean(row[key] for row in lengths) for key in lengths[0]}
    output["scores"] = {key: mean(row[key] for row in metrics) for key in metrics[0]}
    output["occurrence_totals"] = dict(totals)
    output["focus_share_of_predicted_occurrences"] = sum(label_occurrences[term] for term in FOCUS_TERMS) / sum(label_occurrences.values())
    return output


def label_statistics(samples: list[dict[str, Any]], model: str, term: str, is_perfume_equal: bool = False) -> dict[str, Any]:
    """Measure one label's prediction errors and occurrence concentration.

    Args:
        samples (list[dict[str, Any]]): Aligned subgroup, supplied by the caller.
        model (str): Model whose frozen predictions are counted.
        term (str): Exact normalized label; no note-family matching is used.
        is_perfume_equal (bool): Weight reviews by inverse within-perfume count.
    Returns:
        dict[str, Any]: TP/FP/FN/TN, coverage, precision, recall, FPR, and shares.
            Counts are reviews by default, fractional perfume-equivalent units otherwise.
    Raises:
        ValueError: Empty subgroup or invalid model name is supplied.
    """
    # 步骤1: Perfume-equal weights prevent prolific products dominating the sensitivity summary.
    if not samples or model not in MODEL_KEYS:
        raise ValueError(f"标签汇总失败: 无效输入, 上下文: model={model}, samples={len(samples)}")
    perfume_counts = Counter(row["perfume"] for row in samples)
    counts = Counter(); weight_sum = 0.0; predicted_total = 0.0; reference_total = 0.0
    for row in samples:
        reference = flatten_notes(row["reference"]); predicted = flatten_notes(row["models"][model])
        weight = 1 / perfume_counts[row["perfume"]] if is_perfume_equal else 1
        has_reference = term in reference; has_prediction = term in predicted
        if has_prediction:
            cell = "tp" if has_reference else "fp"
        else:
            cell = "fn" if has_reference else "tn"
        counts[cell] += weight; weight_sum += weight
        predicted_total += weight * len(predicted); reference_total += weight * len(reference)
    reference_count = counts["tp"] + counts["fn"]; predicted_count = counts["tp"] + counts["fp"]
    precision = counts["tp"] / predicted_count if predicted_count else None
    recall = counts["tp"] / reference_count if reference_count else None
    output = {key: counts[key] for key in ("tp", "fp", "fn", "tn")}
    output.update({"reference_count": reference_count, "predicted_count": predicted_count,
                   "reference_coverage": reference_count / weight_sum, "prediction_coverage": predicted_count / weight_sum,
                   "precision": precision, "recall": recall, "coverage_difference_pp": 100 * (predicted_count - reference_count) / weight_sum,
                   "false_positive_rate": counts["fp"] / (counts["fp"] + counts["tn"]) if counts["fp"] + counts["tn"] else None,
                   "reference_occurrence_share": reference_count / reference_total,
                   "predicted_occurrence_share": predicted_count / predicted_total if predicted_total else None})
    return output


def build_group(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build complete descriptive results for all labels and selected focus labels.

    Args:
        samples (list[dict[str, Any]]): Overall, Level 2, or Level 3 aligned rows.
    Returns:
        dict[str, Any]: Review summaries, all-label counts, and perfume-equal focus checks.
    Raises:
        ValueError: Downstream aggregation receives empty or invalid data.
    """
    # 步骤1: Audit all labels so three preselected labels are not the only archived evidence.
    terms = set()
    for row in samples:
        terms.update(flatten_notes(row["reference"]))
        for model in MODEL_KEYS:
            terms.update(flatten_notes(row["models"][model]))
    output = {"n_reviews": len(samples), "n_perfumes": len({row["perfume"] for row in samples}), "models": {}}
    reference_occurrences = Counter()
    for row in samples:
        reference_occurrences.update(flatten_notes(row["reference"]))
    output["focus_share_of_reference_occurrences"] = sum(reference_occurrences[term] for term in FOCUS_TERMS) / sum(reference_occurrences.values())
    for model in MODEL_KEYS:
        output["models"][model] = aggregate_model(samples, model)
        output["models"][model]["labels"] = {term: label_statistics(samples, model, term) for term in sorted(terms)}
        output["models"][model]["perfume_equal_focus"] = {term: label_statistics(samples, model, term, True) for term in FOCUS_TERMS}
    return output


def verify_frozen_metrics(groups: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Check that parsing, macro scores, and list counts reproduce frozen outputs.

    Args:
        groups (dict[str, Any]): Newly computed subgroup results.
        frozen (dict[str, Any]): Archived automatic_metrics.json contents.
    Returns:
        None: Successful return means all checked values agree within 1e-10.
    Raises:
        AssertionError: Any aggregate differs from the archived scorer output.
    """
    # 步骤1: This cross-check detects denominator or invalid-output handling changes.
    for group_name, group in groups.items():
        for model, frozen_model in MODEL_KEYS.items():
            previous = frozen["models"][frozen_model][group_name]
            current = group["models"][model]
            for metric in ("precision", "recall", "f1"):
                assert abs(current["scores"][metric] - previous[metric]) < 1e-10, f"冻结指标核查失败: {group_name}/{model}/{metric}"
            assert abs(current["mean_list_count"] - previous["average_note_count"]) < 1e-10, f"冻结输出项数核查失败: {group_name}/{model}"


# ========== Human-readable report and immutable output ==========

def report_group(name: str, group: dict[str, Any]) -> list[str]:
    """Format a subgroup's count and selected-label tables in Markdown.

    Args:
        name (str): Descriptive subgroup name supplied by report builder.
        group (dict[str, Any]): Fully calculated subgroup statistics.
    Returns:
        list[str]: Markdown lines with rates and exact TP/FP/FN counts.
    Raises:
        KeyError: Required analysis fields are absent.
    """
    # 步骤1: Show same-test references next to both predictions and preserve denominators.
    lines = [f"## {name}：{group['n_reviews']}条评论，{group['n_perfumes']}款香水", "",
             "| 模型 | 平均三层列表项数 | 平均跨层去重词项数 | 平均参考去重词项数 | 平均FP项数 | 平均FN项数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for model, values in group["models"].items():
        lines.append(f"| {model} | {values['mean_list_count']:.4f} | {values['mean_unique_count']:.4f} | {values['mean_reference_count']:.4f} | {values['mean_false_positive_count']:.4f} | {values['mean_false_negative_count']:.4f} |")
    lines += ["", "| 词项 | 模型 | 测试参考覆盖率 | 预测覆盖率 | Precision | Recall | TP | FP | FN |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for term in FOCUS_TERMS:
        for model, values in group["models"].items():
            label = values["labels"][term]
            lines.append(f"| {term} | {model} | {label['reference_coverage']:.2%} | {label['prediction_coverage']:.2%} | {label['precision']:.2%} | {label['recall']:.2%} | {label['tp']} | {label['fp']} | {label['fn']} |")
    lines += ["", f"三词合计占参考词项出现次数：{group['focus_share_of_reference_occurrences']:.2%}。"]
    for model, values in group["models"].items():
        lines.append(f"{model}三词合计占预测词项出现次数：{values['focus_share_of_predicted_occurrences']:.2%}。")
    return lines + [""]


def build_report(results: dict[str, Any]) -> str:
    """Create a Chinese interpretation report describing scope and formulas.

    Args:
        results (dict[str, Any]): Validated groups and provenance generated locally.
    Returns:
        str: Complete Markdown report for manuscript revision, without new inference.
    Raises:
        KeyError: A required subgroup is absent.
    """
    # 步骤1: State descriptive scope, exact-string matching, and statistical units upfront.
    lines = ["# 同一测试集上的香调标签预测诊断", "", "统计日期：2026-10-14。仅对冻结结果作描述性后处理，未调用模型或API。", "",
             "## 统计口径", "", "- 主表以评论为单位：Overall 3,289条；Level 2 1,832条；Level 3 1,457条。测试覆盖52款香水。",
             "- 预测来自正式step-700、repair24目录；保留全部样本，沿用canonical_notes中可恢复的层级。无效schema不等于整条预测清空。",
             "- 小写、连续空白合并、精确字符串匹配；三层合并去重；musk不包含white musk或其他词族。",
             "- TP/FP/FN/TN表示该词在一条评论预测集合与所属香水公开参考集合中的存在/缺失。Precision=TP/(TP+FP)，Recall=TP/(TP+FN)。",
             "- 参考覆盖率=(TP+FN)/N；预测覆盖率=(TP+FP)/N。逐标签P/R属于标签二分类统计，不是原论文的逐评论宏平均P/R。",
             "- 三层列表项数保留跨层重复，复现原average_note_count；另报告跨层去重后的词项数，使预测数和参考数可以公平比较。",
             "- all_label_stats.json保存所有词项及按香水等权的三词敏感性统计。等权统计中每条评论权重为1/该款评论数；P/R为加权TP及分母之比。",
             "- 同款评论共享参考，不被解释为独立香水样本；本报告不增加置信区间、显著性或因果归因。",
             "- 字符串未命中公开标签不直接意味着感官错误；结果只评价公开标签预测。词项份额控制总输出数这一描述性尺度，但不完全控制标签共现等因素。", ""]
    for name, group in results["groups"].items():
        lines.extend(report_group(name, group))
    lines += ["## 可支持的解释", "", "1. 用同一测试集参考覆盖率替代训练覆盖率作为过度预测的直接对照。",
              "2. 分开讨论输出总项数扩大、少数标签的预测覆盖率超出参考覆盖率，以及这些标签在全部输出词项中的集中程度。",
              "3. 低Precision与高Recall可说明频繁检出与较多额外标签预测并存；不能仅据此断言错误来自训练先验。",
              "4. 平均输出项数接近参考并不排除标签分配错误：FP与FN可以同时很多。",
              "5. 不把术语种类减少解释为模式坍塌，不改变冻结的领域主结果。", "", "## 复现与核查", "",
              "运行：python scripts/analyze_test_note_overprediction_20261014.py", "",
              "已逐条验证row_id、source_index、香水、Level、system/user提示及参考标签一致。全部分组的宏平均Precision/Recall/F1和平均列表项数均复现冻结指标（容差1e-10）。输入SHA-256与归档及运行前后核查见config.json。"]
    return "\n".join(lines) + "\n"


def persist_results(results: dict[str, Any], paths: dict[str, Path], hashes: dict[str, str], output_directory: Path) -> None:
    """Save a new analysis directory and verify source bytes remain unchanged.

    Args:
        results (dict[str, Any]): Completed and verified statistical tables.
        paths (dict[str, Path]): Frozen input paths recorded for provenance.
        hashes (dict[str, str]): Input digests calculated before the analysis.
        output_directory (Path): New destination supplied by main or --output-dir.
    Returns:
        None: Writes JSON results, configuration, and Markdown report.
    Raises:
        RuntimeError: Source hashes change or new output cannot be written.
        FileExistsError: The proposed output directory already exists.
    """
    # 步骤1: Refuse overwriting earlier post-processing or frozen artifacts.
    try:
        output_directory.mkdir(parents=True, exist_ok=False)
        for name, path in paths.items():
            if source_digest(path) != hashes[name]:
                raise RuntimeError(f"输入保留核查失败: 哈希变化, 上下文: path={path}")
        configuration = {"analysis_date": "2026-10-14", "model_inference": False, "focus_terms": FOCUS_TERMS,
                         "inputs": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
                         "frozen_metrics_reproduced": True, "source_hashes_unchanged": True,
                         "archive_generation_hashes_verified": True, "statistics": "descriptive exact-string counts; review and perfume-equal weights; no hypothesis tests"}
        (output_directory / "config.json").write_text(json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_directory / "all_label_stats.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_directory / "REPORT.md").write_text(build_report(results), encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"保存统计结果失败: {error}, 上下文: output={output_directory}") from error


def main() -> None:
    """Run the frozen-test label audit and preserve a standalone reproducibility record.

    Args:
        None: Uses fixed sources; optional --output-dir sets a new destination.
    Returns:
        None: Persists descriptive artifacts and prints their directory.
    Raises:
        RuntimeError: Input or output operations fail.
        AssertionError: Source hashes, alignment, or metric verification fail.
    """
    # 步骤1: Check frozen generations against the original archive digests.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True, help="Private formal test JSONL; never redistributed with this release.")
    parser.add_argument("--formal-directory", type=Path, required=True, help="Private frozen repair24 run directory.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY, help="New directory; existing directories are never overwritten.")
    arguments = parser.parse_args()
    output_directory = arguments.output_dir.resolve()
    if output_directory.exists():
        raise FileExistsError(f"输出保留核查失败: 目录已存在, 上下文: output={output_directory}; use --output-dir for a new run")
    paths = {"test": arguments.test_file.resolve(),
             "Base": arguments.formal_directory / "generations/base.jsonl", "Finetuned": arguments.formal_directory / "generations/finetuned.jsonl",
             "metrics": arguments.formal_directory / "metrics/automatic_metrics.json"}
    hashes = {name: source_digest(path) for name, path in paths.items()}
    assert hashes["Base"] == "47ed17346dace494fab814cd8eabe1cd64b57c1975702b2b4784105cdac2f678", "归档核查失败: Base生成哈希"
    assert hashes["Finetuned"] == "6b7b5b3e7b135a4e9dfb0b45eecc1f04db6c0fe5f6bf23be37b33e6856c09cbe", "归档核查失败: Finetuned生成哈希"
    # 步骤2: Calculate all groups before writing any outputs.
    samples = load_samples(paths)
    groups = {"overall": build_group(samples)}
    for level in (2, 3):
        groups[f"level_{level}"] = build_group([row for row in samples if row["level"] == level])
    verify_frozen_metrics(groups, read_json(paths["metrics"]))
    # 步骤3: Preserve a new analysis with complete provenance and readable tables.
    persist_results({"groups": groups}, paths, hashes, output_directory)
    print(output_directory)


if __name__ == "__main__":
    # 步骤1: Example: python scripts/analyze_test_note_overprediction_20261014.py
    main()
