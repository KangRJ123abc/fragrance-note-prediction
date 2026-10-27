"""Reconstruct fixed-prior selection and descriptive permutations for release.

This is new release-time code, not the missing July 2026 executor. It uses existing
labels and final predictions only; it cannot train, load a model, or call an API.
Unknown historical search/tie/seed/stratification choices are declared explicitly.
Dependencies are the standard library and the archived NumPy-backed metric module.
"""

from __future__ import annotations

import argparse
import itertools
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from reconstruction_common import (
    REQUIRED_NOTE_KEYS, align_predictions, calculate_set_scores, calculate_sha256,
    create_output_directory, flatten_notes, load_labels, read_records, write_json,
)

# ------ Reconstruction choices, not asserted historical settings ------
RECONSTRUCTION_VERSION = "release_reconstruction_v1"
GROUP_LEVELS = {"overall": None, "level_2": 2, "level_3": 3}
SCORE_TOLERANCE = 1e-12


# ========== Frequency ranking and validation selection ==========

def rank_training_notes(labels: list[dict[str, Any]]) -> dict[str, list[tuple[str, int]]]:
    """Rank each tier by Level-3 training sample coverage and lexical tie order.

    Args:
        labels (list[dict[str, Any]]): Level-3 training labels from load_labels.
    Returns:
        dict[str, list[tuple[str, int]]]: Descending frequencies per required tier.
    Raises:
        ValueError: A tier has no candidate labels.
    """
    # 步骤1: Count a normalized term once per training sample and tier.
    ranked = {}
    for tier in REQUIRED_NOTE_KEYS:
        frequencies = Counter()
        for row in labels:
            frequencies.update(set(row["notes"][tier]))
        if not frequencies:
            raise ValueError(f"统计先验失败: 层级无标签, 上下文: tier={tier}")
        # 步骤2: Alphabetic tie-breaking is an explicit release choice.
        ranked[tier] = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return ranked


def make_prior(ranked: dict[str, list[tuple[str, int]]], counts: tuple[int, ...]) -> dict[str, list[str]]:
    """Create one constant pyramid from tier rankings and candidate counts.

    Args:
        ranked (dict[str, list[tuple[str, int]]]): Level-3 training frequency ranks.
        counts (tuple[int, ...]): Positive top/middle/base counts from grid search.
    Returns:
        dict[str, list[str]]: Exactly three tier lists, containing frequent labels.
    Raises:
        ValueError: Counts are missing, nonpositive, or exceed tier vocabulary.
    """
    # 步骤1: Reject impossible counts instead of silently slicing fewer terms.
    if len(counts) != 3 or any(count < 1 or count > len(ranked[tier]) for tier, count in zip(REQUIRED_NOTE_KEYS, counts)):
        raise ValueError(f"构建先验失败: 数量超出范围, 上下文: counts={counts}")
    return {tier: [term for term, frequency in ranked[tier][:count]]
            for tier, count in zip(REQUIRED_NOTE_KEYS, counts, strict=True)}


def score_constant_prior(prior: dict[str, list[str]], labels: list[dict[str, Any]]) -> dict[str, float]:
    """Score one fixed pyramid using the original review-macro flat and tier F1.

    Args:
        prior (dict[str, list[str]]): One fixed pyramid selected without test labels.
        labels (list[dict[str, Any]]): Validation or test references from load_labels.
    Returns:
        dict[str, float]: F1 and Tier F1, each in [0, 1].
    Raises:
        ValueError: No labels are supplied.
    """
    # 步骤1: Preserve review weighting even when many perfumes share labels.
    if not labels:
        raise ValueError("评价先验失败: 标签为空, 上下文: score_constant_prior")
    predicted = flatten_notes(prior)
    flat_scores = [calculate_set_scores(flatten_notes(row["notes"]), predicted)["f1"] for row in labels]
    tier_scores = [calculate_set_scores(set(row["notes"][tier]), set(prior[tier]))["f1"]
                   for row in labels for tier in REQUIRED_NOTE_KEYS]
    return {"f1": mean(flat_scores), "tier_f1": mean(tier_scores)}


def search_prior(ranked: dict[str, list[tuple[str, int]]], validation: list[dict[str, Any]],
                 maximum_count: int) -> dict[str, Any]:
    """Jointly search tier counts using Level-3 validation flat macro F1 only.

    Args:
        ranked (dict[str, list[tuple[str, int]]]): Level-3 training-only rankings.
        validation (list[dict[str, Any]]): Level-3 validation references, not test.
        maximum_count (int): CLI upper bound per tier; default 15 is reconstructed.
    Returns:
        dict[str, Any]: Selected prior, counts, score and every candidate score.
    Raises:
        ValueError: Count bounds are nonpositive or validation is empty.
    """
    # 步骤1: Aggregate identical references without changing review-macro weights.
    if maximum_count < 1 or not validation:
        raise ValueError("搜索先验失败: 范围或验证集无效, 上下文: search_prior")
    reference_counts = Counter(frozenset(flatten_notes(row["notes"])) for row in validation)
    ranges = [range(1, min(maximum_count, len(ranked[tier])) + 1) for tier in REQUIRED_NOTE_KEYS]
    best = None
    candidate_scores = []
    # 步骤2: Search jointly because cross-tier duplicate terms affect flat F1.
    for counts in itertools.product(*ranges):
        prior = make_prior(ranked, counts)
        predicted = flatten_notes(prior)
        score = sum(weight * calculate_set_scores(set(reference), predicted)["f1"]
                    for reference, weight in reference_counts.items()) / len(validation)
        candidate_scores.append({"counts": counts, "validation_f1": score})
        complexity = (sum(counts), *counts)
        if best is None or score > best["validation_f1"] + SCORE_TOLERANCE or (
                abs(score - best["validation_f1"]) <= SCORE_TOLERANCE and complexity < best["complexity"]):
            best = {"counts": counts, "prior": prior, "validation_f1": score, "complexity": complexity}
    return {**best, "candidate_scores": candidate_scores}


# ========== Descriptive within-group permutations ==========

def permutation_summary(references: list[set[str]], predictions: list[set[str]],
                        repetitions: int, seed: int) -> dict[str, Any]:
    """Shuffle predictions within one supplied group and retain every score.

    Args:
        references (list[set[str]]): Flat test references in verified sample order.
        predictions (list[set[str]]): Aligned final predictions of the same length.
        repetitions (int): CLI repetition count, positive; default 500.
        seed (int): Explicit reconstructed Python random.Random seed for this group.
    Returns:
        dict[str, Any]: True F1, all shuffled scores, their mean and difference.
    Raises:
        ValueError: Inputs are empty, sizes differ, or repetitions are nonpositive.
    """
    # 步骤1: Fix group membership; never mix Level-2/Level-3 prediction marginals.
    if not references or len(references) != len(predictions) or repetitions < 1:
        raise ValueError("置换失败: 输入长度或次数无效, 上下文: permutation_summary")
    generator = random.Random(seed)
    true_f1 = mean(calculate_set_scores(reference, prediction)["f1"]
                   for reference, prediction in zip(references, predictions, strict=True))
    scores = []
    # 步骤2: Each permutation starts from the original order and preserves margins.
    for iteration in range(repetitions):
        order = list(range(len(predictions)))
        generator.shuffle(order)
        scores.append(mean(calculate_set_scores(reference, predictions[index])["f1"]
                           for reference, index in zip(references, order, strict=True)))
    return {"sample_count": len(references), "seed": seed, "true_f1": true_f1,
            "permutation_scores": scores, "permutation_mean": mean(scores),
            "true_pair_advantage": true_f1 - mean(scores)}


def evaluate_groups(labels: list[dict[str, Any]], predictions: list[set[str]],
                    prior: dict[str, list[str]], repetitions: int, seed: int) -> dict[str, Any]:
    """Evaluate one selected prior and independent group-wise shuffles.

    Args:
        labels (list[dict[str, Any]]): All validated test references.
        predictions (list[set[str]]): One verified prediction per test reference.
        prior (dict[str, list[str]]): Validation-selected constant pyramid.
        repetitions (int): Positive number of descriptive shuffles per group.
        seed (int): CLI base seed; group seeds are seed, seed+1, seed+2.
    Returns:
        dict[str, Any]: Overall, Level-2 and Level-3 results with declared seeds.
    Raises:
        ValueError: A requested group is empty or parameters are invalid.
    """
    # 步骤1: Report subgroup shuffles as independent analyses, not subsetted global shuffles.
    results = {}
    for offset, (name, level) in enumerate(GROUP_LEVELS.items()):
        indexes = [index for index, row in enumerate(labels) if level is None or row["level"] == level]
        subset = [labels[index] for index in indexes]
        references = [flatten_notes(row["notes"]) for row in subset]
        shuffled = permutation_summary(references, [predictions[index] for index in indexes], repetitions, seed + offset)
        fixed = score_constant_prior(prior, subset)
        results[name] = {"fixed_prior": fixed, "permutation": shuffled,
                         "finetuned_minus_fixed_prior": shuffled["true_f1"] - fixed["f1"]}
    return results


# ========== Explicit release reconstruction entry ==========

def build_parser() -> argparse.ArgumentParser:
    """Expose formal inputs, new output path and reconstructed numerical choices.

    Args:
        None: The main function obtains actual values from command-line callers.
    Returns:
        argparse.ArgumentParser: Parser with no implicit formal-data execution.
    Raises:
        RuntimeError: No normal runtime exception is expected.
    """
    # 步骤1: Require every data path; an empty command cannot start a formal analysis.
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-file", "val-file", "test-file", "predictions-file", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--maximum-tier-count", type=int, default=15)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def run_reconstruction(arguments: argparse.Namespace) -> None:
    """Read frozen inputs and persist newly reconstructed statistics.

    Args:
        arguments (argparse.Namespace): Explicit input and choice values from CLI.
    Returns:
        None: Writes new result, candidate-score and configuration JSON artifacts.
    Raises:
        RuntimeError: Input hashing or output creation fails.
        ValueError: Pairing, parameters, or data are invalid.
    """
    # 步骤1: Reject existing destinations and invalid bounds before reading data.
    if arguments.output_dir.exists() or arguments.maximum_tier_count < 1 or arguments.permutations < 1:
        raise ValueError("重建失败: 输出已存在或次数无效, 上下文: CLI arguments")
    training = load_labels(arguments.train_file, level=3)
    validation = load_labels(arguments.val_file, level=3)
    labels = load_labels(arguments.test_file)
    predictions = align_predictions(labels, read_records(arguments.predictions_file))
    ranked = rank_training_notes(training)
    selected = search_prior(ranked, validation, arguments.maximum_tier_count)
    groups = evaluate_groups(labels, predictions, selected["prior"], arguments.permutations, arguments.seed)
    # 步骤2: Save separately; never replace historical numbers or claim exact replay.
    create_output_directory(arguments.output_dir)
    write_json(arguments.output_dir / "candidate_scores.json", selected.pop("candidate_scores"))
    write_json(arguments.output_dir / "results.json", {"status": RECONSTRUCTION_VERSION, "selected": selected, "groups": groups})
    save_configuration(arguments, len(training), len(validation), ranked)


def save_configuration(arguments: argparse.Namespace, training_count: int, validation_count: int,
                       ranked: dict[str, list[tuple[str, int]]]) -> None:
    """Persist input checksums and every choice not known from historical code.

    Args:
        arguments (argparse.Namespace): Input paths and explicit reconstruction choices.
        training_count (int): Retained Level-3 training row count from load_labels.
        validation_count (int): Retained Level-3 validation row count from load_labels.
        ranked (dict[str, list[tuple[str, int]]]): Frequency ranks calculated from training.
    Returns:
        None: Writes provenance and training-frequency JSON files.
    Raises:
        RuntimeError: Hashing or writing an output fails.
    """
    # 步骤1: Record input hashes without copying their contents into result metadata.
    sources = {name: {"file_name": getattr(arguments, name).name,
                      "sha256": calculate_sha256(getattr(arguments, name))}
               for name in ("train_file", "val_file", "test_file", "predictions_file")}
    choices = {"status": RECONSTRUCTION_VERSION, "original_executor_recovered": False,
               "training_level": 3, "training_rows": training_count, "validation_level": 3,
               "validation_rows": validation_count, "maximum_tier_count": arguments.maximum_tier_count,
               "objective": "joint flat review-macro F1 on Level-3 validation only",
               "frequency_tie_rule": "descending frequency then lexical ascending",
               "score_tie_rule": "within 1e-12, minimize total notes then top/middle/base counts",
               "permutation_scope": "independent within overall, Level 2, Level 3",
               "rng": "Python random.Random", "group_seeds": [arguments.seed + offset for offset in range(3)],
               "permutations": arguments.permutations, "inferential_tests": False, "inputs": sources}
    write_json(arguments.output_dir / "config.json", choices)
    write_json(arguments.output_dir / "training_frequencies.json", ranked)


if __name__ == "__main__":
    # 步骤1: Example: python scripts/reconstruct_prior_permutation.py --help
    run_reconstruction(build_parser().parse_args())
