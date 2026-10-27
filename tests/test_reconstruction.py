"""Test release reconstructions using invented tiny inputs only.

These tests never read the formal splits, load a model, or run a historical
experiment. They check selection, immutable replacement, normalization, joint
prior search and deterministic group-wise shuffling. Dependencies are unittest
and the adjacent release scripts, including the original NumPy-backed scorer.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any

# ------ Local release modules; GPU dependencies remain deferred ------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reconstruction_common import align_predictions
from reconstruct_length_fallback import (
    build_schema, deduplicate_response, merge_records, select_length_failures, update_record,
)
from reconstruct_prior_permutation import (
    evaluate_groups, make_prior, permutation_summary, rank_training_notes, search_prior,
)


# ========== Invented fixtures ==========

def synthetic_row(source_index: int = 0, finish_reason: str = "length") -> dict[str, Any]:
    """Create an invented first-pass record for pure fallback tests.

    Args:
        source_index (int): Index selected by a unit-test caller, normally 0 or 1.
        finish_reason (str): Synthetic engine finish flag selected by that test.
    Returns:
        dict[str, Any]: No real review, product identity, model path or private data.
    Raises:
        RuntimeError: No normal runtime exception is expected.
    """
    # 步骤1: Fixtures are invented and cannot be mistaken for formal experiment data.
    return {"source_index": source_index, "row_id": source_index, "model_name": "Synthetic-Finetuned",
            "model_path": "synthetic-model", "scent_information_level": 3,
            "messages": [{"role": "system", "content": "Synthetic instruction"},
                         {"role": "user", "content": "Synthetic citrus review"}],
            "reference": '{"top_notes":["bergamot"],"middle_notes":[],"base_notes":[]}',
            "raw_response": "invalid synthetic output", "valid_schema": False,
            "finish_reason": finish_reason, "output_token_count": 256}


def synthetic_labels() -> list[dict[str, Any]]:
    """Create two tiny label rows for validation/search tests.

    Args:
        None: All fixture values are explicitly invented inside this function.
    Returns:
        list[dict[str, Any]]: Two references with identical three-tier note sets.
    Raises:
        RuntimeError: No normal runtime exception is expected.
    """
    # 步骤1: Keep each tier nonempty to permit a one-count-per-tier search.
    notes = {"top_notes": ["bergamot"], "middle_notes": ["jasmine"], "base_notes": ["vanilla"]}
    return [{"source_index": index, "row_id": index, "level": level, "notes": copy.deepcopy(notes)}
            for index, level in enumerate([2, 3])]


# ========== Pure fallback tests ==========

class FallbackTests(unittest.TestCase):
    """Check reconstructed retry behavior without importing any inference engine."""

    def test_only_length_failures_selected(self) -> None:
        """Select synthetic length rows only; no arguments/return; raises assertions."""
        # 步骤1: Normally stopped invalid JSON must remain unselected.
        rows = [synthetic_row(0), synthetic_row(1, "stop")]
        self.assertEqual([row["source_index"] for row in select_length_failures(rows)], [0])

    def test_wrong_initial_budget_rejected(self) -> None:
        """Reject an unsupported synthetic token cap; no arguments/return; raises assertions."""
        # 步骤1: Do not claim the documented policy for a different first-pass cap.
        row = synthetic_row()
        row["output_token_count"] = 512
        with self.assertRaises(ValueError):
            select_length_failures([row])

    def test_previously_repaired_input_rejected(self) -> None:
        """Prevent repeat repair; no arguments/return; raises assertions."""
        # 步骤1: Already reconstructed rows must never enter another silent retry.
        row = synthetic_row()
        row["repair_metadata"] = {}
        with self.assertRaises(ValueError):
            select_length_failures([row])

    def test_schema_is_bounded_and_exact(self) -> None:
        """Inspect the reconstructed schema; no arguments/return; raises assertions."""
        # 步骤1: Enforce documented maxima without inventing unrecorded minima.
        schema = build_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(len(schema["required"]), 3)
        for properties in schema["properties"].values():
            self.assertEqual(properties["maxItems"], 12)
            self.assertNotIn("minItems", properties)

    def test_casefold_dedup_preserves_first_and_tier(self) -> None:
        """Check dedup semantics; no arguments/return; raises assertions."""
        # 步骤1: Do not merge cedarwood with cedar or remove cross-tier duplicates.
        raw = '{"top_notes":["Cedar","cedar","cedarwood"],"middle_notes":[],"base_notes":["cedar"]}'
        response, removed = deduplicate_response(raw)
        self.assertEqual(removed, 1)
        self.assertIn('"top_notes":["Cedar","cedarwood"]', response)
        self.assertIn('"base_notes":["cedar"]', response)

    def test_invalid_retry_retained(self) -> None:
        """Retain failed output; no arguments/return; raises assertions."""
        # 步骤1: A still-invalid completion stays represented in the final sample set.
        updated = update_record(synthetic_row(), "invalid retry", 512, "length", 7)
        self.assertFalse(updated["valid_schema"])
        self.assertTrue(updated["hit_max_new_tokens"])
        self.assertEqual(updated["repair_metadata"]["raw_model_response"], "invalid retry")

    def test_merge_preserves_order_and_originals(self) -> None:
        """Verify immutable index replacement; no arguments/return; raises assertions."""
        # 步骤1: Use non-sorted indexes to expose accidental reordering.
        originals = [synthetic_row(9, "stop"), synthetic_row(2)]
        before = copy.deepcopy(originals)
        repaired = update_record(originals[1], originals[1]["reference"], 20, "stop", 7)
        merged = merge_records(originals, [repaired])
        self.assertEqual([row["source_index"] for row in merged], [9, 2])
        self.assertEqual(merged[0], originals[0])
        self.assertEqual(originals, before)

    def test_merge_rejects_reference_changes(self) -> None:
        """Reject label edits during fallback; no arguments/return; raises assertions."""
        # 步骤1: The new generator cannot modify the evaluation reference.
        original = synthetic_row()
        repaired = update_record(original, original["reference"], 20, "stop", 7)
        repaired["reference"] = "changed synthetic label"
        with self.assertRaises(ValueError):
            merge_records([original], [repaired])


# ========== Tiny statistical reconstruction tests ==========

class PriorPermutationTests(unittest.TestCase):
    """Check algorithms on tiny invented sets, not the formal private data."""

    def test_frequency_ties_are_lexical(self) -> None:
        """Check declared frequency tie rule; no arguments/return; raises assertions."""
        # 步骤1: Equal-frequency terms must not depend on hash iteration order.
        labels = synthetic_labels()
        labels[0]["notes"]["top_notes"] = ["lemon"]
        ranked = rank_training_notes(labels)
        self.assertEqual(ranked["top_notes"], [("bergamot", 1), ("lemon", 1)])

    def test_joint_search_on_tiny_validation(self) -> None:
        """Search an invented one-candidate grid; no arguments/return; raises assertions."""
        # 步骤1: No formal split is read, and no historical number is forced.
        labels = synthetic_labels()
        result = search_prior(rank_training_notes(labels), labels, 1)
        self.assertEqual(result["counts"], (1, 1, 1))
        self.assertEqual(result["validation_f1"], 1.0)
        self.assertEqual(len(result["candidate_scores"]), 1)

    def test_impossible_count_rejected(self) -> None:
        """Reject invalid candidate counts; no arguments/return; raises assertions."""
        # 步骤1: Do not silently truncate beyond the ranked vocabulary length.
        with self.assertRaises(ValueError):
            make_prior(rank_training_notes(synthetic_labels()), (2, 1, 1))

    def test_permutation_repeats_with_fixed_seed(self) -> None:
        """Verify deterministic toy shuffles; no arguments/return; raises assertions."""
        # 步骤1: Test a few synthetic repetitions, never the paper's 500 full shuffles.
        references = [{"a"}, {"b"}, {"c"}]
        first = permutation_summary(references, references, 4, 7)
        second = permutation_summary(references, references, 4, 7)
        self.assertEqual(first, second)
        self.assertEqual(first["true_f1"], 1.0)
        self.assertEqual(len(first["permutation_scores"]), 4)

    def test_groups_keep_declared_seeds(self) -> None:
        """Check independent toy groups; no arguments/return; raises assertions."""
        # 步骤1: Declare group membership and seeds rather than infer old settings.
        labels = synthetic_labels()
        prior = labels[0]["notes"]
        predictions = [{"bergamot", "jasmine", "vanilla"}] * 2
        results = evaluate_groups(labels, predictions, prior, 2, 7)
        self.assertEqual([results[name]["permutation"]["seed"] for name in results], [7, 8, 9])
        self.assertEqual(results["level_2"]["permutation"]["sample_count"], 1)

    def test_prediction_reference_mismatch_rejected(self) -> None:
        """Verify pairing guard; no arguments/return; raises assertions."""
        # 步骤1: Similar row positions do not justify mismatched references.
        row = synthetic_row()
        with self.assertRaises(ValueError):
            align_predictions(synthetic_labels()[:1], [row])


if __name__ == "__main__":
    # 步骤1: Example: python -m unittest discover -s tests -v
    unittest.main()
