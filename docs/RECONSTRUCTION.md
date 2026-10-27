# Release-time reconstruction of missing auxiliary executors

The historical recovery search found supporting configurations, logs, and outputs,
but no original independent executors for the length fallback, fixed-prior search,
or permutation analysis. The scripts below were newly written for public release
on 2026-10-22. They are not recovered originals and have not been run on the formal
dataset or on a model. The paper's archived results are unchanged.

## What is known and what was chosen

| Component | Supported by existing artifacts | Explicit reconstruction choices |
| --- | --- | --- |
| Length fallback | Only length failures; 256-token first pass; same prompts and weights; 512-token retry; repetition penalty 1.05; frequency penalty 0.10; three required tiers with at most 12 items | Exact JSON Schema serialization, StructuredOutputsParams API, explicit new seed, current compatible backend behavior |
| Fixed prior | Frequent Level-3 training labels; candidate counts selected on validation; constant predictions for all test reviews | Level-3 validation-only search; joint counts 1 through 15 by default; lexical frequency ties; fewest total notes then lexical count tuple for F1 ties within 1e-12 |
| Permutations | 500 shuffled pairings; comparison of true and shuffled F1 | Python random.Random; independently shuffle Overall, Level 2 and Level 3; group seeds seed, seed+1, seed+2 |

None of the choices in the right column is claimed to be proven historical code.
The original report gives a fixed-prior flat F1 around 0.2629/0.2636/0.2620 and
permutation means around 0.1883/0.1909/0.1855 for Overall/Level 2/Level 3.
These are historical reference values, not outputs from the new scripts.
The earlier arithmetic reconstruction's (1,1,7) pyramid is not hardcoded into the
new search. It must be selected by the supplied validation data if appropriate.
The archived tier-optimized prior is a separate objective and is not recreated
by the flat-F1 search. Tier F1 reported for the flat-selected prior need not equal
the historical independently tier-optimized baseline.

## Length fallback

`scripts/reconstruct_length_fallback.py` prints a plan by default. It reads a
first-pass generation JSONL and checks indexes and failure flags without a model.
The source_index is a zero-based physical input-line position from the original
loader, not a one-based line number. Blank lines do not renumber later records.

```bash
python scripts/reconstruct_length_fallback.py --input-generations /path/to/first_pass/generations/finetuned.jsonl --output-dir runs/reconstructed_fallback --expected-count 24
```

No command above was run on the formal artifacts during reconstruction. If a user
later elects to generate new retries, adding --execute, --model-dir, and an explicit
--seed is required. The published experiment seed for this independent fallback
is unknown, so the script refuses to silently pick one. Such a run produces new
outputs and must not replace the archived repair24 directory or be described as
an exact replay. GPU behavior has not been validated in this release.

The schema permits empty lists because the historical minimum item count was
not recorded. It enforces exactly three keys, lists of strings, and at most 12
items per tier. Deduplication is casefold-exact within each tier, preserving the
first spelling; it does not merge synonyms or remove cross-tier occurrences.
Invalid or still-truncated retries remain represented and are not silently dropped.
Unselected records remain unchanged. Input text, references and raw new responses
are retained separately, and final replacement uses source_index without reordering.
These outputs may contain private reviews and must not be publicly uploaded.

The new run contains only the selected model's generation side. For a later metric
run, place an unchanged copy of the peer model's archived generation file beside
it, then use the original --phase metrics command in a new directory. Do not
recompute or substitute published paper values without author review.

## Fixed-prior selection and permutations

`scripts/reconstruct_prior_permutation.py` uses existing private splits and final
prediction files; it contains no model or API call. Every input must be provided
explicitly. A future run, if separately requested, can use:

```bash
python scripts/reconstruct_prior_permutation.py --train-file /path/to/train.jsonl --val-file /path/to/val.jsonl --test-file /path/to/test.jsonl --predictions-file /path/to/final/finetuned.jsonl --output-dir runs/reconstructed_prior_permutation --maximum-tier-count 15 --permutations 500 --seed 3407
```

The seed 3407 in this example is a declared reconstruction choice, not a recovered
historical permutation seed. Inputs are hash-recorded. The code selects a prior
using Level-3 training and validation only; test labels do not select counts.
Predictions are paired by source_index, row_id and reference content. The original
parser and original per-review F1 primitive are reused, including recoverable
tiers from schema-invalid outputs. All candidate scores and all permutation
scores are saved in a fresh directory with their configuration. No bootstrap,
domain p-value, or new significance claim is generated.

## Validation performed for this release

Only static parsing, command help and tests on invented tiny records are allowed
in this reconstruction task. Those tests do not establish numerical agreement
with formal results or GPU compatibility. No training, model inference, formal
statistical rerun, or paid API evaluation was performed.

Implementation reference for structured generation:
https://github.com/vllm-project/vllm/blob/main/docs/features/structured_outputs.md
The API reference does not identify the original historical executor or backend.
