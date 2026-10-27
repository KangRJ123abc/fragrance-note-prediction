# Reproduction scope

## Data and checkpoint

The formal split files are now published in the linked ModelScope dataset rather
than embedded in the code repository. Use scripts/download_data.py and the fixed
revision in configs/dataset_sources.json; local SDK login may be required for
file downloads. Downloaded files preserve the original experiment hashes and
splits. The toy example is not a replacement. The final merged model is from step 700, not the
step-1100 early-stopping endpoint. Base refers to the original post-trained
Qwen3.5-4B before domain adaptation, not a separately pretrained `-Base` model.

## Domain generation and metrics

With local split and model paths, the original first-pass generation command is:

```bash
python src/eval_qwen35_4b_chat_l3_unsloth.py --phase generate --model base --project-root . --test-file /path/to/test.jsonl --base-model models/base --run-dir runs/domain_new
python src/eval_qwen35_4b_chat_l3_unsloth.py --phase generate --model finetuned --project-root . --test-file /path/to/test.jsonl --finetuned-model models/finetuned-step700 --run-dir runs/domain_new
```

**Important:** this original generator does not execute the independent historical
length-fallback step. The formal results use a repair24 run, whose settings are
recorded in `configs/protocols.json`. The independent historical fallback executor
was not located in the archive. A newly reconstructed executor is now supplied as
`scripts/reconstruct_length_fallback.py`; it is explicitly not the original and
has not been run on a model. See `docs/RECONSTRUCTION.md`. Consequently,
first-pass output is not guaranteed to reproduce the final paper metrics.
Recoverable objects preserve valid tiers; missing/invalid tiers become empty.
An unrecoverable object becomes three empty predictions. All samples are scored.

The fixed prior used frequency-ranked Level-3 training labels and a validation-
selected joint tier count. A later arithmetic reconstruction reproduced its
rounded scores with top=[bergamot], middle=[jasmine], base=[vanilla, musk,
sandalwood, amber, patchouli, cedar, tonka bean]. The joint count is (1,1,7);
seven is a selected baseline count, not the number of notes produced by Base.
The original prior-search/permutation executors were not located. A new
`scripts/reconstruct_prior_permutation.py` declares its search, tie and RNG rules;
it has not been run on the formal dataset. Do not claim its rules are historical.
The published permutation comparison reports 500 shuffles descriptively, not
an additional domain significance test. Historical domain confidence-interval
claims are not adopted by this release.

## General benchmarks

Install EvalScope 1.4.2 in a client environment separate from vLLM. Example:

```bash
python src/eval_benchmark.py --mode full --project-root . --base-model models/base --finetuned-model models/finetuned-step700 --vllm-executable /path/to/inference_env/bin/vllm --run-dir runs/general_new
python src/analyze_general_benchmarks.py --run-dir runs/general_new --output-dir runs/general_analysis_new
```

C-Eval, CMMLU and MMLU use 100 items per subset, not 100 total. MMLU and GSM8K
are 5-shot; the remaining four benchmarks are 0-shot. Context length is 8,192
and output budget is 1,024. Source benchmark datasets are retrieved by EvalScope
under their own licenses. Historical service credentials are not required.
Paired bootstrap uses 20,000 resamples; exact McNemar tests receive six-test Holm
correction. No private benchmark inputs are included in this folder.

## Exploratory application transfer

```bash
python src/eval_user_request_note_design.py --phase generate --model base --project-root . --requests-file data/user_request_note_design_pilot_v2_thinking.jsonl --base-model models/base --reasoning-mode visible_plan --max-new-tokens 512 --run-dir runs/application_new
```

Generate the other model with the same command and `--model finetuned` plus its
explicit model path. Native thinking stays disabled. Analysis also requires the
Level-3 labels from the published training split to reconstruct the 501-term vocabulary:

```bash
python src/eval_user_request_note_design.py --phase analyze --project-root . --requests-file data/user_request_note_design_pilot_v2_thinking.jsonl --training-file /path/to/train.jsonl --reasoning-mode visible_plan --max-new-tokens 512 --run-dir runs/application_new
```

This is researcher-led exploratory analysis. No questionnaire, paid Judge, human
preference score, or user win rate is produced or required.

## Existing-result audits

The same-test common-note audit uses the published test split and the original
frozen generation files. Those generation archives are not included in this code repository:

```bash
python scripts/analyze_test_note_overprediction_20261014.py --test-file /path/to/test.jsonl --formal-directory /path/to/repair24 --output-dir runs/common_note_audit_new
```

It performs only descriptive post-processing and retains frozen input-hash
verification. Use new output directories and preserve originals. Aggregate public
results are provided as evidence, not as a replacement for the required input artifacts.
