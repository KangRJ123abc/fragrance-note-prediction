# Structured fragrance note prediction from consumer reviews

Code and model access instructions for **Mapping Fragrance Reviews to Structured Olfactory Notes with a Domain-Fine-Tuned Large Language Model**.

The domain-fine-tuned model maps a single anonymized review to the publicly listed top, middle, and base notes of its perfume. These are perfume-level weak labels, not a full reconstruction of an individual reviewer's experience. The published model is merged from the best **step-700** checkpoint. Experiments were frozen in July 2026.

Published resources: [code](https://github.com/KangRJ123abc/fragrance-note-prediction), [formal dataset](https://www.modelscope.cn/datasets/KangRJ123abc/Fragrance-Review-Note-Pyramids), and [step-700 merged model](https://www.modelscope.cn/models/KangRJ123abc/Qwen3.5-4B-Fragrance-Notes). Model and dataset downloads pin immutable revisions, not the mutable master branch.

## What is included

- `src/`: training, dataset construction, domain evaluation, general benchmarks, and exploratory application-transfer code.
- `scripts/`: pinned model/data download, single-review inference, explicit training launcher, file verification, and descriptive post-processing.
- Release-time reconstructed length-fallback, fixed-prior and permutation scripts are included, with provenance and unverified historical choices documented in `docs/RECONSTRUCTION.md`. They have not been run on formal data or a model.
- `configs/`: model sources, artifact hashes, generation settings, and dataset checksums.
- `data/`: the researcher-constructed 24-request bank, a synthetic format example, and access instructions for the separately published formal reviews. Downloaded files are cached locally and ignored by Git.
- `results/`: aggregate domain metrics, common-label statistics, and selected general benchmark results.
- `environment/`: separate historical environment snapshots and minimal dependency lists.

No model weights, credentials, private server paths, canceled Judge runs, or human questionnaire materials are included. Source changes made solely for release are documented in `docs/RELEASE_CHANGES.md`.

## Quick start

Use Linux with an NVIDIA GPU and a compatible CUDA stack. The archived training and inference environments differ; do not combine their dependencies into one environment. See `environment/README.md`.

For CPU-only checks without training or model inference:

```bash
python -m unittest discover -s tests -v
python scripts/verify_files.py --root . --manifest SHA256SUMS.json
python scripts/download_models.py --role base --dry-run
python scripts/download_data.py --dry-run
```

Install the inference environment separately, then use a local copy of the released ModelScope folder:

```bash
python scripts/predict.py --model-dir ../modelscope --review "A synthetic example: bright citrus at first, soft jasmine later, and a warm woody finish."
```

Alternatively, use automatic ModelScope download:

```bash
python scripts/download_models.py --role base
python scripts/predict.py --role finetuned --review "A synthetic example: fresh citrus and a dry woody finish."
```

Both published ModelScope IDs and immutable revisions are configured in `configs/model_sources.json`. The baseline is `unsloth/Qwen3.5-4B`, not the escaped local cache spelling or the separately pretrained `-Base` variant. Its pinned upstream tokenizer config differs from the historical experiment copy only in `tokenizer_class`; the helper applies the included, hash-checked historical config before verifying the frozen artifact manifest. No weights or tokenizer vocabulary are modified. Upstream and fine-tuned file metadata were checked on 2026-10-27 without downloading weights. A missing revision or unrecognized file difference fails validation rather than substituting an updated model.

Downloads are cached under `models/`. Inference generates at most 256 new tokens with thinking disabled; the helper is a first-pass example, not the historical 24-case repair pipeline or a benchmark runner. No guarantee is made that arbitrary review input will produce a valid JSON object.

Text files use LF line endings through `.gitattributes` so `SHA256SUMS.json` agrees across Windows and Linux checkouts. The manifest excludes itself and the local `.git` directory.

## Formal dataset download

The formal 33,385-sample dataset is published separately on ModelScope at revision
`8275415535d4ed2d4f7dfd8ad3ef964a8a3eeae4`. Its three file hashes match the frozen
experiment splits. Install the data-download dependency and request a download:

```bash
pip install -r environment/requirements-data.txt
python scripts/download_data.py
```

The service returned a login requirement for anonymous file requests during the
2026-10-27 check. Authenticate locally through the official ModelScope SDK/CLI
when required; browser login alone is not necessarily SDK login. Never put access
tokens into code or commits. The helper uses the SDK's local authentication state
and refuses to report success unless all three file hashes match.

By default, the snapshot layout is preserved under `data/formal/`: the train,
validation and test files are `data/formal/data/train.jsonl`,
`data/formal/data/validation.jsonl` and `data/formal/data/test.jsonl`.
The downloader prints the verified paths. `--destination` can select another cache
root; `--dry-run` only shows settings without network or credentials. Metadata
and download-endpoint access were checked, but this integration did not perform
an authenticated full download, retraining, inference or formal statistical rerun.

## Training and evaluation

Training is a separate opt-in entry point. It downloads the frozen baseline if needed, validates its hashes, then invokes the historical trainer with explicit paths:

```bash
python scripts/train.py --train-file data/formal/data/train.jsonl --eval-file data/formal/data/validation.jsonl --output-dir runs/new_training --final-dir runs/new_merged
```

Only Level-3 training records are optimized; only Level-3 validation records select the checkpoint. Loss is computed on assistant-answer tokens only. This repository preparation did not run training.

Domain metrics can be recomputed from existing generation files without inference:

```bash
python src/eval_qwen35_4b_chat_l3_unsloth.py --phase metrics --project-root . --test-file /path/to/test.jsonl --run-dir /path/to/existing_generation_run
```

That command writes metrics into the supplied run directory. Work on a copy of archived results. See `docs/REPRODUCIBILITY.md` for generation commands, benchmark environments, the length-fallback boundary, and data access limitations.

## Main findings

Review-macro note-set F1 increased from 0.1513 to 0.2948 overall and from 0.2111 to 0.3316 on Level 3. The output terminology became more aligned with professional note labels, but several frequent labels were overpredicted. General benchmark changes were task-specific. The 24-request transfer analysis is researcher-led and exploratory, not a human-preference experiment or a validated personalized perfume designer.

## Data and license

See `data/README.md` for the public formal dataset, source-rights notice, and distinction between synthetic examples and experimental samples. The raw 119,093-record source spreadsheet is not included. The code is released under Apache-2.0. Model and data rights are separate; the code license does not grant rights to third-party review text or benchmark datasets.
