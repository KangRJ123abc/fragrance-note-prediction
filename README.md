# Structured fragrance note prediction from consumer reviews

Code and model access instructions for **Mapping Fragrance Reviews to Structured Olfactory Notes with a Domain-Fine-Tuned Large Language Model**.

The domain-fine-tuned model maps a single anonymized review to the publicly listed top, middle, and base notes of its perfume. These are perfume-level weak labels, not a full reconstruction of an individual reviewer's experience. The published model is merged from the best **step-700** checkpoint. Experiments were frozen in July 2026.

## What is included

- `src/`: training, dataset construction, domain evaluation, general benchmarks, and exploratory application-transfer code.
- `scripts/`: pinned model download, single-review inference, explicit training launcher, file verification, and descriptive post-processing.
- Release-time reconstructed length-fallback, fixed-prior and permutation scripts are included, with provenance and unverified historical choices documented in `docs/RECONSTRUCTION.md`. They have not been run on formal data or a model.
- `configs/`: model sources, artifact hashes, generation settings, and dataset checksums.
- `data/`: the researcher-constructed 24-request bank and a clearly synthetic format example; no third-party consumer-review text.
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

The baseline ModelScope ID and recorded revision come from the original local download metadata; downloaded files must match the frozen SHA-256 manifest. If that revision is no longer served, the helper fails rather than silently substituting another model. The author's fine-tuned ModelScope ID and fixed revision must be filled in `configs/model_sources.json` after the repository is created. Until then use `--model-dir`.

Downloads are cached under `models/`. Inference generates at most 256 new tokens with thinking disabled; the helper is a first-pass example, not the historical 24-case repair pipeline or a benchmark runner. No guarantee is made that arbitrary review input will produce a valid JSON object.

## Training and evaluation

Training is a separate opt-in entry point. It downloads the frozen baseline if needed, validates its hashes, then invokes the historical trainer with explicit paths:

```bash
python scripts/train.py --train-file /path/to/train.jsonl --eval-file /path/to/val.jsonl --output-dir runs/new_training --final-dir runs/new_merged
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

See `data/README.md` for unavailable third-party review data and the distinction between synthetic examples and formal samples. The code is released under Apache-2.0. Upstream model licensing is separate and retained in the model repository. No rights to third-party review text or benchmark datasets are granted by the code license.
