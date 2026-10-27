# Data access and format

The formal review-to-note dataset is now published separately:
https://www.modelscope.cn/datasets/KangRJ123abc/Fragrance-Review-Note-Pyramids

The pinned revision is 8275415535d4ed2d4f7dfd8ad3ef964a8a3eeae4. The three files
preserve the frozen experiment bytes, with shortened filenames. They are not
embedded in this GitHub repository. The unscreened 119,093-record raw spreadsheet
and historical generation archives are not redistributed by this code release.

Run `python scripts/download_data.py --dry-run` to inspect configuration, or
`python scripts/download_data.py` to request the data through the official
ModelScope SDK. The endpoint required account login for anonymous file requests
on 2026-10-27; use local SDK authentication when required. No token belongs in this
repository. The helper validates frozen hashes and reuses an already valid cache.

Default downloaded paths:
- train: data/formal/data/train.jsonl
- validation: data/formal/data/validation.jsonl
- test: data/formal/data/test.jsonl

The validation filename retains metadata.split=val. Downloaded caches are ignored
by Git. See configs/dataset_sources.json for the repository paths and revision.

The formal dataset has 26,646 training, 3,450 validation, and 3,289 test records,
covering 421, 52, and 52 disjoint perfumes. Only Level-3 training (11,487) and
validation (1,490) samples are used for fitting and checkpoint selection.
The test set includes 1,832 Level-2 and 1,457 Level-3 reviews.

`synthetic_review_example.jsonl` is invented solely to demonstrate the schema.
It is not an experimental sample or a source of reported metrics.
`user_request_note_design_pilot_v2_thinking.jsonl` contains the 24 researcher-
constructed scent briefs used by the formal visible-planning transfer analysis.
The filename is historical: the formal protocol disables native thinking and
uses visible planning. It is not a survey or human-preference dataset.

Checksums under the original frozen filenames are in `configs/dataset_files.json`;
the corresponding published paths are mapped in `configs/dataset_sources.json`.
To use the original trainer, supply local formal split paths explicitly.
Row ID, perfume identity, and source URL must never enter inference prompts.

Code licensing does not authorize reuse or redistribution of third-party data.
Read SOURCE_AND_RIGHTS.md in the dataset repository for source attribution and
rights information; the software's Apache-2.0 license is not a blanket data license.
