# Data access and format

No Fragrantica consumer-review text is included in this public code folder.
Raw reviews, anonymized reviews, product identities, URLs, and generation files
containing the source reviews remain excluded pending the authors' data-sharing
decision. This release must not be described as a fully public training dataset.

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

Checksums for the formal private splits are in `configs/dataset_files.json`.
To use the original trainer, supply local formal split paths explicitly.
Row ID, perfume identity, and source URL must never enter inference prompts.

Code licensing does not authorize reuse or redistribution of third-party data.
