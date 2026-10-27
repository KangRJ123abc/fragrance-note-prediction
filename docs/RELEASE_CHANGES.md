# Release-only changes

Original project files are untouched. The following edits affect only release copies:

1. Training and domain scripts resolve the project root above `src/` rather than treating `src/` as the root.
2. The general benchmark launcher requires an explicit `--vllm-executable`; the personal historical executable path is removed.
3. The application analysis no longer creates questionnaire files or private A/B keys. Historical unused questionnaire helper functions are omitted. The generated report describes researcher-led exploratory analysis only.
4. The historical DeepSeek script remains available for prompt and preprocessing inspection, but its command-line entry refuses execution. This prevents accidental paid API runs; it is not a preprocessing quick-start command.
5. New download, inference, training-launch, verification, and test helpers do not change the frozen training or evaluation settings. They have only CPU/static validation in this release.
6. Historical environment snapshots retain version information but remove local installation paths. Private review data and canceled experimental outputs are excluded.
7. The same-test note audit accepts explicit private input paths, retaining its original frozen-hash checks. No new model generation or statistical test is added.

New wrappers are release tooling, not scripts executed in the July 2026 experiments.

Release-time reconstruction now adds a planning-default length fallback and a
fixed-prior/permutation executor. These are newly written implementations, not
recovered originals. Unknown historical choices are explicit in configuration.
Only synthetic unit tests and static/help checks were run; no formal result was
recomputed and no model was loaded. See `docs/RECONSTRUCTION.md`.

The companion model's config.json contains one metadata-only edit: `model_name`
now records the public upstream ID instead of a private server path. All other
configuration fields and the full weight shards remain unchanged. Public model
hash manifests identify these release bytes, not the unredacted historical config.
