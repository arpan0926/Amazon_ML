# AI Project Context

This file is a handoff for AI assistants working on this repository. Read it before changing the pipeline; then inspect the linked implementation and tests rather than assuming this document is newer than the code.

## Project Goal

Resolve every Source 1 business record to zero or more matching Source 2 and Source 3 records using only `business_name` and `business_address`. Source IDs have prefixes `S1-`, `S2-`, and `S3-`. Do not use external lookups. Do not filter on country: France appears in test data and must remain included.

All source and output data are TSVs. Read with an explicit tab separator and preserve IDs as strings. The leaderboard metric is per-Source-1 macro F0.5, including singletons:

`F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)`

A correct empty prediction for a singleton scores 1.0; any false prediction for it scores 0.0. False merges are costly.

## Current Implementation

The main code is [pipeline.py](code/business_entity_resolution/src/pipeline.py); the root launcher is [business_entity_resolution.py](business_entity_resolution.py).

- `generate_candidate_pairs_duckdb()` registers Source 1 and vendor DataFrames in an in-memory DuckDB connection.
- It constructs exact lowercased/trimmed name, first name word plus first address number, and exact lowercased/trimmed address keys in SQL.
- Three joins are combined with `UNION ALL`, deduplicated by Source 1/vendor ID, ranked by vendor ID, and capped at 30 per Source 1 by default. The function returns exactly `source1_entity_id` and `vendor_entity_id` and closes DuckDB plus calls `gc.collect()`.
- The model path joins those IDs back to names and addresses, calculates chunked RapidFuzz features, trains CatBoost with `GroupKFold` grouped by `source1_entity_id`, chooses a threshold from OOF scores, and scores test candidates.
- Training currently uses a deterministic sample of up to 50,000 Source 1 entities (`--max-training-groups`). Candidate pairs receive binary labels by checking the ground-truth ID lists.
- Similarity features: name token-sort ratio, name token-set ratio, name Jaro-Winkler, address token-set ratio, address partial ratio, absolute name/address length differences, and address digit overlap.
- Inference output `output/candidate_pairs.tsv` uses the challenge validator's aggregated schema (`source1_entity_id`, `candidate_entity_ids`), not the internal two-column DuckDB pair schema. Final matches are a subset of those candidates and use `source1_entity_id`, `matched_entity_ids`.
- Unmatched Source 1 records are emitted with an empty string. Both output files have one row per test Source 1 ID.

Source files are under `student_resource/dataset/{train,test}/` in this workspace. The pipeline also supports a root-level `dataset/` tree. The validator is `student_resource/utils/validate_submission.py`.

## Current Snapshot

The workspace currently has `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `output/oof_predictions.tsv`, a root `matching_results.tsv`, and `SPD_Rangers_submission.zip`. The two submission files were streamed and checked for expected headers and row counts: each has 1,732,544 data rows. The OOF file has 468,980 candidate rows. The ZIP was inspected and contains both outputs plus the source package, package README/requirements, and methodology document.

Recomputing the saved OOF probabilities against the deterministic 50,000-entity training sample, including all sampled Source 1 entities with no generated candidates, gives macro F0.5 **0.654753** at threshold **0.50**. This is a sampled OOF threshold-selection diagnostic, not an independent validation estimate; the same OOF predictions were used to select the threshold. Do not call it a leaderboard score. The real submission validator was not independently run as part of this context-file task; run it before submission.

Four unit tests currently pass. They cover DuckDB's three keys, deduplication, top-k, enrichment into RapidFuzz columns, singleton serialization, threshold tie-breaking, and deterministic training sampling. The DuckDB join was previously timed on a subset of 100,000 Source 1 and 500,000 vendor rows: 0.68 seconds for the join and 2.24 seconds including TSV reads/concatenation. That subset timing does not establish full 12M-row runtime or memory use. The earlier SQLite full-scale attempts were stopped and are obsolete; SQLite code has since been removed.

## Environment And Commands

Use Python 3.13 in this Windows workspace. Plain `python` points to Python 3.14 and previously lacked project dependencies. Pinned dependencies are in [requirements.txt](requirements.txt) and [the package requirements](code/business_entity_resolution/requirements.txt).

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m unittest discover -s tests -v
py -3.13 business_entity_resolution.py
py -3.13 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
py -3.13 package_submission.py --team-name SPD_Rangers
```

The repository has a root README, `.gitignore`, and a GitHub Actions workflow at `.github/workflows/tests.yml`. The workspace was not initialized as a Git repository when checked. Local datasets, output files, macOS metadata, and submission ZIPs are ignored by Git; do not commit challenge data.

## Prioritized Next Steps For F0.5

1. **Validate the current outputs first.** Run the validator command above and fix any formatting or candidate-subset warnings before interpreting model quality.
2. **Measure blocking recall and candidate reduction on a held-out Source 1 split.** Split strictly by Source 1 ID before tuning. Report the fraction of all true vendor IDs present in candidates, candidates per Source 1, candidate reduction ratio, singleton accuracy, and macro F0.5. Include every held-out Source 1 ID, even with no candidates.
3. **Improve top-30 ranking.** Current ranking is lexicographic by vendor ID, not match evidence. This can discard true matches whenever an entity has more than 30 candidates. Compare ranking by key-hit evidence and inexpensive RapidFuzz name/address scores before applying the cap; retain deterministic tie-breaking.
4. **Improve blocking recall without exploding candidates.** The current exact name/address and first-word-plus-number keys are intentionally restrictive. Test additional selective keys such as normalized token/character n-grams, address-number plus postal/locality fragments when derivable from the address string, and alternate name token keys. Keep only allowed input fields. Measure marginal recall and candidate count per key.
5. **Tune the threshold on independent validation.** The reported 0.654753 is optimized on the same OOF set it evaluates. Use a source-entity holdout or nested/grouped validation for threshold selection; calculate the challenge macro metric over all validation Source 1 IDs, including candidate-free singletons.
6. **Tune model and features against precision-heavy score.** Inspect false-positive categories, calibrate probabilities, and compare CatBoost settings/features using the held-out macro F0.5. Avoid any tuning on test labels (none are provided).
7. **Benchmark full-data resources.** The DuckDB path currently loads full vendor DataFrames into pandas before registering them and then materializes candidate pairs. Measure peak RAM, full join time, pair count, and output size on the actual machine. The earlier subset benchmark is not proof that the full run fits memory or completes under five minutes.

Keep tests focused and add a regression test whenever changing SQL keys, tie-breaking, caps, or output schemas. Preserve the challenge's exact TSV headers, empty-string singleton convention, group isolation, and test-country coverage.
