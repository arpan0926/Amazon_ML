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
- It constructs exact lowercased/trimmed name, first name word plus first address number, exact lowercased/trimmed address, and legal-suffix-stripped name keys in SQL.
- Four joins are combined with `UNION ALL`, deduplicated by Source 1/vendor ID, ranked by distinct key-hit count, weighted DuckDB Jaro-Winkler name/address score (best of raw or suffix-stripped name), normalized length gap, and vendor ID, then capped at 30 per Source 1 by default. The function returns exactly `source1_entity_id` and `vendor_entity_id` and closes DuckDB plus calls `gc.collect()`.
- The model path joins those IDs back to names and addresses, calculates chunked RapidFuzz features, trains CatBoost with `GroupKFold` grouped by `source1_entity_id`, chooses a threshold from OOF scores, and scores test candidates.
- Training currently uses a deterministic sample of up to 50,000 Source 1 entities (`--max-training-groups`). Candidate pairs receive binary labels by checking the ground-truth ID lists.
- Similarity features: name token-sort ratio, name token-set ratio, name Jaro-Winkler, address token-set ratio, address partial ratio, absolute name/address length differences, address digit overlap, four exact blocking-key flags, blocking-key match count, and legal-suffix-stripped name similarity. Python and SQL use the same suffix regex.
- Text preprocessing is shared across SQL blocking keys and Python features: NFC Unicode normalization, lowercase, ampersand-to-`and`, punctuation-to-space tokenization, whitespace collapse, and whole-token address aliases (`street/st`, `road/rd`, `avenue/ave`, `boulevard/blvd`, `drive/dr`, `lane/ln`, `highway/hwy`, `suite/ste`, `apartment/apt`). Missing text values normalize to empty; original fields are retained for output.
- Threshold selection deterministically splits the sampled Source 1 groups in half. Subset A selects the threshold; subset B reports the held-out macro F0.5 estimate. `main()` prints both scores and the group counts.
- Threshold selection jointly tunes a global probability threshold and minimum per-Source-1 relative confidence from `(0.0, 0.5, 0.7, 0.8, 0.9, 0.95)`. Inference requires both `p >= threshold` and `p >= relative_confidence * max_p_for_source1`. Saved OOF diagnostics selected threshold 0.79 and relative floor 0.90; held-out estimate 0.771830 versus 0.770841 with no relative floor on the same split. This is a small, provisional validation gain, not a leaderboard score.
- The optional `--report-sample-sensitivity` flag evaluates 10k/50k/100k requested group samples using the same blocking, features, GroupKFold OOF, and nested threshold procedure, then exits before test inference. The default path and 50k default are unchanged.
- Inference output `output/candidate_pairs.tsv` uses the challenge validator's aggregated schema (`source1_entity_id`, `candidate_entity_ids`), not the internal two-column DuckDB pair schema. Final matches are a subset of those candidates and use `source1_entity_id`, `matched_entity_ids`.
- Unmatched Source 1 records are emitted with an empty string. Both output files have one row per test Source 1 ID.

Source files are under `student_resource/dataset/{train,test}/` in this workspace. The pipeline also supports a root-level `dataset/` tree. The validator is `student_resource/utils/validate_submission.py`.

## Current Snapshot

The workspace has `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `output/oof_predictions.tsv`, a root `matching_results.tsv`, and `SPD_Rangers_submission.zip`. These artifacts were generated before changes 1-4 below and are stale relative to the current pipeline; regenerate and validate them before submission. Their earlier headers and row counts were checked (1,732,544 rows in each submission file and 468,980 candidate OOF rows), and the ZIP contents were inspected.

The pre-change saved OOF probabilities scored macro F0.5 **0.654753** at threshold **0.50** on the same data used for threshold selection. This is a historical, potentially optimistic diagnostic, not a current independent validation estimate or leaderboard score. The real submission validator was not independently run as part of the prior context-file task; run it on regenerated outputs.

Fourteen unit tests currently pass. They cover the four DuckDB keys, similarity ranking, deduplication, top-k, attribute enrichment, singleton serialization, preprocessing parity, nested threshold selection, float32 features, class weighting and imbalanced OOF behavior, sensitivity opt-in, and deterministic sampling. On a deterministic 10k train-reference subset, new ranking found 21,465/34,707 true pairs (61.85%) at top-30, versus 21,214/34,707 (61.12%) for the previous key-count/length-gap/ID ranking, with the same 124,904 candidates. This indicates a modest blocking-recall gain, not a measured macro-F0.5 gain. The separate 100k/500k DuckDB timing does not establish full 12M-row runtime or memory use. Earlier SQLite full-scale attempts were stopped and SQLite code has since been removed.

## Environment And Commands

Use Python 3.13 in this Windows workspace. Plain `python` points to Python 3.14 and previously lacked project dependencies. Pinned dependencies are in [requirements.txt](requirements.txt) and [the package requirements](code/business_entity_resolution/requirements.txt).

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m unittest discover -s tests -v
py -3.13 business_entity_resolution.py
py -3.13 business_entity_resolution.py --report-sample-sensitivity
py -3.13 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
py -3.13 package_submission.py --team-name SPD_Rangers
```

The repository has a root README, `.gitignore`, and a GitHub Actions workflow at `.github/workflows/tests.yml`. The workspace was not initialized as a Git repository when checked. Local datasets, output files, macOS metadata, and submission ZIPs are ignored by Git; do not commit challenge data.

## Prioritized Next Steps For F0.5

1. **Validate the current outputs first.** Run the validator command above and fix any formatting or candidate-subset warnings before interpreting model quality.
2. **Measure blocking recall and candidate reduction on a held-out Source 1 split.** Split strictly by Source 1 ID before tuning. Report the fraction of all true vendor IDs present in candidates, candidates per Source 1, candidate reduction ratio, singleton accuracy, and macro F0.5. Include every held-out Source 1 ID, even with no candidates.
3. **Measure the F0.5 effect of reranking and relative filtering.** SQL similarity reranking improved recall@30 by 0.72 percentage points on a 10k sample at unchanged candidate count. The relative-confidence floor showed a small held-out OOF gain. Regenerate predictions and validate both changes together before attributing a leaderboard gain.
4. **Measure the legal-suffix key.** Compare candidate recall, false candidate volume, and per-key contribution on held-out Source 1 groups. Consider other address/name keys only when they materially improve recall without excessive candidate growth.
5. **Re-evaluate threshold performance.** The new selection/held-out split is deterministic and group-safe; regenerate OOF data and compare selection score with held-out estimate. Do not tune repeatedly against the same held-out half and call it an unbiased final estimate.
6. **Tune CatBoost/features against held-out macro F0.5.** Inspect false-positive categories, calibrate probabilities, and compare classifier settings using held-out Source 1 groups. Never tune on test labels.
7. **Benchmark full-data resources.** The DuckDB path loads vendor DataFrames into pandas before registering them and materializes candidate pairs. Measure peak RAM, full join time, pair count, and output size on the actual machine; the subset timing does not establish a five-minute full run.

Keep tests focused and add a regression test whenever changing SQL keys, tie-breaking, caps, or output schemas. Preserve the challenge's exact TSV headers, empty-string singleton convention, group isolation, and test-country coverage.
