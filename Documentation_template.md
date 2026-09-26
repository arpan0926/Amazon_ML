# Methodology

## Methodology Used

The pipeline reads the supplied train and test source files as tab-separated dataframes. Matching uses only `business_name` and `business_address`; it makes no external lookups and does not filter by `country`. Every test Source 1 ID is retained in both output tables, including IDs from the unseen France country.

## Candidate Generation / Blocking

An in-memory DuckDB query creates candidates through three hash joins: exact lowercased/trimmed business name, first business-name word concatenated with the first address number, and exact lowercased/trimmed address. The three join results are combined with `UNION ALL`, deduplicated by Source 1 and vendor ID, ranked by vendor ID within each Source 1 group, and limited to 30 by default. The two-column ID-pair result is joined back to source attributes for model features. The final candidate set sent to inference is written to `output/candidate_pairs.tsv` in the challenge-required aggregated format, making the match output a subset of the audited candidate set.

This approach avoids comparing every Source 1 record with every vendor record. DuckDB builds the in-memory hash tables and performs the multi-key joins in its C++ engine. The 30-pair cap bounds downstream feature and model work; blocking keys and the cap should be evaluated on held-out training data for recall and candidate-set size.

## Model Architecture and Feature Engineering

CatBoostClassifier is trained on RapidFuzz name token-sort ratio, name token-set ratio, name Jaro-Winkler, address token-set ratio, address partial ratio, absolute name/address character-length differences, and address digit overlap. Similarity values and derived features are stored as float32 and computed in chunks without row-wise DataFrame apply.

## Validation and Threshold Selection

Out-of-fold predictions use GroupKFold with `source1_entity_id` as the grouping key, preventing pairs belonging to one reference entity from crossing folds. By default, a deterministic sample of up to 50,000 training Source 1 groups is used to bound model-training memory; threshold selection evaluates macro-averaged F0.5 over those sampled IDs from 0.50 to 0.95. A correct empty prediction for a singleton scores 1.0; an incorrect non-empty prediction scores 0.0.

## Outputs and Reproducibility

The run creates `output/matching_results.tsv` with columns `source1_entity_id` and `matched_entity_ids`, and `output/candidate_pairs.tsv` with columns `source1_entity_id` and `candidate_entity_ids`. Both are tab-separated, contain one row per test Source 1 entity, use empty strings for empty lists, and contain deduplicated comma-separated vendor IDs. The pipeline invokes the provided submission validator before completing.

Run from the repository root with `python business_entity_resolution.py`; pinned dependencies are in `code/business_entity_resolution/requirements.txt`.