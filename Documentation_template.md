# Methodology

## Methodology Used

The pipeline reads the supplied train and test source files as tab-separated dataframes. Matching uses only `business_name` and `business_address`; it makes no external lookups and does not filter by `country`. Text normalization is shared across DuckDB blocking and RapidFuzz features: NFC Unicode normalization, lowercase, `&` expansion to `and`, punctuation-to-space tokenization, whitespace collapse, and whole-token address abbreviation canonicalization. Missing text values normalize to empty strings; original source fields remain unchanged for output. Every test Source 1 ID is retained in both output tables, including IDs from the unseen France country.

## Candidate Generation / Blocking

An in-memory DuckDB query creates candidates through four hash joins: normalized exact name, first name word plus first address number, normalized exact address, and legal-suffix-stripped name. The results are combined with `UNION ALL`, deduplicated by Source 1 and vendor ID, then ranked by distinct matched-key count; a weighted name/address Jaro-Winkler score (favoring the higher raw or suffix-stripped name score); normalized character-length gap; and vendor ID for deterministic ties. Candidates are limited to 30 per Source 1 by default. On a deterministic 10k training-reference sample, this reranking increased recall@30 from 61.12% to 61.85% at the same 124,904 candidate pairs. This is a candidate-recall diagnostic only, not proof of an F0.5 gain. The two-column ID-pair result is joined back to source attributes for model features. The final candidate set sent to inference is written to `output/candidate_pairs.tsv` in the challenge-required aggregated format, making the match output a subset of the audited candidate set.

This approach avoids comparing every Source 1 record with every vendor record. DuckDB builds the in-memory hash tables and performs the multi-key joins in its C++ engine. The 30-pair cap bounds downstream feature and model work; blocking keys and the cap should be evaluated on held-out training data for recall and candidate-set size.

## Model Architecture and Feature Engineering

CatBoostClassifier is trained on RapidFuzz name token-sort ratio, name token-set ratio, name Jaro-Winkler, address token-set ratio, address partial ratio, absolute normalized name/address character-length differences, normalized address digit overlap, exact-match flags for all four blocking keys, a blocking-key match count, and suffix-stripped name similarity. Similarity values and derived features are stored as float32 and computed in chunks without row-wise DataFrame apply. Python feature key normalization uses the same legal-suffix and address-alias rules as the DuckDB SQL key builder. At inference, a candidate must pass the globally selected probability threshold and a selected per-Source-1 relative-confidence floor compared with that entity's strongest candidate. The relative floor is tuned on the threshold-selection half only.

## Validation and Threshold Selection

Out-of-fold predictions use GroupKFold with `source1_entity_id` as the grouping key, preventing pairs belonging to one reference entity from crossing folds. By default, a deterministic sample of up to 50,000 training Source 1 groups is used to bound model-training memory. A deterministic group split selects the probability threshold and relative-confidence floor on one half and reports macro F0.5 on the held-out half, including entities with no candidates. A correct empty prediction for a singleton scores 1.0; an incorrect non-empty prediction scores 0.0. In the current saved OOF diagnostic, threshold 0.79 plus relative floor 0.90 yields a held-out estimate of 0.771830, versus 0.770841 without the relative floor on the same split; treat this small difference as provisional, not leaderboard performance.

## Outputs and Reproducibility

The run creates `output/matching_results.tsv` with columns `source1_entity_id` and `matched_entity_ids`, and `output/candidate_pairs.tsv` with columns `source1_entity_id` and `candidate_entity_ids`. Both are tab-separated, contain one row per test Source 1 entity, use empty strings for empty lists, and contain deduplicated comma-separated vendor IDs. The pipeline invokes the provided submission validator before completing.

Run from the repository root with `python business_entity_resolution.py`; pinned dependencies are in `code/business_entity_resolution/requirements.txt`.