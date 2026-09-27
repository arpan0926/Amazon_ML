# Business Entity Resolution Instructions

- All tabular input and output files are tab-separated TSV files. Read them with `pd.read_csv(file, sep='\t')` and write them with `DataFrame.to_csv(file, sep='\t', index=False)`. Preserve IDs as strings where possible.
- Use only `business_name` and `business_address` for matching features. Do not use external lookups or other data sources. `country` is open-set metadata: never restrict, map, or filter countries; every test Source 1 entity, including France, must be included.
- Apply consistent preprocessing to both blocking and RapidFuzz inputs: NFC Unicode normalization, lowercase, expand `&` to `and`, replace punctuation with spaces, and collapse whitespace. For addresses, canonicalize only defined whole-token street/suite abbreviations using the same Python/SQL mapping. Treat null/NaN text as empty. Keep original source strings for ID joins and output.
- Generate candidates with an in-memory DuckDB connection and C++ hash joins over lowercased exact name, first name word plus first address number, and lowercased exact address. Union the three key joins, deduplicate by Source 1/vendor ID, rank by vendor ID, and cap at 30 candidates per Source 1 by default.
- Always use the C++-backed `rapidfuzz` library for string matching. Never use `fuzzywuzzy` or `difflib`.
- Never use `df.apply(axis=1)` for string comparison over large dataframes. Build features with list comprehensions and `zip()`, or use `rapidfuzz.process.cdist`.
- Process candidate pairs in chunks when needed. Store numeric feature columns as `np.float32` and explicitly garbage-collect temporary lists to limit memory use.
- During cross-validation, use `sklearn.model_selection.GroupKFold`, grouped strictly by `source1_entity_id` (the challenge's Source 1 ID column). Pairs from one Source 1 entity must not appear in both training and validation folds.
- Score thresholds with the specified formula: `F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)`. Avoid false merges; the output must represent unmatched Source 1 entities with an empty string, never `NaN` or `None`.
- Generate both `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Candidate output must be the final exact set scored by inference, contain one row for every Source 1 ID, contain only existing Source 2/3 IDs, have no duplicates, and include every predicted match.
- Before final packaging, run the validator as a Python subprocess: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`.

## Pipeline Input Contract

Source records are `dataset/{train,test}/{train,test}_source{1,2,3}.tsv` with `entity_id`, `business_name`, `business_address`, and `country`. In the workspace, the same tree is discovered under `student_resource/dataset/` when it is not at the repository root. Ground truth is `dataset/train/train_ground_truth.tsv` with `source1_entity_id` and comma-separated `matched_entity_ids`. Submission columns are exactly `source1_entity_id`/`matched_entity_ids` and `source1_entity_id`/`candidate_entity_ids`.