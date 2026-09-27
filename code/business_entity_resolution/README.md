# Business Entity Resolution

Reproducible CatBoost pipeline for the three-source business matching challenge. It uses only business names and addresses for blocking and matching; it does not call external services or use country to filter records. Country labels remain open-set, so all test Source 1 records are retained, including records from countries absent from training.

## Environment

Use Python 3.11 or newer with the pinned dependencies:

```powershell
python -m pip install -r code/business_entity_resolution/requirements.txt
```

The model is CatBoost (Apache-2.0). RapidFuzz provides the C++-backed string similarity functions.

## Input Layout

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

Source TSVs use `entity_id`, `business_name`, `business_address`, and `country`. Ground truth uses `source1_entity_id` and comma-separated `matched_entity_ids`.

## Run

From the repository root (or the unpacked submission archive):

```powershell
python code/business_entity_resolution/src/pipeline.py --train-dir dataset/train --test-dir dataset/test --ground-truth dataset/train/train_ground_truth.tsv --output-dir output
```

In the editable workspace, `python business_entity_resolution.py` is a short equivalent.

Candidate generation uses DuckDB's in-memory C++ hash joins over consistently normalized exact name, first name word plus first address number, exact address, and legal-suffix-stripped name. Text is NFC-normalized, lowercased, ampersands expand to `and`, punctuation is tokenized to spaces, whitespace is collapsed, and whole-token address abbreviations are canonicalized consistently in SQL and Python. Candidate pairs are ranked by distinct key-hit count, a 70/30 weighted DuckDB Jaro-Winkler score for name/address (using the better raw or suffix-stripped name score), normalized character-length gap, and vendor ID for deterministic ties before the top-30 cap. The ID-pair function returns exactly `source1_entity_id` and `vendor_entity_id`; the pipeline enriches those pairs with source attributes before RapidFuzz feature generation. Model training uses a deterministic sample of up to 50,000 Source 1 groups by default (`--max-training-groups`) to bound training memory. Nested threshold selection also tunes a per-Source-1 relative-confidence floor from 0.0 to 0.95: a candidate must pass the global probability threshold and remain sufficiently close to that Source 1's highest candidate probability. A 0.90 relative floor improved saved OOF held-out macro F0.5 slightly (0.770841 to 0.771830) on the current fixed split; this is a validation estimate, not a leaderboard result. Tune with `--max-candidates` and `--max-training-groups`; run `python business_entity_resolution.py --help` for all options. The run writes `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and `output/oof_predictions.tsv`, then invokes the provided validator with `--matching`, `--candidate`, and `--test-dir` arguments. The candidate file uses the challenge-required aggregated header `source1_entity_id`/`candidate_entity_ids`; it is generated from the exact ID pairs sent to feature extraction. Validation can be skipped for local development with `--skip-validation`.

After a validated run, assemble the requested archive from the repository root:

```powershell
python package_submission.py --team-name your_team
```

The script writes `your_team_submission.zip` with both outputs, this code package, and the methodology document.

## Method

The four-key join avoids a full Cartesian comparison. Source TSVs are loaded into pandas DataFrames and registered with an in-memory DuckDB connection; candidate IDs are joined back to their source names and addresses before model scoring.

The model uses name token-sort ratio, name token-set ratio, name Jaro-Winkler, address token-set ratio, address partial ratio, absolute normalized name/address length differences, normalized address digit overlap, one exact-match flag per blocking key, blocking-key match count, and RapidFuzz name ratio after legal suffix removal. Features are generated with chunked `zip()` list comprehensions and stored as float32. GroupKFold splits candidate rows strictly by `source1_entity_id`. Threshold selection uses one deterministic group subset; a separate held-out group subset supplies the reported macro F0.5 estimate, including singleton credit.