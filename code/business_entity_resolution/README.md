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

Candidate generation uses DuckDB's in-memory C++ hash joins over exact normalized business name, first name word plus first address number, and exact normalized address. Matching keys are built inside DuckDB, the three joins are unioned, duplicate ID pairs are removed, and candidates are capped at 30 per Source 1 ID by vendor ID. The ID-pair function returns exactly `source1_entity_id` and `vendor_entity_id`; the pipeline enriches those pairs with source attributes before RapidFuzz feature generation. Model training uses a deterministic sample of up to 50,000 Source 1 groups by default (`--max-training-groups`) to bound training memory. Tune with `--max-candidates` and `--max-training-groups`; run `python business_entity_resolution.py --help` for all options. The run writes `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and `output/oof_predictions.tsv`, then invokes the provided validator with `--matching`, `--candidate`, and `--test-dir` arguments. The candidate file uses the challenge-required aggregated header `source1_entity_id`/`candidate_entity_ids`; it is generated from the exact ID pairs sent to feature extraction. Validation can be skipped for local development with `--skip-validation`.

After a validated run, assemble the requested archive from the repository root:

```powershell
python package_submission.py --team-name your_team
```

The script writes `your_team_submission.zip` with both outputs, this code package, and the methodology document.

## Method

The three-key join is deliberately restrictive and avoids a full Cartesian comparison. Source TSVs are loaded into pandas DataFrames and registered with an in-memory DuckDB connection; candidate IDs are joined back to their source names and addresses before model scoring.

The model uses name token-sort ratio, name token-set ratio, name Jaro-Winkler, address token-set ratio, address partial ratio, absolute name/address length differences, and an address digit overlap flag. Features are generated with chunked `zip()` list comprehensions and stored as float32. GroupKFold splits candidate rows strictly by `source1_entity_id`; threshold selection evaluates macro F0.5 over the sampled training Source 1 groups, assigning 1.0 to a correctly predicted singleton. The final threshold is selected from 0.50 through 0.95 using OOF probabilities.