# Business Entity Resolution

A Python pipeline for matching Source 1 business records against Source 2 and Source 3 using DuckDB candidate generation, RapidFuzz features, CatBoost, and grouped out-of-fold validation.

## Requirements

- Python 3.13 (the pinned CatBoost, DuckDB, and scientific Python wheels were verified with this runtime)
- The challenge TSV files in `dataset/train/` and `dataset/test/`, or under `student_resource/dataset/`

Install the pinned dependencies on Windows:

```powershell
py -3.13 -m pip install -r requirements.txt
```

The challenge data and generated result files are excluded by `.gitignore`; do not commit private or multi-gigabyte source datasets to GitHub.

## Run

From the repository root:

```powershell
py -3.13 business_entity_resolution.py
```

The pipeline also accepts explicit `--train-dir`, `--test-dir`, `--ground-truth`, and `--output-dir` options. Run `py -3.13 business_entity_resolution.py --help` to see all settings. It writes:

- `output/matching_results.tsv`: final challenge submission
- `output/candidate_pairs.tsv`: the final blocked candidates, in the validator's required aggregated format
- `output/oof_predictions.tsv`: sampled out-of-fold scores used for threshold selection

The run invokes `utils/validate_submission.py` (in `student_resource/` in the supplied workspace) after generating both submission files.

## Tests

```powershell
py -3.13 -m unittest discover -s tests -v
```

GitHub Actions runs this test suite on pushes and pull requests.

## Packaging

After a validated run:

```powershell
py -3.13 package_submission.py --team-name SPD_Rangers
```

This creates `<team-name>_submission.zip` in the repository root. The archive contains both required output TSVs, the source package, its README and pinned requirements, and `Documentation_template.md`.

## Project Layout

```text
.
|-- .github/workflows/tests.yml
|-- code/business_entity_resolution/
|   |-- README.md
|   |-- requirements.txt
|   `-- src/pipeline.py
|-- tests/test_pipeline.py
|-- dataset/ or student_resource/dataset/   # local challenge data; ignored by Git
|-- output/                                # generated results; ignored by Git
|-- Documentation_template.md
|-- business_entity_resolution.py
|-- instructions.md
|-- package_submission.py
`-- requirements.txt
```

The matching pipeline uses only business name and address; country is not used to block or filter records. See [Documentation_template.md](Documentation_template.md) for the methodology summary.
