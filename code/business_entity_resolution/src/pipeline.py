"""Block, score, and export business entity matches for the challenge."""

import argparse
import gc
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.model_selection import GroupKFold


ID_COLUMN = "entity_id"
SOURCE1_ID_COLUMN = "source1_entity_id"
MATCH_COLUMN = "matched_entity_ids"
NAME_COLUMN = "business_name"
ADDRESS_COLUMN = "business_address"
FEATURE_COLUMNS = [
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "address_token_set_ratio",
    "address_partial_ratio",
    "name_length_diff",
    "address_length_diff",
    "address_digit_match",
]
DIGIT_PATTERN = re.compile(r"\d+")


def require_columns(frame, required, label):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _create_blocking_keys(con, table_name, id_alias):
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {table_name}_keys AS
        WITH normalized AS (
            SELECT
                entity_id AS {id_alias},
                lower(trim(coalesce(business_name, ''))) AS normalized_name,
                lower(trim(coalesce(business_address, ''))) AS normalized_address,
                regexp_extract(
                    lower(trim(coalesce(business_name, ''))), '^[a-z0-9]+', 0
                ) AS first_word,
                regexp_extract(coalesce(business_address, ''), '[0-9]+', 0)
                    AS first_number
            FROM {table_name}
        )
        SELECT
            {id_alias},
            CASE WHEN normalized_name <> ''
                THEN normalized_name END AS key_exact_name,
            CASE WHEN first_word <> '' AND first_number <> ''
                THEN first_word || '_' || first_number END
                AS key_word_digit,
            CASE WHEN normalized_address <> ''
                THEN normalized_address END AS key_exact_addr
        FROM normalized
        """
    )


def _candidate_join_query(con, top_k):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    _create_blocking_keys(con, "src1", "source1_entity_id")
    _create_blocking_keys(con, "vendor", "vendor_entity_id")
    query = """
        WITH candidate_matches AS (
            SELECT s.source1_entity_id, v.vendor_entity_id
            FROM src1_keys s
            JOIN vendor_keys v ON s.key_exact_name = v.key_exact_name
            WHERE s.key_exact_name IS NOT NULL AND s.key_exact_name <> '_'

            UNION ALL

            SELECT s.source1_entity_id, v.vendor_entity_id
            FROM src1_keys s
            JOIN vendor_keys v ON s.key_word_digit = v.key_word_digit
            WHERE s.key_word_digit IS NOT NULL AND s.key_word_digit <> '_'

            UNION ALL

            SELECT s.source1_entity_id, v.vendor_entity_id
            FROM src1_keys s
            JOIN vendor_keys v ON s.key_exact_addr = v.key_exact_addr
            WHERE s.key_exact_addr IS NOT NULL AND s.key_exact_addr <> '_'
        ), deduplicated AS (
            SELECT DISTINCT source1_entity_id, vendor_entity_id
            FROM candidate_matches
        ), ranked AS (
            SELECT
                source1_entity_id,
                vendor_entity_id,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY vendor_entity_id
                ) AS rn
            FROM deduplicated
        )
        SELECT source1_entity_id, vendor_entity_id
        FROM ranked
        WHERE rn <= ?
    """
    return con.execute(query, [top_k]).df()


def generate_candidate_pairs_duckdb(
    src1_df: pd.DataFrame,
    vendor_df: pd.DataFrame,
    top_k: int = 30,
) -> pd.DataFrame:
    """Return deduplicated, top-k Source 1/vendor ID pairs using DuckDB joins."""
    required = [ID_COLUMN, NAME_COLUMN, ADDRESS_COLUMN]
    require_columns(src1_df, required, "Source 1")
    require_columns(vendor_df, required, "Vendor data")
    con = duckdb.connect(database=":memory:")
    try:
        con.register("src1", src1_df)
        con.register("vendor", vendor_df)
        return _candidate_join_query(con, top_k)
    finally:
        con.close()
        gc.collect()


def _sample_training_records(path, sample_size, seed, read_chunk_size):
    if sample_size < 1:
        raise ValueError("--max-training-groups must be positive")
    random_state = np.random.RandomState(seed)
    sample = pd.DataFrame()
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[ID_COLUMN, NAME_COLUMN, ADDRESS_COLUMN],
        chunksize=read_chunk_size,
    ):
        hashes = pd.util.hash_pandas_object(chunk[ID_COLUMN], index=False).to_numpy(
            dtype=np.uint64
        )
        chunk = chunk.copy()
        chunk["_sample_hash"] = hashes ^ np.uint64(random_state.randint(0, 2**31))
        sample = pd.concat([sample, chunk], ignore_index=True)
        if len(sample) > sample_size:
            sample = sample.nsmallest(sample_size, "_sample_hash")
    return sample.drop(columns="_sample_hash").reset_index(drop=True)


def _load_truth_subset(path, source1_ids, read_chunk_size):
    truth = {}
    source1_ids = set(source1_ids)
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[SOURCE1_ID_COLUMN, MATCH_COLUMN],
        chunksize=read_chunk_size,
    ):
        selected = chunk[chunk[SOURCE1_ID_COLUMN].isin(source1_ids)]
        for source_id, match_string in zip(
            selected[SOURCE1_ID_COLUMN], selected[MATCH_COLUMN]
        ):
            truth[source_id] = {
                item.strip() for item in match_string.split(",") if item.strip()
            }
    return truth


def _resolve_resource_path(path):
    path = Path(path)
    if path.exists():
        return path
    nested = Path("student_resource") / path
    return nested if nested.exists() else path


def engineer_features(pairs, chunk_size=50000):
    """Create RapidFuzz metrics in bounded chunks using zip comprehensions."""
    feature_chunks = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs.iloc[start:start + chunk_size]
        name_left = [value.strip() for value in chunk["name_1"].tolist()]
        name_right = [value.strip() for value in chunk["name_2"].tolist()]
        address_left = [value.strip() for value in chunk["address_1"].tolist()]
        address_right = [value.strip() for value in chunk["address_2"].tolist()]
        values = {
            "name_token_sort_ratio": [
                fuzz.token_sort_ratio(left, right)
                for left, right in zip(name_left, name_right)
            ],
            "name_token_set_ratio": [
                fuzz.token_set_ratio(left, right)
                for left, right in zip(name_left, name_right)
            ],
            "name_jaro_winkler": [
                100.0 * JaroWinkler.normalized_similarity(left, right)
                for left, right in zip(name_left, name_right)
            ],
            "address_token_set_ratio": [
                fuzz.token_set_ratio(left, right)
                for left, right in zip(address_left, address_right)
            ],
            "address_partial_ratio": [
                fuzz.partial_ratio(left, right)
                for left, right in zip(address_left, address_right)
            ],
            "name_length_diff": [
                abs(len(left) - len(right))
                for left, right in zip(name_left, name_right)
            ],
            "address_length_diff": [
                abs(len(left) - len(right))
                for left, right in zip(address_left, address_right)
            ],
            "address_digit_match": [
                float(bool(set(DIGIT_PATTERN.findall(left)) & set(DIGIT_PATTERN.findall(right))))
                for left, right in zip(address_left, address_right)
            ],
        }
        feature_chunks.append(
            pd.DataFrame(values, index=chunk.index, dtype=np.float32)
        )
        del values, name_left, name_right, address_left, address_right
        gc.collect()
    if not feature_chunks:
        return pd.DataFrame(index=pairs.index, columns=FEATURE_COLUMNS, dtype=np.float32)
    return pd.concat(feature_chunks).reindex(columns=FEATURE_COLUMNS)


def truth_mapping(ground_truth):
    require_columns(ground_truth, [SOURCE1_ID_COLUMN, MATCH_COLUMN], "Ground truth")
    return {
        source_id: {item.strip() for item in matches.split(",") if item.strip()}
        for source_id, matches in zip(
            ground_truth[SOURCE1_ID_COLUMN], ground_truth[MATCH_COLUMN]
        )
    }


def attach_labels(pairs, truth):
    return np.asarray(
        [
            int(candidate_id in truth.get(source_id, set()))
            for source_id, candidate_id in zip(
                pairs["source1_entity_id"], pairs["candidate_entity_id"]
            )
        ],
        dtype=np.uint8,
    )


def f0_5_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    denominator = 0.25 * precision + recall
    return 1.25 * precision * recall / denominator if denominator else 0.0


def macro_f0_5(source1_ids, truth, predictions):
    predicted_by_source = defaultdict(set)
    for source_id, candidate_id in predictions:
        predicted_by_source[source_id].add(candidate_id)
    scores = []
    for source_id in source1_ids:
        actual = truth.get(source_id, set())
        predicted = predicted_by_source.get(source_id, set())
        if not actual and not predicted:
            scores.append(1.0)
            continue
        true_positive = len(actual & predicted)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(actual) if actual else 0.0
        denominator = 0.25 * precision + recall
        scores.append(1.25 * precision * recall / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def select_threshold(source1_ids, truth, pairs, probabilities):
    best_threshold = 0.50
    best_score = -1.0
    thresholds = np.round(np.arange(0.50, 0.951, 0.01), 2)
    for threshold in thresholds:
        selected = pairs.loc[probabilities >= threshold]
        predicted = zip(
            selected["source1_entity_id"], selected["candidate_entity_id"]
        )
        score = macro_f0_5(source1_ids, truth, predicted)
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def _new_classifier(random_seed):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=800,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def cross_validate_oof(features, labels, groups, n_splits, random_seed):
    unique_groups = pd.Series(groups).nunique()
    if n_splits < 2 or n_splits > unique_groups:
        raise ValueError(
            f"n_splits must be between 2 and candidate-bearing Source 1 groups ({unique_groups})"
        )
    probabilities = np.full(len(labels), np.nan, dtype=np.float32)
    splitter = GroupKFold(n_splits=n_splits)
    for train_index, valid_index in splitter.split(features, labels, groups=groups):
        fold_labels = labels[train_index]
        if len(np.unique(fold_labels)) < 2:
            probabilities[valid_index] = float(fold_labels[0])
            continue
        model = _new_classifier(random_seed)
        model.fit(features.iloc[train_index], fold_labels)
        probabilities[valid_index] = model.predict_proba(features.iloc[valid_index])[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Some candidate rows did not receive an OOF prediction")
    return probabilities


def aggregate_submission(source1_ids, pairs, probabilities, threshold,
                         source_column, values_column):
    probabilities = np.asarray(probabilities)
    accepted = pairs.loc[
        probabilities >= threshold,
        ["source1_entity_id", "candidate_entity_id"],
    ].drop_duplicates()
    grouped = (
        accepted.groupby("source1_entity_id", sort=False)["candidate_entity_id"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .rename(values_column)
        .reset_index()
    )
    result = source1_ids[[ID_COLUMN]].drop_duplicates().rename(
        columns={ID_COLUMN: source_column}
    )
    result = result.merge(
        grouped.rename(columns={"source1_entity_id": source_column}),
        on=source_column,
        how="left",
        validate="one_to_one",
    )
    result[values_column] = result[values_column].fillna("").astype(str)
    return result


def candidate_submission(source1_ids, pairs):
    grouped = (
        pairs.groupby("source1_entity_id", sort=False)["candidate_entity_id"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .rename("candidate_entity_ids")
        .reset_index()
    )
    result = source1_ids[[ID_COLUMN]].drop_duplicates().rename(
        columns={ID_COLUMN: SOURCE1_ID_COLUMN}
    )
    result = result.merge(
        grouped.rename(columns={"source1_entity_id": SOURCE1_ID_COLUMN}),
        on=SOURCE1_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("").astype(str)
    return result


def _resolve_resource_path(path):
    path = Path(path)
    if path.exists():
        return path
    nested = Path("student_resource") / path
    return nested if nested.exists() else path


def _read_entity_table(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[ID_COLUMN, NAME_COLUMN, ADDRESS_COLUMN],
    )


def _load_vendor_table(paths):
    return pd.concat([_read_entity_table(path) for path in paths], ignore_index=True)


def _attach_candidate_attributes(source1, vendors, candidate_ids):
    if candidate_ids.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "name_1",
                "name_2",
                "address_1",
                "address_2",
            ]
        )
    source_attributes = source1[[ID_COLUMN, NAME_COLUMN, ADDRESS_COLUMN]].rename(
        columns={
            ID_COLUMN: "source1_entity_id",
            NAME_COLUMN: "name_1",
            ADDRESS_COLUMN: "address_1",
        }
    )
    vendor_attributes = vendors[[ID_COLUMN, NAME_COLUMN, ADDRESS_COLUMN]].rename(
        columns={
            ID_COLUMN: "vendor_entity_id",
            NAME_COLUMN: "name_2",
            ADDRESS_COLUMN: "address_2",
        }
    )
    pairs = candidate_ids.merge(
        source_attributes,
        on="source1_entity_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    pairs = pairs.merge(
        vendor_attributes,
        on="vendor_entity_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return pairs.rename(columns={"vendor_entity_id": "candidate_entity_id"})[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "name_1",
            "name_2",
            "address_1",
            "address_2",
        ]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--read-chunk-size", type=int, default=25000)
    parser.add_argument("--max-training-groups", type=int, default=50000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    train_dir = _resolve_resource_path(args.train_dir)
    test_dir = _resolve_resource_path(args.test_dir)
    ground_truth_path = _resolve_resource_path(args.ground_truth)
    output_dir = Path(args.output_dir)
    train_source1_path = train_dir / "train_source1.tsv"
    train_vendor_paths = [train_dir / "train_source2.tsv", train_dir / "train_source3.tsv"]
    test_source1_path = test_dir / "test_source1.tsv"
    test_vendor_paths = [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"]
    required_paths = [
        train_source1_path,
        *train_vendor_paths,
        ground_truth_path,
        test_source1_path,
        *test_vendor_paths,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required challenge file not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_sample = _sample_training_records(
        train_source1_path,
        args.max_training_groups,
        args.random_seed,
        args.read_chunk_size,
    )
    sample_ids = train_sample[ID_COLUMN].tolist()
    truth = _load_truth_subset(ground_truth_path, sample_ids, args.read_chunk_size)
    train_vendors = _load_vendor_table(train_vendor_paths)
    train_candidate_ids = generate_candidate_pairs_duckdb(
        train_sample, train_vendors, top_k=args.max_candidates
    )
    train_pairs = _attach_candidate_attributes(
        train_sample, train_vendors, train_candidate_ids
    )
    del train_candidate_ids, train_vendors
    gc.collect()
    if train_pairs.empty:
        raise ValueError("Blocking produced no training pairs; cannot train a match model")
    labels = attach_labels(train_pairs, truth)
    train_features = engineer_features(train_pairs, args.chunk_size)
    oof_probabilities = cross_validate_oof(
        train_features,
        labels,
        train_pairs["source1_entity_id"].to_numpy(),
        args.n_splits,
        args.random_seed,
    )
    threshold, oof_score = select_threshold(
        sample_ids, truth, train_pairs, oof_probabilities
    )
    oof_output = train_pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    oof_output["label"] = labels
    oof_output["probability"] = oof_probabilities
    oof_output.to_csv(output_dir / "oof_predictions.tsv", sep="\t", index=False)
    del oof_output
    print(
        f"Sampled OOF macro F0.5: {oof_score:.6f} at threshold {threshold:.2f} "
        f"({len(sample_ids):,} Source 1 groups)"
    )

    final_model = None
    constant_probability = None
    if len(np.unique(labels)) < 2:
        constant_probability = float(labels[0])
    else:
        final_model = _new_classifier(args.random_seed)
        final_model.fit(train_features, labels)
    del train_features
    del train_pairs, train_sample, labels, oof_probabilities
    gc.collect()

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    test_source1 = _read_entity_table(test_source1_path)
    test_vendors = _load_vendor_table(test_vendor_paths)
    test_candidate_ids = generate_candidate_pairs_duckdb(
        test_source1, test_vendors, top_k=args.max_candidates
    )
    candidate_submission(
        test_source1,
        test_candidate_ids.rename(columns={"vendor_entity_id": "candidate_entity_id"}),
    ).to_csv(candidate_path, sep="\t", index=False)
    test_pairs = _attach_candidate_attributes(
        test_source1, test_vendors, test_candidate_ids
    )
    del test_candidate_ids, test_vendors
    gc.collect()

    matches_by_source = defaultdict(set)
    for start in range(0, len(test_pairs), args.chunk_size):
        pair_chunk = test_pairs.iloc[start : start + args.chunk_size]
        if final_model is None:
            probabilities = np.full(
                len(pair_chunk), constant_probability, dtype=np.float32
            )
        else:
            features = engineer_features(pair_chunk, args.chunk_size)
            probabilities = final_model.predict_proba(features)[:, 1]
            del features
        accepted = pair_chunk.loc[
            probabilities >= threshold,
            ["source1_entity_id", "candidate_entity_id"],
        ]
        for source_id, candidate_id in zip(
            accepted["source1_entity_id"], accepted["candidate_entity_id"]
        ):
            matches_by_source[source_id].add(candidate_id)
        del pair_chunk, accepted, probabilities
        gc.collect()

    matched = pd.DataFrame(
        [
            (source_id, ",".join(sorted(candidate_ids)))
            for source_id, candidate_ids in matches_by_source.items()
        ],
        columns=[SOURCE1_ID_COLUMN, MATCH_COLUMN],
    )
    submission = test_source1[[ID_COLUMN]].drop_duplicates().rename(
        columns={ID_COLUMN: SOURCE1_ID_COLUMN}
    )
    submission = submission.merge(matched, on=SOURCE1_ID_COLUMN, how="left")
    submission[MATCH_COLUMN] = submission[MATCH_COLUMN].fillna("").astype(str)
    submission.to_csv(matching_path, sep="\t", index=False)
    del test_pairs, test_source1, matches_by_source
    gc.collect()

    if not args.skip_validation:
        root = Path(__file__).resolve().parents[3]
        validator = root / "utils" / "validate_submission.py"
        if not validator.is_file():
            validator = root / "student_resource" / "utils" / "validate_submission.py"
        if not validator.is_file():
            raise FileNotFoundError(f"Submission validator not found: {validator}")
        subprocess.run(
            [
                sys.executable,
                str(validator),
                "--matching",
                str((output_dir / "matching_results.tsv").resolve()),
                "--candidate",
                str((output_dir / "candidate_pairs.tsv").resolve()),
                "--test-dir",
                str(test_dir.resolve()),
            ],
            check=True,
            cwd=root,
        )


if __name__ == "__main__":
    main()