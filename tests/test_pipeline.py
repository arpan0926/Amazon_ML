import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"),
)
import pipeline


class PipelineTests(unittest.TestCase):
    def test_duckdb_three_key_join_deduplicates_and_caps_pairs(self):
        vendors = pd.DataFrame(
            [
                {
                    "entity_id": "S2-01",
                    "business_name": "Alpha Shop",
                    "business_address": "44 Oak Road",
                },
                {
                    "entity_id": "S2-02",
                    "business_name": "Alpha Store",
                    "business_address": "44 Elm Road",
                },
                {
                    "entity_id": "S3-01",
                    "business_name": "Different Name",
                    "business_address": "44 Oak Road",
                },
                {
                    "entity_id": "S2-99",
                    "business_name": "_",
                    "business_address": "",
                },
            ]
        )
        source1 = pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Alpha Shop",
                    "business_address": "44 Oak Road",
                },
                {
                    "entity_id": "S1-2",
                    "business_name": "Unmatched Business",
                    "business_address": "99 Unknown Lane",
                },
                {
                    "entity_id": "S1-3",
                    "business_name": "_",
                    "business_address": "",
                },
            ]
        )
        candidates = pipeline.generate_candidate_pairs_duckdb(
            source1, vendors, top_k=30
        )
        self.assertEqual(
            list(candidates.columns), ["source1_entity_id", "vendor_entity_id"]
        )
        self.assertEqual(
            set(candidates["vendor_entity_id"]), {"S2-01", "S2-02", "S3-01"}
        )
        self.assertEqual(
            candidates.duplicated(["source1_entity_id", "vendor_entity_id"]).sum(), 0
        )
        pairs = pipeline._attach_candidate_attributes(source1, vendors, candidates)
        self.assertEqual(
            list(pairs.columns),
            [
                "source1_entity_id",
                "candidate_entity_id",
                "name_1",
                "name_2",
                "address_1",
                "address_2",
            ],
        )
        candidate_file = pipeline.candidate_submission(source1, pairs)
        self.assertEqual(
            list(candidate_file.columns),
            ["source1_entity_id", "candidate_entity_ids"],
        )
        self.assertEqual(
            candidate_file.loc[
                candidate_file["source1_entity_id"] == "S1-2",
                "candidate_entity_ids",
            ].iloc[0],
            "",
        )
        capped = pipeline.generate_candidate_pairs_duckdb(source1, vendors, top_k=2)
        self.assertEqual(len(capped[capped["source1_entity_id"] == "S1-1"]), 2)

    def test_duckdb_candidate_ranking_keeps_strongest_before_top_k(self):
        source1 = pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Acme Bakery",
                    "business_address": "42 Main Street",
                }
            ]
        )
        vendors = pd.DataFrame(
            [
                {
                    "entity_id": "S2-01",
                    "business_name": "Unrelated Company",
                    "business_address": "42 Main Street",
                },
                {
                    "entity_id": "S2-02",
                    "business_name": "Acme Bakery",
                    "business_address": "99 Pine Road",
                },
                {
                    "entity_id": "S2-03",
                    "business_name": "Acme Coffee",
                    "business_address": "42 Main Street",
                },
                {
                    "entity_id": "S2-04",
                    "business_name": "Completely Different Business",
                    "business_address": "700 Faraway Avenue",
                },
            ]
        )

        candidates = pipeline.generate_candidate_pairs_duckdb(
            source1, vendors, top_k=2
        )

        self.assertEqual(
            set(candidates["vendor_entity_id"]), {"S2-02", "S2-03"}
        )
        self.assertNotIn("S2-01", set(candidates["vendor_entity_id"]))

    def test_duckdb_similarity_breaks_equal_evidence_ranking_ties(self):
        source1 = pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Acme Bakery",
                    "business_address": "42 Main Street",
                }
            ]
        )
        vendors = pd.DataFrame(
            [
                {
                    "entity_id": "S2-01",
                    "business_name": "Zzzz Zzzzzz",
                    "business_address": "42 Main Street",
                },
                {
                    "entity_id": "S2-99",
                    "business_name": "Acmf Bakery",
                    "business_address": "42 Main Street",
                },
            ]
        )

        candidates = pipeline.generate_candidate_pairs_duckdb(
            source1, vendors, top_k=1
        )

        self.assertEqual(candidates["vendor_entity_id"].tolist(), ["S2-99"])

    def test_legal_suffix_key_matches_variants_without_unrelated_expansion(self):
        source1 = pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Acme, Inc.",
                    "business_address": "10 Oak Road",
                }
            ]
        )
        vendors = pd.DataFrame(
            [
                {
                    "entity_id": "S2-01",
                    "business_name": "ACME LLC",
                    "business_address": "77 Pine Road",
                },
                {
                    "entity_id": "S2-02",
                    "business_name": "Acme Restaurant Group",
                    "business_address": "88 Cedar Avenue",
                },
            ]
        )

        candidates = pipeline.generate_candidate_pairs_duckdb(
            source1, vendors, top_k=30
        )

        self.assertEqual(set(candidates["vendor_entity_id"]), {"S2-01"})

    def test_unicode_name_and_address_variants_normalize_consistently(self):
        source1 = pd.DataFrame(
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "École & Fils, Inc.",
                    "business_address": "10 Main Street, Suite #2",
                }
            ]
        )
        vendors = pd.DataFrame(
            [
                {
                    "entity_id": "S2-1",
                    "business_name": "E\u0301COLE and FILS LLC",
                    "business_address": "10 Main St., Ste. 2",
                },
                {
                    "entity_id": "S2-2",
                    "business_name": "Unrelated Workshop",
                    "business_address": "99 Faraway Avenue",
                },
            ]
        )

        candidates = pipeline.generate_candidate_pairs_duckdb(source1, vendors)
        pairs = pipeline._attach_candidate_attributes(source1, vendors, candidates)
        features = pipeline.engineer_features(pairs)

        self.assertEqual(set(candidates["vendor_entity_id"]), {"S2-1"})
        self.assertEqual(
            pipeline._normalize_name(source1.iloc[0]["business_name"]),
            "école and fils inc",
        )
        self.assertEqual(
            pipeline._normalize_name(vendors.iloc[0]["business_name"]),
            "école and fils llc",
        )
        self.assertAlmostEqual(
            float(features.iloc[0]["name_token_sort_ratio"]), 88.8889, places=3
        )
        self.assertEqual(features.iloc[0]["exact_address_key_match"], 1.0)
        self.assertEqual(features.iloc[0]["legal_name_similarity"], 100.0)
        self.assertEqual(pipeline._normalize_name(None), "")
        self.assertEqual(pipeline._normalize_name(float("nan")), "")

    def test_features_and_singleton_output(self):
        pairs = pd.DataFrame(
            [
                {
                    "source1_entity_id": "S1-1",
                    "candidate_entity_id": "S2-1",
                    "name_1": "Acme Bakery",
                    "name_2": "Acme Bakery LLC",
                    "address_1": "12 Main Street",
                    "address_2": "12 Main St",
                }
            ]
        )
        features = pipeline.engineer_features(pairs)
        self.assertTrue(all(dtype == "float32" for dtype in features.dtypes.astype(str)))

        source1 = pd.DataFrame([{"entity_id": "S1-1"}, {"entity_id": "S1-2"}])
        result = pipeline.aggregate_submission(
            source1,
            pairs,
            [0.9],
            0.8,
            pipeline.SOURCE1_ID_COLUMN,
            pipeline.MATCH_COLUMN,
        )
        self.assertEqual(result.loc[1, pipeline.MATCH_COLUMN], "")
        self.assertEqual(
            pipeline.macro_f0_5(
                ["S1-1", "S1-2"],
                {"S1-1": {"S2-1"}, "S1-2": set()},
                [("S1-1", "S2-1")],
            ),
            1.0,
        )

    def test_blocking_features_match_sql_keys_and_are_float32(self):
        pairs = pd.DataFrame(
            [
                {
                    "source1_entity_id": "S1-1",
                    "candidate_entity_id": "S2-1",
                    "name_1": "Acme, Inc.",
                    "name_2": "ACME LLC",
                    "address_1": "10 Main Street",
                    "address_2": "10 Main Street",
                }
            ]
        )

        features = pipeline.engineer_features(pairs)
        feature_row = features.iloc[0]

        expected_columns = {
            "exact_name_key_match",
            "word_digit_key_match",
            "exact_address_key_match",
            "legal_name_key_match",
            "blocking_key_match_count",
            "legal_name_similarity",
        }
        self.assertTrue(expected_columns.issubset(features.columns))
        self.assertTrue(all(dtype == "float32" for dtype in features.dtypes.astype(str)))
        self.assertEqual(feature_row["exact_name_key_match"], 0.0)
        self.assertEqual(feature_row["word_digit_key_match"], 1.0)
        self.assertEqual(feature_row["exact_address_key_match"], 1.0)
        self.assertEqual(feature_row["legal_name_key_match"], 1.0)
        self.assertEqual(feature_row["blocking_key_match_count"], 3.0)
        self.assertEqual(feature_row["legal_name_similarity"], 100.0)

    def test_threshold_ties_prefer_higher_cutoff(self):
        pairs = pd.DataFrame(
            [
                {
                    "source1_entity_id": "S1-1",
                    "candidate_entity_id": "S2-1",
                }
            ]
        )
        threshold, _, _, _ = pipeline.select_threshold(
            ["S1-1", "S1-2"], {"S1-1": set(), "S1-2": set()},
            pd.concat([pairs, pairs.assign(source1_entity_id="S1-2")], ignore_index=True),
            [0.1, 0.1],
        )
        self.assertEqual(threshold, 0.95)

    def test_threshold_is_selected_and_reported_on_separate_group_halves(self):
        source_ids = ["S1-A", "S1-B", "S1-C", "S1-D"]
        splitter = pipeline.GroupShuffleSplit(
            n_splits=1, test_size=0.5, random_state=17
        )
        selection_indices, heldout_indices = next(
            splitter.split(source_ids, groups=source_ids)
        )
        selection_ids = [source_ids[index] for index in selection_indices]
        heldout_ids = [source_ids[index] for index in heldout_indices]
        truth = {
            selection_ids[0]: {"V-selection-positive"},
            selection_ids[1]: set(),
            heldout_ids[0]: {"V-heldout-positive"},
            heldout_ids[1]: set(),
        }
        pairs = pd.DataFrame(
            [
                {"source1_entity_id": selection_ids[0], "candidate_entity_id": "V-selection-positive"},
                {"source1_entity_id": selection_ids[1], "candidate_entity_id": "V-selection-negative"},
                {"source1_entity_id": heldout_ids[0], "candidate_entity_id": "V-heldout-positive"},
                {"source1_entity_id": heldout_ids[1], "candidate_entity_id": "V-heldout-negative"},
            ]
        )
        probabilities = [0.82, 0.60, 0.54, 0.90]

        threshold, relative_confidence, selection_score, heldout_score = pipeline.select_threshold(
            source_ids, truth, pairs, probabilities, random_seed=17
        )

        self.assertEqual(threshold, 0.82)
        self.assertEqual(relative_confidence, 0.95)
        self.assertEqual(selection_score, 1.0)
        self.assertEqual(heldout_score, 0.0)

        heldout_mask = pairs["source1_entity_id"].isin(heldout_ids).to_numpy()
        heldout_pairs = pairs.loc[heldout_mask]
        heldout_probabilities = pd.Series(probabilities).to_numpy()[heldout_mask]
        heldout_sweep = []
        for candidate_threshold in np.round(
            np.arange(
                pipeline.THRESHOLD_SWEEP_MIN,
                pipeline.THRESHOLD_SWEEP_MAX + pipeline.THRESHOLD_SWEEP_STEP / 2,
                pipeline.THRESHOLD_SWEEP_STEP,
            ),
            2,
        ):
            selected = heldout_pairs.loc[
                heldout_probabilities >= candidate_threshold
            ]
            heldout_sweep.append(
                (
                    candidate_threshold,
                    pipeline.macro_f0_5(
                        heldout_ids,
                        truth,
                        zip(
                            selected["source1_entity_id"],
                            selected["candidate_entity_id"],
                        ),
                    ),
                )
            )
        heldout_optimum = max(heldout_sweep, key=lambda result: (result[1], result[0]))[0]
        self.assertEqual(heldout_optimum, 0.95)
        self.assertNotEqual(threshold, heldout_optimum)

    def test_threshold_sweep_finds_optimum_below_half(self):
        source_ids = [f"S1-{index}" for index in range(8)]
        splitter = pipeline.GroupShuffleSplit(
            n_splits=1, test_size=0.5, random_state=23
        )
        selection_indices, _ = next(
            splitter.split(source_ids, groups=source_ids)
        )
        selection_ids = [source_ids[index] for index in selection_indices]
        truth = {source_id: {f"V-{source_id}"} for source_id in selection_ids}
        pairs = pd.DataFrame(
            [
                {
                    "source1_entity_id": source_id,
                    "candidate_entity_id": f"V-{source_id}",
                }
                for source_id in selection_ids
            ]
        )

        threshold, _, selection_score, _ = pipeline.select_threshold(
            source_ids,
            truth,
            pairs,
            np.full(len(pairs), 0.20, dtype=np.float32),
            random_seed=23,
        )

        self.assertEqual(threshold, 0.20)
        self.assertEqual(selection_score, 1.0)

    def test_relative_confidence_filter_is_applied_per_source1_group(self):
        pairs = pd.DataFrame(
            [
                {"source1_entity_id": "S1-1", "candidate_entity_id": "V1-best"},
                {"source1_entity_id": "S1-1", "candidate_entity_id": "V1-weak"},
                {"source1_entity_id": "S1-2", "candidate_entity_id": "V2-best"},
                {"source1_entity_id": "S1-2", "candidate_entity_id": "V2-close"},
            ]
        )

        mask = pipeline._candidate_acceptance_mask(
            pairs,
            np.asarray([0.8, 0.5, 0.8, 0.75]),
            threshold=0.4,
            relative_confidence=0.9,
        )

        self.assertEqual(mask.tolist(), [True, False, True, True])

    def test_training_sampler_respects_group_limit(self):
        rows = pd.DataFrame(
            [
                {
                    "entity_id": f"S1-{index}",
                    "business_name": f"Business {index}",
                    "business_address": f"{index} Main Road",
                }
                for index in range(12)
            ]
        )
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "source1.tsv"
            rows.to_csv(path, sep="\t", index=False)
            sample = pipeline._sample_training_records(path, 5, 42, 3)
        self.assertEqual(len(sample), 5)
        self.assertEqual(sample["entity_id"].nunique(), 5)

    def test_sample_sensitivity_is_opt_in(self):
        report_args = {
            "source1_path": Path("train_source1.tsv"),
            "vendor_paths": [Path("train_source2.tsv"), Path("train_source3.tsv")],
            "ground_truth_path": Path("train_ground_truth.tsv"),
            "n_splits": 5,
            "max_candidates": 30,
            "chunk_size": 50000,
            "read_chunk_size": 25000,
            "random_seed": 42,
        }

        with patch.object(pipeline, "report_sample_sensitivity") as report:
            self.assertFalse(
                pipeline._maybe_report_sample_sensitivity(False, report_args)
            )
            report.assert_not_called()

            self.assertTrue(
                pipeline._maybe_report_sample_sensitivity(True, report_args)
            )
            report.assert_called_once_with(**report_args)

    def test_classifier_uses_balanced_class_weights(self):
        model = pipeline._new_classifier(random_seed=42)
        self.assertEqual(model.get_params()["auto_class_weights"], "Balanced")

    def test_imbalanced_oof_keeps_positive_probabilities_above_floor(self):
        group_count = 20
        rows = []
        labels = []
        groups = []
        for group_index in range(group_count):
            source_id = f"S1-{group_index}"
            for pair_index in range(11):
                rows.append(
                    {
                        "positive_signal": np.float32(pair_index == 0),
                        "weak_signal": np.float32(0.2 if pair_index == 0 else 0.0),
                    }
                )
                labels.append(int(pair_index == 0))
                groups.append(source_id)
        features = pd.DataFrame(rows, dtype=np.float32)

        probabilities = pipeline.cross_validate_oof(
            features,
            np.asarray(labels, dtype=np.uint8),
            np.asarray(groups),
            n_splits=5,
            random_seed=42,
        )

        positive_mean = float(probabilities[np.asarray(labels) == 1].mean())
        negative_mean = float(probabilities[np.asarray(labels) == 0].mean())
        self.assertGreater(positive_mean, 0.3)
        self.assertGreater(positive_mean, negative_mean)


if __name__ == "__main__":
    unittest.main()