import unittest
from pathlib import Path
import sys

import pandas as pd

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

    def test_threshold_ties_prefer_higher_cutoff(self):
        pairs = pd.DataFrame(
            [
                {
                    "source1_entity_id": "S1-1",
                    "candidate_entity_id": "S2-1",
                }
            ]
        )
        threshold, _ = pipeline.select_threshold(
            ["S1-1"], {"S1-1": set()}, pairs, [0.1]
        )
        self.assertEqual(threshold, 0.95)

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


if __name__ == "__main__":
    unittest.main()