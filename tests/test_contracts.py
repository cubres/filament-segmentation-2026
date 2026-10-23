"""Contract checks; no network, notebook edits, or test-label access."""
import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import numpy as np
import pandas as pd
import solar_threads as st


class Contracts(unittest.TestCase):
    def test_coco_round_trip(self):
        mask = np.zeros((2048, 2048), np.uint8)
        mask[7:11, 50:70] = 1
        encoded = st.encode_mask(mask)
        self.assertTrue(np.array_equal(mask, st.coco_mask.decode(encoded)))

    def test_perfect_and_empty_pq(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[5:15, 20:30] = 1
        encoded = st.encode_mask(mask)
        self.assertEqual(st.aggregate_pq([st.pq_counts([encoded], [encoded])]), 1)
        self.assertEqual(st.aggregate_pq([st.pq_counts([encoded], [])]), 0)
        self.assertEqual(st.pq_counts([], [encoded])["fp"], 1)

    def test_sparse_rle_matches_dense_rle(self):
        labels = np.zeros((64, 65), np.int32)
        labels[0, 0] = 1
        labels[2:15, 3:30] = 1
        labels[-8:, -9:] = 2
        labels[21:31, 41:54] = 3
        rles = st.labels_to_rles(labels)
        self.assertEqual(len(rles), 3)
        for label, rle in enumerate(rles, 1):
            self.assertTrue(np.array_equal(st.coco_mask.decode(rle), labels == label))
            self.assertEqual(rle["counts"], st.encode_mask(labels == label)["counts"])

    def test_disjoint_instances_and_seed_rejection(self):
        contrast = np.zeros((64, 64), np.float32)
        contrast[8:18, 8:18] = 0.05
        contrast[10:15, 10:15] = 0.10
        contrast[40:50, 40:50] = 0.05
        config = st.Config(low=0.04, high=0.08, min_area=5, min_core_area=3, closing_size=1)
        labels = st.predict_features(contrast, np.ones_like(contrast, bool), config)
        self.assertEqual(labels[12, 12], 1)
        self.assertEqual(labels[45, 45], 0)
        self.assertEqual((labels > 0).sum(), 100)

    def test_month_split_is_stable_and_disjoint(self):
        names = [f"2015{month:02d}{day:02d}105534Ch.jpeg" for month in range(1, 13) for day in (1, 2)]
        a, b = st.split_names(names, 50, 50)
        self.assertEqual((a, b), st.split_names(names[::-1], 50, 50))
        self.assertFalse({n[:6] for n in a} & {n[:6] for n in b})

    def test_submission_rejects_overlap_and_foreign_image(self):
        mask = np.zeros((2048, 2048), np.uint8)
        mask[10:20, 10:20] = 1
        counts = st.encode_mask(mask)["counts"].decode()
        good = pd.DataFrame([{"filament_id": "test_1", "segmentation_rle": counts}])
        self.assertEqual(st.validate_submission(good, {"test"})["rows"], 1)
        with self.assertRaises(ValueError):
            st.validate_submission(good, {"other"})
        overlap = pd.concat([good, good.assign(filament_id="test_2")], ignore_index=True)
        with self.assertRaises(ValueError):
            st.validate_submission(overlap, {"test"})

    def test_submission_records_empty_images_without_inventing_masks(self):
        mask = np.zeros((2048, 2048), np.uint8)
        mask[10:20, 10:20] = 1
        frame = pd.DataFrame([{"filament_id": "one_1", "segmentation_rle": st.encode_mask(mask)["counts"].decode()}])
        self.assertEqual(st.validate_submission(frame, {"one", "two"})["zero_prediction_images"], ["two"])



if __name__ == "__main__":
    unittest.main()
