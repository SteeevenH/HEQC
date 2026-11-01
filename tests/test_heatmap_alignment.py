from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constants import LABELS
from geometry import GridGeometry
from heatmaps import build_spatial_heatmaps, threshold_relative_scores


class HeatmapAlignmentTests(unittest.TestCase):
    def test_threshold_is_mapped_to_midpoint(self) -> None:
        scores = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        mapped = threshold_relative_scores(scores, 0.5)
        np.testing.assert_allclose(mapped, scores, atol=1e-6)
        mapped_high_threshold = threshold_relative_scores(
            np.asarray([0.0, 0.8, 1.0], dtype=np.float32),
            0.8,
        )
        np.testing.assert_allclose(mapped_high_threshold, [0.0, 0.5, 1.0])

    def test_patch_signal_stays_on_its_spatial_side(self) -> None:
        geometry = GridGeometry.create(768, 512, 512, 256)
        probabilities = np.zeros((len(geometry), len(LABELS)), dtype=np.float32)
        probabilities[0, 0] = 0.95
        probabilities[1, 0] = 0.05
        thresholds = {label: 0.5 for label in LABELS}
        result = build_spatial_heatmaps(
            probabilities,
            geometry,
            (96, 64),
            thresholds,
            smoothing_radius_override=0.0,
        )
        foreign = result.relative_maps["foreign"]
        self.assertGreater(float(foreign[32, 12]), float(foreign[32, 84]))
        self.assertTrue(bool(result.coverage_mask.all()))
        stacked = np.stack([result.relative_maps[label] for label in LABELS])
        np.testing.assert_allclose(result.overall_relative_map, stacked.max(axis=0))


if __name__ == "__main__":
    unittest.main()
