from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constants import LABELS
from thresholds import load_threshold_profiles


class ThresholdTests(unittest.TestCase):
    def test_efficientnet_threshold_file_is_canonical(self) -> None:
        path = PROJECT_ROOT / "model" / "threshold_efficientnet_b0.json"
        profiles = load_threshold_profiles(path)
        self.assertEqual(tuple(profiles), ("balanced", "recall"))
        for values in profiles.values():
            self.assertEqual(tuple(values), LABELS)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values.values()))


if __name__ == "__main__":
    unittest.main()
