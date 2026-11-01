from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualization import format_magnification


class VisualizationNameTests(unittest.TestCase):
    def test_thumbnail_magnification_filename_component(self) -> None:
        self.assertEqual(format_magnification(1.25), "1p25x")
        self.assertEqual(format_magnification(2.0), "2x")
        self.assertEqual(format_magnification(0.625), "0p625x")


if __name__ == "__main__":
    unittest.main()
