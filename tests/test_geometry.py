from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from geometry import GridGeometry, window_count


class GeometryTests(unittest.TestCase):
    def test_window_count_includes_padded_edge_patch(self) -> None:
        self.assertEqual(window_count(512, 512, 256), 1)
        self.assertEqual(window_count(513, 512, 256), 2)
        self.assertEqual(window_count(1024, 512, 256), 3)

    def test_coordinates_and_padding(self) -> None:
        geometry = GridGeometry.create(700, 600, 512, 256)
        self.assertEqual((geometry.row_count, geometry.column_count), (2, 2))
        self.assertEqual(geometry.position(0), (0, 0, 0, 0, 512, 512))
        self.assertEqual(geometry.position(3), (1, 1, 256, 256, 444, 344))
        arrays = geometry.arrays()
        self.assertEqual(int(arrays["was_padded"].sum()), 3)

    def test_stride_cannot_leave_gaps(self) -> None:
        with self.assertRaises(ValueError):
            GridGeometry.create(1024, 1024, 512, 513)


if __name__ == "__main__":
    unittest.main()
