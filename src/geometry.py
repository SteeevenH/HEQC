"""
WSI patch grid geometry and coordinate utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def window_count(dimension: int, patch_size: int, stride: int) -> int:
    """Return the number of windows required for full coverage."""
    if dimension <= 0:
        raise ValueError("WSI level dimensions must be positive")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if dimension <= patch_size:
        return 1
    return int((dimension - patch_size + stride - 1) // stride + 1)


@dataclass(frozen=True)
class GridGeometry:
    """Represent patch grid geometry in WSI level coordinates."""
    level_width: int
    level_height: int
    patch_size: int
    stride: int
    row_count: int
    column_count: int

    @classmethod
    def create(
        cls,
        level_width: int,
        level_height: int,
        patch_size: int,
        stride: int,
    ) -> GridGeometry:
        """Initialize a grid."""
        if stride > patch_size:

            raise ValueError("stride cannot exceed the model patch size")
        return cls(
            level_width=int(level_width),
            level_height=int(level_height),
            patch_size=int(patch_size),
            stride=int(stride),
            row_count=window_count(level_height, patch_size, stride),
            column_count=window_count(level_width, patch_size, stride),
        )

    def __len__(self) -> int:
        return self.row_count * self.column_count

    def position(self, index: int) -> tuple[int, int, int, int, int, int]:
        """Return patch coordinates and dimensions."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = index // self.column_count
        column = index % self.column_count
        x = column * self.stride
        y = row * self.stride
        valid_width = min(self.patch_size, self.level_width - x)
        valid_height = min(self.patch_size, self.level_height - y)
        return row, column, x, y, valid_width, valid_height

    def arrays(self) -> dict[str, np.ndarray]:
        """Generate coordinates and padding metadata for all patches."""
        indices = np.arange(len(self), dtype=np.int64)
        rows = indices // self.column_count
        columns = indices % self.column_count
        x = columns * self.stride
        y = rows * self.stride
        valid_width = np.minimum(self.patch_size, self.level_width - x)
        valid_height = np.minimum(self.patch_size, self.level_height - y)
        return {
            "index": indices,
            "row": rows,
            "column": columns,
            "x_level": x,
            "y_level": y,
            "valid_width": valid_width,
            "valid_height": valid_height,
            "was_padded": (
                (valid_width < self.patch_size) | (valid_height < self.patch_size)
            ),
        }

    def to_dict(self) -> dict[str, int]:
        return {
            "level_width": self.level_width,
            "level_height": self.level_height,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "total_patches": len(self),
        }
