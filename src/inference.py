"""
WSI patch loading and batched model inference.

Reads patches directly from WSIs using OpenSlide and
performs batched inference with PyTorch.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from constants import IMAGENET_MEAN, IMAGENET_STD
from geometry import GridGeometry
from models import DeviceInfo, predict_tensor_batch
from wsi import read_rgb_patch


class WSIPatchDataset(Dataset):
    """Read prediction patches directly from a WSI without saving patch files."""

    def __init__(
        self,
        wsi_path: Path,
        prediction_level: int,
        prediction_downsample: float,
        geometry: GridGeometry,
    ) -> None:
        self.wsi_path = str(wsi_path)
        self.prediction_level = int(prediction_level)
        self.prediction_downsample = float(prediction_downsample)
        self.geometry = geometry
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._slide: openslide.OpenSlide | None = None

    def __len__(self) -> int:
        return len(self.geometry)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the OpenSlide handle when serializing the dataset."""
        state = self.__dict__.copy()
        state["_slide"] = None
        return state

    def _get_slide(self) -> openslide.OpenSlide:
        """Lazily open the WSI for the current dataset instance."""
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.wsi_path)
        return self._slide

    def close(self) -> None:
        """Close the WSI handle if it is open."""
        if self._slide is not None:
            self._slide.close()
            self._slide = None

    def __del__(self) -> None:
        with suppress(openslide.OpenSlideError, OSError):
            self.close()

    def __getitem__(self, index: int) -> torch.Tensor:
        """Read and normalize a patch at the specified grid index."""
        _, _, x, y, _, _ = self.geometry.position(index)
        patch = read_rgb_patch(
            self._get_slide(),
            self.prediction_level,
            self.prediction_downsample,
            x,
            y,
            self.geometry.patch_size,
        )
        try:
            return self.transform(patch)
        finally:
            patch.close()


def run_patch_inference(
    wsi_path: Path,
    slide_id: str,
    model: nn.Module,
    device_info: DeviceInfo,
    prediction_level: int,
    prediction_downsample: float,
    geometry: GridGeometry,
    num_labels: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    use_amp: bool,
) -> tuple[np.ndarray, float]:
    """
    Run batched WSI inference and return probabilities with prediction time.

    Preserves grid order so each output row corresponds to its patch index.
    Prediction time accumulates the durations reported by predict_tensor_batch.
    """
    dataset = WSIPatchDataset(
        wsi_path,
        prediction_level,
        prediction_downsample,
        geometry,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory and device_info.device.type == "cuda"),
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    loader = DataLoader(dataset, **loader_kwargs)

    probability_batches: list[np.ndarray] = []
    model_prediction_seconds = 0.0
    progress = tqdm(
        total=len(dataset),
        desc=f"Predicting {slide_id}",
        unit="patch",
        dynamic_ncols=True,
    )
    try:
        for images in loader:
            probabilities, elapsed = predict_tensor_batch(
                model,
                images,
                device_info,
                use_amp,
            )
            if probabilities.shape[1] != num_labels:
                raise RuntimeError(
                    f"Model produced {probabilities.shape[1]} outputs; expected "
                    f"{num_labels}"
                )
            probability_batches.append(probabilities)
            model_prediction_seconds += elapsed
            progress.update(len(probabilities))
    finally:
        progress.close()
        dataset.close()
        del loader

    if not probability_batches:
        raise RuntimeError("No prediction batches were produced")
    probabilities = np.concatenate(probability_batches, axis=0)
    if probabilities.shape != (len(geometry), num_labels):
        raise RuntimeError(
            f"Prediction shape mismatch: received {probabilities.shape}, expected "
            f"{(len(geometry), num_labels)}"
        )
    return probabilities, float(model_prediction_seconds)
