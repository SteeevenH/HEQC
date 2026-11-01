"""
Model initialization, weight loading, and device-aware inference.

Supports ResNet18, EfficientNet-B0, and MobileNetV3-Large with
validated model weights, optional mixed precision, and inference timing.
"""
from __future__ import annotations

import hashlib
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision import models

from constants import SUPPORTED_MODEL_NAMES


@dataclass(frozen=True)
class DeviceInfo:
    """Configuration and identification of the selected compute device."""
    device: torch.device
    mode: str
    name: str
    cuda_device_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "mode": self.mode,
            "name": self.name,
            "cuda_device_index": self.cuda_device_index,
        }


def resolve_device(device_setting: str, cuda_device_index: int) -> DeviceInfo:
    """Select an available CUDA, MPS, or CPU device."""
    requested = device_setting.strip().lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("DEVICE must be auto, cuda, mps, or cpu")

    use_cuda = requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    )
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        count = torch.cuda.device_count()
        if cuda_device_index < 0 or cuda_device_index >= count:
            raise ValueError(
                f"CUDA_DEVICE_INDEX={cuda_device_index} is invalid; visible "
                f"CUDA device count is {count}"
            )
        torch.cuda.set_device(cuda_device_index)
        device = torch.device(f"cuda:{cuda_device_index}")
        return DeviceInfo(
            device=device,
            mode="single_cuda",
            name=torch.cuda.get_device_name(cuda_device_index),
            cuda_device_index=cuda_device_index,
        )

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested == "mps" or (requested == "auto" and mps_available):
        if not mps_available:
            raise RuntimeError("MPS was requested but is unavailable")
        return DeviceInfo(torch.device("mps"), "single_mps", "Apple MPS", None)

    return DeviceInfo(torch.device("cpu"), "cpu", "CPU", None)


def create_model(model_name: str, num_labels: int) -> nn.Module:
    """Initialize a supported architecture with a custom output head."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_labels,
        )
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_labels,
        )
    else:
        raise ValueError(
            f"Unsupported model name {model_name!r}; choose from "
            f"{SUPPORTED_MODEL_NAMES}"
        )
    return model


def load_tensor_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load and validate a tensor-only model state dictionary."""
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Could not safely load model weights from {path}. The file must "
            "contain a tensor state_dict compatible with torch.load(weights_only=True)."
        ) from error

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError(f"Model file does not contain a state_dict: {path}")
    if "model_state_dict" in checkpoint and isinstance(
        checkpoint["model_state_dict"], dict
    ):
        checkpoint = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]

    state_dict: dict[str, torch.Tensor] = {}
    for raw_name, value in checkpoint.items():
        if not torch.is_tensor(value):
            raise TypeError(
                f"Model entry {raw_name!r} is not a tensor. Publish a pure "
                "state_dict without training metadata."
            )
        # Remove DataParallel prefixes for single-device compatibility.
        name = str(raw_name)
        name = name.removeprefix("module.")
        if name in state_dict:
            raise ValueError(f"Duplicate model key after normalization: {name}")
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Model tensor contains NaN or Inf: {name}")
        state_dict[name] = value
    return state_dict


def load_model(
    path: Path,
    model_name: str,
    num_labels: int,
    device_info: DeviceInfo,
) -> nn.Module:
    """Load validated model weights and prepare the model for inference."""
    model = create_model(model_name, num_labels)
    state_dict = load_tensor_state_dict(path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Model weights do not strictly match architecture {model_name!r} "
            f"with {num_labels} outputs: {path}"
        ) from error
    model.to(device_info.device)
    model.eval()
    return model


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 checksum of a file using streamed reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _autocast_context(device: torch.device, use_amp: bool):
    """Configure automatic mixed precision for supported compute devices."""
    if not use_amp:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def predict_tensor_batch(
    model: nn.Module,
    images: torch.Tensor,
    device_info: DeviceInfo,
    use_amp: bool,
) -> tuple[np.ndarray, float]:
    """Return sigmoid probabilities and model forward-pass time for a batch."""
    device = device_info.device
    images = images.to(device, non_blocking=device.type == "cuda")

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode(), _autocast_context(device, use_amp):
            logits = model(images)
        end_event.record()
        end_event.synchronize()
        model_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
    else:
        if device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), _autocast_context(device, use_amp):
            logits = model(images)
        if device.type == "mps":
            torch.mps.synchronize()
        model_seconds = time.perf_counter() - started

    probabilities = torch.sigmoid(logits.float()).detach().cpu().numpy()
    if probabilities.ndim != 2:
        raise RuntimeError(
            f"Model output must be two-dimensional, received {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Model output contains NaN or Inf")
    return probabilities.astype(np.float32, copy=False), model_seconds
