"""Load, validate, and select class-specific prediction thresholds."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from constants import LABELS, THRESHOLD_PROFILES


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file and validate its top-level object structure."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in threshold file: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"Threshold file must contain a JSON object: {path}")
    return payload


def load_threshold_profiles(path: Path) -> dict[str, dict[str, float]]:
    """Load and validate all threshold profiles and class-specific values."""
    if not path.is_file():
        raise FileNotFoundError(f"Threshold file not found: {path}")

    payload = load_json_object(path)
    unexpected_profiles = sorted(set(payload) - set(THRESHOLD_PROFILES))
    missing_profiles = sorted(set(THRESHOLD_PROFILES) - set(payload))
    if unexpected_profiles or missing_profiles:
        raise ValueError(
            "Threshold profiles must be exactly balanced and recall; "
            f"missing={missing_profiles}, unexpected={unexpected_profiles}"
        )

    profiles: dict[str, dict[str, float]] = {}
    expected_labels = set(LABELS)
    for profile_name in THRESHOLD_PROFILES:
        raw_profile = payload[profile_name]
        if not isinstance(raw_profile, dict):
            raise TypeError(f"Threshold profile {profile_name!r} must be an object")
        missing = sorted(expected_labels - set(raw_profile))
        unexpected = sorted(set(raw_profile) - expected_labels)
        if missing or unexpected:
            raise ValueError(
                f"Threshold profile {profile_name!r} must use exactly the five "
                f"canonical labels; missing={missing}, unexpected={unexpected}"
            )

        values: dict[str, float] = {}
        for label in LABELS:
            try:
                value = float(raw_profile[label])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Threshold {profile_name}/{label} is not numeric"
                ) from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Threshold {profile_name}/{label} must be within [0, 1]"
                )
            values[label] = value
        profiles[profile_name] = values
    return profiles


def select_threshold_profile(
    profiles: dict[str, dict[str, float]],
    profile_name: str,
) -> dict[str, float]:
    """Return the thresholds associated with the selected profile."""
    if profile_name not in THRESHOLD_PROFILES:
        raise ValueError(
            f"THRESHOLD_PROFILE must be one of {THRESHOLD_PROFILES}, "
            f"received {profile_name!r}"
        )
    return dict(profiles[profile_name])