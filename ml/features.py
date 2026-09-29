"""Pure geometry on normalized MediaPipe image coordinates (x, y, visibility)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = ("left_knee", "right_knee", "left_hip", "right_hip", "torso_lean", "knee_width_ratio")
JOINTS = np.array([11, 12, 23, 24, 25, 26, 27, 28])
Array = NDArray[np.float64]


def angle(a: Array, b: Array, c: Array) -> float:
    """Interior 2D angle, or NaN for missing/degenerate geometry."""
    u, v = np.asarray(a)[:2] - np.asarray(b)[:2], np.asarray(c)[:2] - np.asarray(b)[:2]
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if not np.isfinite(denominator) or denominator < 1e-8:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denominator, -1, 1))))


def frame_features(landmarks: Array, min_visibility: float = 0.7) -> Array:
    """Return six features or six NaNs; require all eight joints to be reliable.

    Input is (33, 3): x, y, visibility, NOT x, y, z. Angles intentionally use
    normalized image coordinates as specified; camera aspect ratio affects them.
    """
    p = np.asarray(landmarks, dtype=float)
    if p.shape != (33, 3):
        raise ValueError("Expected (33, 3) x/y/visibility landmarks.")
    valid = p[JOINTS]
    if (not np.isfinite(valid).all() or np.any(valid[:, 2] < min_visibility)
            or np.any(valid[:, :2] < 0) or np.any(valid[:, :2] > 1)):
        return np.full(6, np.nan)
    shoulders, hips = p[[11, 12], :2].mean(0), p[[23, 24], :2].mean(0)
    ankle_width = np.linalg.norm(p[27, :2] - p[28, :2])
    values = np.array([
        angle(p[23], p[25], p[27]), angle(p[24], p[26], p[28]),
        angle(p[11], p[23], p[25]), angle(p[12], p[24], p[26]),
        angle(shoulders, hips, hips + [0, -1]),
        np.linalg.norm(p[25, :2] - p[26, :2]) / ankle_width if ankle_width > 1e-8 else np.nan,
    ])
    return values if np.isfinite(values).all() else np.full(6, np.nan)


def resample(values: Array, frames: int = 32, times: Array | None = None) -> Array:
    """Interpolate finite (N, 6) rows on timestamps, preserving gaps in time."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < 2 or frames < 2:
        raise ValueError("Need at least two rows of six features and frames >= 2.")
    times = np.arange(len(values), dtype=float) if times is None else np.asarray(times, dtype=float)
    if (times.shape != (len(values),) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(values).all()):
        raise ValueError("Features must be finite and timestamps strictly increasing.")
    target = np.linspace(times[0], times[-1], frames)
    return np.stack([np.interp(target, times, values[:, i]) for i in range(6)], axis=1)


def sequence_features(landmarks: Array, frames: int = 32) -> Array:
    """Apply identical training/inference filtering; require eight valid frames."""
    values = np.asarray([frame_features(frame) for frame in landmarks])
    if values.size == 0:
        raise ValueError("Need at least 8 valid pose frames.")
    valid = np.isfinite(values).all(axis=1)
    if valid.sum() < 8:
        raise ValueError("Need at least 8 valid pose frames.")
    return resample(values[valid], frames, np.flatnonzero(valid))


def summary_features(x: Array) -> Array:
    """Min, max, mean, range: (..., T, 6) -> (..., 24)."""
    return np.concatenate((x.min(-2), x.max(-2), x.mean(-2), np.ptp(x, axis=-2)), axis=-1)
