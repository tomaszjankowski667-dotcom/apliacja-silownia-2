"""Deterministic runtime settings for the computer-vision pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionRuntimeConfig:
    pose_confidence_attempts: tuple[float, ...] = (0.35, 0.30)
    opencv_threads: int = 1


DEFAULT_RUNTIME_CONFIG = VisionRuntimeConfig()


def configure_deterministic_runtime() -> VisionRuntimeConfig:
    """Set CPU controls before importing MediaPipe, TensorFlow Lite, or OpenCV."""
    environment = {
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for key, value in environment.items():
        os.environ[key] = value
    return DEFAULT_RUNTIME_CONFIG