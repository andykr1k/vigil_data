from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..camera import Frame, Intrinsics


@dataclass
class PoseResult:
    joints: np.ndarray  # (J, 4) xyz metres in camera frame + confidence (skeleton.JOINTS order)
    kp2d: np.ndarray  # (J, 3) pixel u, v + confidence
    vertices: np.ndarray | None = None  # (V, 3) camera frame, metres


class PoseEstimator(ABC):
    name: str
    faces: np.ndarray | None = None  # (F, 3) triangle indices when the backend produces a mesh

    @abstractmethod
    def estimate(self, frame: Frame, bbox: np.ndarray, K: Intrinsics) -> PoseResult | None:
        """Estimate the pose of the person inside `bbox` (xyxy pixels)."""

    def estimate_batch(self, frames: list[Frame], bboxes: list[np.ndarray],
                       Ks: list[Intrinsics]) -> list[PoseResult | None]:
        """One person per camera. Backends that can batch across cameras override this."""
        return [self.estimate(f, b, K) for f, b, K in zip(frames, bboxes, Ks)]

    def warmup(self, batch: int) -> None:
        """Called once on the inference thread before the loop (e.g. to compile)."""
