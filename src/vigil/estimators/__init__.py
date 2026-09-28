from __future__ import annotations

from ..config import Config
from .base import PoseEstimator, PoseResult


def build_estimator(cfg: Config) -> PoseEstimator:
    est = cfg.estimator
    if est.backend == "sam3d_body":
        from .sam3d_body import Sam3dBodyEstimator

        return Sam3dBodyEstimator(est.sam3d_body, cfg.resolve(est.sam3d_body.repo_dir), est.device)
    if est.backend == "vitpose_depth":
        from .vitpose_depth import VitposeDepthEstimator

        return VitposeDepthEstimator(est.vitpose_depth, est.device)
    raise ValueError(f"unknown backend {est.backend!r}")


__all__ = ["PoseEstimator", "PoseResult", "build_estimator"]
