"""Probe tracking across the camera rig: detect per camera, fuse in world, filter."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Config
from .aruco import DetectedMarkerPose, detect_marker_poses, make_detector
from .filtering.pose import FILTER_METHODS, ObjectPoseFilter
from .filtering.presets import load_filter_preset_file
from .geometry import ProbeGeometry
from .object_pose import ObjectPose, estimate_object_pose, mean_rotation

log = logging.getLogger(__name__)


@dataclass
class CameraObservation:
    serial: str
    markers: list[DetectedMarkerPose]
    pose: ObjectPose | None  # cube in this camera's optical frame (unfiltered)


class ProbeTracker:
    def __init__(self, cfg: Config):
        pc = cfg.probe
        self.geometry = ProbeGeometry.clarius(pc.marker_length_m, pc.cube_size_m, pc.tip_in_object_m)
        self.detector = make_detector(pc.dictionary)
        self.filter = ObjectPoseFilter()
        preset = cfg.resolve(pc.filter_preset)
        if preset.is_file():
            try:
                method, tuning = load_filter_preset_file(preset)
                self.filter.set_method(method)
                self.filter.set_tuning(tuning)
            except (OSError, ValueError) as e:
                log.warning("ignoring probe filter preset %s: %s", preset, e)
        if pc.filter_method:
            self.filter.set_method(pc.filter_method)

    def observe(self, rgb: np.ndarray, K: np.ndarray, dist: np.ndarray,
                serial: str) -> CameraObservation:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        markers = detect_marker_poses(gray, self.detector, self.geometry, K, dist)
        pose = estimate_object_pose(markers, self.geometry, K, dist)
        return CameraObservation(serial, markers, pose)

    def update(self, world_poses: list[ObjectPose], timestamp: float) -> ObjectPose | None:
        """Fuse per-camera cube poses (already in the world frame) and run the filter."""
        return self.filter.update(fuse_poses(world_poses), timestamp)

    def set_method(self, method: str) -> None:
        if method not in FILTER_METHODS:
            raise ValueError(f"unknown probe filter {method!r}")
        self.filter.set_method(method)

    def tip_world(self, pose: ObjectPose) -> np.ndarray:
        return pose.rotation_matrix @ self.geometry.tip_array + pose.position


def fuse_poses(poses: list[ObjectPose]) -> ObjectPose | None:
    """Weight each camera by how many tags it saw (more corners → better PnP)."""
    if not poses:
        return None
    if len(poses) == 1:
        return poses[0]
    best = max(poses, key=lambda p: len(p.marker_ids))
    # A camera that disagrees badly (e.g. a single-tag pose flip) is dropped, not averaged.
    poses = [p for p in poses
             if np.linalg.norm(p.position - best.position) < 0.05
             and _angle_deg(p.rotation_matrix, best.rotation_matrix) < 20]
    w = np.array([len(p.marker_ids) for p in poses], dtype=np.float64)
    position = (w[:, None] * np.array([p.position for p in poses])).sum(0) / w.sum()
    rotation = mean_rotation([p.rotation_matrix for p in poses], list(w))
    ids = tuple(sorted({i for p in poses for i in p.marker_ids}))
    return ObjectPose(ids, position, rotation)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))
