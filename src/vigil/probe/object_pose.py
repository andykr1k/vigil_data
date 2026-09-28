"""Marker-to-cube pose conversion (ported from DataCollection's vision/object_pose.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .aruco import DetectedMarkerPose
from .geometry import ProbeGeometry


@dataclass(frozen=True)
class ObjectPose:
    marker_ids: tuple[int, ...]
    position: np.ndarray
    rotation_matrix: np.ndarray


def _single_marker_object_pose(marker_pose: DetectedMarkerPose,
                               geometry: ProbeGeometry) -> ObjectPose:
    mount = geometry.mounts[marker_pose.marker_id]
    rotation_camera_from_marker, _ = cv2.Rodrigues(marker_pose.rotation_vector)
    # camera_T_marker = camera_T_object @ object_T_marker
    rotation_camera_from_object = rotation_camera_from_marker @ mount.rotation_object_from_marker.T
    position = marker_pose.translation - rotation_camera_from_object @ np.asarray(mount.center)
    return ObjectPose((marker_pose.marker_id,), position, rotation_camera_from_object)


def estimate_object_pose(
    marker_poses: Iterable[DetectedMarkerPose],
    geometry: ProbeGeometry,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> ObjectPose | None:
    """Cube pose in the camera frame.

    All visible tags' corners go into one SQPNP solve (+ LM refinement); if that
    fails or is implausible, fall back to the largest single tag.
    """
    candidates = sorted(
        (p for p in marker_poses if p.marker_id in geometry.mounts),
        key=lambda p: p.marker_id,
    )
    if not candidates:
        return None
    fallback = _single_marker_object_pose(max(candidates, key=lambda p: p.pixel_area), geometry)
    if len(candidates) == 1:
        return fallback

    object_points = np.concatenate([geometry.marker_corners_in_object(p.marker_id)
                                    for p in candidates])
    image_points = np.concatenate([p.image_corners for p in candidates]).astype(np.float64)
    try:
        # Unlike IPPE_SQUARE, SQPNP supports corners on different box faces.
        success, rotation_vector, translation = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            return fallback
        rotation_vector, translation = cv2.solvePnPRefineLM(
            object_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation)
        rotation, _ = cv2.Rodrigues(rotation_vector)
    except cv2.error:
        return fallback

    position = translation.reshape(3)
    points_in_camera = (rotation @ object_points.T).T + position
    if (not np.isfinite(rotation).all() or not np.isfinite(position).all()
            or np.any(points_in_camera[:, 2] <= 0)):
        return fallback
    return ObjectPose(tuple(p.marker_id for p in candidates), position, rotation)


def mean_rotation(rotations: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Chordal L2 mean of rotation matrices (projection of the weighted sum onto SO(3))."""
    w = np.ones(len(rotations)) if weights is None else np.asarray(weights, dtype=np.float64)
    m = sum(wi * r for wi, r in zip(w, rotations))
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt
