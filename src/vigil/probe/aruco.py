"""ArUco detection and per-tag pose (ported from DataCollection's vision/aruco.py).

Drawing was dropped: tag outlines and the tip dot are drawn by the dashboard instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import ProbeGeometry, marker_object_points


@dataclass(frozen=True)
class DetectedMarkerPose:
    """One marker's pose and detected image corners."""

    marker_id: int
    rotation_vector: np.ndarray
    translation: np.ndarray
    pixel_area: float
    image_corners: np.ndarray


def make_detector(dictionary_name: str) -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    # Sub-pixel corners noticeably steady the PnP pose.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def generate_tag_image(dictionary_name: str, marker_id: int, size_px: int = 600,
                       margin_px: int = 50) -> np.ndarray:
    """Printable tag with a white margin (same layout as DataCollection's generator)."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px, borderBits=1)
    return cv2.copyMakeBorder(marker, margin_px, margin_px, margin_px, margin_px,
                              cv2.BORDER_CONSTANT, value=255)


def estimate_marker_pose(corners: np.ndarray, marker_length: float, camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    success, rotation, translation = cv2.solvePnP(
        marker_object_points(marker_length),
        corners.reshape(4, 2),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None
    return rotation, translation.reshape(3)


def detect_marker_poses(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    geometry: ProbeGeometry,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[DetectedMarkerPose]:
    """Poses of every configured tag visible in `image` (gray or color)."""
    corners, ids, _rejected = detector.detectMarkers(image)
    if ids is None:
        return []
    out: list[DetectedMarkerPose] = []
    for marker_corners, marker_id_value in zip(corners, ids.flat):
        marker_id = int(marker_id_value)
        if marker_id not in geometry.mounts:
            continue  # someone else's tag
        pose = estimate_marker_pose(marker_corners, geometry.marker_length, camera_matrix,
                                    dist_coeffs)
        if pose is None:
            continue
        rotation, position = pose
        quad = marker_corners.reshape(4, 2).astype(np.float32)
        out.append(DetectedMarkerPose(
            marker_id=marker_id,
            rotation_vector=rotation.copy(),
            translation=position.copy(),
            pixel_area=abs(float(cv2.contourArea(quad))),
            image_corners=quad.astype(np.float64),
        ))
    return out
