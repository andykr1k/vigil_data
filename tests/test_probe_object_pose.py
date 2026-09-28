"""Ported from DataCollection/tests/test_object_pose.py."""

from dataclasses import replace
from itertools import combinations

import cv2
import numpy as np
import pytest

from vigil.probe.aruco import DetectedMarkerPose
from vigil.probe.geometry import ProbeGeometry, marker_object_points
from vigil.probe.object_pose import estimate_object_pose as _estimate

GEOMETRY = ProbeGeometry.clarius(0.04, (0.05, 0.05, 0.05), (0.0, 0.1925096, -0.003594))
MARKER_MOUNTS = GEOMETRY.mounts


def estimate_object_pose(observations, camera_matrix, dist_coeffs):
    return _estimate(observations, GEOMETRY, camera_matrix, dist_coeffs)


CAMERA_MATRIX = np.array(
    [[1200.0, 0.0, 960.0], [0.0, 1210.0, 540.0], [0.0, 0.0, 1.0]]
)
DIST_COEFFS = np.array([0.04, -0.02, 0.001, -0.002, 0.01])
EXPECTED_ROTATION, _ = cv2.Rodrigues(np.array([0.3, -0.4, 0.2]))
EXPECTED_POSITION = np.array([0.08, -0.03, 0.75])


def test_five_marker_mounts_leave_attachment_face_unused() -> None:
    assert tuple(MARKER_MOUNTS) == (0, 1, 2, 3, 4)
    assert MARKER_MOUNTS[4].face == "bottom (-Y)"
    np.testing.assert_allclose(MARKER_MOUNTS[4].center, [0.0, -0.025, 0.0])


def make_observation(marker_id: int) -> DetectedMarkerPose:
    mount = MARKER_MOUNTS[marker_id]
    marker_rotation = EXPECTED_ROTATION @ mount.rotation_object_from_marker
    rotation_vector, _ = cv2.Rodrigues(marker_rotation)
    translation = EXPECTED_POSITION + EXPECTED_ROTATION @ np.asarray(mount.center)
    image_corners, _ = cv2.projectPoints(
        marker_object_points(0.04),
        rotation_vector,
        translation,
        CAMERA_MATRIX,
        DIST_COEFFS,
    )
    image_corners = image_corners.reshape(4, 2)
    return DetectedMarkerPose(
        marker_id=marker_id,
        rotation_vector=rotation_vector,
        translation=translation,
        pixel_area=abs(cv2.contourArea(image_corners.astype(np.float32))),
        image_corners=image_corners,
    )


@pytest.mark.parametrize(
    "marker_ids",
    [ids for count in (1, 2, 3) for ids in combinations(MARKER_MOUNTS, count)],
)
def test_joint_pose_uses_every_visible_tag(marker_ids: tuple[int, ...]) -> None:
    pose = estimate_object_pose(
        [make_observation(marker_id) for marker_id in marker_ids],
        CAMERA_MATRIX,
        DIST_COEFFS,
    )

    assert pose is not None
    assert pose.marker_ids == marker_ids
    np.testing.assert_allclose(pose.position, EXPECTED_POSITION, atol=1e-6)
    np.testing.assert_allclose(pose.rotation_matrix, EXPECTED_ROTATION, atol=1e-6)


def test_visible_tag_set_changes_between_frames() -> None:
    for marker_ids in ((1,), (1, 2), (1, 2, 4), (2, 4), (4,), ()):
        pose = estimate_object_pose(
            [make_observation(marker_id) for marker_id in marker_ids],
            CAMERA_MATRIX,
            DIST_COEFFS,
        )
        if not marker_ids:
            assert pose is None
        else:
            assert pose is not None
            assert pose.marker_ids == marker_ids
            np.testing.assert_allclose(pose.position, EXPECTED_POSITION, atol=1e-6)


def test_joint_pose_uses_corners_not_averaged_individual_poses() -> None:
    observations = [
        replace(
            make_observation(marker_id),
            translation=np.array([2.0, 3.0, 4.0]),
            rotation_vector=np.zeros((3, 1)),
        )
        for marker_id in (1, 2, 4)
    ]
    pose = estimate_object_pose(observations, CAMERA_MATRIX, DIST_COEFFS)

    assert pose is not None
    assert pose.marker_ids == (1, 2, 4)
    np.testing.assert_allclose(pose.position, EXPECTED_POSITION, atol=1e-6)
    np.testing.assert_allclose(pose.rotation_matrix, EXPECTED_ROTATION, atol=1e-6)


def test_unconfigured_tags_do_not_contribute() -> None:
    unknown = replace(make_observation(0), marker_id=49)
    assert estimate_object_pose([unknown], CAMERA_MATRIX, DIST_COEFFS) is None

    pose = estimate_object_pose(
        [make_observation(2), unknown, make_observation(1)],
        CAMERA_MATRIX,
        DIST_COEFFS,
    )
    assert pose is not None
    assert pose.marker_ids == (1, 2)


@pytest.mark.parametrize("failure", ["false", "error", "behind_camera"])
def test_failed_joint_solve_falls_back_to_largest_tag(monkeypatch, failure) -> None:
    observations = [
        replace(make_observation(1), pixel_area=1000.0),
        replace(make_observation(2), pixel_area=2000.0),
    ]

    def failed_solve(*args, **kwargs):
        if failure == "error":
            raise cv2.error("synthetic solve failure")
        return failure != "false", np.zeros((3, 1)), np.array([[0.0], [0.0], [-1.0]])

    monkeypatch.setattr(cv2, "solvePnP", failed_solve)
    monkeypatch.setattr(cv2, "solvePnPRefineLM", lambda *args: args[-2:])
    pose = estimate_object_pose(observations, CAMERA_MATRIX, DIST_COEFFS)

    assert pose is not None
    assert pose.marker_ids == (2,)
    np.testing.assert_allclose(pose.position, EXPECTED_POSITION, atol=1e-9)
    np.testing.assert_allclose(pose.rotation_matrix, EXPECTED_ROTATION, atol=1e-9)
