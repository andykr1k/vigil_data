"""Probe solver: outlier tags, single-tag flips, joint multi-camera + depth refinement."""

import cv2
import numpy as np
import pytest

from vigil.probe.aruco import DetectedMarkerPose
from vigil.probe.geometry import ProbeGeometry
from vigil.probe.object_pose import ObjectPose
from vigil.probe.solver import CameraView, clean_view, refine_multiview

G = ProbeGeometry.clarius(0.04, (0.05, 0.05, 0.05), (0.0, 0.1925096, -0.003594))
K = np.array([[604.0, 0, 424], [0, 604.0, 240], [0, 0, 1]])
D = np.zeros(5)
rng = np.random.default_rng(1)


def rot(v):
    return cv2.Rodrigues(np.asarray(v, float))[0]


def to_h(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def observe(T_cam_obj, ids, noise_px=0.0, shift=None):
    out = []
    for i in ids:
        P = G.marker_corners_in_object(i)
        Pc = (T_cam_obj[:3, :3] @ P.T).T + T_cam_obj[:3, 3]
        uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, D)
        uv = uv.reshape(4, 2) + rng.normal(0, noise_px, (4, 2))
        if shift is not None and i == shift[0]:
            uv = uv + shift[1]
        out.append(DetectedMarkerPose(i, np.zeros((3, 1)), np.zeros(3), 1000.0, uv))
    return out


def depth_image(T_cam_obj, ids, bias=0.0, noise=0.0):
    """Depth map with the true per-pixel depth over each tag's surface (± bias/noise)."""
    d = np.zeros((480, 848), np.float32)
    vs, us = np.mgrid[0:480, 0:848]
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], -1)
    for i in ids:
        P = G.marker_corners_in_object(i)
        Pc = (T_cam_obj[:3, :3] @ P.T).T + T_cam_obj[:3, 3]
        uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, D)
        mask = np.zeros_like(d, np.uint8)
        cv2.fillConvexPoly(mask, np.round(uv.reshape(4, 2)).astype(np.int32), 1)
        n = np.cross(Pc[1] - Pc[0], Pc[3] - Pc[0])
        z = (n @ Pc[0]) / (rays @ n)  # ray–plane intersection
        noisy = z + bias + rng.normal(0, noise, z.shape)
        d[mask > 0] = noisy[mask > 0]
    return d


def pose_err(pose, T):
    return (np.linalg.norm(pose.position - T[:3, 3]) * 1000,
            np.degrees(np.arccos(np.clip((np.trace(pose.rotation_matrix.T @ T[:3, :3]) - 1) / 2, -1, 1))))


T_TRUE = to_h(rot([0.5, -0.6, 0.2]), np.array([0.05, -0.02, 0.8]))


def test_bad_tag_is_rejected_and_pose_recovered():
    markers = observe(T_TRUE, (0, 2, 4), noise_px=0.2, shift=(4, np.array([9.0, -7.0])))
    view = clean_view(markers, G, K, D)
    assert view.rejected == [4]
    assert [m.marker_id for m in view.markers] == [0, 2]
    dt, da = pose_err(view.pose, T_TRUE)
    assert dt < 3 and da < 1


def test_single_tag_flip_resolved_with_previous_pose():
    # A tag seen nearly face-on at distance: the mirrored tilt fits the corners almost as well.
    T = to_h(rot([0.0, 0.35, 0.0]) @ rot([0.1, 0, 0]), np.array([0.0, 0.0, 1.4]))
    markers = observe(T, (0,), noise_px=0.3)
    near_truth = ObjectPose((0,), T[:3, 3], T[:3, :3] @ rot([0.02, 0.01, 0]))
    good = clean_view(markers, G, K, D, prev_pose=near_truth)
    assert pose_err(good.pose, T)[1] < 12  # single small tag: a few degrees of noise
    # The other IPPE candidate is a genuinely different (mirrored) pose — and here it even
    # has the *lower* reprojection error, so "pick the best fit" alone would be wrong.
    from vigil.probe.object_pose import _single_marker_object_pose
    from vigil.probe.geometry import marker_object_points

    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(marker_object_points(0.04), markers[0].image_corners,
                                                K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    cands = [_single_marker_object_pose(DetectedMarkerPose(0, r, tv.reshape(3), 1.0, markers[0].image_corners), G)
             for r, tv in zip(rvecs, tvecs)]
    wrong = max(cands, key=lambda c: pose_err(c, T)[1])
    assert pose_err(wrong, T)[1] > pose_err(good.pose, T)[1] + 15
    # History pointing at the other candidate selects it: the choice follows the previous pose.
    bad = clean_view(markers, G, K, D, prev_pose=wrong)
    assert pose_err(bad.pose, T)[1] == pytest.approx(pose_err(wrong, T)[1])
    assert good.flip_resolved and bad.flip_resolved


def test_two_disagreeing_tags_keep_the_larger():
    markers = observe(T_TRUE, (0, 2), noise_px=0.2, shift=(2, np.array([12.0, 9.0])))
    markers[0] = DetectedMarkerPose(0, markers[0].rotation_vector, markers[0].translation,
                                    5000.0, markers[0].image_corners)
    view = clean_view(markers, G, K, D)
    assert view.rejected == [2] and [m.marker_id for m in view.markers] == [0]


def test_multiview_refine_beats_single_camera_depth_error():
    T_w_c2 = to_h(rot([0.1, -1.1, 0.05]), np.array([0.9, 0.05, 0.35]))
    errs_single, errs_joint = [], []
    for _ in range(20):
        T_c1 = T_TRUE
        T_c2 = np.linalg.inv(T_w_c2) @ T_TRUE
        m1, m2 = observe(T_c1, (0, 2), noise_px=0.6), observe(T_c2, (2, 1), noise_px=0.6)
        single = clean_view(m1, G, K, D).pose
        errs_single.append(pose_err(single, T_TRUE)[0])
        views = [CameraView(np.eye(4), K, D, m1), CameraView(T_w_c2, K, D, m2)]
        ref = refine_multiview(views, G, single, use_depth=False)
        errs_joint.append(pose_err(ref.pose, T_TRUE)[0])
    assert np.mean(errs_joint) < 0.6 * np.mean(errs_single)


def test_depth_residuals_tighten_single_camera_distance():
    errs_px, errs_depth = [], []
    for _ in range(20):
        m = observe(T_TRUE, (0,), noise_px=0.6)
        d = depth_image(T_TRUE, (0,), noise=0.003)
        init = clean_view(m, G, K, D).pose
        a = refine_multiview([CameraView(np.eye(4), K, D, m, None)], G, init, use_depth=False)
        b = refine_multiview([CameraView(np.eye(4), K, D, m, d)], G, init, use_depth=True)
        errs_px.append(abs(a.pose.position[2] - T_TRUE[2, 3]) * 1000)
        errs_depth.append(abs(b.pose.position[2] - T_TRUE[2, 3]) * 1000)
        assert b.depth_samples == 5  # centre + 4 inner points of the tag
    assert np.mean(errs_depth) < 0.7 * np.mean(errs_px)
