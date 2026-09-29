"""Calibration health monitor."""

import cv2
import numpy as np
import pytest

from vigil.probe.geometry import ProbeGeometry
from vigil.probe.object_pose import ObjectPose
from vigil.rig import RigCalibrator, RigHealth, to_h

G = ProbeGeometry.clarius(0.04, (0.05, 0.05, 0.05), (0.0, 0.1925096, -0.003594))
K = np.array([[604.0, 0, 424], [0, 604.0, 240], [0, 0, 1]])
D = np.zeros(5)
rng = np.random.default_rng(7)
T_W_C2 = to_h(cv2.Rodrigues(np.array([0.05, -1.0, 0.03]))[0], np.array([0.8, 0.02, 0.4]))


def rot(v):
    return cv2.Rodrigues(np.asarray(v, float))[0]


def corners(T_cam_obj, ids, noise):
    tags = []
    for i in ids:
        P = G.marker_corners_in_object(i)
        Pc = (T_cam_obj[:3, :3] @ P.T).T + T_cam_obj[:3, 3]
        if (Pc[:, 2] <= 0).any():
            continue
        uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, D)
        tags.append((P, uv.reshape(4, 2) + rng.normal(0, noise, (4, 2))))
    return tags


def noisy_pose(T, mm, deg):
    return ObjectPose((0,), T[:3, 3] + rng.normal(0, mm / 1000, 3), T[:3, :3] @ rot(rng.normal(0, np.radians(deg), 3)))


def test_health_flags_a_bumped_camera():
    h = RigHealth()
    T = to_h(np.eye(3), np.array([0, 0, 1.0]))
    for i in range(40):
        h.add("world", {"world": noisy_pose(T, 2, 0.3), "cam2": noisy_pose(T, 2, 0.3)}, now=i / 30)
    assert h.report(now=40 / 30)["cam2"]["state"] == "good"
    bumped = to_h(rot([0, 0.08, 0]), np.array([0.03, 0, 1.0]))  # 3 cm / 4.6° off
    for i in range(40, 140):
        h.add("world", {"world": noisy_pose(T, 2, 0.3), "cam2": noisy_pose(bumped, 2, 0.3)}, now=i / 30)
    assert h.report(now=140 / 30)["cam2"]["state"] == "poor"
