"""Multi-view triangulation of joints and body/probe time alignment."""

from types import SimpleNamespace

import cv2
import numpy as np

from vigil.body import BodyState, BodyWorker, fuse_people, triangulate_joints
from vigil.camera import Intrinsics
from vigil.estimators.base import PoseResult
from vigil.rig import to_h
from vigil.skeleton import JOINT_INDEX, JOINTS, empty_joints

K = Intrinsics(848, 480, 604.0, 604.0, 424.0, 240.0)
rng = np.random.default_rng(5)


def rot(v):
    return cv2.Rodrigues(np.asarray(v, float))[0]


def cam(T_world_camera):
    return SimpleNamespace(T_world_camera=T_world_camera, camera=SimpleNamespace(intrinsics=K), index=0)


CAMS = [cam(np.eye(4)), cam(to_h(rot([0.05, -0.9, 0.02]), np.array([1.0, 0.0, 0.5]))),
        cam(to_h(rot([0.0, 0.7, 0.0]), np.array([-0.9, 0.05, 0.4])))]
TRUE = {"left_hip": (0.1, 0.0, 2.0), "right_hip": (-0.1, 0.0, 2.0), "left_knee": (0.12, 0.45, 1.95),
        "right_knee": (-0.1, 0.45, 2.02), "left_ankle": (0.12, 0.88, 2.0), "right_ankle": (-0.1, 0.88, 2.05)}


def view(c, depth_error=None, swap=None, noise=1.0):
    """What one camera's estimator returns: depth-lifted joints (world) + its 2D keypoints."""
    j = empty_joints()
    kp = np.zeros((len(JOINTS), 3), np.float32)
    T_cw = np.linalg.inv(c.T_world_camera)
    for name, X in TRUE.items():
        Xc = T_cw[:3, :3] @ np.array(X) + T_cw[:3, 3]
        uv, _ = cv2.projectPoints(Xc[None], np.zeros(3), np.zeros(3), K.matrix().astype(float), np.zeros(5))
        kp[JOINT_INDEX[name]] = (*(uv.reshape(2) + rng.normal(0, noise, 2)), 0.9)
        lifted = np.array(X, float)
        if depth_error and name in depth_error:  # depth hit an occluder in front of the joint
            ray = lifted - c.T_world_camera[:3, 3]
            lifted = lifted - depth_error[name] * ray / np.linalg.norm(ray)
        j[JOINT_INDEX[name]] = (*lifted, 0.9)
    if swap:
        a, b = JOINT_INDEX[swap[0]], JOINT_INDEX[swap[1]]
        kp[[a, b]] = kp[[b, a]]
    j[JOINT_INDEX["pelvis"]] = (0, 0, 2.0, 0.9)
    return PoseResult(joints=j, kp2d=kp)


def err_mm(joints, name):
    return np.linalg.norm(joints[JOINT_INDEX[name], :3] - np.array(TRUE[name])) * 1000


def test_triangulation_fixes_an_occluded_depth_reading():
    views = [(CAMS[0], view(CAMS[0], depth_error={"left_knee": 0.4})), (CAMS[1], view(CAMS[1]))]
    fused = fuse_people(views, None)
    assert err_mm(fused.joints, "left_knee") > 150  # depth averaging is pulled off by ~20 cm
    tri, n = triangulate_joints(fused, views)
    assert n == len(TRUE)
    assert err_mm(tri.joints, "left_knee") < 15


def test_reprojection_guard_drops_a_bad_view():
    # Camera 3 confuses left and right knee; with three views that view is dropped.
    views = [(c, view(c, swap=("left_knee", "right_knee") if i == 2 else None)) for i, c in enumerate(CAMS)]
    tri, _ = triangulate_joints(fuse_people(views, None), views)
    assert err_mm(tri.joints, "left_knee") < 15 and err_mm(tri.joints, "right_knee") < 15
    # With only two views that disagree there is no majority: keep the depth-lifted joint.
    two = [views[0], views[2]]
    fused = fuse_people(two, None)
    tri2, _ = triangulate_joints(fused, two)
    np.testing.assert_allclose(tri2.joints[JOINT_INDEX["left_knee"]], fused.joints[JOINT_INDEX["left_knee"]])


def test_joints_at_extrapolates_to_the_probe_time():
    w = BodyWorker.__new__(BodyWorker)
    import threading
    w._lock = threading.Lock()
    j = empty_joints()
    j[JOINT_INDEX["left_knee"]] = (0.1, 0.4, 2.0, 0.9)
    vel = np.zeros((len(JOINTS), 3))
    vel[JOINT_INDEX["left_knee"]] = (0.5, 0.0, 0.0)  # moving 0.5 m/s in x
    w.state = BodyState(joints=j, timestamp=10.0, velocity=vel)
    at = w.joints_at(10.04)  # probe frame 40 ms later
    assert abs(at[JOINT_INDEX["left_knee"], 0] - 0.12) < 1e-6
    assert abs(w.joints_at(11.0)[JOINT_INDEX["left_knee"], 0] - 0.15) < 1e-6  # clamped to 100 ms
