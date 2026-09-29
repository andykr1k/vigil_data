"""Bundle-refined rig calibration, pivot calibration of the tip, calibration health."""

import cv2
import numpy as np
import pytest

from vigil.probe.geometry import ProbeGeometry
from vigil.probe.object_pose import ObjectPose
from vigil.probe.pivot import PivotCalibrator, load_tip, save_tip, solve_pivot
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


def _pnp(tags):
    from vigil.probe.aruco import DetectedMarkerPose
    from vigil.probe.object_pose import estimate_object_pose

    ids = [next(i for i in G.mounts if np.allclose(G.marker_corners_in_object(i), P)) for P, _ in tags]
    return estimate_object_pose([DetectedMarkerPose(i, np.zeros((3, 1)), np.zeros(3), 1000.0, uv)
                                 for i, (_, uv) in zip(ids, tags)], G, K, D)


def test_bundle_refinement_fits_corners_and_is_at_least_as_accurate():
    """Per-frame poses and the bundle both come from the same noisy corners (fair test)."""
    errs_b, errs_a = [], []
    for _ in range(6):
        cal = RigCalibrator(target=40, geometry=G)
        for spot in range(4):
            T_wo = to_h(rot(rng.normal(0, 0.5, 3)), np.array([0.25 * spot - 0.35, 0.1, 1.3]))
            T_co = np.linalg.inv(T_W_C2) @ T_wo
            for _ in range(12):
                cw, cc = corners(T_wo, (0, 2, 3, 4), 0.2), corners(T_co, (0, 1, 2, 3, 4), 0.2)
                cal.add("world", {"world": _pnp(cw), "cam2": _pnp(cc)},
                        {"world": (K, D, cw), "cam2": (K, D, cc)})
        solved = cal.solve("cam2")
        if solved is None:
            continue
        T, meta = solved
        assert meta.get("method") == "bundle adjustment"
        assert meta["reprojection_px"] < meta["reprojection_px_before"]
        avg = RigCalibrator(target=40)
        avg.samples, avg.observations = cal.samples, {}
        errs_b.append(np.linalg.norm(T[:3, 3] - T_W_C2[:3, 3]) * 1000)
        errs_a.append(np.linalg.norm(avg.solve("cam2")[0][:3, 3] - T_W_C2[:3, 3]) * 1000)
    assert len(errs_b) >= 4
    assert np.mean(errs_b) < 4.0 and np.mean(errs_b) <= np.mean(errs_a) * 1.1


def test_pivot_calibration_recovers_the_true_tip():
    true_tip = np.array([0.0012, 0.1941, -0.0051])  # the attachment differs from CAD by ~3 mm
    pivot = np.array([0.1, 0.3, 1.2])
    cal = PivotCalibrator(target=200)
    for i in range(200):
        R = rot([0.4 * np.sin(i / 17), 0.5 * np.cos(i / 23), 0.3 * np.sin(i / 11)]) @ rot([np.pi / 2, 0, 0])
        t = pivot - R @ true_tip + rng.normal(0, 0.0008, 3)  # ~1 mm tracking noise
        if i % 40 == 0:
            t = t + 0.03  # a bump: gross outlier
        cal.add(ObjectPose((0,), t, R))
    assert cal.finished() and cal.progress()["spread_deg"] > 25
    res = solve_pivot(cal.poses)
    assert np.linalg.norm(res.tip - true_tip) * 1000 < 1.0
    assert np.linalg.norm(res.pivot - pivot) * 1000 < 1.0
    assert res.rejected >= 5 and res.rms_mm < 2.0


def test_pivot_needs_rotation_and_round_trips(tmp_path):
    assert solve_pivot([ObjectPose((0,), np.zeros(3), np.eye(3))] * 5) is None
    res = solve_pivot([ObjectPose((0,), np.array([0, 0, 1.0]) - rot([0.3 * np.sin(i), 0.2 * i / 30, 0]) @ G.tip_array,
                                  rot([0.3 * np.sin(i), 0.2 * i / 30, 0])) for i in range(30)])
    save_tip(tmp_path / "tip.yaml", res, G.tip_array)
    np.testing.assert_allclose(load_tip(tmp_path / "tip.yaml"), res.tip, atol=1e-6)


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
