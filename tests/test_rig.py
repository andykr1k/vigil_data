"""Rig extrinsic calibration, multi-camera probe fusion, and probe-tip geometry."""

import cv2
import numpy as np
import pytest

from vigil.estimators.base import PoseResult
from vigil.body import fuse_people as _fuse_people
from vigil.pipeline import _nearest_segment
from vigil.probe.object_pose import ObjectPose, mean_rotation
from vigil.probe.tracker import fuse_poses
from vigil.rig import RigCalibrator, Rig, to_h
from vigil.skeleton import JOINT_INDEX, empty_joints

rng = np.random.default_rng(3)


def rot(v) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(v, dtype=np.float64))[0]


# Camera 2 sits 1.2 m to the right, turned 40° towards camera 1's view.
T_WORLD_CAM2 = to_h(rot([0.05, -0.7, 0.02]), np.array([1.2, 0.05, 0.3]))


def cube_views(T_world_cube, noise_m=0.0, noise_deg=0.0):
    """The same cube seen from both cameras, optionally with PnP-like noise."""
    T_cam2_cube = np.linalg.inv(T_WORLD_CAM2) @ T_world_cube

    def obs(T):
        R = T[:3, :3] @ rot(rng.normal(0, np.radians(noise_deg), 3))
        return ObjectPose((0, 2), T[:3, 3] + rng.normal(0, noise_m, 3), R)

    return {"world": obs(T_world_cube), "cam2": obs(T_cam2_cube)}


def test_calibration_recovers_extrinsic_from_still_cube_poses():
    cal = RigCalibrator(target=40)
    for spot in range(4):  # user moves the cube to four places, holding still at each
        T = to_h(rot(rng.normal(0, 0.6, 3)), np.array([0.2 * spot - 0.3, 0.1, 1.5]))
        for _ in range(15):
            cal.add("world", cube_views(T, noise_m=0.002, noise_deg=0.3))
    T, meta = cal.solve("cam2")
    np.testing.assert_allclose(T[:3, 3], T_WORLD_CAM2[:3, 3], atol=0.004)
    angle = np.degrees(np.arccos((np.trace(T[:3, :3].T @ T_WORLD_CAM2[:3, :3]) - 1) / 2))
    assert angle < 0.5
    # The first frames at each spot fill the stillness window.
    assert meta["samples"] >= 30
    assert meta["stderr_mm"] < meta["spread_mm"]


def test_calibration_ignores_moving_cube_and_rejects_flips():
    cal = RigCalibrator(target=10)
    T = to_h(rot([0.3, 0.2, 0.1]), np.array([0.0, 0.0, 1.4]))
    for i in range(30):  # moving: every frame 1 cm away from the last
        cal.add("world", cube_views(to_h(T[:3, :3], T[:3, 3] + [0.01 * i, 0, 0])))
    assert cal.progress().get("cam2", 0) == 0

    for i in range(30):
        views = cube_views(T, noise_m=0.001)
        if i % 10 == 0:  # an IPPE-style flip in camera 2
            p = views["cam2"]
            views["cam2"] = ObjectPose(p.marker_ids, p.position, p.rotation_matrix @ rot([0, np.pi, 0]))
        cal.add("world", views)
    T_est, meta = cal.solve("cam2")
    assert meta["rejected"] >= 2
    np.testing.assert_allclose(T_est[:3, 3], T_WORLD_CAM2[:3, 3], atol=0.003)


def test_calibration_needs_enough_samples():
    cal = RigCalibrator(target=60)
    T = to_h(np.eye(3), np.array([0.0, 0.0, 1.0]))
    for _ in range(5):
        cal.add("world", cube_views(T))
    assert cal.solve("cam2") is None


def test_extrinsics_round_trip_and_rereference(tmp_path):
    """Saved extrinsics are re-expressed when a different camera becomes the world."""
    rig = Rig.__new__(Rig)
    rig.extrinsics_path = tmp_path / "extrinsics.yaml"
    rig._ref, rig._ref_T, rig._meta = None, {}, {}

    class Cam:
        def __init__(self, serial):
            self.serial, self.T_world_camera = serial, None

        calibrated = property(lambda self: self.T_world_camera is not None)

    rig.cameras = [Cam("A"), Cam("B")]
    rig.cameras[0].T_world_camera = np.eye(4)
    rig.set_extrinsic("B", T_WORLD_CAM2, {"stderr_mm": 1.0})
    np.testing.assert_allclose(rig.cameras[1].T_world_camera, T_WORLD_CAM2)

    # Re-open with B as the world camera: A must come out as the inverse.
    rig2 = Rig.__new__(Rig)
    rig2.extrinsics_path = rig.extrinsics_path
    rig2._ref, rig2._ref_T, rig2._meta = None, {}, {}
    rig2.cameras = [Cam("B"), Cam("A")]
    rig2.cameras[0].T_world_camera = np.eye(4)
    rig2._load()
    np.testing.assert_allclose(rig2.cameras[1].T_world_camera, np.linalg.inv(T_WORLD_CAM2),
                               atol=1e-5)


def test_probe_fusion_weights_by_tags_and_drops_disagreeing_camera():
    R = rot([0.1, 0.2, 0.3])
    a = ObjectPose((0, 2, 4), np.array([0.0, 0.0, 1.0]), R)
    b = ObjectPose((2,), np.array([0.004, 0.0, 1.0]), R)
    fused = fuse_poses([a, b])
    np.testing.assert_allclose(fused.position, [0.001, 0, 1.0])
    assert fused.marker_ids == (0, 2, 4)

    flipped = ObjectPose((1,), np.array([0.0, 0.0, 1.0]), R @ rot([np.pi, 0, 0]))
    fused = fuse_poses([a, flipped])
    np.testing.assert_allclose(fused.rotation_matrix, R, atol=1e-9)


def test_mean_rotation_of_symmetric_perturbations_is_the_centre():
    R = rot([0.4, -0.2, 0.9])
    rs = [R @ rot([0.1, 0, 0]), R @ rot([-0.1, 0, 0])]
    np.testing.assert_allclose(mean_rotation(rs), R, atol=1e-9)


def _person(offset, conf=1.0):
    j = empty_joints()
    for name, xyz in {"left_hip": (0.1, 0, 2), "right_hip": (-0.1, 0, 2), "left_knee": (0.1, 0.45, 2),
                      "left_ankle": (0.1, 0.9, 2)}.items():
        j[JOINT_INDEX[name]] = (*(np.array(xyz) + offset), conf)
    j[JOINT_INDEX["pelvis"]] = (*(np.array((0, 0, 2.0)) + offset), conf)
    return PoseResult(joints=j, kp2d=np.zeros((len(j), 3)))


def test_people_fusion_averages_agreeing_views_and_ignores_other_person():
    a, b = _person(np.array([0.02, 0, 0])), _person(np.array([-0.02, 0, 0]))
    fused = _fuse_people([(None, a), (None, b)], prev_pelvis=None)
    np.testing.assert_allclose(fused.joints[JOINT_INDEX["left_knee"], :3], [0.1, 0.45, 2], atol=1e-6)

    stranger = _person(np.array([2.0, 0, 1.0]))
    fused = _fuse_people([(None, a), (None, stranger)], prev_pelvis=np.array([0.0, 0, 2]))
    np.testing.assert_allclose(fused.joints[JOINT_INDEX["pelvis"], :3], [0.02, 0, 2], atol=1e-6)


def test_nearest_segment_reports_distance_to_bone_axis():
    joints = _person(np.zeros(3)).joints
    near = _nearest_segment(np.array([0.1 + 0.03, 0.675, 2.0]), joints)  # beside the mid-shin
    assert near["segment"] == "left_shin"
    assert near["distance_mm"] == pytest.approx(30.0, abs=0.01)
    assert near["along"] == pytest.approx(0.5, abs=1e-6)
    assert _nearest_segment(np.zeros(3), None) is None


def test_keypoint_box_follows_confident_keypoints():
    from vigil.body import _keypoint_box

    kp = np.zeros((21, 3), np.float32)
    kp[:5] = [[100, 50, 0.9], [200, 60, 0.9], [150, 300, 0.8], [120, 400, 0.5], [180, 410, 0.4]]
    kp[5] = [900, 900, 0.1]  # low confidence: ignored
    box = _keypoint_box(kp)
    np.testing.assert_allclose(box, [90, 14, 210, 446])
    assert _keypoint_box(np.zeros((21, 3), np.float32)) is None


def test_real_calibration_metadata_saves_and_reloads(tmp_path):
    """Regression: numpy scalars in the solved metadata crashed yaml.safe_dump."""
    import yaml

    cal = RigCalibrator(target=20)
    T = to_h(rot([0.2, -0.3, 0.1]), np.array([0.0, 0.1, 1.4]))
    for _ in range(30):
        cal.add("world", cube_views(T, noise_m=0.001, noise_deg=0.2))
    T_est, meta = cal.solve("cam2")

    rig = Rig.__new__(Rig)
    rig.extrinsics_path = tmp_path / "extrinsics.yaml"
    rig._ref, rig._ref_T, rig._meta = None, {}, {}

    class Cam:
        def __init__(self, serial):
            self.serial, self.T_world_camera = serial, None

    rig.cameras = [Cam("A"), Cam("B")]
    rig.cameras[0].T_world_camera = np.eye(4)
    rig.set_extrinsic("B", T_est, meta)
    saved = yaml.safe_load(rig.extrinsics_path.read_text())
    assert saved["cameras"]["B"]["samples"] == meta["samples"]
    assert isinstance(saved["cameras"]["B"]["stderr_mm"], float)


def test_voxel_dedupe_merges_overlapping_points():
    from vigil.pipeline import voxel_dedupe

    a = np.array([[0.001, 0.0, 1.0], [0.004, 0.002, 1.003], [0.5, 0.0, 1.0]], np.float32)
    rgb = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.uint8)
    xyz, col = voxel_dedupe(a, rgb, 0.01)
    assert len(xyz) == 2 and len(col) == 2


def _bare_pipeline():
    from vigil.config import Config
    from vigil.pipeline import Pipeline

    return Pipeline(Config(), publish=lambda msg, binaries: None)


def test_depth_window_is_clamped_and_never_empty():
    p = _bare_pipeline()
    p._set_depth_range({"min": 0.6, "max": 1.4})
    assert p.depth_range == (0.6, 1.4)
    p._set_depth_range({"min": -3, "max": 99})  # clamped to the sensor limits
    assert p.depth_range == (0.2, 6.0)
    p._set_depth_range({"min": 1.0, "max": 0.5})  # inverted → a 2 cm sliver, not empty
    assert p.depth_range == (1.0, 1.02)


def test_masked_feed_blacks_out_pixels_outside_the_window():
    import cv2

    from vigil.camera import Frame

    p = _bare_pipeline()
    color = np.full((480, 848, 3), 200, np.uint8)
    depth = np.full((480, 848), 3.0, np.float32)
    depth[:, :424] = 1.0  # left half near, right half far
    frame = Frame(color, depth, 0.0, 0)
    p._set_depth_range({"min": 0.5, "max": 1.5, "mask_feeds": True})
    img = cv2.imdecode(np.frombuffer(p._preview(frame), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    assert img[h // 2, w // 4].mean() > 150  # near half kept
    assert img[h // 2, 3 * w // 4].mean() < 20  # far half removed


def test_frame_buffer_returns_frame_nearest_in_time():
    from vigil.capture import FrameBuffer

    fb = FrameBuffer(64 * 48)
    try:
        for i in range(8):  # 30 fps; the buffer keeps the newest few
            fb.write(np.full((48, 64, 3), i, np.uint8), np.full((48, 64), 1000, np.uint16),
                     index=i, ts=100.0 + i / 30)
        assert fb.read(0.001).index == 7
        f = fb.read(0.001, near_ts=100.0 + 5.2 / 30)
        assert f.index == 5 and f.color[0, 0, 0] == 5
        assert abs(f.depth[0, 0] - 1.0) < 1e-6
    finally:
        fb.close(unlink=True)
