"""Rig calibration from fixed world tags (testing.probe = 0), on rendered images.

Tags lie flat on a table; two cameras look down at it from opposite sides (like the rig, which
films the leg from both sides). The calibrator only gets the images and depth.
"""

import cv2
import numpy as np
import pytest

from vigil.config import WorldTagsConfig, load_config
from vigil.probe.aruco import generate_tag_image
from vigil.rig import to_h
from vigil.world_tags import TagInput, TagRigCalibrator, kabsch

W, H = 848, 480
K = np.array([[604.0, 0, 424], [0, 604.0, 240], [0, 0, 1]])
D = np.zeros(5)
L = 0.08  # printed black square
CUBE = {0, 1, 2, 3, 4}
# Tag centres on the table (metres) and their yaw: an irregular layout, nothing measured.
LAYOUT = {5: (-0.25, -0.15, 0.3), 6: (0.22, -0.18, -0.5), 7: (0.05, 0.2, 1.1),
          8: (-0.2, 0.22, 2.0), 2: (0.3, 0.15, 0.0)}  # 2 is a probe-cube tag lying around


def look_at(eye, target=(0, 0, 0)):
    """camera → table transform for an optical frame (z forward, y down) at `eye`."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return to_h(np.column_stack([x, np.cross(z, x), z]), np.asarray(eye, float))


T_TABLE_A = look_at(np.array([0.1, -0.9, 1.1]))
T_TABLE_B = look_at(np.array([-0.05, 0.95, 1.0]), (0.03, 0.0, 0.0))
T_WORLD_B = np.linalg.inv(T_TABLE_A) @ T_TABLE_B  # what calibration should recover


def tag_corners_table(marker_id, size):
    """Corners of the tag image incl. its white margin (TL, TR, BR, BL) on the table."""
    cx, cy, yaw = LAYOUT[marker_id]
    ax, ay = np.array([np.cos(yaw), np.sin(yaw), 0]), np.array([-np.sin(yaw), np.cos(yaw), 0])
    h = size / 2
    c = np.array([cx, cy, 0.0])
    return np.array([c - h * ax + h * ay, c + h * ax + h * ay, c + h * ax - h * ay, c - h * ax - h * ay])


def render(T_table_cam, ids, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 120, np.float32)
    T_cam_table = np.linalg.inv(T_table_cam)
    for i in ids:
        tag = generate_tag_image("DICT_6X6_50", i).astype(np.float32)
        P = tag_corners_table(i, L * 700 / 600) @ T_cam_table[:3, :3].T + T_cam_table[:3, 3]
        uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, D)
        src = np.float32([[0, 0], [700, 0], [700, 700], [0, 700]])
        Hm = cv2.getPerspectiveTransform(src, uv.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(tag, Hm, (W, H), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.ones_like(tag), Hm, (W, H))
        img = img * (1 - mask) + warped
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    # Depth of the table plane (z = 0) along every pixel ray.
    v, u = np.mgrid[0:H, 0:W].astype(float)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d_table = rays @ T_table_cam[:3, :3].T
    depth = (-T_table_cam[2, 3] / d_table[..., 2]).astype(np.float32)
    return np.dstack([img] * 3), np.where(depth > 0, depth, 0)


def calibrate(cfg, ids_a, ids_b, frames=12):
    cal = TagRigCalibrator(cfg, "A", target=frames, exclude_ids=CUBE)
    for f in range(frames):
        cal.add([TagInput("A", K, D, *render(T_TABLE_A, ids_a, seed=2 * f)),
                 TagInput("B", K, D, *render(T_TABLE_B, ids_b, seed=2 * f + 1))])
    return cal


def angle_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def test_cameras_on_opposite_sides_are_placed_from_flat_tags():
    cfg = WorldTagsConfig(marker_length_m=L, min_views=5)
    cal = calibrate(cfg, [5, 6, 7, 8, 2], [5, 6, 7, 8, 2])
    assert cal.finished(["B"])
    assert cal.tags_seen()["A"] == [5, 6, 7, 8]  # the probe-cube tag is ignored
    T, meta = cal.solve("B")
    assert np.linalg.norm(T[:3, 3] - T_WORLD_B[:3, 3]) < 0.01, (T, T_WORLD_B)
    assert angle_deg(T[:3, :3], T_WORLD_B[:3, :3]) < 0.5
    assert meta["source"] == "tags" and meta["tags"] == [5, 6, 7, 8] and meta["warnings"] == []
    assert meta["rms_mm"] < 5
    assert all(abs(r - 1) < 0.03 for r in meta["depth_scale"].values()), meta


def test_only_tags_both_cameras_see_count():
    cfg = WorldTagsConfig(marker_length_m=L, min_views=5)
    cal = calibrate(cfg, [5, 6, 7], [6, 7, 8])
    T, meta = cal.solve("B")
    assert meta["tags"] == [6, 7]
    assert any("fewer than 3" in w for w in meta["warnings"])
    assert np.linalg.norm(T[:3, 3] - T_WORLD_B[:3, 3]) < 0.05


def test_no_shared_tag_fails_and_explicit_ids_are_respected():
    cfg = WorldTagsConfig(marker_length_m=L, min_views=5)
    assert calibrate(cfg, [5, 6], [7, 8], frames=6).solve("B") is None
    cal = calibrate(WorldTagsConfig(marker_length_m=L, min_views=5, ids=[5, 7, 8]), [5, 6, 7, 8], [5, 6, 7, 8], 6)
    assert cal.solve("B")[1]["tags"] == [5, 7, 8]


def test_wrong_marker_length_is_flagged_by_depth():
    cfg = WorldTagsConfig(marker_length_m=0.07, min_views=5)  # printed 8 cm, configured 7 cm
    _, meta = calibrate(cfg, [5, 6, 7, 8], [5, 6, 7, 8], frames=6).solve("B")
    assert any("marker_length_m" in w for w in meta["warnings"]), meta
    assert meta["depth_scale"]["A"] == pytest.approx(0.08 / 0.07, rel=0.03)


def test_a_camera_whose_depth_reads_short_is_rescaled_not_dropped():
    """One RealSense reading 5% short shrinks its tag layout; plain depth then drops the outer
    tags (what the real rig showed: 0.94 depth/PnP on CAM 2, tags 5 and 7 dropped)."""
    cfg = WorldTagsConfig(marker_length_m=L, min_views=5)
    cal = TagRigCalibrator(cfg, "A", target=8, exclude_ids=CUBE)
    for f in range(8):
        img_b, d_b = render(T_TABLE_B, [5, 6, 7, 8], seed=2 * f + 1)
        cal.add([TagInput("A", K, D, *render(T_TABLE_A, [5, 6, 7, 8], seed=2 * f)),
                 TagInput("B", K, D, img_b, d_b * 0.95)])
    T, meta = cal.solve("B")
    assert meta["tags"] == [5, 6, 7, 8] and meta["rejected_tags"] == []
    assert meta["distance_from"]["B"] == "depth, rescaled"
    assert meta["depth_correction"] == pytest.approx(1 / 0.95, abs=0.015)
    assert np.linalg.norm(T[:3, 3] - T_WORLD_B[:3, 3]) < 0.01 and angle_deg(T[:3, :3], T_WORLD_B[:3, :3]) < 0.5
    assert any("depth calibration" in w for w in meta["warnings"])
    assert "reads" in TagRigCalibrator.summary("B", meta)


def test_kabsch_recovers_a_rigid_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    T = to_h(cv2.Rodrigues(np.array([0.3, -2.0, 0.5]))[0], np.array([0.4, -0.1, 1.2]))
    np.testing.assert_allclose(kabsch(src, src @ T[:3, :3].T + T[:3, 3]), T, atol=1e-9)


def test_default_config_parses_the_new_keys():
    cfg = load_config()
    assert cfg.testing.probe in (0, 1)
    assert cfg.world_tags.ids is None or all(i not in CUBE for i in cfg.world_tags.ids)


def test_depth_correction_is_applied_and_composes_on_recalibration(tmp_path):
    """The rig applies a camera's depth correction to its frames; a later tag calibration sees
    corrected depth, so what it measures multiplies onto what's already applied."""
    from types import SimpleNamespace

    from vigil.rig import Rig, RigCamera

    rig = Rig.__new__(Rig)
    rig.extrinsics_path = tmp_path / "extrinsics.yaml"
    rig.cameras = [RigCamera(0, SimpleNamespace(serial="A", depth_correction=1.0), np.eye(4)),
                   RigCamera(1, SimpleNamespace(serial="B", depth_correction=1.0), None)]
    rig._ref, rig._ref_T, rig._meta = None, {}, {}
    rig.set_extrinsic("B", np.eye(4), {"source": "tags", "depth_correction": 1.05})
    assert rig.cameras[1].camera.depth_correction == pytest.approx(1.05)
    rig.set_extrinsic("B", np.eye(4), {"source": "tags", "depth_correction": 1.01})  # residual
    assert rig.cameras[1].camera.depth_correction == pytest.approx(1.0605)
    rig.set_extrinsic("B", np.eye(4), {"source": "probe"})  # a probe recalibration keeps it
    assert rig.cameras[1].camera.depth_correction == pytest.approx(1.0605)
    assert rig.cameras[0].camera.depth_correction == 1.0

    reloaded = Rig.__new__(Rig)
    reloaded.extrinsics_path = rig.extrinsics_path
    reloaded.cameras = [RigCamera(0, SimpleNamespace(serial="A"), np.eye(4)),
                        RigCamera(1, SimpleNamespace(serial="B"), None)]
    reloaded._ref, reloaded._ref_T, reloaded._meta = None, {}, {}
    reloaded._load()
    assert reloaded.cameras[1].camera.depth_correction == pytest.approx(1.0605)
