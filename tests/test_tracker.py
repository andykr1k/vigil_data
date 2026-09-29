"""End to end on rendered images: real ArUco detection → cleaning → joint solve → world pose,
for a colour camera and one whose tags are seen by a separate (IR-like) sensor."""

import cv2
import numpy as np

from vigil.config import Config
from vigil.probe.aruco import generate_tag_image
from vigil.probe.tracker import ProbeInput, ProbeTracker
from vigil.rig import to_h

K = np.array([[610.0, 0, 424], [0, 610.0, 240], [0, 0, 1]])
D = np.zeros(5)


def rot(v):
    return cv2.Rodrigues(np.asarray(v, float))[0]


def render(tracker, T_img_obj, size=(848, 480)):
    """Gray image of the cube's visible tags (with their white margins) seen from a camera."""
    g = tracker.geometry
    img = np.full((size[1], size[0]), 90, np.uint8)
    faces = []
    for i, m in g.mounts.items():
        R = T_img_obj[:3, :3]
        centre = R @ np.asarray(m.center) + T_img_obj[:3, 3]
        normal = R @ m.rotation_object_from_marker[:, 2]
        if normal @ centre < -1e-3:  # facing the camera
            faces.append((centre[2], i, m))
    for _, i, m in sorted(faces, reverse=True):  # far faces first
        tag = generate_tag_image("DICT_6X6_50", i)
        s = g.marker_length * 700 / 600 / 2  # tag incl. white margin, half size
        corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        P = (m.rotation_object_from_marker @ corners.T).T + np.asarray(m.center)
        Pc = (T_img_obj[:3, :3] @ P.T).T + T_img_obj[:3, 3]
        uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, D)
        h, w = tag.shape
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), uv.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(tag, H, size, flags=cv2.INTER_AREA, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(tag, 255), H, size, flags=cv2.INTER_NEAREST)
        img[mask > 0] = warped[mask > 0]
    return img


def err(pose, T):
    return (np.linalg.norm(pose.position - T[:3, 3]) * 1000,
            np.degrees(np.arccos(np.clip((np.trace(pose.rotation_matrix.T @ T[:3, :3]) - 1) / 2, -1, 1))))


def test_rendered_two_camera_tracking_with_an_ir_sensor():
    tracker = ProbeTracker(Config())
    T_world_obj = to_h(rot([0.4, 0.5, 0.1]), np.array([0.05, 0.0, 0.7]))
    # Camera 1 = world, tags seen in colour. Camera 2 sees the tags through its IR sensor,
    # which sits 15 mm to the side of its colour camera (like a D435's left IR imager).
    T_w_c2 = to_h(rot([0.05, -0.8, 0.02]), np.array([0.6, 0.02, 0.3]))
    T_c2_ir = to_h(np.eye(3), np.array([-0.015, 0.0, 0.0]))  # IR sensor → colour camera
    img1 = cv2.cvtColor(render(tracker, T_world_obj), cv2.COLOR_GRAY2RGB)
    img2 = render(tracker, np.linalg.inv(T_w_c2 @ T_c2_ir) @ T_world_obj)
    inputs = [ProbeInput("cam1", np.eye(4), K, D, img1, None),
              ProbeInput("cam2", T_w_c2, K, D, img2, None, T_c2_ir, "infrared")]
    frame = tracker.track(inputs, timestamp=1.0)

    assert set(frame.world_poses) == {"cam1", "cam2"}
    cams = {"cam1": np.eye(4), "cam2": T_w_c2 @ T_c2_ir}
    for serial, pose in frame.world_poses.items():
        # Each camera alone: precise across its view, weak along its viewing ray.
        d = (pose.position - T_world_obj[:3, 3]) * 1000
        ray = T_world_obj[:3, 3] - cams[serial][:3, 3]
        ray /= np.linalg.norm(ray)
        across = np.linalg.norm(d - (d @ ray) * ray)
        # < 15°: a small single tag is noisy in rotation, but not flipped (a flip is ~35°+).
        assert across < 6 and abs(d @ ray) < 25 and err(pose, T_world_obj)[1] < 15, (serial, d)
    # Per-camera poses are reported in the colour camera frame (what rig calibration needs):
    # exactly the world pose seen through that camera's extrinsic, IR offset included.
    c2 = frame.cam_poses["cam2"]
    T_c2 = to_h(c2.rotation_matrix, c2.position)
    w2 = frame.world_poses["cam2"]
    np.testing.assert_allclose(T_w_c2 @ T_c2, to_h(w2.rotation_matrix, w2.position), atol=1e-9)
    # The joint solve over both cameras' corners.
    assert frame.refined is not None and frame.refined.rms_px < 1.0
    dt, da = err(frame.measured, T_world_obj)
    assert dt < 3 and da < 1.5
    tip_true = T_world_obj[:3, :3] @ tracker.geometry.tip_array + T_world_obj[:3, 3]
    assert np.linalg.norm(tracker.tip_world(frame.measured) - tip_true) * 1000 < 6
