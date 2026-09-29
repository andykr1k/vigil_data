"""Probe tracking across the camera rig.

Per frame: detect tags in every camera → clean each view (outlier tags, single-tag flips
resolved against the previous pose) → fuse the calibrated views into an initial world
pose → refine jointly over every corner in every camera plus measured depth → filter.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Config
from .aruco import DetectedMarkerPose, detect_marker_poses, make_detector
from .filtering.pose import FILTER_METHODS, ObjectPoseFilter
from .filtering.presets import load_filter_preset_file
from .geometry import ProbeGeometry
from .object_pose import ObjectPose, mean_rotation
from .pivot import load_tip
from .solver import CameraView, CleanView, Refined, clean_view, refine_multiview

log = logging.getLogger(__name__)


@dataclass
class ProbeInput:
    """One camera's contribution for a frame."""

    serial: str
    T_world_cam: np.ndarray | None  # colour camera → world; None for an uncalibrated camera
    K: np.ndarray  # intrinsics of the image the tags are detected in
    dist: np.ndarray
    image: np.ndarray  # RGB colour image, or gray projector-off IR image
    depth: np.ndarray | None  # aligned to `image` (None for IR: not aligned to it)
    T_cam_img: np.ndarray | None = None  # image sensor → colour camera frame (IR: T_color_ir)
    source: str = "color"


@dataclass
class ProbeFrame:
    estimate: ObjectPose | None  # filtered, world frame
    measured: ObjectPose | None  # unfiltered joint solve, world frame
    views: dict[str, CleanView] = field(default_factory=dict)  # per camera, image-sensor frame
    cam_poses: dict[str, ObjectPose] = field(default_factory=dict)  # per camera, colour frame
    world_poses: dict[str, ObjectPose] = field(default_factory=dict)  # per calibrated camera
    sources: dict[str, str] = field(default_factory=dict)  # "color" / "infrared" per camera
    refined: Refined | None = None


def _to_h(p: ObjectPose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = p.rotation_matrix, p.position
    return T


def _from_h(T: np.ndarray, ids: tuple[int, ...]) -> ObjectPose:
    return ObjectPose(ids, T[:3, 3].copy(), T[:3, :3].copy())


class ProbeTracker:
    def __init__(self, cfg: Config):
        pc = cfg.probe
        self.cfg = pc
        self.cad_tip = np.asarray(pc.tip_in_object_m, dtype=np.float64)
        self.tip_path = cfg.resolve(pc.tip_calibration_path)
        tip = load_tip(self.tip_path)
        self.tip_source = "pivot calibration" if tip is not None else "CAD"
        self.geometry = ProbeGeometry.clarius(pc.marker_length_m, pc.cube_size_m,
                                              tuple(tip if tip is not None else self.cad_tip))
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aruco")
        self._frame = 0
        self.filter = ObjectPoseFilter()
        self._last: ObjectPose | None = None
        preset = cfg.resolve(pc.filter_preset)
        if preset.is_file():
            try:
                method, tuning = load_filter_preset_file(preset)
                self.filter.set_method(method)
                self.filter.set_tuning(tuning)
            except (OSError, ValueError) as e:
                log.warning("ignoring probe filter preset %s: %s", preset, e)
        if pc.filter_method:
            self.filter.set_method(pc.filter_method)

    def _detect_input(self, c: "ProbeInput") -> list[DetectedMarkerPose]:
        roi = self._roi(c)
        if roi is not None:
            x0, y0, x1, y1 = roi
            found = self.detect(c.image[y0:y1, x0:x1], c.K, c.dist, offset=(x0, y0))
            if found:
                return found
        return self.detect(c.image, c.K, c.dist)

    def _roi(self, c: "ProbeInput") -> tuple[int, int, int, int] | None:
        """Image window around the last known cube position (None → search the full frame)."""
        if self._last is None or c.T_world_cam is None or self._frame % 15 == 0:
            return None  # also a periodic full-frame pass, in case the ROI locked onto a stale spot
        T_wi = c.T_world_cam @ (np.eye(4) if c.T_cam_img is None else c.T_cam_img)
        T_io = np.linalg.inv(T_wi) @ _to_h(self._last)
        # The cube's 8 corners (±half size) projected into this image, padded.
        h = np.asarray(self.geometry.cube_size) / 2
        box = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        Pc = box @ T_io[:3, :3].T + T_io[:3, 3]
        if (Pc[:, 2] <= 0.05).any():
            return None
        uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), c.K, c.dist)
        uv = uv.reshape(-1, 2)
        (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
        pad = max(40.0, 0.75 * max(u1 - u0, v1 - v0))  # room for motion between frames
        H, W = c.image.shape[:2]
        x0, y0 = int(max(0, u0 - pad)), int(max(0, v0 - pad))
        x1, y1 = int(min(W, u1 + pad)), int(min(H, v1 + pad))
        if x1 - x0 < 32 or y1 - y0 < 32 or (x1 - x0) * (y1 - y0) > 0.6 * W * H:
            return None
        return x0, y0, x1, y1

    def set_tip(self, tip: np.ndarray, source: str) -> None:
        self.geometry = dataclasses.replace(self.geometry, tip=tuple(float(v) for v in tip))
        self.tip_source = source

    def detect(self, image: np.ndarray, K: np.ndarray, dist: np.ndarray,
               offset: tuple[int, int] = (0, 0)) -> list[DetectedMarkerPose]:
        """Tags in `image`, which may be a crop whose top-left sits at `offset` in the frame."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if offset != (0, 0):
            K = K.copy()
            K[0, 2] -= offset[0]  # intrinsics of the crop
            K[1, 2] -= offset[1]
        detector = getattr(self._local, "detector", None)
        if detector is None:  # one detector per thread: OpenCV doesn't promise thread safety
            detector = self._local.detector = make_detector(self.cfg.dictionary)
        found = detect_marker_poses(gray, detector, self.geometry, K, dist)
        if offset == (0, 0):
            return found
        shift = np.array(offset, dtype=np.float64)
        return [dataclasses.replace(m, image_corners=m.image_corners + shift) for m in found]

    def track(self, inputs: list[ProbeInput], timestamp: float) -> ProbeFrame:
        out = ProbeFrame(None, None)
        # ArUco detection dominates the cost: cameras run in parallel (OpenCV releases the
        # GIL), and each searches only around where the cube was last seen when it can.
        detections = dict(zip([c.serial for c in inputs],
                              self._pool.map(self._detect_input, inputs)))
        self._frame += 1

        def T_wi(c):  # image sensor → world (None if the camera isn't calibrated)
            if c.T_world_cam is None:
                return None
            return c.T_world_cam @ (np.eye(4) if c.T_cam_img is None else c.T_cam_img)

        def solve(c, reference):
            T = T_wi(c)
            prev = (None if reference is None or T is None
                    else _from_h(np.linalg.inv(T) @ _to_h(reference), reference.marker_ids))
            return clean_view(detections[c.serial], self.geometry, c.K, c.dist,
                              prev_pose=prev, max_err_px=self.cfg.max_tag_error_px)

        views = {c.serial: solve(c, self._last) for c in inputs}
        if self._last is None:
            # No history yet: a single tag's two mirror-image poses can't be told apart from
            # one image, but any camera that sees 2+ tags can — use it to resolve the others.
            anchors = [_from_h(T_wi(c) @ _to_h(views[c.serial].pose), views[c.serial].pose.marker_ids)
                       for c in inputs
                       if T_wi(c) is not None and views[c.serial].pose is not None
                       and len(views[c.serial].markers) >= 2]
            reference = fuse_poses(anchors)
            if reference is not None:
                for c in inputs:
                    if len(views[c.serial].markers) == 1:
                        views[c.serial] = solve(c, reference)

        for c in inputs:
            view = views[c.serial]
            out.views[c.serial] = view
            out.sources[c.serial] = c.source
            if view.pose is None:
                continue
            ids = view.pose.marker_ids
            T_ci = np.eye(4) if c.T_cam_img is None else c.T_cam_img
            out.cam_poses[c.serial] = _from_h(T_ci @ _to_h(view.pose), ids)
            if T_wi(c) is not None:
                out.world_poses[c.serial] = _from_h(T_wi(c) @ _to_h(view.pose), ids)

        init = fuse_poses(list(out.world_poses.values()))
        measured = init
        if init is not None and self.cfg.joint_solve:
            views = [CameraView(c.T_world_cam @ (np.eye(4) if c.T_cam_img is None else c.T_cam_img),
                                c.K, c.dist, out.views[c.serial].markers,
                                c.depth if self.cfg.use_depth else None)
                     for c in inputs
                     if c.T_world_cam is not None and c.serial in out.world_poses]
            out.refined = refine_multiview(views, self.geometry, init, use_depth=self.cfg.use_depth)
            if out.refined is not None:
                measured = out.refined.pose
        out.measured = measured
        out.estimate = self.filter.update(measured, timestamp)
        self._last = out.estimate or measured
        return out

    def set_method(self, method: str) -> None:
        if method not in FILTER_METHODS:
            raise ValueError(f"unknown probe filter {method!r}")
        self.filter.set_method(method)

    def tip_world(self, pose: ObjectPose) -> np.ndarray:
        return pose.rotation_matrix @ self.geometry.tip_array + pose.position


def fuse_poses(poses: list[ObjectPose]) -> ObjectPose | None:
    """Initial guess from several cameras: weight by how many tags each saw."""
    if not poses:
        return None
    if len(poses) == 1:
        return poses[0]
    best = max(poses, key=lambda p: len(p.marker_ids))
    # A camera that disagrees badly (e.g. a single-tag pose flip) is dropped, not averaged.
    poses = [p for p in poses
             if np.linalg.norm(p.position - best.position) < 0.05
             and _angle_deg(p.rotation_matrix, best.rotation_matrix) < 20]
    w = np.array([len(p.marker_ids) for p in poses], dtype=np.float64)
    position = (w[:, None] * np.array([p.position for p in poses])).sum(0) / w.sum()
    rotation = mean_rotation([p.rotation_matrix for p in poses], list(w))
    ids = tuple(sorted({i for p in poses for i in p.marker_ids}))
    return ObjectPose(ids, position, rotation)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))
