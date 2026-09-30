"""Body pose on its own thread: track the subject per camera (person detector boxes), batch
the pose model across cameras, fuse in the world frame, smooth. The procedure picks the
target region (leg or chest): what is highlighted, measured and anchored across views.
Runs as fast as the world camera delivers frames.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from .camera import Frame
from .config import Config
from .detect_worker import DetectorProcess
from .estimators import PoseEstimator, PoseResult
from .filters import OneEuroFilter
from .rig import Rig, RigCamera, transform
from .skeleton import JOINT_INDEX, JOINTS, LEG_JOINTS, REGIONS, Region, fill_derived, leg_angles

log = logging.getLogger(__name__)

DETECTION_MAX_AGE_S = 0.5


@dataclass
class BodyState:
    seq: int = 0
    person: dict | None = None  # JSON-ready (world frame)
    joints: np.ndarray | None = None  # (J, 4) smoothed world joints
    vertices: np.ndarray | None = None  # (V, 3) smoothed world mesh
    views: dict[int, dict] = field(default_factory=dict)  # cam index → bbox / kp2d / score
    people: int = 0
    fps: float = 0.0
    pose_ms: float = 0.0
    detection_age_ms: float | None = None
    timestamp: float = 0.0  # capture time of the frames the joints came from
    velocity: np.ndarray | None = None  # (J, 3) m/s, from the previous update
    triangulated: int = 0  # joints placed by multi-view triangulation this update


class BodyWorker:
    def __init__(self, cfg: Config, rig: Rig, estimator: PoseEstimator, detector: DetectorProcess,
                 region: Region = REGIONS["lower_limb"]):
        self.cfg, self.rig, self.estimator, self.detector = cfg, rig, estimator, detector
        self.region = region
        self.state = BodyState()
        self.ready = threading.Event()
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._main, name="body", daemon=True)
        n = len(rig.cameras)
        self._track_box: list[np.ndarray | None] = [None] * n  # last box used per camera
        self._kp_box: list[np.ndarray | None] = [None] * n  # box around last keypoints
        sm = cfg.smoothing
        self._joint_filter = OneEuroFilter(sm.min_cutoff, sm.beta)
        self._vert_filter = OneEuroFilter(sm.min_cutoff, sm.beta)
        self._prev_anchor: np.ndarray | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def latest(self) -> BodyState:
        with self._lock:
            return self.state

    def _cams(self) -> list[RigCamera]:
        return [rc for rc in self.rig.cameras if rc.calibrated
                and (self.cfg.estimator.cameras == "all" or rc.index == 0)]

    def _main(self) -> None:
        try:
            # Compiling with CUDA graphs must happen on the thread that runs the model.
            self.estimator.warmup(batch=len(self._cams()))
            self.ready.set()
            self._loop()
        except Exception as e:
            log.error("body worker failed:\n%s", traceback.format_exc())
            self.error = f"{type(e).__name__}: {e}"
            self.ready.set()

    def _loop(self) -> None:
        frames: dict[int, Frame] = {}
        fps, last_t, seq = 0.0, time.perf_counter(), 0
        while not self._stop.is_set():
            cams = self._cams()
            world = self.rig.world
            prev = frames.get(0)
            f0 = world.camera.latest(prev.index if prev else -1, timeout=0.1)
            if f0 is None:
                continue
            frames[0] = f0
            for rc in cams:
                if rc.index != 0:
                    # Pair by capture time, so every view shows the same instant.
                    f = rc.camera.paired(f0.timestamp)
                    if f is not None:
                        frames[rc.index] = f
            t0 = time.perf_counter()

            batch_cams, batch_boxes, people, det_age = [], [], 0, None
            for rc in cams:
                frame = frames.get(rc.index)
                if frame is None:
                    continue
                dets = self.detector.latest(rc.index)
                if dets is not None:
                    age = frame.timestamp - dets.timestamp
                    det_age = age if det_age is None else max(det_age, age)
                    people = max(people, len(dets.boxes))
                    if age > DETECTION_MAX_AGE_S:
                        dets = None
                box = self._track(rc.index, dets, frame)
                if box is not None:
                    batch_cams.append(rc)
                    batch_boxes.append(box)

            results = self.estimator.estimate_batch(
                [frames[rc.index] for rc in batch_cams], batch_boxes,
                [rc.camera.intrinsics for rc in batch_cams])
            pose_ms = (time.perf_counter() - t0) * 1000

            views: dict[int, dict] = {}
            world_results: list[tuple[RigCamera, PoseResult]] = []
            for rc, box, result in zip(batch_cams, batch_boxes, results):
                if result is None:
                    self._track_box[rc.index] = self._kp_box[rc.index] = None
                    continue
                self._track_box[rc.index] = box
                self._kp_box[rc.index] = _keypoint_box(result.kp2d)
                h, w = frames[rc.index].color.shape[:2]
                views[rc.index] = {
                    "bbox": [round(float(v), 4) for v in box / [w, h, w, h]],
                    "kp2d": {JOINTS[i]: [round(float(u / w), 4), round(float(v / h), 4),
                                         round(float(c), 2)]
                             for i, (u, v, c) in enumerate(result.kp2d) if c > 0},
                }
                world_results.append((rc, to_world(result, rc.T_world_camera)))

            state = self._fuse_and_smooth(world_results, f0.timestamp)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / (now - last_t) if fps else 1.0 / (now - last_t)
            last_t = now
            seq += 1
            state.seq, state.views, state.people, state.fps, state.pose_ms = seq, views, people, fps, pose_ms
            state.detection_age_ms = None if det_age is None else det_age * 1000
            with self._lock:
                self.state = state

    def _track(self, i: int, dets, frame: Frame) -> np.ndarray | None:
        """Box to run the pose model on for camera i, without waiting for the detector."""
        boxes = dets.boxes if dets is not None else np.zeros((0, 4))
        prev = self._track_box[i]
        if prev is not None:
            if len(boxes):
                ious = np.array([_iou(b, prev) for b in boxes])
                if ious.max() > 0.3:
                    return boxes[int(ious.argmax())]
            # Detector hasn't caught up (or missed): follow our own keypoints.
            return self._kp_box[i]
        return self._select(boxes, frame)

    def _select(self, boxes: np.ndarray, frame: Frame) -> np.ndarray | None:
        if len(boxes) == 0:
            return None
        dist = np.array([_box_depth(frame.depth, b) for b in boxes])
        eligible = dist <= self.cfg.detector.max_distance_m
        if not eligible.any():
            return None
        idx = np.flatnonzero(eligible)
        mode = self.cfg.detector.select
        if mode == "confident":
            return boxes[idx[0]]  # sorted by score
        if mode == "largest":
            b = boxes[idx]
            return b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
        return boxes[idx[np.argmin(dist[idx])]]

    def _fuse_and_smooth(self, results, timestamp: float) -> BodyState:
        r = self.region
        kept = agreeing_views(results, self._prev_anchor, r.anchor)
        fused = fuse_people(kept, None, r.anchor, r.joints)
        n_tri = 0
        if fused is not None and self.cfg.estimator.triangulate and len(kept) >= 2:
            fused, n_tri = triangulate_joints(fused, kept, self.cfg.estimator.triangulation_min_conf,
                                              self.cfg.estimator.triangulation_max_px)
        if fused is None:
            self._joint_filter.reset()
            self._vert_filter.reset()
            self._prev_anchor = None
            return BodyState()
        joints = fused.joints.copy()
        verts = fused.vertices
        if self.cfg.smoothing.enabled:
            joints[:, :3] = self._joint_filter(joints[:, :3], timestamp)
            if verts is not None:
                verts = self._vert_filter(verts, timestamp)
        anchor = joints[JOINT_INDEX[r.anchor]]
        self._prev_anchor = anchor[:3].copy() if anchor[3] > 0 else None
        angles = leg_angles(joints) if r.name == "leg" else {}
        person = {
            "region": r.name,
            "cameras": [rc.index for rc, _ in kept],
            "joints": joints_json(joints),
            "angles": {k: (None if v is None else round(v, 1)) for k, v in angles.items()},
            "has_mesh": verts is not None,
            "triangulated": n_tri,
        }
        prev = self.state
        velocity = None
        if prev.joints is not None and 0 < timestamp - prev.timestamp < 0.2:
            velocity = np.nan_to_num((joints[:, :3] - prev.joints[:, :3]) / (timestamp - prev.timestamp))
        return BodyState(person=person, joints=joints, vertices=verts, timestamp=timestamp,
                         velocity=velocity, triangulated=n_tri)

    def joints_at(self, t: float) -> np.ndarray | None:
        """Joints predicted at capture time `t` (e.g. a probe frame's), so body and probe are
        compared at the same instant. Linear extrapolation, at most 100 ms."""
        st = self.latest()
        if st.joints is None:
            return None
        if st.velocity is None:
            return st.joints
        out = st.joints.copy()
        out[:, :3] += st.velocity * float(np.clip(t - st.timestamp, -0.1, 0.1))
        return out


def _keypoint_box(kp2d: np.ndarray, min_conf: float = 0.3) -> np.ndarray | None:
    pts = kp2d[kp2d[:, 2] >= min_conf, :2]
    if len(pts) < 4:
        return None
    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
    mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    return np.array([x1 - mx, y1 - my, x2 + mx, y2 + my], np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _box_depth(depth: np.ndarray, box: np.ndarray) -> float:
    """Median valid depth over the central part of a person box (inf if none)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    region = depth[int(y1 + 0.2 * h): int(y2 - 0.2 * h): 2, int(x1 + 0.3 * w): int(x2 - 0.3 * w): 2]
    valid = region[region > 0]
    return float(np.median(valid)) if valid.size > 20 else float("inf")


def to_world(result: PoseResult, T: np.ndarray) -> PoseResult:
    joints = result.joints.copy()
    joints[:, :3] = transform(T, joints[:, :3])
    verts = None if result.vertices is None else transform(T, result.vertices).astype(np.float32)
    return PoseResult(joints=joints, kp2d=result.kp2d, vertices=verts)


def agreeing_views(results: list[tuple[RigCamera, PoseResult]], prev_anchor: np.ndarray | None,
                   anchor_joint: str = "pelvis") -> list[tuple[RigCamera, PoseResult]]:
    """The views that agree on where the target is (anchor joint ≤ 0.5 m apart)."""
    if len(results) <= 1:
        return results
    pel = JOINT_INDEX[anchor_joint]

    def pelvis(r: PoseResult):
        return r.joints[pel, :3] if r.joints[pel, 3] > 0 else None

    anchor = prev_anchor
    if anchor is None:
        anchor = next((pelvis(r) for _, r in results if pelvis(r) is not None), None)
    if anchor is None:
        return results
    agree = [(rc, r) for rc, r in results
             if pelvis(r) is None or np.linalg.norm(pelvis(r) - anchor) < 0.5]
    return agree or [min(results, key=lambda x: np.linalg.norm(
        (pelvis(x[1]) if pelvis(x[1]) is not None else np.full(3, 1e3)) - anchor))]


def fuse_people(results: list[tuple[RigCamera, PoseResult]], prev_anchor: np.ndarray | None,
                anchor_joint: str = "pelvis", focus: list[str] = LEG_JOINTS) -> PoseResult | None:
    """Merge one body seen by several cameras (all in the world frame)."""
    results = agreeing_views(results, prev_anchor, anchor_joint)
    if not results:
        return None
    if len(results) == 1:
        return results[0][1]

    stack = np.stack([r.joints for _, r in results])  # (C, J, 4)
    conf = np.where(np.isfinite(stack[..., 0]), stack[..., 3], 0.0)
    wsum = conf.sum(0)
    xyz = (np.nan_to_num(stack[..., :3]) * conf[..., None]).sum(0) / np.maximum(wsum, 1e-6)[:, None]
    joints = np.concatenate([xyz, conf.max(0)[:, None]], axis=1).astype(np.float32)
    joints[wsum <= 0, :3] = np.nan
    # Mesh from the camera that sees the target most confidently.
    idx = [JOINT_INDEX[j] for j in focus]
    best = max(results, key=lambda x: float(x[1].joints[idx, 3].sum()))[1]
    return PoseResult(joints=joints, kp2d=best.kp2d, vertices=best.vertices)


def triangulate_joints(fused: PoseResult, views: list[tuple[RigCamera, PoseResult]],
                       min_conf: float = 0.4, max_px: float = 15.0,
                       max_depth_disagreement_m: float = 0.3) -> tuple[PoseResult, int]:
    """Replace depth-lifted joints with multi-view triangulation where 2+ cameras see them.

    Depth lifting samples the depth map at the keypoint, which fails when the joint is
    occluded (depth hits whatever is in front) or at silhouette edges; rays from calibrated
    cameras don't. Each triangulated joint must reproject within `max_px` in every view used.
    """
    import cv2

    Ps, uvs, confs = [], [], []
    for rc, r in views:
        K = rc.camera.intrinsics
        Km = K.matrix().astype(np.float64)
        T_cw = np.linalg.inv(rc.T_world_camera)
        Ps.append((Km, K.dist_coeffs(), T_cw))
        uvs.append(r.kp2d[:, :2].astype(np.float64))
        confs.append(r.kp2d[:, 2])
    joints = fused.joints.copy()
    n_done = 0
    for j in range(len(joints)):
        cand = [i for i in range(len(views)) if confs[i][j] >= min_conf]
        use = _consensus(j, cand, Ps, uvs, confs, max_px)
        if len(use) < 2:
            continue
        X, errs = _dlt(j, use, Ps, uvs, confs)
        if X is None or max(errs) > max_px:
            continue
        # Two views can't expose a wrong correspondence (e.g. swapped knees still give rays
        # that nearly meet — at the wrong place). Require agreement with at least one view's
        # depth-based estimate; an occluded joint still passes via the unoccluded camera.
        lifted = [r.joints[j, :3] for _, r in views if r.joints[j, 3] > 0 and np.isfinite(r.joints[j, 0])]
        if lifted and min(np.linalg.norm(X - q) for q in lifted) > max_depth_disagreement_m:
            continue
        joints[j] = (*X, float(np.mean([confs[i][j] for i in use])))
        n_done += 1
    fill_derived(joints)
    return PoseResult(joints=joints, kp2d=fused.kp2d, vertices=fused.vertices), n_done


def _consensus(j, cand, Ps, uvs, confs, max_px):
    """Largest set of views that agree on joint j. A single bad view (e.g. left/right
    swapped) drags a least-squares point towards itself and makes *good* views look worst,
    so hypothesise from every pair and keep the pair whose point most views agree with."""
    if len(cand) <= 2:
        return cand
    best = []
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            X, _ = _dlt(j, [cand[a], cand[b]], Ps, uvs, confs)
            if X is None:
                continue
            inliers = [i for i in cand if _reproj(X, i, j, Ps, uvs) <= max_px]
            if len(inliers) > len(best):
                best = inliers
    return best


def _reproj(X, i, j, Ps, uvs):
    import cv2

    Km, dist, T_cw = Ps[i]
    Xc = T_cw[:3, :3] @ X + T_cw[:3, 3]
    if Xc[2] <= 0.05:
        return np.inf
    uv, _ = cv2.projectPoints(Xc.reshape(1, 3), np.zeros(3), np.zeros(3), Km, dist)
    return float(np.linalg.norm(uv.reshape(2) - uvs[i][j]))


def _dlt(j, use, Ps, uvs, confs):
    import cv2

    A = []
    for i in use:
        Km, dist, T_cw = Ps[i]
        # Undistort to normalised coordinates, then project with [R|t] only.
        xn = cv2.undistortPoints(uvs[i][j].reshape(1, 1, 2), Km, dist).reshape(2)
        P = T_cw[:3]
        w = float(confs[i][j])
        A.append(w * (xn[0] * P[2] - P[0]))
        A.append(w * (xn[1] * P[2] - P[1]))
    _, _, vt = np.linalg.svd(np.asarray(A))
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-9:
        return None, []
    X = Xh[:3] / Xh[3]
    errs = []
    for i in use:
        Km, dist, T_cw = Ps[i]
        Xc = T_cw[:3, :3] @ X + T_cw[:3, 3]
        if Xc[2] <= 0.05:
            return None, []
        uv, _ = cv2.projectPoints(Xc.reshape(1, 3), np.zeros(3), np.zeros(3), Km, dist)
        errs.append(float(np.linalg.norm(uv.reshape(2) - uvs[i][j])))
    return X, errs


def joints_json(joints: np.ndarray) -> dict[str, list[float]]:
    return {
        JOINTS[i]: [round(float(x), 4), round(float(y), 4), round(float(z), 4), round(float(c), 3)]
        for i, (x, y, z, c) in enumerate(joints)
        if c > 0 and np.isfinite(x)
    }
