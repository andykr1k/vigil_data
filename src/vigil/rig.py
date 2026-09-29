"""Multiple RealSense cameras sharing one world frame (the first camera's optical frame)."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .camera import connected_serials
from .capture import CameraClient
from .config import CamerasConfig
from .probe.object_pose import ObjectPose, mean_rotation

log = logging.getLogger(__name__)


def to_h(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


@dataclass
class RigCamera:
    index: int
    camera: CameraClient
    T_world_camera: np.ndarray | None  # None until calibrated (except the world camera)

    @property
    def serial(self) -> str:
        return self.camera.serial

    @property
    def calibrated(self) -> bool:
        return self.T_world_camera is not None

    def describe(self) -> dict:
        K = self.camera.intrinsics
        return {
            "index": self.index,
            "serial": self.serial,
            "name": self.camera.name,
            "usb": self.camera.usb,
            "width": K.width, "height": K.height,
            "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy,
            "fps": self.camera.mode[2],
            "state": self.camera.state,
            "T_world_camera": None if self.T_world_camera is None
            else [float(v) for v in self.T_world_camera.reshape(-1)],
        }


class Rig:
    def __init__(self, cfg: CamerasConfig, extrinsics_path: Path, infrared: bool = False):
        self.cfg = cfg
        self.extrinsics_path = extrinsics_path
        self.warnings: list[str] = []
        serials = cfg.serials or connected_serials()
        if not serials:
            raise RuntimeError("No RealSense connected")
        self.cameras: list[RigCamera] = []
        for serial in serials:
            role = (1 if not self.cameras else 2) if cfg.hardware_sync else 0
            try:
                cam = CameraClient(cfg, serial, sync_role=role, infrared=infrared)
            except RuntimeError as e:
                self.warnings.append(str(e))
                log.warning("%s", e)
                continue
            if cam.usb.startswith("2"):
                self.warnings.append(f"{serial} is on USB {cam.usb}: limited to "
                                     f"{cam.mode[0]}x{cam.mode[1]}@{cam.mode[2]}")
            self.cameras.append(RigCamera(len(self.cameras), cam, None))
        if not self.cameras:
            raise RuntimeError("; ".join(self.warnings))
        self.cameras[0].T_world_camera = np.eye(4)

        # Extrinsics are stored relative to a reference serial; re-express them in our world.
        self._ref: str | None = None
        self._ref_T: dict[str, np.ndarray] = {}
        self._meta: dict[str, dict] = {}
        self._load()

    @property
    def world(self) -> RigCamera:
        return self.cameras[0]

    def _load(self) -> None:
        if not self.extrinsics_path.is_file():
            return
        data = yaml.safe_load(self.extrinsics_path.read_text()) or {}
        self._ref = data.get("reference")
        for serial, entry in (data.get("cameras") or {}).items():
            self._ref_T[str(serial)] = np.asarray(entry["T_reference_camera"], dtype=np.float64)
            self._meta[str(serial)] = {k: v for k, v in entry.items() if k != "T_reference_camera"}
        self._apply()

    def _apply(self) -> None:
        T_ref_world = self._ref_T.get(self.world.serial)
        for rc in self.cameras[1:]:
            T_ref_cam = self._ref_T.get(rc.serial)
            rc.T_world_camera = (None if T_ref_world is None or T_ref_cam is None
                                 else np.linalg.inv(T_ref_world) @ T_ref_cam)

    def set_extrinsic(self, serial: str, T_world_camera: np.ndarray, meta: dict) -> None:
        w = self.world.serial
        if self._ref == w:
            self._ref_T[serial] = T_world_camera
        elif self._ref is not None and w in self._ref_T:
            self._ref_T[serial] = self._ref_T[w] @ T_world_camera
        else:
            self._ref, self._ref_T, self._meta = w, {w: np.eye(4), serial: T_world_camera}, {}
        self._meta[serial] = {k: v.item() if isinstance(v, np.generic) else v for k, v in meta.items()}
        self._apply()
        self._save()

    def _save(self) -> None:
        doc = {
            "reference": self._ref,
            "cameras": {
                s: {"T_reference_camera": [[round(float(v), 6) for v in row] for row in T],
                    **self._meta.get(s, {})}
                for s, T in self._ref_T.items()
            },
        }
        self.extrinsics_path.parent.mkdir(parents=True, exist_ok=True)
        header = ("# Camera extrinsics, written by CALIBRATE RIG (probe cube seen by both cameras).\n"
                  "# T_reference_camera maps camera optical-frame points into the reference camera.\n")
        self.extrinsics_path.write_text(header + yaml.safe_dump(doc, sort_keys=False))

    def close(self) -> None:
        for rc in self.cameras:
            rc.camera.close()


@dataclass
class RigCalibrator:
    """Collects simultaneous cube poses from the world camera and each other camera.

    Every still frame gives T_world_cam = T_world_cube @ inv(T_cam_cube); the robust
    mean over many frames is the extrinsic.
    """

    target: int
    timeout_s: float = 60.0
    geometry: object | None = None  # ProbeGeometry; enables the joint (bundle) refinement
    started: float = field(default_factory=time.monotonic)
    samples: dict[str, list[np.ndarray]] = field(default_factory=dict)
    # Raw corner observations per sample, for the joint refinement: (world obs, camera obs)
    # where each obs is (K, dist, [(object corners (4,3), image corners (4,2)), ...]).
    observations: dict[str, list[tuple | None]] = field(default_factory=dict)
    _recent: deque = field(default_factory=lambda: deque(maxlen=5))

    def add(self, world_serial: str, poses: dict[str, ObjectPose],
            observations: dict[str, tuple] | None = None) -> None:
        wp = poses.get(world_serial)
        if wp is None:
            self._recent.clear()
            return
        self._recent.append(wp)
        # The cameras aren't hardware-synced: only use frames where the cube has been
        # still for a few frames. The window spread separates hand motion from PnP jitter.
        if len(self._recent) < self._recent.maxlen:
            return
        centre = np.mean([p.position for p in self._recent], axis=0)
        spread = max(np.linalg.norm(p.position - centre) for p in self._recent)
        turned = max(_angle_deg(self._recent[0].rotation_matrix, p.rotation_matrix)
                     for p in self._recent)
        if spread > 0.006 or turned > 2.0:
            return
        T_world_cube = to_h(wp.rotation_matrix, wp.position)
        for serial, p in poses.items():
            if serial == world_serial:
                continue
            T_cam_cube = to_h(p.rotation_matrix, p.position)
            self.samples.setdefault(serial, []).append(T_world_cube @ np.linalg.inv(T_cam_cube))
            obs = None
            if observations and world_serial in observations and serial in observations:
                obs = (observations[world_serial], observations[serial], T_world_cube)
            self.observations.setdefault(serial, []).append(obs)

    def progress(self) -> dict[str, int]:
        return {s: len(v) for s, v in self.samples.items()}

    def finished(self, serials: list[str]) -> bool:
        timed_out = time.monotonic() - self.started > self.timeout_s
        return timed_out or all(len(self.samples.get(s, [])) >= self.target for s in serials)

    def solve(self, serial: str) -> tuple[np.ndarray, dict] | None:
        Ts = self.samples.get(serial, [])
        if len(Ts) < 10:
            return None
        t = np.array([T[:3, 3] for T in Ts])
        R = [T[:3, :3] for T in Ts]
        # Reject flips / bad frames: far from the median translation or the mean rotation.
        med = np.median(t, axis=0)
        dist = np.linalg.norm(t - med, axis=1)
        mad = np.median(dist)
        R0 = mean_rotation(R)
        ang = np.array([_angle_deg(R0, r) for r in R])
        keep = (dist <= max(3 * mad, 0.01)) & (ang <= 5.0)
        if keep.sum() < 8:
            return None
        R_mean = mean_rotation([r for r, k in zip(R, keep) if k])
        t_mean = t[keep].mean(axis=0)
        n = int(keep.sum())
        spread_mm = float(np.sqrt(np.mean(np.sum((t[keep] - t_mean) ** 2, axis=1))) * 1000)
        rot_spread = float(np.sqrt(np.mean([_angle_deg(R_mean, r) ** 2
                                            for r, k in zip(R, keep) if k])))
        meta = {
            "samples": n,
            "rejected": int((~keep).sum()),
            # Per-frame scatter, and the standard error of the averaged extrinsic.
            "spread_mm": round(spread_mm, 2),
            "spread_deg": round(rot_spread, 3),
            "stderr_mm": round(float(spread_mm / np.sqrt(n)), 2),
            "stderr_deg": round(float(rot_spread / np.sqrt(n)), 3),
            "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        T = to_h(R_mean, t_mean)
        # Joint refinement: the extrinsic plus every sample's cube pose, fitted to the raw
        # tag corners seen by both cameras (instead of averaging per-frame estimates).
        obs = [o for o, k in zip(self.observations.get(serial, []), keep) if k and o is not None]
        if len(obs) >= 8:
            refined = _bundle_extrinsic(T, obs)
            if refined is not None and refined[2] <= refined[1] * 1.05:
                T = refined[0]
                meta["method"] = "bundle adjustment"
                meta["reprojection_px_before"] = round(refined[1], 3)
                meta["reprojection_px"] = round(refined[2], 3)
        return T, meta


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def _se3(x: np.ndarray) -> np.ndarray:
    import cv2

    R, _ = cv2.Rodrigues(x[:3])
    return to_h(R, x[3:6])


def _xi(T: np.ndarray) -> np.ndarray:
    import cv2

    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


def _bundle_extrinsic(T_init: np.ndarray, samples: list[tuple]):
    """Least squares over [extrinsic, cube pose per sample] against all observed corners.

    Returns (T_world_cam, rms_px_before, rms_px_after) or None.
    """
    import cv2
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    def project(T_cam_obj, obs):
        """Reprojection residuals (px) of every corner. (Depth isn't used here: with many
        frames averaged it doesn't beat multi-frame PnP along the ray, and it would import
        the depth sensor's scale bias into the extrinsic.)"""
        K, dist, tags = obs[:3]
        if len(obs) > 3 and obs[3] is not None:  # tags seen by another sensor (IR) of this camera
            T_cam_obj = obs[3] @ T_cam_obj
        P = np.concatenate([t[0] for t in tags])
        uv = np.concatenate([t[1] for t in tags])
        rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
        proj, _ = cv2.projectPoints(P, rvec, T_cam_obj[:3, 3], K, dist)
        return (proj.reshape(-1, 2) - uv).ravel()

    n = len(samples)
    x0 = np.concatenate([_xi(T_init)] + [_xi(s[2]) for s in samples])

    def residuals(x):
        T_wc = _se3(x[:6])
        T_cw = np.linalg.inv(T_wc)
        out = []
        for i, (obs_w, obs_c, _) in enumerate(samples):
            T_wo = _se3(x[6 + 6 * i: 12 + 6 * i])
            out.append(project(T_wo, obs_w))
            out.append(project(T_cw @ T_wo, obs_c))
        return np.concatenate(out)

    r0 = residuals(x0)
    # Sparsity: sample i's world residuals depend on its pose; its camera residuals on its
    # pose and the extrinsic.
    def n_res(obs):  # 2 residuals (u, v) per observed corner
        return 2 * sum(len(t[1]) for t in obs[2])

    sizes = [(n_res(s[0]), n_res(s[1])) for s in samples]
    J = lil_matrix((len(r0), len(x0)), dtype=int)
    row = 0
    for i, (nw, nc) in enumerate(sizes):
        J[row: row + nw + nc, 6 + 6 * i: 12 + 6 * i] = 1
        J[row + nw: row + nw + nc, :6] = 1
        row += nw + nc
    try:
        sol = least_squares(residuals, x0, jac_sparsity=J, loss="soft_l1", f_scale=2.0,
                            x_scale="jac", max_nfev=3000)
    except (ValueError, np.linalg.LinAlgError):
        return None
    def rms_px(x):
        return float(np.sqrt(np.mean(residuals(x).reshape(-1, 2) ** 2) * 2))

    return _se3(sol.x[:6]), rms_px(x0), rms_px(sol.x)


class RigHealth:
    """Does the calibration still hold? Whenever the world camera and another calibrated
    camera each see the probe on their own, their world-frame cube poses should agree.
    A rolling median of the disagreement exposes a bumped camera or a bad calibration."""

    def __init__(self, window: int = 90, max_age_s: float = 30.0):
        self.window, self.max_age_s = window, max_age_s
        self.history: dict[str, deque] = {}

    def add(self, world_serial: str, world_poses: dict[str, ObjectPose], now: float) -> None:
        ref = world_poses.get(world_serial)
        if ref is None:
            return
        for serial, p in world_poses.items():
            if serial == world_serial:
                continue
            h = self.history.setdefault(serial, deque(maxlen=self.window))
            h.append((now, float(np.linalg.norm(p.position - ref.position)) * 1000,
                      _angle_deg(p.rotation_matrix, ref.rotation_matrix)))

    def report(self, now: float) -> dict[str, dict]:
        out = {}
        for serial, h in self.history.items():
            recent = [x for x in h if now - x[0] <= self.max_age_s]
            if len(recent) < 10:
                continue
            mm = float(np.median([x[1] for x in recent]))
            deg = float(np.median([x[2] for x in recent]))
            state = "good" if mm < 10 and deg < 2 else "fair" if mm < 25 and deg < 5 else "poor"
            out[serial] = {"mm": round(mm, 1), "deg": round(deg, 2), "n": len(recent), "state": state}
        return out
