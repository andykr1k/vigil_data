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
    def __init__(self, cfg: CamerasConfig, extrinsics_path: Path):
        self.cfg = cfg
        self.extrinsics_path = extrinsics_path
        self.warnings: list[str] = []
        serials = cfg.serials or connected_serials()
        if not serials:
            raise RuntimeError("No RealSense connected")
        self.cameras: list[RigCamera] = []
        for serial in serials:
            try:
                cam = CameraClient(cfg, serial)
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
    started: float = field(default_factory=time.monotonic)
    samples: dict[str, list[np.ndarray]] = field(default_factory=dict)
    _recent: deque = field(default_factory=lambda: deque(maxlen=5))

    def add(self, world_serial: str, poses: dict[str, ObjectPose]) -> None:
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
        return to_h(R_mean, t_mean), meta


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


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
