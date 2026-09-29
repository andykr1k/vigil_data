"""Pivot calibration of the probe tip.

Rest the tip in a fixed divot and rock/rotate the probe around it. Every tracked pose
(R_i, t_i) of the cube then satisfies  R_i · tip + t_i = pivot  for one unknown tip
offset (cube frame) and one unknown pivot point (world frame) — linear least squares.
The result replaces the CAD tip offset, absorbing how the attachment was actually built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .object_pose import ObjectPose


@dataclass
class PivotResult:
    tip: np.ndarray  # cube frame, metres
    pivot: np.ndarray  # world frame, metres
    rms_mm: float
    samples: int
    rejected: int
    spread_deg: float  # largest rotation between any sample and the mean


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def solve_pivot(poses: list[ObjectPose]) -> PivotResult | None:
    if len(poses) < 20:
        return None
    R = np.stack([p.rotation_matrix for p in poses])
    t = np.stack([p.position for p in poses])
    keep = np.ones(len(poses), bool)
    for _ in range(3):  # solve, drop gross outliers (bumps, bad frames), re-solve
        A = np.concatenate([R[keep], -np.broadcast_to(np.eye(3), (int(keep.sum()), 3, 3))], axis=2)
        x, *_ = np.linalg.lstsq(A.reshape(-1, 6), -t[keep].reshape(-1), rcond=None)
        res = np.linalg.norm(np.einsum("nij,j->ni", R, x[:3]) + t - x[3:], axis=1)
        new_keep = res <= max(3 * np.median(res[keep]), 0.002)
        if (new_keep == keep).all():
            break
        keep = new_keep
    mean_R = R[keep][len(R[keep]) // 2]
    spread = max(_angle_deg(mean_R, r) for r in R[keep])
    return PivotResult(x[:3], x[3:], float(np.sqrt(np.mean(res[keep] ** 2)) * 1000),
                       int(keep.sum()), int((~keep).sum()), spread)


@dataclass
class PivotCalibrator:
    target: int = 300  # ~10 s at 30 fps
    min_spread_deg: float = 25.0
    poses: list[ObjectPose] = field(default_factory=list)

    def add(self, pose: ObjectPose | None) -> None:
        if pose is not None:
            self.poses.append(pose)

    def progress(self) -> dict:
        spread = 0.0
        if len(self.poses) >= 2:
            ref = self.poses[0].rotation_matrix
            spread = max(_angle_deg(ref, p.rotation_matrix) for p in self.poses[-60:])
        return {"samples": len(self.poses), "target": self.target, "spread_deg": round(spread, 1)}

    def finished(self) -> bool:
        return len(self.poses) >= self.target


def save_tip(path: Path, result: PivotResult, cad_tip: np.ndarray) -> None:
    doc = {
        "tip_in_object_m": [round(float(v), 6) for v in result.tip],
        "rms_mm": round(result.rms_mm, 2),
        "samples": result.samples,
        "rejected": result.rejected,
        "rotation_spread_deg": round(result.spread_deg, 1),
        "change_from_cad_mm": round(float(np.linalg.norm(result.tip - cad_tip)) * 1000, 2),
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Probe tip from pivot calibration (overrides probe.tip_in_object_m).\n"
                    + yaml.safe_dump(doc, sort_keys=False))


def load_tip(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    tip = doc.get("tip_in_object_m")
    return None if tip is None else np.asarray(tip, dtype=np.float64)
