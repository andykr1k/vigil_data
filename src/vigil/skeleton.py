"""Canonical joint set shared by all backends and the dashboard, plus leg joint angles."""

from __future__ import annotations

import numpy as np

JOINTS: list[str] = [
    "nose", "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "pelvis",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_big_toe", "right_big_toe",
    "left_small_toe", "right_small_toe",
]
JOINT_INDEX = {name: i for i, name in enumerate(JOINTS)}

LEG_JOINTS = [
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
]

BONES: list[tuple[str, str]] = [
    # legs
    ("pelvis", "left_hip"), ("pelvis", "right_hip"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("right_ankle", "right_heel"),
    ("left_heel", "left_big_toe"), ("right_heel", "right_big_toe"),
    ("left_ankle", "left_big_toe"), ("right_ankle", "right_big_toe"),
    ("left_big_toe", "left_small_toe"), ("right_big_toe", "right_small_toe"),
    # upper body
    ("pelvis", "neck"), ("neck", "nose"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
]


def empty_joints() -> np.ndarray:
    """(J, 4) array of x, y, z, confidence; missing joints are NaN with confidence 0."""
    out = np.full((len(JOINTS), 4), np.nan, dtype=np.float32)
    out[:, 3] = 0.0
    return out


def fill_derived(joints: np.ndarray) -> None:
    """Fill pelvis / neck from hips / shoulders when the backend doesn't provide them."""
    for derived, (a, b) in (("pelvis", ("left_hip", "right_hip")),
                            ("neck", ("left_shoulder", "right_shoulder"))):
        d, ia, ib = JOINT_INDEX[derived], JOINT_INDEX[a], JOINT_INDEX[b]
        if joints[d, 3] <= 0 and joints[ia, 3] > 0 and joints[ib, 3] > 0:
            joints[d, :3] = (joints[ia, :3] + joints[ib, :3]) / 2
            joints[d, 3] = min(joints[ia, 3], joints[ib, 3])


def _angle(joints: np.ndarray, a: str, b: str, c: str) -> float | None:
    """Angle ABC in degrees, or None if any joint is missing."""
    pa, pb, pc = (joints[JOINT_INDEX[n]] for n in (a, b, c))
    if min(pa[3], pb[3], pc[3]) <= 0:
        return None
    v1, v2 = pa[:3] - pb[:3], pc[:3] - pb[:3]
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))


def leg_angles(joints: np.ndarray) -> dict[str, float | None]:
    """Clinical-style leg angles in degrees.

    knee flexion : 0 = straight leg
    hip flexion  : 0 = thigh in line with the trunk (neck→pelvis)
    ankle        : angle between shin and foot (≈90 standing; >90 = plantarflexion)
    """
    out: dict[str, float | None] = {}
    for side in ("left", "right"):
        knee = _angle(joints, f"{side}_hip", f"{side}_knee", f"{side}_ankle")
        hip = _angle(joints, "neck", f"{side}_hip", f"{side}_knee")
        ankle = _angle(joints, f"{side}_knee", f"{side}_ankle", f"{side}_big_toe")
        out[f"{side}_knee"] = None if knee is None else 180.0 - knee
        out[f"{side}_hip"] = None if hip is None else 180.0 - hip
        out[f"{side}_ankle"] = ankle
    return out
