"""Meta SAM 3D Body (MHR body model) backend.

The upstream repo isn't pip-installable, so `vigil setup` clones it into
third_party/ and we import it from there. Weights are gated on Hugging Face
(HF_TOKEN in .env).
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

from ..camera import Frame, Intrinsics
from ..config import Sam3dBodyConfig
from ..geometry import sample_depth
from ..skeleton import JOINT_INDEX, empty_joints, fill_derived
from .base import PoseEstimator, PoseResult

# Indices into the MHR-70 keypoint set (sam_3d_body/metadata/mhr70.py).
MHR70 = {
    "nose": 0, "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10, "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14,
    "left_big_toe": 15, "left_small_toe": 16, "left_heel": 17,
    "right_big_toe": 18, "right_small_toe": 19, "right_heel": 20,
    "right_wrist": 41, "left_wrist": 62, "neck": 69,
}
_MAP = [(src, JOINT_INDEX[name]) for name, src in MHR70.items()]
# Joints whose measured depth is trusted for anchoring (big, well-textured body parts).
_ANCHOR = [MHR70[n] for n in (
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder",
)]


@contextlib.contextmanager
def _quiet():
    """The upstream estimator prints on every call."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class Sam3dBodyEstimator(PoseEstimator):
    name = "sam3d_body"

    def __init__(self, cfg: Sam3dBodyConfig, repo_dir: Path, device: str):
        if not device.startswith("cuda"):
            raise RuntimeError("SAM 3D Body inference requires CUDA (upstream hard-codes it).")
        if not (repo_dir / "sam_3d_body").is_dir():
            raise RuntimeError(f"SAM 3D Body code not found at {repo_dir}. Run `uv run vigil setup`.")
        sys.path.insert(0, str(repo_dir))
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf

        self.cfg = cfg
        with _quiet():
            model, model_cfg = load_sam_3d_body_hf(cfg.hf_repo_id)
            self.estimator = SAM3DBodyEstimator(model, model_cfg)
        self.faces = self.estimator.faces.astype(np.int32)

    @torch.inference_mode()
    def estimate(self, frame: Frame, bbox: np.ndarray, K: Intrinsics) -> PoseResult | None:
        cam_int = torch.from_numpy(K.matrix())[None] if self.cfg.use_camera_intrinsics else None
        with _quiet():
            outs = self.estimator.process_one_image(
                frame.color,
                bboxes=bbox[None].astype(np.float32),
                cam_int=cam_int,
                inference_type=self.cfg.inference_type,
            )
        if not outs:
            return None
        o = outs[0]
        cam_t = o["pred_cam_t"].astype(np.float32)
        kp3d = o["pred_keypoints_3d"].astype(np.float32) + cam_t
        verts = o["pred_vertices"].astype(np.float32) + cam_t
        uv = o["pred_keypoints_2d"][:, :2].astype(np.float32)

        if self.cfg.anchor_to_depth:
            s = self._depth_scale(frame.depth, uv, kp3d)
            kp3d *= s
            verts *= s

        h, w = frame.depth.shape
        in_view = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        joints = empty_joints()
        kp2d = np.zeros((len(JOINT_INDEX), 3), dtype=np.float32)
        for src, dst in _MAP:
            conf = 1.0 if in_view[src] else 0.4  # out-of-frame joints are extrapolated
            joints[dst] = (*kp3d[src], conf)
            kp2d[dst] = (*uv[src], conf)
        fill_derived(joints)
        return PoseResult(joints=joints, kp2d=kp2d, vertices=verts)

    def _depth_scale(self, depth: np.ndarray, uv: np.ndarray, kp3d: np.ndarray) -> float:
        """Scale about the camera centre that makes predicted depth match the sensor.

        Monocular mesh recovery can't tell a big far person from a small near one;
        scaling about the optical centre keeps the 2D projection identical while
        fixing distance (and body size) from the measured depth.
        """
        idx = np.array(_ANCHOR)
        measured = sample_depth(depth, uv[idx], patch=5)
        pred = kp3d[idx, 2]
        target = measured + self.cfg.limb_radius_m
        ok = (measured > 0) & (np.abs(target - pred) < 1.5) & (pred > 0.1)
        if ok.sum() < 3:
            return 1.0
        return float(np.median(target[ok] / pred[ok]))
