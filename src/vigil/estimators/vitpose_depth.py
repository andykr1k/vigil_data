"""ViTPose (transformers) 2D keypoints lifted to metric 3D with the RealSense depth map.

Fast path for 30 fps: crops are warped with cv2 (the processor uses scipy, ~9 ms),
every camera's crop goes through one batched forward pass, and the model is compiled
with CUDA graphs (torch.compile "reduce-overhead": ~4 ms for two crops vs ~16 ms eager).
Crop geometry and heatmap decoding reuse transformers' own ViTPose helpers.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation
from transformers.models.vitpose.image_processing_vitpose import (
    box_to_center_and_scale,
    get_keypoint_predictions,
    get_warp_matrix,
    transform_preds,
)

from ..camera import Frame, Intrinsics
from ..config import VitposeDepthConfig
from ..geometry import deproject, sample_depth
from ..skeleton import JOINT_INDEX, empty_joints, fill_derived
from .base import PoseEstimator, PoseResult

log = logging.getLogger(__name__)

# COCO-17 keypoint order produced by ViTPose.
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
_MAP = [(i, JOINT_INDEX[n]) for i, n in enumerate(COCO17) if n in JOINT_INDEX]


class VitposeDepthEstimator(PoseEstimator):
    name = "vitpose_depth"

    def __init__(self, cfg: VitposeDepthConfig, device: str):
        self.cfg = cfg
        self.device = device
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        model = VitPoseForPoseEstimation.from_pretrained(cfg.model_id).to(device).eval()
        if cfg.half:
            model = model.half()
        self.model = model
        self.dtype = torch.float16 if cfg.half else torch.float32
        # ViTPose+ checkpoints are multi-dataset MoE models; expert 0 is COCO.
        self.moe = getattr(model.config.backbone_config, "num_experts", 1) > 1
        size = self.processor.size
        self.in_w, self.in_h = size["width"], size["height"]
        self.mean = np.asarray(self.processor.image_mean, np.float32) * 255
        self.std = np.asarray(self.processor.image_std, np.float32) * 255
        self.batch = 1
        self._forward = self.model

    def warmup(self, batch: int) -> None:
        """Fix the batch size (one crop per camera) and compile; must run on the inference thread."""
        self.batch = max(1, batch)
        if self.cfg.compile and self.device.startswith("cuda"):
            log.info("compiling ViTPose (batch %d) — first run takes ~20-30 s", self.batch)
            self._forward = torch.compile(self.model, mode="reduce-overhead")
        with torch.inference_mode():
            # Built inside inference_mode like real inputs; otherwise dynamo's dispatch-key
            # guard fails on the first real frame and recompiles (a multi-second stall).
            x = torch.zeros(self.batch, 3, self.in_h, self.in_w, device=self.device, dtype=self.dtype)
            for _ in range(3):  # compile + record the CUDA graph
                self._run(x)
        torch.cuda.synchronize()

    def _run(self, pixels: torch.Tensor) -> np.ndarray:
        extra = {}
        if self.moe:
            extra["dataset_index"] = torch.zeros(len(pixels), dtype=torch.long, device=self.device)
        heat = self._forward(pixel_values=pixels, **extra).heatmaps
        return heat.float().cpu().numpy()

    def _crop(self, rgb: np.ndarray, bbox: np.ndarray):
        x1, y1, x2, y2 = bbox
        center, scale = box_to_center_and_scale(
            [x1, y1, x2 - x1, y2 - y1], image_width=self.in_w, image_height=self.in_h)
        M = get_warp_matrix(0, center * 2.0, np.array((self.in_w, self.in_h)) - 1.0, scale * 200.0)
        crop = cv2.warpAffine(rgb, M, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR)
        return (crop.astype(np.float32) - self.mean) / self.std, center, scale

    @torch.inference_mode()
    def estimate_batch(self, frames: list[Frame], bboxes: list[np.ndarray],
                       Ks: list[Intrinsics]) -> list[PoseResult | None]:
        n = len(frames)
        if n == 0:
            return []
        B = max(self.batch, n)  # pad to the compiled batch size (static shapes for CUDA graphs)
        batch = np.zeros((B, self.in_h, self.in_w, 3), np.float32)
        centers = np.zeros((n, 2), np.float32)
        scales = np.zeros((n, 2), np.float32)
        for i, (f, b) in enumerate(zip(frames, bboxes)):
            batch[i], centers[i], scales[i] = self._crop(f.color, b)
        pixels = torch.from_numpy(batch).to(self.device, non_blocking=True)
        pixels = pixels.permute(0, 3, 1, 2).to(self.dtype).contiguous()
        heat = self._run(pixels)[:n]
        preds, scores = decode_heatmaps(heat, centers, scales)
        return [self._lift(f, preds[i], scores[i, :, 0], K) for i, (f, K) in enumerate(zip(frames, Ks))]

    def estimate(self, frame: Frame, bbox: np.ndarray, K: Intrinsics) -> PoseResult | None:
        return self.estimate_batch([frame], [bbox], [K])[0]

    def _lift(self, frame: Frame, uv: np.ndarray, scores: np.ndarray, K: Intrinsics) -> PoseResult | None:
        kp2d = np.zeros((len(JOINT_INDEX), 3), dtype=np.float32)
        for src, dst in _MAP:
            kp2d[dst] = (*uv[src], scores[src])

        z = sample_depth(frame.depth, uv, self.cfg.depth_patch_px)
        good = (scores >= self.cfg.min_score) & (z > 0)
        if good.sum() < 3:
            return None
        # Reject depth samples that hit the background (e.g. between the legs).
        body_z = np.median(z[good])
        good &= np.abs(z - body_z) < self.cfg.max_depth_jump_m

        # Depth hits the skin/clothing surface; the joint centre is ~one limb radius further.
        xyz = deproject(uv, z, K)
        ray = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-6)
        xyz = xyz + ray * self.cfg.limb_radius_m

        joints = empty_joints()
        for src, dst in _MAP:
            if good[src]:
                joints[dst] = (*xyz[src], scores[src])
        fill_derived(joints)
        return PoseResult(joints=joints, kp2d=kp2d)


def decode_heatmaps(heat: np.ndarray, centers: np.ndarray, scales: np.ndarray,
                    kernel: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """transformers' ViTPose decoding (argmax + DARK refinement + inverse crop transform),
    with the per-heatmap scipy Gaussian replaced by one multi-channel cv2 blur (~6 ms → 0.2 ms).
    """
    n, k, h, w = heat.shape
    coords, scores = get_keypoint_predictions(heat)
    # scipy gaussian_filter(sigma=0.8, radius=r, mode="reflect") == cv2 BORDER_REFLECT.
    maps = np.ascontiguousarray(heat.reshape(n * k, h, w).transpose(1, 2, 0))
    blurred = np.empty_like(maps)
    for c0 in range(0, n * k, 512):  # cv2 filters take at most 512 channels
        blurred[..., c0:c0 + 512] = cv2.GaussianBlur(
            maps[..., c0:c0 + 512], (kernel, kernel), 0.8, borderType=cv2.BORDER_REFLECT
        ).reshape(h, w, -1)
    logh = np.log(np.clip(blurred.transpose(2, 0, 1).reshape(n, k, h, w), 0.001, 50))

    # DARK: one Newton step on the log-heatmap around the integer peak.
    pad = np.pad(logh, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    x = coords[..., 0].astype(int) + 1
    y = coords[..., 1].astype(int) + 1
    ni, ki = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")

    def at(dy, dx):
        return pad[ni, ki, y + dy, x + dx]

    i_ = at(0, 0)
    dx = 0.5 * (at(0, 1) - at(0, -1))
    dy = 0.5 * (at(1, 0) - at(-1, 0))
    dxx = at(0, 1) - 2 * i_ + at(0, -1)
    dyy = at(1, 0) - 2 * i_ + at(-1, 0)
    dxy = 0.5 * (at(1, 1) - at(0, 1) - at(1, 0) + 2 * i_ - at(0, -1) - at(-1, 0) + at(-1, -1))
    hess = np.stack([np.stack([dxx, dxy], -1), np.stack([dxy, dyy], -1)], -2)
    hess = np.linalg.inv(hess + np.finfo(np.float32).eps * np.eye(2))
    coords = coords - np.einsum("ijmn,ijn->ijm", hess, np.stack([dx, dy], -1))
    preds = np.stack([transform_preds(coords[i], centers[i], scales[i], [h, w]) for i in range(n)])
    return preds, scores
