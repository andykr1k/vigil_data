"""Depth sampling, deprojection, point clouds and ground-plane estimation.

All 3D coordinates are in the RealSense color camera frame (OpenCV convention):
x right, y down, z forward, metres.
"""

from __future__ import annotations

import numpy as np

from .camera import Intrinsics


def sample_depth(depth: np.ndarray, uv: np.ndarray, patch: int) -> np.ndarray:
    """Median of valid depth in a (2*patch+1)^2 window around each (u, v). 0 where none valid."""
    h, w = depth.shape
    uv = np.round(np.asarray(uv)).astype(int).reshape(-1, 2)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    off = np.arange(-patch, patch + 1)
    xs = np.clip(uv[:, 0, None, None] + off[None, None, :], 0, w - 1)
    ys = np.clip(uv[:, 1, None, None] + off[None, :, None], 0, h - 1)
    win = depth[ys, xs].reshape(len(uv), -1).astype(np.float32)
    # Clipped windows repeat edge pixels; fine for a median of a few-pixel border.
    win[win <= 0] = np.nan
    valid = ~np.isnan(win).all(axis=1)
    out = np.zeros(len(uv), dtype=np.float32)
    if valid.any():
        out[valid] = np.nanmedian(win[valid], axis=1)
    out[~inside] = 0
    return out


def deproject(uv: np.ndarray, z: np.ndarray, K: Intrinsics) -> np.ndarray:
    x = (uv[:, 0] - K.cx) * z / K.fx
    y = (uv[:, 1] - K.cy) * z / K.fy
    return np.stack([x, y, z], axis=-1)


def point_cloud(
    color: np.ndarray, depth: np.ndarray, K: Intrinsics, stride: int, zmin: float, zmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Subsampled (N, 3) float32 xyz and (N, 3) uint8 rgb."""
    d = depth[::stride, ::stride]
    c = color[::stride, ::stride]
    vs, us = np.mgrid[0: depth.shape[0]: stride, 0: depth.shape[1]: stride]
    mask = (d > zmin) & (d < zmax)
    z = d[mask]
    uv = np.stack([us[mask], vs[mask]], axis=-1).astype(np.float32)
    return deproject(uv, z, K).astype(np.float32), c[mask]


def fit_floor(
    xyz: np.ndarray,
    max_tilt_deg: float,
    iters: int = 300,
    inlier_m: float = 0.03,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float] | None:
    """RANSAC the dominant roughly-horizontal plane below the camera.

    Returns (n, d) with unit n pointing up (towards the camera side) and n·p + d = 0,
    so d is the camera height above the floor. None if no plausible floor is found.
    """
    if len(xyz) < 500:
        return None
    rng = rng or np.random.default_rng()
    # Only points below the optical axis can be floor (camera y points down).
    cand = xyz[xyz[:, 1] > 0.1]
    if len(cand) < 300:
        return None
    if len(cand) > 8000:
        cand = cand[rng.choice(len(cand), 8000, replace=False)]

    tri = cand[rng.integers(0, len(cand), size=(iters, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n, tri = n[ok] / norm[ok, None], tri[ok]
    n *= np.where(n[:, 1] > 0, -1.0, 1.0)[:, None]  # make every normal point up (-y)
    up = np.array([0.0, -1.0, 0.0])
    ok = n @ up > np.cos(np.radians(max_tilt_deg))
    n, tri = n[ok], tri[ok]
    if len(n) == 0:
        return None
    d = -np.einsum("ij,ij->i", n, tri[:, 0])
    ok = d > 0.2  # camera must be at least 20 cm above it
    n, d = n[ok], d[ok]
    if len(n) == 0:
        return None

    counts = (np.abs(cand @ n.T + d) < inlier_m).sum(axis=0)
    best = int(np.argmax(counts))
    if counts[best] < 0.05 * len(cand):
        return None

    # Least-squares refinement on the inliers.
    inl = cand[np.abs(cand @ n[best] + d[best]) < inlier_m]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[1] > 0:
        normal = -normal
    return normal.astype(np.float32), float(-normal @ centroid)
