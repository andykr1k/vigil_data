"""Leg surface meshes from a recording's RGB-D frames.

TSDF fusion: every camera's depth frames in a time window are cut down to the leg, fused
into a truncated signed distance volume, and the zero crossing is extracted with marching
cubes. Only voxels some camera actually observed take part, so the result is the open
surface the cameras saw (the top of the leg), not a closed reconstruction.

Leg region ("upper half-capsule"): a pixel is kept when its 3D point lies within `radius`
of a detected leg bone (hip–knee–ankle–foot, from the recorded pose) and no more than
`below` under that bone, measured along the floor's up direction. A support-plane cut then
removes what the leg rests on, and an optional knee cut keeps only the thigh or the shin.

All geometry is in the recording's world frame (the first camera's colour optical frame,
metres). Input is a recording written by vigil.recorder: meta.json, frames.jsonl and the
per-camera colour / aligned-depth images.
"""

from __future__ import annotations

import bisect
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from ..camera import Intrinsics
from ..geometry import deproject
from ..rig import transform
from ..skeleton import LEG_JOINTS
from .holes import fill_holes

log = logging.getLogger(__name__)

LEG_NAMES = [j for j in LEG_JOINTS if j != "pelvis"]  # the leg itself (pelvis bones reach into the torso)
MAX_VOXELS = 8_000_000  # ~160 MB of volume; the voxel size grows to stay under it
MIN_JOINT_CONF = 0.3



@dataclass
class TsdfParams:
    window_s: float = 0.5          # fuse frames within ± this of the current frame
    max_frames: int = 30           # per camera; evenly subsampled beyond this
    cameras: list[int] | None = None  # None = every camera
    voxel_mm: float = 3.0
    trunc_mm: float = 12.0
    depth_min: float | None = None  # metres; None = the recording's depth window
    depth_max: float | None = None
    region: str = "pose"           # "pose" (half-capsule around leg bones) | "window" (depth window only)
    leg: str = "both"              # "left" | "right" | "both"
    radius_cm: float = 11.0
    below_cm: float = 5.0
    support: bool = True           # find the surface the leg rests on and cut it away
    support_margin_cm: float = 1.5
    edge_filter: bool = True       # drop flying pixels at depth discontinuities
    align: bool = True             # ICP-snap every depth frame onto a reference frame before fusing
    blend: bool = True             # feather each frame toward its mask edge and grazing angles (no seams)
    largest: bool = True           # keep only the largest connected piece
    smooth: int = 4                # Taubin smoothing iterations
    fill_holes: bool = True        # close holes smaller than max_hole_cm (never the outer edge)
    max_hole_cm: float = 5.0
    section: str = "above"         # "whole" | "above" (knee → hip, the thigh) | "below" (knee → foot)
    knee_shift_cm: float = 0.0     # move the knee cut along the leg (+ = toward the hip)

    @classmethod
    def from_dict(cls, d: dict) -> TsdfParams:
        names = cls.__dataclass_fields__.keys()
        p = cls(**{k: v for k, v in d.items() if k in names})
        p.voxel_mm = float(np.clip(p.voxel_mm, 1.0, 20.0))
        p.trunc_mm = float(np.clip(p.trunc_mm, 2 * p.voxel_mm, 100.0))
        p.window_s = float(np.clip(p.window_s, 0.0, 10.0))
        p.max_frames = int(np.clip(p.max_frames, 1, 300))
        p.smooth = int(np.clip(p.smooth, 0, 100))
        p.max_hole_cm = float(np.clip(p.max_hole_cm, 0.0, 50.0))
        return p


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float32, world frame, metres
    faces: np.ndarray     # (F, 3) uint32
    normals: np.ndarray   # (V, 3) float32
    colors: np.ndarray    # (V, 3) uint8
    stats: dict = field(default_factory=dict)

    def to_binary(self) -> bytes:
        """u32 V, u32 F, f32 xyz[V*3], f32 normals[V*3], u8 rgb[V*3] (padded to 4), u32 faces[F*3]."""
        rgb = self.colors.tobytes()
        rgb += b"\0" * (-len(rgb) % 4)
        return (np.array([len(self.vertices), len(self.faces)], "<u4").tobytes()
                + self.vertices.astype("<f4").tobytes() + self.normals.astype("<f4").tobytes()
                + rgb + self.faces.astype("<u4").tobytes())

    def to_ply(self) -> bytes:
        v = np.empty(len(self.vertices), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                                                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for i, k in enumerate("xyz"):
            v[k] = self.vertices[:, i]
            v["n" + k] = self.normals[:, i]
        for i, k in enumerate(("red", "green", "blue")):
            v[k] = self.colors[:, i]
        f = np.empty(len(self.faces), dtype=[("n", "u1"), ("i", "<u4", (3,))])
        f["n"], f["i"] = 3, self.faces
        header = ("ply\nformat binary_little_endian 1.0\ncomment vigil TSDF leg surface, "
                  "world frame (camera 1 colour optical), metres\n"
                  f"element vertex {len(v)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property float nx\nproperty float ny\nproperty float nz\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"element face {len(f)}\nproperty list uchar uint vertex_indices\nend_header\n")
        return header.encode() + v.tobytes() + f.tobytes()

    @classmethod
    def from_ply(cls, data: bytes, stats: dict | None = None) -> Mesh:
        """Read back a PLY written by `to_ply` (binary: xyz, normals, rgb; triangle faces)."""
        end = data.index(b"end_header\n") + len(b"end_header\n")
        header = data[:end].decode()
        nv = int(header.split("element vertex ")[1].split()[0])
        nf = int(header.split("element face ")[1].split()[0])
        vd = np.dtype([("p", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 3)])
        v = np.frombuffer(data, vd, nv, end)
        f = np.frombuffer(data, np.dtype([("k", "u1"), ("i", "<u4", 3)]), nf, end + nv * vd.itemsize)
        return cls(v["p"].copy(), f["i"].copy(), v["n"].copy(), v["c"].copy(), stats or {})

    def to_obj(self) -> bytes:
        c = self.colors / 255.0
        lines = ["# vigil TSDF leg surface, world frame (camera 1 colour optical), metres"]
        lines += [f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {q[0]:.3f} {q[1]:.3f} {q[2]:.3f}"
                  for p, q in zip(self.vertices, c)]
        lines += [f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}" for n in self.normals]
        lines += [f"f {a}//{a} {b}//{b} {d}//{d}" for a, b, d in self.faces + 1]
        return ("\n".join(lines) + "\n").encode()


class MeshError(ValueError):
    """A build that can't go ahead (no frames, no leg found, …): shown to the user."""


# ---------------------------------------------------------------------- inputs
@dataclass
class _View:
    cam: int
    depth: Path
    color: Path
    joints: dict
    up: np.ndarray
    t: float


def _camera_matrices(meta: dict) -> dict[int, dict]:
    out = {}
    for c in meta["cameras"]:
        if c.get("T_world_camera") is None:
            continue  # uncalibrated: can't place it in the world frame
        T = np.array(c["T_world_camera"], float).reshape(4, 4)
        K = Intrinsics(c["width"], c["height"], c["fx"], c["fy"], c["cx"], c["cy"],
                       tuple(c.get("dist") or (0.0,) * 5))
        out[c["index"]] = {"fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"],
                           "w": c["width"], "h": c["height"], "K": K, "T": T, "T_inv": np.linalg.inv(T)}
    return out


def _up(frame: dict, cams: dict) -> np.ndarray:
    """World 'up': the floor normal if known, else opposite the cameras' mean viewing ray."""
    n = (frame.get("floor") or {}).get("normal")
    if n is not None:
        return np.asarray(n, float) / np.linalg.norm(n)
    look = np.mean([c["T"][:3, 2] for c in cams.values()], axis=0)
    return -look / np.linalg.norm(look)


def _leg_bones(joints: dict, leg: str, bones: list) -> list[tuple[np.ndarray, np.ndarray]]:
    names = {j for j in LEG_NAMES if leg == "both" or j.startswith(leg + "_")}
    ok = {k: np.asarray(v[:3], float) for k, v in joints.items()
          if k in names and (len(v) < 4 or v[3] >= MIN_JOINT_CONF)}
    segs = [(ok[a], ok[b]) for a, b in bones if a in ok and b in ok]
    if not segs:  # bones broken up (a joint missing): still keep each joint as a sphere
        segs = [(p, p) for p in ok.values()]
    return segs


def _select_views(lines: list, index: int, p: TsdfParams, cams: dict) -> list[_View]:
    t = [ln["t"] for ln in lines]
    lo = bisect.bisect_left(t, t[index] - p.window_s)
    hi = bisect.bisect_right(t, t[index] + p.window_s)
    # Joints for each line: its own, else the nearest line in the window that has some.
    with_joints = [i for i in range(lo, hi) if (lines[i]["frame"].get("person") or {}).get("joints")]
    per_cam: dict[int, list[_View]] = {}
    for i in range(lo, hi):
        ln = lines[i]
        if with_joints:
            j = min(with_joints, key=lambda k: abs(k - i))
            joints = lines[j]["frame"]["person"]["joints"]
        else:
            joints = {}
        for f in ln["files"]:
            cam = f.get("cam")
            if cam is None or "depth" not in f or cam not in cams:
                continue
            if p.cameras is not None and cam not in p.cameras:
                continue
            per_cam.setdefault(cam, []).append(
                _View(cam, Path(f["depth"]), Path(f["color"]), joints, _up(ln["frame"], cams), ln["t"]))
    views = []
    for cam, vs in sorted(per_cam.items()):
        if len(vs) > p.max_frames:  # evenly spread, always including the centre
            keep = np.unique(np.round(np.linspace(0, len(vs) - 1, p.max_frames)).astype(int))
            vs = [vs[k] for k in keep]
        views.extend(vs)
    return views


# ---------------------------------------------------------------------- segmentation
def _deproject(depth_m: np.ndarray, c: dict) -> np.ndarray:
    """Every pixel of a depth image as a world point (H, W, 3); 0-depth pixels land on the camera."""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xyz = deproject(np.stack([u.ravel(), v.ravel()], 1), depth_m.ravel(), c["K"])
    return transform(c["T"].astype(np.float32), xyz).reshape(h, w, 3).astype(np.float32)


def _normals_grid(world: np.ndarray, valid: np.ndarray, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel surface normals of an organised point grid (central differences)."""
    n = np.zeros_like(world)
    ok = np.zeros(valid.shape, bool)
    s = step
    dx = world[s:-s, 2 * s:] - world[s:-s, :-2 * s]
    dy = world[2 * s:, s:-s] - world[:-2 * s, s:-s]
    c = np.cross(dx, dy)
    norm = np.linalg.norm(c, axis=-1, keepdims=True)
    n[s:-s, s:-s] = c / np.maximum(norm, 1e-12)
    ok[s:-s, s:-s] = (valid[s:-s, 2 * s:] & valid[s:-s, :-2 * s] & valid[2 * s:, s:-s]
                      & valid[:-2 * s, s:-s] & valid[s:-s, s:-s] & (norm[..., 0] > 0))
    return n, ok


def _edge_mask(depth_m: np.ndarray) -> np.ndarray:
    """False at depth discontinuities (RealSense 'flying pixels' between surfaces)."""
    h, w = depth_m.shape
    pad = np.pad(depth_m, 1, mode="edge")
    jump = np.zeros_like(depth_m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                np.maximum(jump, np.abs(depth_m - pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]), out=jump)
    return jump < 0.01 + 0.02 * depth_m


def _segment(depth_m: np.ndarray, view: _View, c: dict, p: TsdfParams, bones: list,
             window: tuple[float, float]) -> np.ndarray:
    """Depth with everything but the leg's upper surface zeroed."""
    keep = (depth_m > window[0]) & (depth_m < window[1])
    if p.edge_filter:
        keep &= _edge_mask(depth_m)
    if p.region == "pose":
        pts = _deproject(np.where(keep, depth_m, 0), c)[keep]
        r, below = p.radius_cm / 100, p.below_cm / 100
        inside = np.zeros(len(pts), bool)
        for a, b in bones:
            ab = b - a
            L2 = float(ab @ ab)
            s = np.clip(((pts - a) @ ab) / L2, 0, 1) if L2 > 1e-9 else np.zeros(len(pts))
            q = a + s[:, None] * ab
            rel = pts - q
            inside |= ((rel ** 2).sum(1) <= r * r) & (rel @ view.up >= -below)
        keep[keep] = inside
    return np.where(keep, depth_m, 0).astype(np.float32)


def _local_below(pts: np.ndarray, sd: np.ndarray, on: np.ndarray, n: np.ndarray,
                 cell: float = 0.03) -> float:
    """Share of the points near a plane's inliers (in-plane, within a cell) that lie under it.

    A sloping leg can have most of itself above a plane that slices through its shin; right
    next to that slice, though, the shin's underside is below it. A real support only has
    its own rim below it there.
    """
    a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    ij = np.floor(np.stack([pts @ a, pts @ b], 1) / cell).astype(np.int64)
    key = lambda q: (q[:, 0] + 100_000) * 200_003 + (q[:, 1] + 100_000)
    near_cells = np.unique(np.concatenate([key(ij[on] + [di, dj])
                                           for di in (-1, 0, 1) for dj in (-1, 0, 1)]))
    m = np.isin(key(ij), near_cells) & ~on & (np.abs(sd) > 0.01)
    return float((sd[m] < 0).mean()) if m.any() else 0.0


def _support_plane(pts: np.ndarray, normals: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The flat, roughly horizontal surface the leg rests on (stool, bed, table), if any.

    RANSAC over planes within 30° of horizontal. Only points whose own surface normal
    agrees with the plane count as on it, so a plane slicing through the leg's sides (whose
    normals point sideways) scores nothing. A support also has (almost) everything else
    above it; a plane through the leg's curved top has much of the leg below it, so those
    are rejected. And the leg must rest on it: at least 20 % of the points more than 2 cm
    above it (a plane across the very top of the leg has none). The least-squares refit is
    checked the same way. Returns (unit normal pointing up, d) with the plane n·x = d.
    """
    if len(pts) < 500:
        return None
    rng = np.random.default_rng(0)  # fresh per call: the same input always gives the same plane
    pick = rng.choice(len(pts), min(len(pts), 20000), replace=False)
    sample, snorm = pts[pick], normals[pick]

    def on_plane(n, d):
        return (np.abs(sample @ n - d) < 0.008) & (np.abs(snorm @ n) > np.cos(np.radians(25)))

    def supports_leg(n, d, on):
        sd = sample @ n - d
        below, above = int((sd < -0.01).sum()), int((sd > 0.01).sum())
        return (below <= 0.15 * (below + above) and _local_below(sample, sd, on, n) <= 0.22
                and (sd > 0.02).sum() >= 0.2 * len(sample))  # the leg rests *on* it

    best, best_count = None, 0
    for _ in range(400):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n @ up < 0:
            n = -n
        if n @ up < np.cos(np.radians(30)):
            continue
        d = float(n @ a)
        on = on_plane(n, d)
        count = int(on.sum())
        if count <= best_count or not supports_leg(n, d, on):
            continue
        best, best_count = (n, d), count
    if best is None or best_count < max(150, 0.015 * len(sample)):
        return None
    n, d = best  # refine on its inliers (least squares), if the refit still qualifies
    inl = sample[on_plane(n, d)]
    centre = inl.mean(0)
    n2 = np.linalg.svd(inl - centre, full_matrices=False)[2][-1]
    n2 = n2 if n2 @ up > 0 else -n2
    d2 = float(n2 @ centre)
    return (n2, d2) if supports_leg(n2, d2, on_plane(n2, d2)) else (n, d)


# ---------------------------------------------------------------------- alignment
def _icp(src: np.ndarray, dst: np.ndarray, dst_n: np.ndarray, tree: cKDTree,
         iterations: int = 30, lock_axis: np.ndarray | None = None,
         translate_only: bool = False) -> tuple[np.ndarray, float, float] | None:
    """Point-to-plane ICP moving `src` onto the surface (dst, dst_n). Returns the 4×4 world
    correction and the RMS point-to-plane distance before and after, or None if it can't lock on.

    A leg is close to a cylinder, so sliding along its axis barely changes the fit and ICP
    can drift along it; `lock_axis` removes the translation along that direction.
    `translate_only` solves for a shift alone, using the best-matching 60 % of pairs: for two
    cameras that share only a strip of the leg, rotation is poorly constrained."""
    T = np.eye(4)
    cur = src.astype(np.float64)
    rms0 = None
    for it in range(iterations):
        max_d = 0.03 if it < 5 else 0.015 if it < 15 else 0.008
        d, j = tree.query(cur, distance_upper_bound=max_d)
        ok = np.isfinite(d)
        if ok.sum() < 150:
            return None
        s, q, n = cur[ok], dst[j[ok]], dst_n[j[ok]]
        r = ((q - s) * n).sum(1)
        rms = float(np.sqrt(np.mean(r ** 2)))
        if rms0 is None:
            rms0 = rms
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.003)  # Huber: outliers count less
        if translate_only:
            keep = np.abs(r) <= np.quantile(np.abs(r), 0.6)
            x = np.zeros(6)
            x[3:] = np.linalg.lstsq(n[keep] * w[keep, None], r[keep] * w[keep], rcond=None)[0]
        else:
            A = np.hstack([np.cross(s, n), n]) * w[:, None]
            x = np.linalg.lstsq(A, r * w, rcond=None)[0]
        if lock_axis is not None:
            x[3:] -= (x[3:] @ lock_axis) * lock_axis
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, x[3:]
        cur = cur @ R.T + x[3:]
        T = step @ T
        if np.linalg.norm(x) < 1e-5:
            break
    d, j = tree.query(cur, distance_upper_bound=0.008)
    ok = np.isfinite(d)
    if ok.sum() < 150:
        return None
    rms1 = float(np.sqrt(np.mean((((dst[j[ok]] - cur[ok]) * dst_n[j[ok]]).sum(1)) ** 2)))
    return T, rms0, rms1


BLEND_PX = 20  # feather width at a frame's mask edge, in pixels


def _view_weights(depth_m: np.ndarray, c: dict) -> np.ndarray:
    """Per-pixel confidence for fusion: ramps 0 → 1 over BLEND_PX from the edge of the frame's
    leg mask, times cos(angle between the camera ray and the surface).

    Without it, where one camera's coverage stops the fused surface jumps from "average of
    both cameras" to "this camera only", and any disagreement between them shows as a step
    (a seam). Feathering makes that hand-over gradual. The cos term also undoes the 1/cos
    stretch of ray-distance SDFs at grazing angles, where depth is noisiest anyway.
    """
    mask = (depth_m > 0).astype(np.uint8)
    ramp = np.clip(cv2.distanceTransform(mask, cv2.DIST_L2, 5) / BLEND_PX, 0, 1)
    world = _deproject(depth_m, c)
    nrm, ok = _normals_grid(world, mask > 0)
    ray = world - c["T"][:3, 3].astype(np.float32)
    ray /= np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), 1e-6)
    cos = np.where(ok, np.abs((nrm * ray).sum(-1)), 0.3)
    return (ramp * np.clip(cos, 0.05, 1.0)).astype(np.float32)


def _metres(e: dict) -> np.ndarray:
    """A segmented frame's depth (stored as uint16 mm) in metres, 0 = masked out."""
    return e["seg"].astype(np.float32) * 0.001


def _apply(e: dict, T: np.ndarray) -> None:
    e["cam"] = _with_correction(e["cam"], T)
    e["pts"] = transform(T, e["pts"])
    e["nrm"] = e["nrm"] @ T[:3, :3].T


def _align(entries: list[dict], t_mid: float, progress) -> tuple[dict | None, int]:
    rng = np.random.default_rng(1)

    def fit(e, target, lock, max_t, max_deg, shift_only=False):
        if len(target["pts"]) < 500 or len(e["pts"]) < 300:
            return None
        src = e["pts"][rng.choice(len(e["pts"]), min(len(e["pts"]), 4000), replace=False)]
        res = _icp(src, target["pts"], target["nrm"], target.setdefault("tree", cKDTree(target["pts"])),
                   lock_axis=lock, translate_only=shift_only)
        if res is None:
            return None
        T, rms0, rms1 = res
        ang = np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3]).as_rotvec()))
        if rms1 >= rms0 or np.linalg.norm(T[:3, 3]) > max_t or ang > max_deg:
            return None
        return res

    anchors = {}
    for e in entries:
        cam = e["view"].cam
        if cam not in anchors or abs(e["view"].t - t_mid) < abs(anchors[cam]["view"].t - t_mid):
            anchors[cam] = e
    ref = anchors[min(anchors)]
    centred = ref["pts"] - ref["pts"].mean(0)
    axis = np.linalg.svd(centred[:: max(1, len(centred) // 5000)], full_matrices=False)[2][0]

    before, after, failed, done = [], [], 0, 0
    # Anchors of the other cameras → the reference (cross-camera calibration).
    for cam, a in anchors.items():
        if a is ref:
            continue
        res = fit(a, ref, axis, 0.04, 1, shift_only=True)
        if res is None:
            failed += 1
            continue
        _apply(a, res[0])
        a["corr"] = res[0]
        before.append(res[1])
        after.append(res[2])
    for a in anchors.values():
        a.pop("tree", None)  # anchors may have moved: rebuild their trees
    # Every other frame → its own camera's anchor (motion within the window).
    for e in entries:
        done += 1
        a = anchors[e["view"].cam]
        if e is a:
            continue
        res = fit(e, a, None, 0.06, 10)
        if res is None:
            failed += 1
            _apply(e, a.get("corr", np.eye(4)))  # at least take its camera's calibration fix
            continue
        _apply(e, res[0])
        before.append(res[1])
        after.append(res[2])
        progress("aligning", done / len(entries))
    for e in entries:
        e.pop("tree", None)
        e.pop("corr", None)
    stats = None
    if before:
        stats = {"aligned": len(before), "failed": failed,
                 "rms_before_mm": round(float(np.median(before)) * 1000, 2),
                 "rms_after_mm": round(float(np.median(after)) * 1000, 2)}
    return stats, failed


def _knee_section(entries: list[dict], pts: np.ndarray, p: TsdfParams, backend: str | None,
                  limb_radius: float) -> dict:
    """Where to cut the leg at the knee: a plane across the leg's long axis.

    The recorded knee keypoints are unreliable one by one (some land on the stool or the
    floor), so: take the leg's axis from its points (it is long and nearly straight), move
    every frame's knee keypoints by that frame's alignment, keep those lying on the leg
    surface, and use their median position along the axis. The ankles' median says which
    end is the foot. Positions are measured along `axis`, which points foot → hip.

    vitpose_depth joints sit `limb_radius` behind the surface the camera saw (its
    limb_radius_m setting); that push is undone before the on-the-leg test. Other backends (SAM 3D Body) put joints
    inside the limb from a body model, so they get a looser test instead.
    """
    undo = limb_radius if backend == "vitpose_depth" else 0.0
    max_d = 0.03 if undo else 0.09
    centre = pts.mean(0)
    sample = pts[:: max(1, len(pts) // 20000)]
    axis = np.linalg.svd(sample - centre, full_matrices=False)[2][0]
    tree = cKDTree(sample)
    world_cam = min(e["view"].cam for e in entries)  # the pose was computed from this camera's depth

    def candidates(part: str) -> tuple[list[float], int]:
        ts, seen = [], 0
        for e in entries:
            if e["view"].cam != world_cam:
                continue
            corr = e["cam"]["T"] @ np.linalg.inv(e["orig_T"])  # this frame's alignment
            cam_centre = e["orig_T"][:3, 3]
            for side in ("left", "right"):
                j = e["view"].joints.get(f"{side}_{part}")
                if j is None or (len(j) > 3 and j[3] < MIN_JOINT_CONF):
                    continue
                seen += 1
                q = np.asarray(j[:3], float)
                ray = (q - cam_centre) / max(np.linalg.norm(q - cam_centre), 1e-6)
                q = q - undo * ray  # back onto the surface the camera measured
                q = transform(corr, q)
                if tree.query(q)[0] <= max_d:  # on the leg, not the stool / floor
                    ts.append(float((q - centre) @ axis))
        return ts, seen

    warnings = []
    knees, knees_seen = candidates("knee")
    if not knees:
        raise MeshError(f"Couldn't place the knee: none of the {knees_seen} knee detections in this window "
                        "lie on the leg. Try another frame or a longer window, or use WHOLE LEG.")
    t_knee = float(np.median(knees))
    ankles, _ = candidates("ankle")
    hips, _ = candidates("hip")
    if ankles:
        foot_ahead = np.median(ankles) > t_knee
    elif hips:
        foot_ahead = np.median(hips) < t_knee
    else:
        # No usable ankle/hip: the foot end is the one with less leg beyond the knee
        # (the thigh runs on out of view toward the body).
        s_ = (sample - centre) @ axis
        foot_ahead = (s_.max() - t_knee) < (t_knee - s_.min())
        warnings.append("No ankle or hip on the leg: guessed which end is the foot.")
    if foot_ahead:  # make the axis point foot → hip
        axis, t_knee = -axis, -t_knee
    t_cut = t_knee + p.knee_shift_cm / 100
    if len(knees) < 3:
        warnings.append(f"Knee placed from only {len(knees)} detection(s); check the gold ring.")
    return {"centre": centre, "axis": axis, "t": t_cut, "knee": centre + t_cut * axis,
            "knees_used": len(knees), "knees_seen": knees_seen, "warnings": warnings}


def _with_correction(c: dict, T_corr: np.ndarray) -> dict:
    T = T_corr @ c["T"]
    return {**c, "T": T, "T_inv": np.linalg.inv(T)}


# ---------------------------------------------------------------------- fusion
class _Volume:
    def __init__(self, lo: np.ndarray, hi: np.ndarray, voxel: float, trunc: float):
        self.voxel, self.trunc = voxel, trunc
        self.origin = lo
        self.shape = tuple(np.maximum(np.ceil((hi - lo) / voxel).astype(int) + 1, 2))
        n = int(np.prod(self.shape))
        self.tsdf = np.ones(n, np.float32)
        self.weight = np.zeros(n, np.float32)
        self.color = np.zeros((n, 3), np.float32)
        self.cweight = np.zeros(n, np.float32)
        self.active = np.arange(n)
        self.points = self._centres(self.active)

    def _centres(self, flat: np.ndarray) -> np.ndarray:
        ijk = np.stack(np.unravel_index(flat, self.shape), 1).astype(np.float32)
        return ijk * self.voxel + self.origin.astype(np.float32)

    def restrict(self, point_sets) -> None:
        """Only voxels within the truncation band of the measured points can hold surface:
        integrate just those (the rest stay unobserved and are skipped when meshing).
        `point_sets` is an iterable of (N, 3) arrays, consumed one at a time."""
        occ = np.zeros(self.shape, bool)
        for pts in point_sets:
            ijk = np.round((pts - self.origin) / self.voxel).astype(int)
            ijk = ijk[((ijk >= 0) & (ijk < self.shape)).all(1)]
            occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        reach = int(np.ceil(self.trunc / self.voxel)) + 1
        band = binary_dilation(occ, np.ones((3, 3, 3), bool), iterations=reach)
        self.active = np.flatnonzero(band.ravel())
        self.points = self._centres(self.active)

    def integrate(self, depth_m: np.ndarray, rgb: np.ndarray | None, c: dict,
                  weights: np.ndarray | None = None) -> None:
        """Fold one depth frame in. `weights` (per pixel, 0–1) scales how much each pixel's
        measurement counts; None = every pixel counts 1."""
        Ti = c["T_inv"].astype(np.float32)
        pc = transform(Ti, self.points)
        z = pc[:, 2]
        front = z > 0.05
        zs = np.where(front, z, 1.0)
        u = np.round(pc[:, 0] * c["fx"] / zs + c["cx"]).astype(np.int32)
        v = np.round(pc[:, 1] * c["fy"] / zs + c["cy"]).astype(np.int32)
        h, w = depth_m.shape
        ok = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        loc = np.flatnonzero(ok)
        d = depth_m[v[loc], u[loc]]
        sdf = d - z[loc]
        wv = weights[v[loc], u[loc]] if weights is not None else np.ones(len(loc), np.float32)
        upd = (d > 0) & (sdf >= -self.trunc) & (wv > 0)
        loc, sdf, wv = loc[upd], sdf[upd], wv[upd]
        idx = self.active[loc]
        val = np.minimum(1.0, sdf / self.trunc)
        w_old = self.weight[idx]
        self.tsdf[idx] = (self.tsdf[idx] * w_old + val * wv) / (w_old + wv)
        self.weight[idx] = w_old + wv
        if rgb is not None:
            near = np.abs(sdf) < self.trunc
            ci, cl, cwv = idx[near], loc[near], wv[near]
            px = rgb[v[cl], u[cl]].astype(np.float32)
            cw = self.cweight[ci]
            self.color[ci] = (self.color[ci] * cw[:, None] + px * cwv[:, None]) / (cw + cwv)[:, None]
            self.cweight[ci] = cw + cwv

    def extract(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = self.tsdf.reshape(self.shape)
        obs = self.weight.reshape(self.shape) > 0
        if not (obs.any() and (T[obs] < 0).any() and (T[obs] > 0).any()):
            raise MeshError("No surface inside the volume: the leg region came out empty. "
                            "Try a larger radius, or region = depth window.")
        verts, faces, _, _ = marching_cubes(T, level=0.0, allow_degenerate=False)
        # A cube that touches an unobserved voxel has a fake crossing (unseen space counts
        # as empty): drop every triangle in such a cube, leaving only the observed surface.
        o = obs
        cube_ok = (o[:-1, :-1, :-1] & o[1:, :-1, :-1] & o[:-1, 1:, :-1] & o[:-1, :-1, 1:]
                   & o[1:, 1:, :-1] & o[1:, :-1, 1:] & o[:-1, 1:, 1:] & o[1:, 1:, 1:])
        cell = np.minimum(np.floor(verts).astype(int), np.array(cube_ok.shape) - 1)
        v_ok = cube_ok[cell[:, 0], cell[:, 1], cell[:, 2]]
        faces = faces[v_ok[faces].all(1)]
        ni = np.clip(np.round(verts).astype(int), 0, np.array(self.shape) - 1)
        flat = np.ravel_multi_index(ni.T, self.shape)
        colors = np.where(self.cweight[flat, None] > 0, self.color[flat], 150).astype(np.uint8)
        return (verts * self.voxel + self.origin).astype(np.float32), faces, colors


# ---------------------------------------------------------------------- mesh cleanup
def _compact(verts, faces, colors):
    used = np.unique(faces)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces], colors[used]


def _largest_piece(verts, faces, colors):
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adj = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, label = connected_components(adj, directed=False)
    sizes = np.bincount(label[faces[:, 0]])
    faces = faces[label[faces[:, 0]] == sizes.argmax()]
    return _compact(verts, faces, colors)


def _taubin(verts, faces, iterations, lam=0.5, mu=-0.53):
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0  # duplicates summed: back to a plain adjacency
    deg = np.asarray(A.sum(1)).ravel()
    W = sparse.diags(1.0 / np.maximum(deg, 1)) @ A
    v = verts.astype(np.float64)
    for _ in range(iterations):
        v = v + lam * (W @ v - v)
        v = v + mu * (W @ v - v)
    return v.astype(np.float32)


def _normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return (vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)).astype(np.float32), fn


# ---------------------------------------------------------------------- entry point
def build_tsdf(path: Path, meta: dict, lines: list, index: int, params: TsdfParams,
               progress=lambda stage, frac: None, limb_radius: float = 0.06) -> Mesh:
    """Mesh the leg at frame `index` of the recording at `path` (`lines` = its frames.jsonl).

    `limb_radius`: how far vitpose_depth placed joints behind the measured surface when the
    recording was made (cfg.estimator.vitpose_depth.limb_radius_m)."""
    t_start = time.monotonic()
    p = params
    cams = _camera_matrices(meta)
    if not cams:
        raise MeshError("No calibrated cameras in this recording.")
    window = (p.depth_min if p.depth_min is not None else (meta.get("depth_window") or [0.2, 6])[0],
              p.depth_max if p.depth_max is not None else (meta.get("depth_window") or [0.2, 6])[1])
    views = _select_views(lines, index, p, cams)
    if not views:
        raise MeshError("No depth frames from the selected cameras around this frame.")
    bones_def = meta["hello"].get("bones", [])

    # 1. Segment every view down to the leg (and sample its points + normals).
    progress("segmenting", 0.0)
    entries, joints_used = [], set()  # {view, seg, cam (pose, maybe corrected), pts, nrm}
    for k, view in enumerate(views):
        c = cams[view.cam]
        raw = cv2.imread(str(path / view.depth), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        depth_m = raw.astype(np.float32) * 0.001
        bones = []
        if p.region == "pose":
            bones = _leg_bones(view.joints, p.leg, bones_def)
            if not bones:
                continue
            joints_used |= {k2 for k2 in view.joints if k2 in LEG_NAMES}
        seg = _segment(depth_m, view, c, p, bones, window)
        if not seg.any():
            continue
        world = _deproject(seg, c)
        nrm, nok = _normals_grid(world, seg > 0)
        step = max(1, int(nok.sum()) // 20000)
        entries.append({"view": view, "seg": np.round(seg * 1000).astype(np.uint16), "cam": c, "orig_T": c["T"],
                        "pts": world[nok][::step], "nrm": nrm[nok][::step]})
        progress("segmenting", (k + 1) / len(views))
    if not entries:
        if p.region == "pose":
            raise MeshError("No leg joints detected in this time window (or the leg region was empty). "
                            "Move to a frame with a detected leg, or use region = depth window.")
        raise MeshError("Nothing inside the depth window.")
    warnings, support, align_stats = [], None, None

    # 2. Align. Each camera's frame nearest the current time is snapped onto the world
    #    camera's (absorbing calibration error, with the slide along the leg locked); every
    #    other frame is snapped onto its own camera's anchor (absorbing leg motion).
    if p.align and len(entries) > 1:
        align_stats, failed = _align(entries, lines[index]["t"], progress)
        if failed > len(entries) / 2:
            warnings.append(f"Alignment failed for {failed} of {len(entries) - 1} frames (kept as recorded).")

    # 3. Cut away what the leg rests on.
    pts = np.concatenate([e["pts"] for e in entries])
    if p.support:
        nrm = np.concatenate([e["nrm"] for e in entries])
        support = _support_plane(pts, nrm, views[len(views) // 2].up)
        if support is not None:
            n, d = support
            cut = d + p.support_margin_cm / 100
            for e in entries:
                world = _deproject(_metres(e), e["cam"])
                e["seg"][(world @ n.astype(np.float32)) < cut] = 0
            pts = pts[pts @ n >= cut]
            if not len(pts):
                raise MeshError("Everything lies on the support surface; turn 'cut support' off.")

    # 3b. Keep one side of the knee.
    section = None
    if p.section in ("above", "below"):
        section = _knee_section(entries, pts, p, meta["hello"].get("backend"), limb_radius)
        u, t_cut = np.asarray(section["axis"]), section["t"]
        for e in entries:
            s_ = (_deproject(_metres(e), e["cam"]) - section["centre"]) @ u.astype(np.float32)
            e["seg"][(s_ < t_cut) if p.section == "above" else (s_ > t_cut)] = 0
        s_pts = (pts - section["centre"]) @ u
        pts = pts[(s_pts >= t_cut) if p.section == "above" else (s_pts <= t_cut)]
        if len(pts) < 100 or not any(e["seg"].any() for e in entries):
            raise MeshError("Nothing left on that side of the knee. Check KNEE SHIFT, or use WHOLE LEG.")
        warnings.extend(section.pop("warnings"))

    # 4. Volume around the leg points (robust bounds: ignore the 0.5 % most extreme points).
    margin = p.trunc_mm / 1000 + 2 * p.voxel_mm / 1000
    lo = np.percentile(pts, 0.5, axis=0) - margin
    hi = np.percentile(pts, 99.5, axis=0) + margin
    voxel = p.voxel_mm / 1000
    n_vox = np.prod(np.ceil((hi - lo) / voxel) + 1)
    if n_vox > MAX_VOXELS:
        voxel *= float((n_vox / MAX_VOXELS) ** (1 / 3)) * 1.01
        warnings.append(f"Region is large: voxel size raised to {voxel * 1000:.1f} mm.")
    vol = _Volume(lo, hi, voxel, max(p.trunc_mm / 1000, 2 * voxel))
    vol.restrict(_deproject(_metres(e), e["cam"])[e["seg"] > 0] for e in entries)

    # 5. Fuse.
    for k, e in enumerate(entries):
        bgr = cv2.imread(str(path / e["view"].color), cv2.IMREAD_COLOR)
        rgb = None if bgr is None or bgr.shape[:2] != e["seg"].shape else bgr[:, :, ::-1]
        depth = _metres(e)
        vol.integrate(depth, rgb, e["cam"], _view_weights(depth, e["cam"]) if p.blend else None)
        progress("fusing", (k + 1) / len(entries))

    # 6. Surface, cleanup, smoothing.
    progress("meshing", 0.0)
    verts, faces, colors = vol.extract()
    if not len(faces):
        raise MeshError("The fused surface came out empty (nothing seen consistently).")
    verts, faces, colors = _compact(verts, faces, colors)
    if p.largest:
        verts, faces, colors = _largest_piece(verts, faces, colors)
    holes = None
    if p.fill_holes and p.max_hole_cm > 0:
        progress("filling holes", 0.5)
        verts, faces, colors, holes = fill_holes(verts, faces, colors, p.max_hole_cm / 100)
    if p.smooth:
        progress("smoothing", 0.5)
        verts = _taubin(verts, faces, p.smooth)
    normals, fn = _normals(verts, faces)
    # Face the cameras: flip the winding if most of the surface points away from them.
    centres = np.array([e["cam"]["T"][:3, 3] for e in entries]).mean(0)
    if np.sum(((centres - verts) * normals).sum(1) > 0) < len(verts) / 2:
        faces, normals, fn = faces[:, ::-1].copy(), -normals, -fn
    area = float(np.linalg.norm(fn, axis=1).sum() / 2)

    per_cam = {}
    for e in entries:
        per_cam[e["view"].cam] = per_cam.get(e["view"].cam, 0) + 1
    t0 = lines[index]["t"] - lines[0]["t"]
    stats = {
        "method": "tsdf", "frame": index, "seq": lines[index]["seq"], "time_s": round(t0, 2),
        "views": per_cam, "voxel_mm": round(voxel * 1000, 2), "grid": [int(n) for n in vol.shape],
        "vertices": int(len(verts)), "faces": int(len(faces)), "area_cm2": round(area * 1e4, 1),
        "joints": sorted(joints_used), "align": align_stats, "holes": holes,
        "section": None if section is None else {
            "mode": p.section, "knee": [round(float(x), 4) for x in section["knee"]],
            "axis": [round(float(x), 4) for x in section["axis"]], "shift_cm": p.knee_shift_cm,
            "knees_used": section["knees_used"], "knees_seen": section["knees_seen"]},
        "support": None if support is None else {"normal": [round(float(x), 4) for x in support[0]],
                                                  "d": round(support[1], 4)},
        "build_s": round(time.monotonic() - t_start, 2),
        "params": asdict(p), "warnings": warnings,
    }
    log.info("TSDF mesh: %s", {k: v for k, v in stats.items() if k != "params"})
    return Mesh(verts.astype(np.float32), faces.astype(np.uint32), normals, colors, stats)
