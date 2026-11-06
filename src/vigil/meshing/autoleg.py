"""Automatic leg prompts: which leg, the thigh, the knee and the hip, without clicks.

Two Sapiens2 models (Meta, 2026; open weights, in `transformers`) look at each camera's colour
image:

* Body-part segmentation labels every pixel; among its 29 classes are Left/Right_Upper_Leg and
  Left/Right_Lower_Leg. On bare skin that gives the leg's side and the knee (where the upper and
  lower leg meet), and the hip where the thigh meets the torso or the other thigh. Where the
  thigh meets clothing (shorts, trousers pushed up to the knee) that's a hem, not the hip: then
  the hip comes from the pose keypoint, and + prompts also go along the covered thigh toward it.
* Pose (308 keypoints; 9–14 are the hips, knees and ankles) is the fallback where the
  segmentation has no leg classes. Trousers do that: they come out as Lower_Clothing, with no
  side and no thigh/shin split.

Several frames around the chosen one and every camera vote on the knee and hip. Each camera's
votes must be steady over time (jitter), and the cameras must agree with each other
(disagreement); the consensus is the mean of the cameras' medians. Votes are compared along the
leg's axis only: the cameras see the leg from opposite sides, so their surface points at the knee
are a leg's thickness apart even when they agree on where it is.

The result is the prompts the clicks make (legmask.Prompts: + leg, − not leg, knee, hip), so SAM 2
tracking, the depth filters, the support cut, the knee/hip cuts and every reconstruction method
run unchanged, and a person can correct an automatic result with clicks.
"""

from __future__ import annotations

import bisect
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .legmask import _cam_frames
from .tsdf import MeshError, _camera_matrices, _deproject, _normals_grid, _support_plane, _up

log = logging.getLogger(__name__)

DEFAULT_SIZE = "0.4b"  # facebook/sapiens2-{seg,pose}-<size>: 0.4b, 0.8b, 1b or 5b
PART_NAMES = ("Background Apparel Eyeglass Face_Neck Hair Left_Foot Left_Hand Left_Lower_Arm Left_Lower_Leg "
              "Left_Shoe Left_Sock Left_Upper_Arm Left_Upper_Leg Lower_Clothing Right_Foot Right_Hand "
              "Right_Lower_Arm Right_Lower_Leg Right_Shoe Right_Sock Right_Upper_Arm Right_Upper_Leg Torso "
              "Upper_Clothing Lower_Lip Upper_Lip Lower_Teeth Upper_Teeth Tongue").split()  # sapiens2 docs/SEG.md
UPPER = {"left": PART_NAMES.index("Left_Upper_Leg"), "right": PART_NAMES.index("Right_Upper_Leg")}
LOWER = {"left": PART_NAMES.index("Left_Lower_Leg"), "right": PART_NAMES.index("Right_Lower_Leg")}
TORSO = PART_NAMES.index("Torso")
CLOTHING = (PART_NAMES.index("Lower_Clothing"), PART_NAMES.index("Apparel"))
KEYPOINTS = {"left": {"hip": 9, "knee": 11, "ankle": 13}, "right": {"hip": 10, "knee": 12, "ankle": 14}}
OTHER = {"left": "right", "right": "left"}

MIN_PART_PX = 1500      # thigh pixels for the segmentation to count in a camera
MIN_KP_SCORE = 0.5      # keypoint confidence for the pose fallback
MIN_HIP_EDGE_PX = 150   # thigh–torso boundary pixels for the segmentation's hip to count
NOT_LEG_MIN_PX = 40     # a − prompt from pose must be this far (px) from the chosen leg's bones
OUTLIER_M = 0.06        # a vote further than this along the leg from its camera's median is dropped
JITTER_M = 0.025        # a camera's votes: median distance from their median, at most
DISAGREE_M = 0.08       # the cameras' medians: furthest apart along the leg, at most
SAME_LEG_M = 0.2        # cameras' medians further apart than this across the leg: different legs
THIGH_M = (0.2, 0.7)    # plausible knee–hip distance along the leg

_lock = threading.Lock()
_models: dict[str, tuple] = {}
_cache: OrderedDict = OrderedDict()  # (recording, colour file, size) -> analysis
CACHE_SIZE = 256


class AutoError(MeshError):
    """The automatic prompts couldn't be trusted for this frame; `str()` says why."""


def _load(size: str):
    if size not in _models:
        try:
            import torch
            from transformers import AutoImageProcessor, Sapiens2ForPoseEstimation, Sapiens2ForSemanticSegmentation
        except ImportError as e:
            raise MeshError("Sapiens2 needs torch and transformers. Run `uv sync`, then restart.") from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        seg_id, pose_id = f"facebook/sapiens2-seg-{size}", f"facebook/sapiens2-pose-{size}"
        log.info("loading %s and %s on %s", seg_id, pose_id, device)
        _models[size] = (
            torch, device, dtype,
            AutoImageProcessor.from_pretrained(seg_id),
            Sapiens2ForSemanticSegmentation.from_pretrained(seg_id, dtype=dtype).to(device).eval(),
            AutoImageProcessor.from_pretrained(pose_id),
            Sapiens2ForPoseEstimation.from_pretrained(pose_id, dtype=dtype).to(device).eval(),
        )
    return _models[size]


def analyse(images: list[np.ndarray], size: str = DEFAULT_SIZE, batch: int = 4) -> list[dict]:
    """Body parts (class ids, H×W) and keypoints (308×2 px, 308 scores) for RGB images."""
    torch, device, dtype, sp, sm, pp, pm = _load(size)
    out = []
    with _lock, torch.inference_mode():
        for k in range(0, len(images), batch):
            ims = images[k:k + batch]
            sizes = [im.shape[:2] for im in ims]
            boxes = [[[0.0, 0.0, float(w), float(h)]] for h, w in sizes]  # the whole image: one person
            x = sp(ims, return_tensors="pt")["pixel_values"].to(device, dtype)
            parts = sp.post_process_semantic_segmentation(sm(pixel_values=x), target_sizes=sizes)
            x = pp(ims, boxes=boxes, return_tensors="pt")["pixel_values"].to(device, dtype)
            pose = pm(pixel_values=x)
            pose.heatmaps = pose.heatmaps.float()
            kps = pp.post_process_pose_estimation(pose, boxes=boxes)
            out += [{"parts": p.cpu().numpy().astype(np.uint8), "kp": q[0]["keypoints"].cpu().numpy(),
                     "score": q[0]["scores"].cpu().numpy()} for p, q in zip(parts, kps)]
    return out


# ---------------------------------------------------------------------- views
@dataclass
class View:
    cam: int
    line: int            # frames.jsonl line the images come from
    rgb: np.ndarray
    depth: np.ndarray    # metres
    c: dict              # camera matrices (mesh._camera_matrices)
    a: dict              # analyse() output


def load_views(path: Path, meta: dict, lines: list, indices: list[int], size: str = DEFAULT_SIZE,
               analyse_fn=None) -> dict[tuple[int, int], View]:
    """Every calibrated camera's frame nearest each replay index, analysed (cached).
    Keyed by (camera, requested index)."""
    analyse_fn = analyse_fn or (lambda ims: analyse(ims, size))
    cams = _camera_matrices(meta)
    t = [ln["t"] for ln in lines]
    todo, found = [], []
    for cam in sorted(cams):
        frames = _cam_frames(lines, cam)
        if not frames:
            continue
        ft = [t[li] for li, _ in frames]
        for idx in indices:
            j = bisect.bisect_left(ft, t[idx])
            j = min((k for k in (j - 1, j) if 0 <= k < len(frames)), key=lambda k: abs(ft[k] - t[idx]))
            li, f = frames[j]
            bgr = cv2.imread(str(path / f["color"]), cv2.IMREAD_COLOR)
            raw = cv2.imread(str(path / f["depth"]), cv2.IMREAD_UNCHANGED)
            if bgr is None or raw is None:
                continue
            key = (str(path), f["color"], size)
            found.append((cam, idx, li, np.ascontiguousarray(bgr[:, :, ::-1]), raw.astype(np.float32) * 0.001, key))
            if key not in _cache:
                todo.append(found[-1])
    if todo:
        for item, a in zip(todo, analyse_fn([it[3] for it in todo])):
            _cache[item[5]] = a
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    views = {}
    for cam, idx, li, rgb, depth, key in found:
        a = _cache.get(key)
        if a is not None:
            _cache.move_to_end(key)
            views[(cam, idx)] = View(cam, li, rgb, depth, cams[cam], a)
    return views


# ---------------------------------------------------------------------- geometry
def _world(v: View, uv: np.ndarray) -> np.ndarray:
    """World points of pixels (N×2, x/y) with depth; rows without depth are dropped."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, v.depth.shape[1] - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, v.depth.shape[0] - 1)
    z = v.depth[y, x]
    ok = z > 0
    c = v.c
    p = np.stack([(x[ok] - c["cx"]) * z[ok] / c["fx"], (y[ok] - c["cy"]) * z[ok] / c["fy"], z[ok]], 1)
    return p @ c["T"][:3, :3].T + c["T"][:3, 3]


def _pixel_point(v: View, x: float, y: float, r: int = 3) -> np.ndarray | None:
    """World point at a pixel: median depth over a (2r+1)² patch. None without depth."""
    h, w = v.depth.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    patch = v.depth[max(yi - r, 0):yi + r + 1, max(xi - r, 0):xi + r + 1]
    patch = patch[patch > 0]
    if not len(patch):
        return None
    z, c = float(np.median(patch)), v.c
    p = np.array([(xi - c["cx"]) * z / c["fx"], (yi - c["cy"]) * z / c["fy"], z])
    return c["T"][:3, :3] @ p + c["T"][:3, 3]


def project(c: dict, X: np.ndarray) -> tuple[float, float, float]:
    q = c["T_inv"][:3, :3] @ np.asarray(X, float) + c["T_inv"][:3, 3]
    return q[0] * c["fx"] / q[2] + c["cx"], q[1] * c["fy"] / q[2] + c["cy"], q[2]


def _boundary(a: np.ndarray, b: np.ndarray, r: int = 4) -> np.ndarray:
    """Pixels (N×2, x/y) where regions a and b touch (both dilated by r)."""
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    both = (cv2.dilate(a.astype(np.uint8), k) > 0) & (cv2.dilate(b.astype(np.uint8), k) > 0)
    y, x = np.nonzero(both)
    return np.stack([x, y], 1).astype(float)


def _spread(pts: np.ndarray, n: int) -> list[list[float]]:
    """n well-spread points (farthest-point sampling from the one nearest the centroid)."""
    if not len(pts):
        return []
    pick = [int(np.argmin(((pts - pts.mean(0)) ** 2).sum(1)))]
    d = ((pts - pts[pick[0]]) ** 2).sum(1)
    for _ in range(min(n, len(pts)) - 1):
        pick.append(int(d.argmax()))
        d = np.minimum(d, ((pts - pts[pick[-1]]) ** 2).sum(1))
    return [pts[k].tolist() for k in pick]


def _inner(mask: np.ndarray, depth: np.ndarray, r: int = 7) -> np.ndarray:
    """Pixels (x/y) well inside a mask that have depth."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    y, x = np.nonzero(m & (depth > 0))
    return np.stack([x, y], 1).astype(float)


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    s = np.clip((p - a) @ ab / max(float(ab @ ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + s * ab)))


def _inside(p: np.ndarray, shape: tuple, margin: float) -> bool:
    return margin <= p[0] < shape[1] - margin and margin <= p[1] < shape[0] - margin


# ---------------------------------------------------------------------- one camera
def read_leg(v: View, side: str) -> dict | None:
    """What one camera says about `side`'s leg: prompts (pixels), knee / hip (world, if seen),
    thigh points (world) and a score. None if it can't see that leg."""
    parts, kp, sc = v.a["parts"], v.a["kp"], v.a["score"]
    upper, lower = parts == UPPER[side], parts == LOWER[side]
    other = (parts == UPPER[OTHER[side]]) | (parts == LOWER[OTHER[side]])
    if upper.sum() >= MIN_PART_PX:  # bare leg: the segmentation says it all
        inner = _inner(upper, v.depth)
        if len(inner) < 50:
            return None
        knee_px, knee = _boundary(upper, lower), None
        if len(knee_px):
            w = _world(v, knee_px)
            knee = np.median(w, 0) if len(w) >= 20 else None
        knee_uv = np.median(knee_px, 0) if len(knee_px) else None
        hip_px, hip, hip_uv = _boundary(upper, (parts == TORSO) | (parts == UPPER[OTHER[side]])), None, None
        if len(hip_px) >= MIN_HIP_EDGE_PX:
            w = _world(v, hip_px)
            # A speck of torso / other thigh labelled next to the knee isn't the hip.
            if len(w) >= 20 and (knee is None or np.linalg.norm(np.median(w, 0) - knee) >= THIGH_M[0]):
                hip, hip_uv = np.median(w, 0), np.median(hip_px, 0)
        i = KEYPOINTS[side]
        H, sh = kp[i["hip"]], float(sc[i["hip"]])
        if hip is None and sh >= MIN_KP_SCORE and _inside(H, parts.shape, 12):
            hip = _pixel_point(v, *H)
            hip_uv = H.copy() if hip is not None else None
        leg, thigh = _spread(inner, 3), list(_world(v, inner[:: max(1, len(inner) // 400)]))
        if knee_uv is not None and knee is not None and sh >= 0.3:
            # The segmentation may only see a bare strip above the knee (clothing covers the rest
            # of the thigh): + prompts along the covered thigh too, toward the hip keypoint.
            for s in (0.3, 0.55, 0.8):
                p = knee_uv + s * (H - knee_uv)
                if not _inside(p, parts.shape, 8) or other[int(p[1]), int(p[0])]:
                    continue
                if parts[int(p[1]), int(p[0])] not in (UPPER[side], *CLOTHING):
                    continue
                w = _pixel_point(v, *p)
                if w is not None and np.linalg.norm(w - knee) < 0.6:
                    leg.append(p.tolist())
                    thigh.append(w)
        not_leg = _spread(_inner(other, v.depth), 2) if other.sum() >= 800 else []
        return {"source": "parts", "score": min(upper.sum() / 20000, 1.0), "leg": leg, "not": not_leg,
                "knee": knee, "knee_uv": knee_uv, "hip": hip, "hip_uv": hip_uv, "thigh": np.array(thigh)}

    # Pose: knee plus the hip or the ankle, for the thigh's direction and the + prompts.
    i = KEYPOINTS[side]
    K, H, A = kp[i["knee"]], kp[i["hip"]], kp[i["ankle"]]
    sk, sh, sa = float(sc[i["knee"]]), float(sc[i["hip"]]), float(sc[i["ankle"]])
    if sk < MIN_KP_SCORE or not _inside(K, parts.shape, 4):
        return None
    knee = _pixel_point(v, *K)
    if knee is None:
        return None
    use_h, use_a = sh >= 0.3, sa >= MIN_KP_SCORE
    if not (use_h or use_a):
        return None
    leg, thigh = [], []
    for end, ts, ok in ((H, (0.3, 0.55, 0.8), use_h), (A, (0.5,), use_a)):
        for s in ts if ok else ():
            p = K + s * (end - K)
            w = _pixel_point(v, *p) if _inside(p, parts.shape, 8) else None
            if w is not None and np.linalg.norm(w - knee) < 0.45:  # on the leg, not the floor behind it
                leg.append(p.tolist())
                if end is H:
                    thigh.append(w)
    if not leg:
        return None
    bones = [(K, H)] * use_h + [(K, A)] * use_a
    not_leg = []
    for name in ("knee", "ankle"):  # the other leg, where it's clearly apart from this one
        q, s = kp[KEYPOINTS[OTHER[side]][name]], float(sc[KEYPOINTS[OTHER[side]][name]])
        if s >= MIN_KP_SCORE and _inside(q, parts.shape, 8) and min(_seg_dist(q, a, b) for a, b in bones) >= NOT_LEG_MIN_PX:
            not_leg.append(q.tolist())
    hip = _pixel_point(v, *H) if sh >= MIN_KP_SCORE and _inside(H, parts.shape, 12) else None
    return {"source": "pose", "score": sk * max(sh, sa), "leg": leg, "not": not_leg,
            "knee": knee, "knee_uv": K.copy(), "hip": hip, "hip_uv": H.copy() if hip is not None else None,
            "thigh": np.array(thigh) if thigh else knee[None]}


# ---------------------------------------------------------------------- voting
def leg_axis(reads: list[dict]) -> np.ndarray | None:
    """Unit vector knee → hip: from hip votes if there are any, else the thigh points' main
    direction, oriented away from the knees."""
    knees = np.array([r["knee"] for r in reads if r["knee"] is not None])
    hips = np.array([r["hip"] for r in reads if r["hip"] is not None])
    if not len(knees):
        return None
    k = np.median(knees, 0)
    if len(hips):
        d = np.median(hips, 0) - k
        if np.linalg.norm(d) > 0.1:
            return d / np.linalg.norm(d)
    pts = np.concatenate([r["thigh"] for r in reads if len(r["thigh"])])
    if len(pts) < 10:
        return None
    centre = pts.mean(0)
    d = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
    if np.linalg.norm(centre - k) < 0.03:
        return None
    return d if (centre - k) @ d > 0 else -d


def vote(votes: list[tuple[int, np.ndarray]], axis: np.ndarray) -> dict:
    """Consensus of (camera, 3D point) votes along `axis` (positions in metres along it).

    Per camera: the median, dropping votes more than OUTLIER_M from it, and the jitter (median
    distance from it). Across cameras: the consensus is the mean of their medians, `disagree`
    the spread of the medians along the leg and `across` the furthest any two are apart across it."""
    per = {}
    for cam in sorted({c for c, _ in votes}):
        P = np.array([p for c, p in votes if c == cam])
        s = P @ axis
        keep = np.abs(s - np.median(s)) <= OUTLIER_M
        m = float(np.median(s[keep]))
        per[cam] = {"s": m, "jitter": float(np.median(np.abs(s[keep] - m))), "n": int(keep.sum()),
                    "dropped": int((~keep).sum()), "point": np.median(P[keep], 0)}
    meds = np.array([v["s"] for v in per.values()])
    pts = np.array([v["point"] for v in per.values()])
    flat = pts - np.outer(pts @ axis, axis)
    across = max((float(np.linalg.norm(a - b)) for a in flat for b in flat), default=0.0)
    return {"s": float(meds.mean()), "cameras": per, "jitter": max(v["jitter"] for v in per.values()),
            "disagree": float(meds.max() - meds.min()), "across": across, "n": len(votes)}


def check_vote(v: dict, what: str) -> list[str]:
    """Why a vote can't be trusted (raises), or warnings about it."""
    if v["n"] < 2:
        raise AutoError(f"Only {v['n']} {what} estimate(s) around this frame; need two to trust one.")
    if v["jitter"] > JITTER_M:
        cam = max(v["cameras"], key=lambda c: v["cameras"][c]["jitter"])
        raise AutoError(f"The {what} jumps around over time in CAM {cam + 1} "
                        f"({v['jitter'] * 100:.1f} cm; at most {JITTER_M * 100:.1f}).")
    if v["across"] > SAME_LEG_M:
        raise AutoError(f"The cameras' {what} estimates are {v['across'] * 100:.0f} cm apart across the leg: "
                        "probably different legs.")
    if v["disagree"] > DISAGREE_M:
        raise AutoError(f"The cameras disagree on the {what} by {v['disagree'] * 100:.1f} cm along the leg "
                        f"(at most {DISAGREE_M * 100:.0f}).")
    if len(v["cameras"]) == 1:
        return [f"Only CAM {next(iter(v['cameras'])) + 1} sees the {what}; no second camera to confirm it."]
    return []


def support_plane(views: list[View], near: np.ndarray, up: np.ndarray, radius: float = 0.4):
    """The surface the leg rests on, from depth around `near` (world points on the leg)."""
    if not len(near):
        return None
    tree = cKDTree(near)
    pts, nrm = [], []
    for v in views:
        world = _deproject(v.depth, v.c)
        n, ok = _normals_grid(world, v.depth > 0)
        w, nn = world[ok][::7], n[ok][::7]
        keep = tree.query(w, distance_upper_bound=radius)[0] < np.inf
        pts.append(w[keep]); nrm.append(nn[keep])
    if not pts or sum(len(p) for p in pts) < 500:
        return None
    return _support_plane(np.concatenate(pts).astype(float), np.concatenate(nrm).astype(float), up)


def on_support(thigh: np.ndarray, plane) -> float | None:
    """Fraction of thigh points 0–35 cm above the support plane (a leg resting on it)."""
    if plane is None or not len(thigh):
        return None
    n, d = plane
    h = thigh @ n - d
    return float(((h > -0.01) & (h < 0.35)).mean())


def width_knee(points: np.ndarray, knee: np.ndarray, axis: np.ndarray, search: float = 0.15) -> float | None:
    """Cross-check from the leg's shape alone: where along the leg (metres from `knee`, + toward
    the hip) the leg is narrowest within ±search. The thigh tapers to the knee and the calf
    widens below it, so on a bare leg that's at or just below the knee. Rough: a sanity check."""
    s = (points - knee) @ axis
    r = (points - knee) - np.outer(s, axis)
    keep = np.abs(s) <= search + 0.05
    if keep.sum() < 300:
        return None
    s, r = s[keep], r[keep]
    lat = np.linalg.svd(r - r.mean(0), full_matrices=False)[2][0]  # across the leg, as the cameras see it
    x = r @ lat
    edges = np.arange(-search - 0.05, search + 0.05 + 1e-9, 0.01)
    width = np.full(len(edges) - 1, np.nan)
    for k in range(len(width)):
        sel = (s >= edges[k]) & (s < edges[k + 1])
        if sel.sum() >= 20:
            width[k] = np.percentile(x[sel], 95) - np.percentile(x[sel], 5)
    centres = (edges[:-1] + edges[1:]) / 2
    ok = ~np.isnan(width) & (np.abs(centres) <= search)
    if ok.sum() < 8:
        return None
    smooth = np.convolve(np.nan_to_num(width, nan=np.nanmax(width)), np.ones(3) / 3, mode="same")
    return float(centres[ok][np.argmin(smooth[ok])])


def _vote_report(v: dict) -> dict:
    return {"votes": v["n"], "jitter_cm": round(v["jitter"] * 100, 2), "disagree_cm": round(v["disagree"] * 100, 2),
            "cameras": {int(c): {"along_cm": round(x["s"] * 100, 1), "votes": x["n"], "dropped": x["dropped"]}
                        for c, x in v["cameras"].items()}}


# ---------------------------------------------------------------------- prompts
def propose(path: Path, meta: dict, lines: list, index: int, side: str = "auto", size: str = DEFAULT_SIZE,
            vote_window_s: float = 0.3, vote_frames: int = 5, analyse_fn=None) -> dict:
    """Automatic prompts for replay frame `index`, in the clicks' format, plus what they came from.

    Raises AutoError when the leg can't be found or the votes disagree."""
    if side not in ("auto", "left", "right"):
        raise MeshError(f"Unknown leg {side!r}: use left, right or auto.")
    t = [ln["t"] for ln in lines]
    lo = bisect.bisect_left(t, t[index] - vote_window_s)
    hi = bisect.bisect_right(t, t[index] + vote_window_s) - 1
    idxs = [index] + [i for i in np.unique(np.linspace(lo, hi, vote_frames).round().astype(int)).tolist() if i != index]
    views = load_views(path, meta, lines, idxs, size, analyse_fn)
    if not views:
        raise AutoError("No calibrated camera has colour and depth at this frame.")
    cams = _camera_matrices(meta)
    up = _up(lines[index]["frame"], cams)
    here_views = [v for (cam, i), v in views.items() if i == index]

    options = []
    for s in (("left", "right") if side == "auto" else (side,)):
        reads = {key: read_leg(v, s) for key, v in views.items()}
        reads = {key: r for key, r in reads.items() if r is not None}
        here = {cam: r for (cam, i), r in reads.items() if i == index}
        if here:
            options.append((s, reads, here))
    if not options:
        raise AutoError(f"No camera sees the {'leg' if side == 'auto' else side + ' leg'} "
                        "(no thigh in the body-part segmentation and no confident knee keypoint).")

    near = np.concatenate([r["thigh"] for _, reads, _ in options for r in reads.values()])
    plane = support_plane(here_views, near, up)
    scored = []
    for s, reads, here in options:
        rest = on_support(np.concatenate([r["thigh"] for r in here.values()]), plane)
        score = sum(r["score"] for r in here.values()) + (0.5 * rest if rest is not None else 0)
        scored.append((score, s, reads, here, rest))
    scored.sort(key=lambda x: -x[0])
    score, s, reads, here, rest = scored[0]
    warnings = []
    if side == "auto" and len(scored) > 1 and scored[1][0] > 0.8 * score:
        warnings.append(f"The left and right legs scored alike ({scored[0][0]:.2f} vs {scored[1][0]:.2f}); "
                        "set the leg explicitly if this picked the wrong one.")

    axis = leg_axis(list(reads.values()))
    if axis is None:
        raise AutoError("Couldn't tell which way the thigh runs (no hip and too little of the thigh).")
    kv = vote([(key[0], r["knee"]) for key, r in reads.items() if r["knee"] is not None], axis)
    warnings += check_vote(kv, "knee")

    hips = [(key[0], r["hip"]) for key, r in reads.items() if r["hip"] is not None]
    hip_s, hv = None, None
    if hips:
        hv = vote(hips, axis)
        try:
            warnings += check_vote(hv, "hip")
            hip_s = hv["s"]
        except AutoError as e:  # the thigh just isn't cut at the hip
            warnings.append(f"{e} Not cut at the hip.")
        if hip_s is not None and not THIGH_M[0] <= hip_s - kv["s"] <= THIGH_M[1]:
            warnings.append(f"The hip estimate is {(hip_s - kv['s']) * 100:.0f} cm from the knee along the leg; "
                            "not cut at the hip.")
            hip_s = None
    if rest is not None and rest < 0.5:
        warnings.append(f"Only {rest * 100:.0f} % of the thigh rests on the support surface; check it's the right leg.")

    def place(target_s: float, key: str) -> dict | None:
        """A click whose depth lands at `target_s` along the leg: searched along the leg's image
        direction from each camera's own estimate; the camera that gets closest wins."""
        best = None
        for cam, r in here.items():
            if r[key] is None or r[key + "_uv"] is None:
                continue
            v = views[(cam, index)]
            x0, y0, _ = project(v.c, r[key])
            x1, y1, _ = project(v.c, r[key] + 0.05 * axis)
            d = np.array([x1 - x0, y1 - y0])
            if np.linalg.norm(d) < 1e-6:
                continue
            d /= np.linalg.norm(d)
            for step in np.arange(-120, 121, 2.0):
                uv = r[key + "_uv"] + step * d
                w = _pixel_point(v, *uv) if _inside(uv, v.depth.shape, 4) else None
                if w is None or np.linalg.norm(w - r[key]) > 0.3:
                    continue
                err = abs(float(w @ axis) - target_s)
                if best is None or err < best[0]:
                    best = (err, cam, uv)
        if best is None:
            return None
        return {"cam": int(best[1]), "x": round(float(best[2][0]), 1), "y": round(float(best[2][1]), 1),
                "error_cm": round(best[0] * 100, 1)}

    clicks = []
    for cam, r in sorted(here.items()):
        clicks += [{"cam": int(cam), "x": round(x, 1), "y": round(y, 1), "kind": "leg"} for x, y in r["leg"]]
        clicks += [{"cam": int(cam), "x": round(x, 1), "y": round(y, 1), "kind": "not"} for x, y in r["not"]]
    knee_click = place(kv["s"], "knee")
    if knee_click is None:
        raise AutoError("No camera at this frame has a knee estimate to cut at.")
    clicks.append({k: knee_click[k] for k in ("cam", "x", "y")} | {"kind": "knee"})
    hip_click = place(hip_s, "hip") if hip_s is not None else None
    if hip_click is not None:
        clicks.append({k: hip_click[k] for k in ("cam", "x", "y")} | {"kind": "hip"})
    elif hip_s is not None:
        warnings.append("No camera at this frame sees the hip; not cut at the hip.")

    sources = sorted({r["source"] for r in here.values()})
    return {
        "prompts": {"index": index, "clicks": clicks},
        "side": s, "source": sources[0] if len(sources) == 1 else "mixed", "model": f"sapiens2-{size}",
        "cameras": {int(cam): {"source": r["source"], "score": round(r["score"], 3)} for cam, r in here.items()},
        "scores": {o[1]: round(o[0], 3) for o in scored},
        "knee": _vote_report(kv) | {"click_error_cm": knee_click["error_cm"]},
        "hip": None if hip_click is None else _vote_report(hv) | {"length_cm": round((hip_s - kv["s"]) * 100, 1),
                                                                  "click_error_cm": hip_click["error_cm"]},
        "axis": axis.round(4).tolist(), "knee_s": kv["s"],
        "on_support": None if rest is None else round(rest, 3),
        "plane": None if plane is None else {"normal": plane[0].round(4).tolist(), "d": round(float(plane[1]), 4)},
        "warnings": warnings,
    }
