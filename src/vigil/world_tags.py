"""Rig calibration from ArUco tags fixed in the scene (used when `testing.probe` is 0).

The tags never move, so unlike the probe there's no need to hold anything still or to have
the cameras capture at the same instant: every camera just watches the tags for a couple of
seconds. Per camera and tag, the per-frame PnP corners (camera frame) are reduced to their
median, which removes jitter and the occasional mirror-image flip of a single-tag pose.

Camera k is then placed by one rigid fit (Kabsch) of its tag corners onto the world camera's,
over every tag both cameras see. With 3+ tags spread out, the rotation comes from where the
tags are relative to each other, not from each small tag's orientation (the noisiest part of
a tag pose: 1° of tag tilt at 1.5 m moves the camera by 2.6 cm). The tags don't need a known
layout, and they don't need to be coplanar; coplanar is fine as long as they aren't in a line.

Each tag's distance can come from PnP or from the measured depth inside it (the tag slid along
its viewing ray). Neither is always right: PnP to a small tag seen at an angle runs 1-4 cm long,
and a RealSense whose depth calibration has drifted reads a few percent short, which shrinks
that camera's whole layout and makes the outer tags misfit. A depth scale error is a single
factor per camera, and with 3+ tags spread out it shows as the two layouts differing in size,
so it can be measured (a similarity fit) and taken out: "depth, rescaled" scales the other
camera's depth to the world camera's. With `use_depth` every candidate (each camera: depth or
PnP, plus depth rescaled) is fitted and the one where the most tags agree, then the smallest
error, is kept; `distance_from` and `depth_correction` in the result say which. The depth / PnP
ratio per camera is reported too: far from 1 on one camera only means its depth (or intrinsics)
is off; on both cameras, `marker_length_m` is.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np

from .config import WorldTagsConfig
from .probe.aruco import estimate_marker_pose, make_detector
from .probe.geometry import marker_object_points

MAX_DEPTH_CORRECTION = 0.12  # a rescale of more than 12% isn't a depth calibration drift
MAX_TAG_RESIDUAL_MM = 10.0  # a tag whose corners don't fit the others by this much is dropped
SCALE_WARN = 0.05  # depth vs PnP distance disagreeing by more than 5% → marker length is off
SCALE_TRUST = 0.3  # beyond 30% the depth probably hit something else: keep the PnP distance


@dataclass
class TagInput:
    serial: str
    K: np.ndarray
    dist: np.ndarray
    image: np.ndarray  # RGB
    depth: np.ndarray | None  # metres, aligned to the image


class TagRigCalibrator:
    """Same interface as rig.RigCalibrator: add() per frame, progress(), finished(), solve()."""

    source = "tags"

    def __init__(self, cfg: WorldTagsConfig, world_serial: str, target: int,
                 exclude_ids: set[int] = frozenset(), timeout_s: float = 20.0):
        self.cfg, self.world_serial, self.target = cfg, world_serial, target
        self.ids = None if cfg.ids is None else set(cfg.ids)
        self.exclude = set(exclude_ids) if cfg.ids is None else set()
        self.timeout_s = timeout_s
        self.started = time.monotonic()
        self.object_pts = marker_object_points(cfg.marker_length_m)
        self.corners: dict[str, dict[int, list[np.ndarray]]] = {}  # serial → tag → [4x3 camera frame, PnP]
        self.depth_corners: dict[str, dict[int, list[np.ndarray]]] = {}  # same, at the measured depth
        self.scale: dict[str, list[float]] = {}  # serial → depth / PnP distance per tag sighting
        self.frames: dict[str, int] = {}
        self._local = threading.local()  # one detector per thread, as in the probe tracker
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="world-tags")

    def wanted(self, marker_id: int) -> bool:
        return marker_id not in self.exclude if self.ids is None else marker_id in self.ids

    def detect(self, c: TagInput) -> tuple[dict[int, tuple[np.ndarray, np.ndarray | None]], list[float]]:
        """Corners (4x3, camera frame) of every wanted tag in one image, from PnP and (when
        there's depth) slid to the measured depth; and the depth / PnP ratios."""
        gray = c.image if c.image.ndim == 2 else cv2.cvtColor(c.image, cv2.COLOR_RGB2GRAY)
        detector = getattr(self._local, "detector", None)
        if detector is None:
            detector = self._local.detector = make_detector(self.cfg.dictionary)
        found, ids, _ = detector.detectMarkers(gray)
        out, ratios = {}, []
        if ids is None:
            return out, ratios
        for quad, marker_id in zip(found, ids.flat):
            marker_id = int(marker_id)
            if not self.wanted(marker_id):
                continue
            pose = estimate_marker_pose(quad, self.cfg.marker_length_m, c.K, c.dist)
            if pose is None:
                continue
            R = cv2.Rodrigues(pose[0])[0]
            pts, at_depth = self.object_pts @ R.T + pose[1], None
            if c.depth is not None and pose[1][2] > 0:
                z = _depth_inside(c.depth, quad.reshape(4, 2))
                if z is not None:
                    ratio = z / pose[1][2]
                    ratios.append(ratio)
                    if abs(ratio - 1) < SCALE_TRUST:
                        at_depth = pts * ratio  # same pixels, measured distance
            out[marker_id] = (pts, at_depth)
        return out, ratios

    def add(self, inputs: list[TagInput]) -> None:
        for c, (tags, ratios) in zip(inputs, self._pool.map(self.detect, inputs)):
            if not tags:
                continue
            self.frames[c.serial] = self.frames.get(c.serial, 0) + 1
            per_tag = self.corners.setdefault(c.serial, {})
            per_tag_d = self.depth_corners.setdefault(c.serial, {})
            for marker_id, (pts, at_depth) in tags.items():
                per_tag.setdefault(marker_id, []).append(pts)
                if at_depth is not None:
                    per_tag_d.setdefault(marker_id, []).append(at_depth)
            self.scale.setdefault(c.serial, []).extend(ratios)

    def shared(self, serial: str) -> list[int]:
        """Tags seen often enough by both the world camera and `serial`."""
        n = self.cfg.min_views
        world, other = self.corners.get(self.world_serial, {}), self.corners.get(serial, {})
        return sorted(i for i in world.keys() & other.keys()
                      if len(world[i]) >= n and len(other[i]) >= n)

    def progress(self) -> dict[str, int]:
        """Frames that count for each camera: both it and the world camera saw a shared tag."""
        world = self.frames.get(self.world_serial, 0)
        return {s: min(world, n) if self.shared(s) else 0
                for s, n in self.frames.items() if s != self.world_serial}

    def finished(self, serials: list[str]) -> bool:
        p = self.progress()
        timed_out = time.monotonic() - self.started > self.timeout_s
        return timed_out or all(p.get(s, 0) >= self.target for s in serials)

    def tags_seen(self) -> dict[str, list[int]]:
        return {s: sorted(t) for s, t in self.corners.items()}

    def _layout(self, serial: str, ids: list[int], source: str) -> dict[int, np.ndarray]:
        """Median corners per tag; "depth" uses the depth-slid sightings where there are enough."""
        out = {}
        for i in ids:
            d = self.depth_corners.get(serial, {}).get(i, [])
            use = d if source == "depth" and len(d) >= max(3, self.cfg.min_views // 2) else self.corners[serial][i]
            out[i] = np.median(use, axis=0)
        return out

    @staticmethod
    def _fit(world: dict, other: dict, ids: list[int]):
        """Rigid fit, dropping the worst tag while it misfits by more than MAX_TAG_RESIDUAL_MM."""
        ids, rejected = list(ids), []
        while True:
            T = kabsch(np.concatenate([other[i] for i in ids]), np.concatenate([world[i] for i in ids]))
            res = {i: float(np.sqrt(np.mean(np.sum((other[i] @ T[:3, :3].T + T[:3, 3] - world[i]) ** 2, 1)))) * 1000
                   for i in ids}
            worst = max(ids, key=res.get)
            if len(ids) <= 2 or res[worst] <= MAX_TAG_RESIDUAL_MM:
                break
            ids.remove(worst)
            rejected.append(worst)
        rms = float(np.sqrt(np.mean([res[i] ** 2 for i in ids])))
        return T, ids, rejected, rms

    def solve(self, serial: str) -> tuple[np.ndarray, dict] | None:
        shared = self.shared(serial)
        if not shared:
            return None
        label = {"depth": "depth", "pnp": "tag pose", "scaled": "depth, rescaled"}
        combos = ([("depth", "depth"), ("depth", "scaled"), ("depth", "pnp"), ("pnp", "depth"), ("pnp", "pnp")]
                  if self.cfg.use_depth else [("pnp", "pnp")])
        fits = []
        for sw, so in combos:  # in order of preference on a tie
            world = self._layout(self.world_serial, shared, sw)
            other = self._layout(serial, shared, "depth" if so == "scaled" else so)
            k = 1.0
            if so == "scaled":
                if len(shared) < 3:
                    continue
                ids = shared
                for _ in range(2):  # scale from the tags that fit, then refit
                    k = similarity_scale(np.concatenate([other[i] for i in ids]),
                                         np.concatenate([world[i] for i in ids]))
                    if abs(k - 1) > MAX_DEPTH_CORRECTION:
                        break
                    _, ids, _, _ = self._fit(world, {i: other[i] * k for i in shared}, shared)
                if abs(k - 1) > MAX_DEPTH_CORRECTION:
                    continue
                other = {i: p * k for i, p in other.items()}
            T, ids, rejected, rms = self._fit(world, other, shared)
            fits.append({"world": sw, "other": so, "k": k, "T": T, "ids": ids, "rejected": rejected,
                         "rms": rms, "world_pts": world})
        best = min(fits, key=lambda f: (len(f["rejected"]), round(f["rms"] * 2) / 2))
        T, ids, rejected, rms, world = best["T"], best["ids"], best["rejected"], best["rms"], best["world_pts"]
        jitter = {i: _jitter_mm(self.corners[serial][i]) for i in ids}
        centres = np.array([world[i].mean(0) for i in ids])
        spread = np.linalg.svd(centres - centres.mean(0), compute_uv=False) if len(ids) > 1 else np.zeros(3)
        scale = {s: float(np.median(self.scale[s])) for s in (self.world_serial, serial) if self.scale.get(s)}
        distance_from = {self.world_serial: label[best["world"]], serial: label[best["other"]]}
        warnings = []
        if len(ids) < 3 or spread[1] / np.sqrt(len(ids)) < 0.03:
            warnings.append("fewer than 3 shared tags spread out (not in a line): the rotation "
                            "comes from single-tag orientation, which is the least accurate part")
        bad = {s: r for s, r in scale.items() if abs(r - 1) > SCALE_WARN}
        if len(bad) == len(scale) and bad:
            warnings.append(f"measured depth / tag distance = {', '.join(f'{r:.3f}' for r in bad.values())} on "
                            "every camera: check world_tags.marker_length_m (the printed black square)")
        elif bad:
            s_, r = next(iter(bad.items()))
            warnings.append(f"camera {s_}: measured depth / tag distance = {r:.3f} (the other camera "
                            "agrees): its depth calibration is probably off")
        meta = {
            "source": "tags",
            "samples": min(self.frames.get(self.world_serial, 0), self.frames.get(serial, 0)),
            "tags": ids,
            "rejected_tags": rejected,
            "rms_mm": round(rms, 2),  # how well the tag corners of both cameras fit one transform
            "distance_from": distance_from,  # per camera: measured depth or tag pose (PnP)
            # The other camera's depth was multiplied by this to match the world camera's
            # (1 = not rescaled). Away from 1: that camera's depth calibration has drifted.
            "depth_correction": round(best["k"], 4),
            "fits": [{"distance_from": {self.world_serial: label[f["world"]], serial: label[f["other"]]},
                      "tags": f["ids"], "rms_mm": round(f["rms"], 2), "depth_correction": round(f["k"], 4)}
                     for f in fits],
            "tag_jitter_mm": round(float(np.median([jitter[i] for i in ids])), 2),
            "depth_scale": {s: round(r, 4) for s, r in scale.items()},
            "warnings": warnings,
            "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return T, meta

    @staticmethod
    def summary(serial: str, meta: dict) -> str:
        text = (f"{serial}: tags {', '.join(map(str, meta['tags']))} · fit {meta['rms_mm']} mm "
                f"over {meta['samples']} frames")
        pnp = [s for s, src in (meta.get("distance_from") or {}).items() if src == "tag pose"]
        if pnp:
            text += f" · distance from tag pose for {', '.join(pnp)}"
        k = meta.get("depth_correction", 1.0)
        if abs(k - 1) > 0.005:
            text += f" · {serial}'s depth reads {abs(1 - 1 / k) * 100:.1f}% {'short' if k > 1 else 'long'}, corrected"
        if meta["rejected_tags"]:
            text += f" (dropped tag {', '.join(map(str, meta['rejected_tags']))})"
        if meta["warnings"]:
            text += " — " + "; ".join(meta["warnings"])
        return text

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def similarity_scale(src: np.ndarray, dst: np.ndarray) -> float:
    """Scale s of the best dst ≈ s·R·src + t (Umeyama). src is in its camera's frame, so s is
    how much that camera's distances must be multiplied by to match dst's."""
    sc, dc = src - src.mean(0), dst - dst.mean(0)
    u, d, vt = np.linalg.svd(dc.T @ sc / len(src))
    sign = np.sign(np.linalg.det(u @ vt))
    return float((d[0] + d[1] + sign * d[2]) / np.mean(np.sum(sc ** 2, 1)))


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 rigid transform T minimising |T·src − dst| for matched (N, 3) points."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cd - R @ cs
    return T


def _depth_inside(depth: np.ndarray, quad: np.ndarray, shrink: float = 0.5) -> float | None:
    """Median depth over the tag's inner region (edges often hit the background), as the
    depth at its centre: the inner quad is symmetric about the centre on a flat tag."""
    centre = quad.mean(0)
    inner = (centre + shrink * (quad - centre)).round().astype(np.int32)
    x0, y0 = np.maximum(inner.min(0), 0)
    x1, y1 = np.minimum(inner.max(0) + 1, [depth.shape[1], depth.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, inner - [x0, y0], 1)
    vals = depth[y0:y1, x0:x1][(mask > 0) & (depth[y0:y1, x0:x1] > 0)]
    return float(np.median(vals)) if vals.size >= 3 else None


def _jitter_mm(samples: list[np.ndarray]) -> float:
    """Median distance of a tag's per-frame centre from its median centre."""
    c = np.array([s.mean(0) for s in samples])
    return float(np.median(np.linalg.norm(c - np.median(c, 0), axis=1))) * 1000
