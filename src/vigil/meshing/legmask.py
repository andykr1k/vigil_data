"""Leg masks from clicks (SAM 2.1): the leg region that doesn't depend on the recorded pose.

You click the leg (+), anything that isn't the leg (−) and optionally the knee on one replay
frame, or draw a BOX around the thigh (a SAM box prompt; tsdf.py also crops to it in 3D). SAM 2's video model segments the leg there and tracks it through every frame the mesh
needs, separately for each camera. A camera you didn't click in gets its prompts from a
camera you did: that camera's mask is lifted to 3D with its depth, projected into the other
camera, and the points that camera actually sees (its own depth agrees) become + prompts.
The cameras look at the leg from opposite sides, so a single click usually isn't visible in
the other camera; the whole mask always shares some surface with it.

The masks only say which pixels are leg. tsdf.py still applies the depth window, the edge
filter and the support cut on top, so SAM never has the last word on its own.

Uses torch + transformers (installed by `uv sync`). The weights
(facebook/sam2.1-hiera-small, ~180 MB) download on first use.
"""

from __future__ import annotations

import base64
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .tsdf import MeshError, _camera_matrices

log = logging.getLogger(__name__)

MODEL_ID = "facebook/sam2.1-hiera-small"
MAX_TRACK_FRAMES = 300  # per camera, from the clicked frame to the farthest frame the mesh needs
TRANSFER_POINTS = 5     # + prompts carried into a camera that wasn't clicked
KINDS = {"leg": 1, "not": 0, "knee": None, "hip": None, "box": None}  # knee / hip: where to cut; box: its own prompt
MIN_BOX_PX = 8

_lock = threading.Lock()
_model = None


@dataclass
class Prompts:
    index: int          # replay frame (0-based line of frames.jsonl) the clicks were made on
    clicks: list[dict]  # {"cam", "x", "y" (image pixels), "kind": "leg" | "not" | "knee" | "hip" | "box"}
                        # a box also has "x2", "y2": the opposite corner

    @classmethod
    def parse(cls, d: dict | None, n_lines: int) -> Prompts:
        if not d or not d.get("clicks"):
            raise MeshError("No leg prompts: press AUTO, or click the leg and knee in a camera feed.")
        try:
            index = int(d["index"])
            clicks = [{"cam": int(c["cam"]), "x": float(c["x"]), "y": float(c["y"]), "kind": str(c["kind"])}
                      | ({"x2": float(c["x2"]), "y2": float(c["y2"])} if c.get("kind") == "box" else {})
                      for c in d["clicks"]]
        except (KeyError, TypeError, ValueError) as e:
            raise MeshError("The leg clicks are malformed.") from e
        if not 0 <= index < n_lines or any(c["kind"] not in KINDS for c in clicks):
            raise MeshError("The leg clicks are malformed.")
        if not any(c["kind"] in ("leg", "box") for c in clicks):
            raise MeshError("Add at least one LEG (+) click, or draw a BOX around the thigh.")
        return cls(index, clicks)

    def points(self, cam: int) -> tuple[list[list[float]], list[int]]:
        """This camera's + / − clicks as SAM point prompts."""
        mine = [c for c in self.clicks if c["cam"] == cam and KINDS[c["kind"]] is not None]
        return [[c["x"], c["y"]] for c in mine], [KINDS[c["kind"]] for c in mine]

    def boxes(self) -> dict[int, list[float]]:
        """Each camera's BOX as [x0, y0, x1, y1] (its last one; smaller than MIN_BOX_PX ignored)."""
        out = {}
        for c in self.clicks:
            if c["kind"] == "box":
                x0, x1 = sorted((c["x"], c["x2"]))
                y0, y1 = sorted((c["y"], c["y2"]))
                if x1 - x0 >= MIN_BOX_PX and y1 - y0 >= MIN_BOX_PX:
                    out[c["cam"]] = [x0, y0, x1, y1]
        return out

    def first(self, kind: str) -> dict | None:
        return next((c for c in self.clicks if c["kind"] == kind), None)


def _load():
    global _model
    if _model is None:
        try:
            import torch
            from transformers import Sam2VideoModel, Sam2VideoProcessor
        except ImportError as e:
            raise MeshError("SAM 2 isn't installed. Run `uv sync`, then restart vigil.") from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        log.info("loading %s on %s", MODEL_ID, device)
        processor = Sam2VideoProcessor.from_pretrained(MODEL_ID)
        model = Sam2VideoModel.from_pretrained(MODEL_ID).to(device, dtype=dtype).eval()
        _model = (torch, processor, model, device, dtype)
    return _model


def _track(images: list[np.ndarray], start: int, points: list, labels: list,
           box: list[float] | None = None) -> list[np.ndarray]:
    """SAM 2 video: prompt `images[start]` (points, a box or both), propagate both ways; one
    bool mask per image."""
    torch, processor, model, device, dtype = _load()
    h, w = images[0].shape[:2]
    session = processor.init_video_session(video=images, inference_device=device,
                                           video_storage_device="cpu", dtype=dtype)
    prompt = {"input_points": [[points]], "input_labels": [[labels]]} if points else {}
    if box is not None:
        prompt["input_boxes"] = [[box]]
    processor.add_inputs_to_inference_session(session, frame_idx=start, obj_ids=1, **prompt)
    masks: list[np.ndarray | None] = [None] * len(images)
    with torch.inference_mode():
        model(inference_session=session, frame_idx=start)
        for reverse in (False, True):
            for out in model.propagate_in_video_iterator(session, start_frame_idx=start, reverse=reverse):
                m = processor.post_process_masks([out.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                masks[out.frame_idx] = m[0, 0].cpu().numpy().astype(bool)
    return [m if m is not None else np.zeros((h, w), bool) for m in masks]


def _cam_frames(lines: list, cam: int) -> list[tuple[int, dict]]:
    """(line index, file entry) of every frame with this camera's colour and depth."""
    return [(i, f) for i, ln in enumerate(lines) for f in ln["files"]
            if f.get("cam") == cam and "color" in f and "depth" in f]


def _depth(path: Path, f: dict) -> np.ndarray:
    raw = cv2.imread(str(path / f["depth"]), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise MeshError(f"Missing depth image {f['depth']}.")
    return raw.astype(np.float32) * 0.001


def _transfer(mask: np.ndarray, src_depth: np.ndarray, cs: dict, dst_depth: np.ndarray, cd: dict
              ) -> list[list[float]]:
    """+ prompts for camera `cd` from camera `cs`'s mask: masked pixels (eroded away from the
    edge) lifted to 3D, projected into cd, kept where cd's own depth agrees within 2 cm (cd
    sees that surface), then TRANSFER_POINTS well spread ones (farthest-point sampling)."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    v, u = np.nonzero(m & (src_depth > 0))
    if len(u) == 0:
        return []
    z = src_depth[v, u]
    cam = np.stack([(u - cs["cx"]) * z / cs["fx"], (v - cs["cy"]) * z / cs["fy"], z], 1)
    T = cd["T_inv"] @ cs["T"]
    q = cam @ T[:3, :3].T + T[:3, 3]
    front = q[:, 2] > 0.05
    q = q[front]
    uu = q[:, 0] * cd["fx"] / q[:, 2] + cd["cx"]
    vv = q[:, 1] * cd["fy"] / q[:, 2] + cd["cy"]
    h, w = dst_depth.shape
    ok = (uu >= 8) & (uu < w - 8) & (vv >= 8) & (vv < h - 8)
    ui, vi, zq = uu[ok].astype(int), vv[ok].astype(int), q[ok, 2]
    dd = dst_depth[vi, ui]
    seen = (dd > 0) & (np.abs(dd - zq) < 0.02)
    pts = np.stack([ui[seen], vi[seen]], 1).astype(float)
    if len(pts) < 50:
        return []
    pick = [int(np.argmin(((pts - pts.mean(0)) ** 2).sum(1)))]
    dist = ((pts - pts[pick[0]]) ** 2).sum(1)
    for _ in range(TRANSFER_POINTS - 1):
        pick.append(int(dist.argmax()))
        dist = np.minimum(dist, ((pts - pts[pick[-1]]) ** 2).sum(1))
    return [pts[k].tolist() for k in pick]


def segment(path: Path, meta: dict, lines: list, prompts: Prompts,
            wanted: dict[int, list[int]] | None = None, progress=lambda stage, frac: None) -> dict[int, dict]:
    """Leg masks per camera.

    `wanted`: camera -> line indices the mesh needs masks for; None = just the clicked frame
    (the preview). Returns camera -> {"masks": {line: bool mask}, "line": the frame the prompts
    went on, "source": "clicks" | "CAM n" (prompts carried over from that camera) | None,
    "points": [[x, y, label]], "box": [x0, y0, x1, y1] | None}. A camera with no source has no masks.
    """
    cams = _camera_matrices(meta)
    t = [ln["t"] for ln in lines]
    boxes = prompts.boxes()
    clicked = sorted({c["cam"] for c in prompts.clicks if KINDS[c["kind"]] is not None} | set(boxes))
    order = [c for c in clicked if c in cams] + [c for c in sorted(cams) if c not in clicked]
    if wanted is not None:
        order = [c for c in order if c in wanted or c in clicked]
    out: dict[int, dict] = {}
    with _lock:
        for k, cam in enumerate(order):
            frames = _cam_frames(lines, cam)
            if not frames:
                continue
            p0 = min(range(len(frames)), key=lambda j: abs(t[frames[j][0]] - t[prompts.index]))
            pos = {li: j for j, (li, _) in enumerate(frames)}
            need = [pos[li] for li in (wanted or {}).get(cam, []) if li in pos]
            lo, hi = min([p0, *need]), max([p0, *need])
            if hi - lo + 1 > MAX_TRACK_FRAMES:
                raise MeshError(f"CAM {cam + 1}: the clicks are {hi - lo + 1} frames from what this mesh needs "
                                f"(at most {MAX_TRACK_FRAMES}). Click the leg again on this frame.")
            res = {"masks": {}, "line": frames[p0][0], "source": None, "points": [], "box": boxes.get(cam)}
            out[cam] = res
            points, labels = prompts.points(cam)
            if points or res["box"]:
                res["source"] = "clicks"
            else:
                src = next((c for c in out if out[c]["source"] == "clicks"), None)
                if src is not None:
                    src_frame = next(f for li, f in _cam_frames(lines, src) if li == out[src]["line"])
                    points = _transfer(out[src]["masks"][out[src]["line"]], _depth(path, src_frame), cams[src],
                                       _depth(path, frames[p0][1]), cams[cam])
                    labels = [1] * len(points)
                    if points:
                        res["source"] = f"CAM {src + 1}"
            if not points and not res["box"]:
                continue
            res["points"] = [[x, y, l] for (x, y), l in zip(points, labels)]
            seq = list(range(lo, hi + 1))
            images = []
            for j in seq:
                bgr = cv2.imread(str(path / frames[j][1]["color"]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise MeshError(f"Missing colour image {frames[j][1]['color']}.")
                images.append(np.ascontiguousarray(bgr[:, :, ::-1]))
            masks = _track(images, p0 - lo, points, labels, res["box"])
            res["masks"] = {frames[j][0]: m for j, m in zip(seq, masks)}
            progress("segmenting (SAM 2)", (k + 1) / len(order))
    return out


def click_point(path: Path, meta: dict, lines: list, prompts: Prompts, click: dict) -> dict | None:
    """A click as a world point (its camera's depth, median over 7×7 px; recorded calibration),
    with the camera and time it was seen at. None if there's no depth there."""
    cams = _camera_matrices(meta)
    c = cams.get(click["cam"])
    frames = _cam_frames(lines, click["cam"])
    if c is None or not frames:
        return None
    t = [ln["t"] for ln in lines]
    li, f = min(frames, key=lambda fr: abs(t[fr[0]] - t[prompts.index]))
    d = _depth(path, f)
    x, y = int(round(click["x"])), int(round(click["y"]))
    patch = d[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4]
    patch = patch[patch > 0]
    if not len(patch):
        return None
    z = float(np.median(patch))
    p = np.array([(x - c["cx"]) * z / c["fx"], (y - c["cy"]) * z / c["fy"], z])
    return {"point": c["T"][:3, :3] @ p + c["T"][:3, 3], "cam": click["cam"], "t": t[li]}


def overlay_png(mask: np.ndarray) -> str:
    """A mask as a translucent amber RGBA PNG (base64), drawn over the camera feed."""
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = (255, 210, 40, 110)  # RGBA amber; reordered to cv2's BGRA below
    edge = mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    rgba[edge] = (255, 230, 120, 255)
    return base64.b64encode(cv2.imencode(".png", rgba[:, :, [2, 1, 0, 3]])[1].tobytes()).decode()
