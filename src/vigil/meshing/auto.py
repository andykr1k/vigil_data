"""Autonomous thigh meshes: choose frames, prompt, track, build, check; no clicks.

For each recording:

1. Score candidate frames (one every STRIDE seconds): how much of the leg the cameras see with
   depth, how confident the leg detection is, and how still the leg is.
2. Best frame first: automatic prompts (autoleg.propose: which leg, + / − points, knee and hip).
3. Checks before building: SAM 2's mask on that frame keeps the + points, avoids the − points,
   doesn't spill onto the support, and the leg's shape puts the knee where the votes did.
4. Build (the same TSDF pipeline as BUILD MESH) and check it: the tracked mask stays steady, the
   support cut didn't remove most of it, the knee cut sits on the leg, the area is plausible.
5. Accepted meshes are saved to the library like any other; a rejected frame records why and
   the next candidate (outside the accepted windows) is tried.

`vigil auto` (cli.py) runs this over recordings and writes a report (JSON + CSV) and per-attempt
preview images to <meshes>/auto_reports/<run>/. SCAN THIGH in the dashboard runs it on the
scan it just recorded (pipeline.py).
"""

from __future__ import annotations

import bisect
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import autoleg, legmask
from .autoleg import AutoError
from .library import MeshLibrary
from .tsdf import MeshError, TsdfParams, build_tsdf

log = logging.getLogger(__name__)

STRIDE_S = 1.0
MAX_MASK_CV = 0.35       # tracked mask area: std / mean over the window, per camera
MAX_SUPPORT_LEAK = 0.4   # fraction of the leg region the support cut removed
MAX_PLANE_LEAK = 0.25    # fraction of the preview mask lying on the support plane
MIN_PROMPT_HIT = 0.6     # + prompts inside the preview mask, at least
AREA_CM2 = (150, 4000)   # plausible thigh surface
WIDTH_KNEE_CM = 8        # leg-shape knee vs voted knee, warning above this


def load_recording(path: Path) -> tuple[dict, list]:
    meta = json.loads((path / "meta.json").read_text())
    lines = [json.loads(ln) for ln in open(path / "frames.jsonl") if ln.strip()]
    if not lines:
        raise MeshError(f"{path.name} has no frames.")
    return meta, lines


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


# ---------------------------------------------------------------------- frame choice
def _leg_region(v: autoleg.View, r: dict) -> np.ndarray:
    """Pixels of the leg as one camera sees it: the thigh mask, or a band along the pose bones."""
    if r["source"] == "parts":
        return np.isin(v.a["parts"], (autoleg.UPPER["left"], autoleg.UPPER["right"],
                                      autoleg.LOWER["left"], autoleg.LOWER["right"]))
    m = np.zeros(v.depth.shape, np.uint8)
    pts = [r["knee_uv"], *r["leg"]]
    for p in pts[1:]:
        cv2.line(m, tuple(int(round(c)) for c in pts[0]), tuple(int(round(c)) for c in p), 1, 40)
    return m > 0


def score_frames(path: Path, meta: dict, lines: list, side: str = "auto", size: str = autoleg.DEFAULT_SIZE,
                 stride_s: float = STRIDE_S, margin_s: float = 0.5, analyse_fn=None,
                 progress=lambda stage, frac: None) -> list[dict]:
    """Candidate frames, best first: {index, score, side, why}. Frames within `margin_s` of
    either end are skipped (the window needs frames on both sides)."""
    t = np.array([ln["t"] for ln in lines])
    targets = np.arange(t[0] + margin_s, t[-1] - margin_s + 1e-9, stride_s)
    idxs = sorted({int(np.argmin(np.abs(t - x))) for x in targets})
    out = []
    for k in range(0, len(idxs), 8):  # a few at a time: the analysis cache stays warm
        chunk = idxs[k:k + 8]
        views = autoleg.load_views(path, meta, lines, chunk, size, analyse_fn)
        for idx in chunk:
            here = {cam: v for (cam, i), v in views.items() if i == idx}
            best = None
            for s in (("left", "right") if side == "auto" else (side,)):
                total, why = 0.0, []
                for cam, v in here.items():
                    r = autoleg.read_leg(v, s)
                    if r is None:
                        continue
                    region = _leg_region(v, r)
                    depth_ok = float((v.depth[region] > 0).mean()) if region.any() else 0.0
                    motion = _motion(path, lines, v, region, t)
                    stillness = max(0.0, 1 - motion / 0.08)
                    total += r["score"] * depth_ok * stillness
                    why.append(f"CAM {cam + 1}: {r['source']} {r['score']:.2f} · depth {depth_ok:.0%} · "
                               f"motion {motion:.3f}")
                if best is None or total > best["score"]:
                    best = {"index": idx, "score": round(total, 3), "side": s, "why": why}
            if best and best["score"] > 0:
                out.append(best)
        progress("scoring frames", min(1.0, (k + 8) / len(idxs)))
    return sorted(out, key=lambda c: -c["score"])


def _motion(path: Path, lines: list, v: autoleg.View, region: np.ndarray, t: np.ndarray, dt: float = 0.2) -> float:
    """Mean absolute grey-level change (0–1) inside `region` between this frame and the same
    camera's frame `dt` later."""
    frames = legmask._cam_frames(lines, v.cam)
    ft = [t[li] for li, _ in frames]
    j = min(bisect.bisect_left(ft, t[v.line] + dt), len(frames) - 1)
    bgr = cv2.imread(str(path / frames[j][1]["color"]), cv2.IMREAD_GRAYSCALE)
    if bgr is None or bgr.shape != region.shape or not region.any():
        return 1.0
    grey = cv2.cvtColor(v.rgb, cv2.COLOR_RGB2GRAY)
    return float(np.abs(grey.astype(np.int16) - bgr.astype(np.int16))[region].mean() / 255)


# ---------------------------------------------------------------------- checks
def precheck(path: Path, meta: dict, lines: list, info: dict) -> dict:
    """SAM 2 on the chosen frame with the automatic prompts, before tracking a whole window.
    Returns the measurements; raises AutoError when they rule the frame out."""
    prompts = legmask.Prompts.parse(info["prompts"], len(lines))
    res = legmask.segment(path, meta, lines, prompts)
    cams = autoleg._camera_matrices(meta)
    out, warnings, leg_pts = {"cameras": {}}, [], []
    plane = info.get("plane")
    for cam, r in res.items():
        mask = r["masks"].get(r["line"])
        if mask is None:
            continue
        mine = [c for c in prompts.clicks if c["cam"] == cam]
        inside = lambda c: bool(mask[min(int(c["y"]), mask.shape[0] - 1), min(int(c["x"]), mask.shape[1] - 1)])
        legs = [c for c in mine if c["kind"] == "leg"]
        hit = np.mean([inside(c) for c in legs]) if legs else None
        nots = sum(inside(c) for c in mine if c["kind"] == "not")
        frame = next(f for li, f in legmask._cam_frames(lines, cam) if li == r["line"])
        depth = legmask._depth(path, frame)
        v = autoleg.View(cam, r["line"], None, depth, cams[cam], {})
        y, x = np.nonzero(mask & (depth > 0))
        pts = autoleg._world(v, np.stack([x, y], 1)[:: max(1, len(x) // 5000)]) if len(x) else np.zeros((0, 3))
        leg_pts.append(pts)
        leak = None
        if plane is not None and len(pts):
            h = pts @ np.asarray(plane["normal"]) - plane["d"]
            leak = float((np.abs(h) < 0.015).mean())
        out["cameras"][cam] = {"source": r["source"], "mask_px": int(mask.sum()),
                               "leg_hit": None if hit is None else round(float(hit), 2), "not_hit": int(nots),
                               "plane_leak": None if leak is None else round(leak, 3)}
        if hit is not None and hit < MIN_PROMPT_HIT:
            raise AutoError(f"CAM {cam + 1}: SAM 2's mask misses {1 - hit:.0%} of the + LEG prompts.")
        if nots:
            raise AutoError(f"CAM {cam + 1}: SAM 2's mask covers {nots} − NOT LEG prompt(s) (the other leg?).")
        if leak is not None and leak > MAX_PLANE_LEAK:
            raise AutoError(f"CAM {cam + 1}: {leak:.0%} of SAM 2's mask lies on the support surface (stool leak).")
    if not out["cameras"]:
        raise AutoError("SAM 2 found no leg with the automatic prompts.")
    pts = np.concatenate(leg_pts) if leg_pts else np.zeros((0, 3))
    axis, wk = np.asarray(info["axis"]), None
    if len(pts):  # the mask point at the voted knee's position along the leg
        wk = autoleg.width_knee(pts, pts[np.argmin(np.abs(pts @ axis - info["knee_s"]))], axis)
    out["width_knee_cm"] = None if wk is None else round(wk * 100, 1)
    if wk is not None and abs(wk) * 100 > WIDTH_KNEE_CM:
        warnings.append(f"The leg's shape puts the knee {wk * 100:+.0f} cm from the voted knee (along the leg).")
    out["warnings"] = warnings
    return out


def postcheck(stats: dict) -> list[str]:
    """Checks on a built mesh's stats; raises AutoError, returns warnings."""
    seg = stats.get("segmentation") or {}
    for cam, c in (seg.get("cameras") or {}).items():
        cv = c.get("mask_px_cv")
        if cv is not None and cv > MAX_MASK_CV:
            raise AutoError(f"CAM {int(cam) + 1}: the tracked leg's area varies {cv:.0%} over the window "
                            "(the tracker lost it or switched legs).")
    leak = seg.get("support_removed")
    if leak is not None and leak > MAX_SUPPORT_LEAK:
        raise AutoError(f"The support cut removed {leak:.0%} of the leg region (the mask spilled onto the support).")
    for w in stats.get("warnings", []):
        if "KNEE click isn't on the leg surface" in w:
            raise AutoError("The knee cut isn't on the leg surface.")
    area = stats.get("area_cm2") or 0
    if not AREA_CM2[0] <= area <= AREA_CM2[1]:
        raise AutoError(f"Implausible thigh area: {area:.0f} cm² (expected {AREA_CM2[0]}–{AREA_CM2[1]}).")
    return [w for w in stats.get("warnings", [])]


# ---------------------------------------------------------------------- previews
def save_preview(path: Path, meta: dict, lines: list, info: dict, out: Path) -> None:
    """Each camera's frame with the automatic prompts drawn on, side by side."""
    t = [ln["t"] for ln in lines]
    idx = info["prompts"]["index"]
    colours = {"leg": (168, 255, 61), "not": (110, 61, 255), "knee": (61, 210, 255), "hip": (255, 160, 60)}
    tiles = []
    for cam in sorted(autoleg._camera_matrices(meta)):
        frames = legmask._cam_frames(lines, cam)
        if not frames:
            continue
        li, f = min(frames, key=lambda fr: abs(t[fr[0]] - t[idx]))
        img = cv2.imread(str(path / f["color"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for c in info["prompts"]["clicks"]:
            if c["cam"] == cam:
                cv2.circle(img, (int(c["x"]), int(c["y"])), 7, colours[c["kind"]], -1)
                cv2.circle(img, (int(c["x"]), int(c["y"])), 7, (10, 10, 10), 1)
                cv2.putText(img, c["kind"].upper(), (int(c["x"]) + 9, int(c["y"]) + 4), 0, 0.45, colours[c["kind"]], 1)
        cv2.putText(img, f"CAM {cam + 1} · {info['side']} leg · {info['source']}", (8, 18), 0, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    if tiles:
        h = min(im.shape[0] for im in tiles)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.hstack([cv2.resize(im, (round(im.shape[1] * h / im.shape[0]), h)) for im in tiles]))


# ---------------------------------------------------------------------- one recording
def run_recording(path: Path, library: MeshLibrary, *, side: str = "auto", count: int = 1,
                  attempts: int = 5, window_s: float = 0.5, stride_s: float = STRIDE_S,
                  size: str = autoleg.DEFAULT_SIZE, params: dict | None = None, preview_dir: Path | None = None,
                  build_fn=None, progress=lambda stage, frac: None) -> dict:
    """Up to `count` accepted meshes (non-overlapping windows) from at most `attempts` tries."""
    meta, lines = load_recording(path)
    t = [ln["t"] for ln in lines]
    started = time.monotonic()
    ranked = score_frames(path, meta, lines, side, size, stride_s, window_s, progress=progress)
    report = {"recording": path.name, "side": side, "method": "tsdf", "window_s": window_s,
              "candidates": len(ranked), "accepted": [], "attempts": []}
    accepted_t: list[float] = []
    for cand in ranked:
        if len(report["accepted"]) >= count or len(report["attempts"]) >= attempts:
            break
        idx = cand["index"]
        if any(abs(t[idx] - a) < 2 * window_s for a in accepted_t):
            continue
        rec = {"frame": idx + 1, "index": idx, "time_s": round(t[idx] - t[0], 2), "score": cand["score"],
               "why": cand["why"], "status": "rejected"}
        report["attempts"].append(rec)
        try:
            info = autoleg.propose(path, meta, lines, idx, side, size)
            rec.update(side=info["side"], source=info["source"], knee=info["knee"], hip=info["hip"],
                       prompt_warnings=info["warnings"])
            if preview_dir is not None:
                prev = preview_dir / f"{path.name}_frame{idx + 1:05d}.jpg"
                save_preview(path, meta, lines, info, prev)
                rec["preview"] = prev.name
            rec["precheck"] = precheck(path, meta, lines, info)
            base = {**(params or {}), "region": "sam", "section": "knee", "prompts": info["prompts"],
                    "window_s": window_s}
            if build_fn is not None:
                mesh = build_fn("tsdf", path, meta, lines, idx, base)
            else:
                mesh = build_tsdf(path, meta, lines, idx, TsdfParams.from_dict(base))
            rec["mesh_warnings"] = postcheck(mesh.stats)
            mesh.stats["auto"] = _jsonable({k: v for k, v in info.items() if k not in ("prompts",)}
                                           | {"precheck": rec["precheck"], "score": cand["score"]})
            entry = library.save(mesh, path, lines, idx, window_s)
            rec.update(status="accepted", stem=entry["stem"], area_cm2=mesh.stats.get("area_cm2"))
            report["accepted"].append(entry["stem"])
            accepted_t.append(t[idx])
        except MeshError as e:  # AutoError included: this window is skipped, with the reason
            rec["reason"] = str(e)
            log.info("%s frame %d rejected: %s", path.name, idx + 1, e)
    if not report["attempts"]:
        report["note"] = "No frame showed the leg well enough to try."
    report["seconds"] = round(time.monotonic() - started, 1)
    return _jsonable(report)


# ---------------------------------------------------------------------- batch (vigil auto)
def recordings_in(args: list[Path]) -> list[Path]:
    """Recording folders (with a meta.json) among `args`, or inside them."""
    out = []
    for p in args:
        p = p.resolve()
        if (p / "meta.json").is_file():
            out.append(p)
        elif p.is_dir():
            out += sorted(d.parent for d in p.glob("*/meta.json"))
    return out


def run_batch(recs: list[Path], meshes: Path, *, side: str = "auto", count: int = 1, attempts: int = 5,
              window_s: float = 0.5, stride_s: float = STRIDE_S, size: str = autoleg.DEFAULT_SIZE,
              say=print) -> Path:
    """run_recording over `recs`, saving to the library at `meshes`; returns the report folder
    (<meshes>/auto_reports/<run>/: report.json, report.csv, previews/)."""
    library = MeshLibrary(meshes)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = meshes / "auto_reports" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"side": side, "count": count, "attempts": attempts, "window_s": window_s, "stride_s": stride_s,
                "model_size": size}
    reports = []
    for path in recs:
        say(f"{path.name}: scoring frames…")
        try:
            r = run_recording(path, library, side=side, count=count, attempts=attempts, window_s=window_s,
                              stride_s=stride_s, size=size, preview_dir=out_dir / "previews")
        except MeshError as e:
            r = {"recording": path.name, "error": str(e), "accepted": [], "attempts": []}
        reports.append(r)
        for a in r["attempts"]:
            say(f"  frame {a['frame']:>5} ({a['time_s']:.1f} s)  {a['status'].upper():8}  "
                f"{a.get('stem') or a.get('reason', '')}")
        say(f"  → {len(r['accepted'])} accepted")
    (out_dir / "report.json").write_text(json.dumps({"run": run, "meshes": str(meshes), "settings": settings,
                                                     "recordings": reports}, indent=2))
    with open(out_dir / "report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "frame", "time_s", "status", "side", "source", "knee_disagree_cm", "stem", "reason"])
        for r in reports:
            for a in r["attempts"]:
                w.writerow([r["recording"], a["frame"], a["time_s"], a["status"], a.get("side"), a.get("source"),
                            (a.get("knee") or {}).get("disagree_cm"), a.get("stem", ""), a.get("reason", "")])
    say(f"Report: {out_dir / 'report.json'}")
    return out_dir
