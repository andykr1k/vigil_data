"""HTTP endpoints behind the dashboard's MESH METHODS, FLATTEN MESH and SAVED MESHES panels.

A mesh is built from the recording being replayed (`pipeline.player`) at its current or a
given frame, saved under Meshes/<recording>/ (library.py), and announced over the dashboard's
websocket via `pipeline.publish` (build progress, "meshes changed"). FLATTEN MESH unrolls a
saved mesh (flatten.py); it needs no replay. LEG REGION = SAM 2: /leg-mask previews the clicked
leg (legmask.py) and /auto-prompts places the clicks itself (autoleg.py), on the replayed frame.
Prefix /api/surface: /api/mesh/* is the SAM 3D Body mesh.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from .flatten import flat_binary, flatten_mesh
from .library import MeshLibrary
from .tsdf import MeshError, TsdfParams, build_tsdf

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/surface")
_building = threading.Lock()


def meshes_root(request: Request) -> Path:
    """Where meshes are saved: app.state.meshes_root if set, else a Meshes folder next to the
    recordings folder (cfg.recording.dir)."""
    root = getattr(request.app.state, "meshes_root", None)
    if root is None:
        cfg = request.app.state.pipeline.cfg
        root = cfg.resolve(cfg.recording.dir).parent / "Meshes"
    return Path(root)


def _build(pipeline, library: MeshLibrary, body: dict) -> dict:
    player = pipeline.player
    if player is None:
        raise MeshError("Replay a recording first: meshes are built from recorded depth.")
    if body.get("method", "tsdf") != "tsdf":
        raise MeshError(f"Unknown mesh method {body.get('method')!r}.")
    if not _building.acquire(blocking=False):
        raise MeshError("A mesh is already being built.")
    try:
        last = [0.0]

        def progress(stage: str, frac: float) -> None:
            now = time.monotonic()
            if frac >= 1 or now - last[0] > 0.1:
                last[0] = now
                pipeline.publish({"type": "mesh_progress", "stage": stage, "progress": round(frac, 3)}, [])

        meta = json.loads((player.path / "meta.json").read_text())
        index = int(body["index"]) if body.get("index") is not None else player.index
        index = min(max(index, 0), len(player.lines) - 1)
        params = TsdfParams.from_dict(body.get("params") or {})
        mesh = build_tsdf(player.path, meta, player.lines, index, params, progress,
                          limb_radius=pipeline.cfg.estimator.vitpose_depth.limb_radius_m)
        entry = library.save(mesh, player.path, player.lines, index, params.window_s)
    finally:
        _building.release()
        pipeline.publish({"type": "mesh_progress", "stage": "done", "progress": 1}, [])
    pipeline.publish({"type": "meshes_changed", "recording": entry["recording"]}, [])
    return {"recording": entry["recording"], "stem": entry["stem"], "entry": entry, "stats": mesh.stats}


def _replayed(pipeline) -> tuple:
    player = pipeline.player
    if player is None:
        raise MeshError("Replay a recording first: the leg is found on recorded frames.")
    return player, json.loads((player.path / "meta.json").read_text())


def _leg_mask(pipeline, body: dict) -> dict:
    """SAM 2 preview: the clicked leg on the clicked frame, in every calibrated camera."""
    from . import legmask

    player, meta = _replayed(pipeline)
    prompts = legmask.Prompts.parse(body, len(player.lines))
    result = legmask.segment(player.path, meta, player.lines, prompts)
    cameras = {}
    for cam, r in result.items():
        mask = r["masks"].get(r["line"])
        cameras[cam] = {"source": r["source"], "points": r["points"],
                        "pixels": 0 if mask is None else int(mask.sum()),
                        "png": None if mask is None else legmask.overlay_png(mask)}
    return {"index": prompts.index, "cameras": cameras}


def _auto_prompts(pipeline, body: dict) -> dict:
    """AUTO: the clicks for a frame (default: the replayed one), from Sapiens2."""
    from . import autoleg
    from .auto import _jsonable

    player, meta = _replayed(pipeline)
    index = int(body["index"]) if body.get("index") is not None else player.index
    index = min(max(index, 0), len(player.lines) - 1)
    info = autoleg.propose(player.path, meta, player.lines, index, body.get("side") or "auto",
                           body.get("size") or autoleg.DEFAULT_SIZE)
    return _jsonable({k: v for k, v in info.items() if k not in ("knee_s", "plane")})


def _flatten(pipeline, library: MeshLibrary, body: dict) -> dict:
    try:
        mesh, source = library.load(str(body.get("recording", "")), str(body.get("stem", "")))
    except (ValueError, OSError) as e:
        raise MeshError("The mesh to flatten could not be found.") from e
    if not source:
        raise MeshError("The mesh to flatten is missing its saved .json.")
    if not _building.acquire(blocking=False):
        raise MeshError("A mesh is already being built or flattened.")
    try:
        last = [0.0]

        def progress(stage: str, frac: float) -> None:
            now = time.monotonic()
            if frac >= 1 or now - last[0] > 0.1:
                last[0] = now
                pipeline.publish({"type": "flatten_progress", "stage": stage, "progress": round(frac, 3)}, [])

        result = flatten_mesh(mesh, str(body.get("method", "scaf")), body.get("iterations", 50), progress)
        entry = library.save_flat(result, source)
    finally:
        _building.release()
        pipeline.publish({"type": "flatten_progress", "stage": "done", "progress": 1}, [])
    pipeline.publish({"type": "meshes_changed", "recording": entry["recording"]}, [])
    return {"recording": entry["recording"], "stem": entry["stem"], "entry": entry, "stats": result.flat.stats}


async def _run(fn, *args) -> Response:
    try:
        result = await asyncio.to_thread(fn, *args)
    except MeshError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001: show it in the panel rather than a bare 500
        log.exception("%s failed", fn.__name__.strip("_"))
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    return JSONResponse(result)


@router.post("/build")
async def build(request: Request, body: dict) -> Response:
    return await _run(_build, request.app.state.pipeline, MeshLibrary(meshes_root(request)), body)


@router.post("/leg-mask")
async def leg_mask(request: Request, body: dict) -> Response:
    """Body: index (replay frame), clicks [{cam, x, y, kind: leg | not | knee | hip | box (+ x2, y2)}]."""
    return await _run(_leg_mask, request.app.state.pipeline, body)


@router.post("/auto-prompts")
async def auto_prompts(request: Request, body: dict) -> Response:
    """Body: index (replay frame; default the current one), side (auto | left | right)."""
    return await _run(_auto_prompts, request.app.state.pipeline, body)


@router.post("/flatten")
async def flatten(request: Request, body: dict) -> Response:
    """Body: recording, stem (a saved surface mesh), method ("scaf" | "slim" | "arap"), iterations."""
    return await _run(_flatten, request.app.state.pipeline, MeshLibrary(meshes_root(request)), body)


@router.get("")
def list_meshes(request: Request, recording: str | None = None) -> Response:
    root = meshes_root(request)
    return JSONResponse({"root": str(root), "items": MeshLibrary(root).list(recording)})


def _recording_dir(request: Request, recording: str) -> Path:
    cfg = request.app.state.pipeline.cfg
    root = getattr(request.app.state, "recordings_root", None) or cfg.resolve(cfg.recording.dir)
    if not recording or "/" in recording or recording.startswith("."):
        raise ValueError("bad recording name")
    return Path(root) / recording


@router.get("/{recording}/{stem}/report")
def report(request: Request, recording: str, stem: str) -> Response:
    """Everything the pipeline page shows for one saved mesh: its sidecar, the recording's
    cameras (to project the knee / hip into the images) and the flattened meshes made from it."""
    library = MeshLibrary(meshes_root(request))
    try:
        entry = json.loads(library.path(recording, stem, "json").read_text())
    except (ValueError, OSError):
        return Response(status_code=404)
    cameras = []
    try:
        meta = json.loads((_recording_dir(request, recording) / "meta.json").read_text())
        cameras = [{k: c.get(k) for k in ("index", "serial", "width", "height", "fx", "fy", "cx", "cy",
                                           "T_world_camera")} for c in meta.get("cameras", [])]
    except (ValueError, OSError):
        pass
    flats = [e for e in library.list(recording)
             if e.get("method") == "flat" and (e.get("source") or {}).get("stem") == stem]
    return JSONResponse({"entry": entry, "cameras": cameras, "flats": flats})


@router.get("/{recording}/{stem}/image/{cam}/{kind}")
def stage_image(request: Request, recording: str, stem: str, cam: int, kind: str) -> Response:
    """The images behind the pipeline page, for camera `cam` on the frame the stage map follows
    (or, for meshes saved before stage maps, the mesh's own frame): kind = color (JPEG),
    depth (colourised PNG), label (the stage map as a PNG, raw values 0-6)."""
    import cv2
    import numpy as np

    library = MeshLibrary(meshes_root(request))
    try:
        entry = json.loads(library.path(recording, stem, "json").read_text())
        rec = _recording_dir(request, recording)
        info = ((entry.get("stages") or {}).get("cameras") or {}).get(str(cam))
        if kind == "label":
            label = library.stage_label(recording, stem, cam)
            if label is None:
                return Response(status_code=404)
            return Response(cv2.imencode(".png", label)[1].tobytes(), media_type="image/png")
        if info is not None:
            rel = info["color" if kind == "color" else "depth"]
        else:  # older mesh: the recording's files at the mesh's frame
            line = None
            with open(rec / "frames.jsonl") as f:
                for i, ln in enumerate(f):
                    if i == entry["index"]:
                        line = json.loads(ln)
                        break
            files = [x for x in (line or {}).get("files", []) if x.get("cam") == cam]
            rel = next((x.get("color" if kind == "color" else "depth") for x in files
                        if x.get("color" if kind == "color" else "depth")), None)
        if rel is None or kind not in ("color", "depth"):
            return Response(status_code=404)
        path = (rec / rel).resolve()
        if rec.resolve() not in path.parents or not path.is_file():
            return Response(status_code=404)
    except (ValueError, OSError, KeyError):
        return Response(status_code=404)
    if kind == "color":
        return FileResponse(path, media_type="image/jpeg")
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return Response(status_code=404)
    valid = depth > 0
    lo, hi = (np.percentile(depth[valid], [2, 98]) if valid.any() else (0, 1))
    norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    img = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return Response(cv2.imencode(".png", img)[1].tobytes(), media_type="image/png")


@router.get("/{recording}/{stem}.{ext}")
def mesh_file(request: Request, recording: str, stem: str, ext: str) -> Response:
    library = MeshLibrary(meshes_root(request))
    try:
        if ext == "flat":  # a flattened mesh, for the dashboard's flat view (flatten.flat_binary)
            arrays, _ = library.load_flat(recording, stem)
            return Response(flat_binary(arrays), media_type="application/octet-stream")
        if ext in ("bin", "obj"):
            mesh, _ = library.load(recording, stem)
            if ext == "bin":
                return Response(mesh.to_binary(), media_type="application/octet-stream")
            return Response(mesh.to_obj(), media_type="application/octet-stream",
                            headers={"Content-Disposition": f'attachment; filename="{recording}_{stem}.obj"'})
        path = library.path(recording, stem, ext)
    except (ValueError, OSError):
        return Response(status_code=404)
    if not path.is_file():
        return Response(status_code=404)
    if ext == "ply":
        return FileResponse(path, media_type="application/octet-stream", filename=f"{recording}_{stem}.ply")
    return FileResponse(path)


@router.delete("/{recording}/{stem}")
def delete_mesh(request: Request, recording: str, stem: str) -> Response:
    try:
        MeshLibrary(meshes_root(request)).delete(recording, stem)
    except ValueError:
        return Response(status_code=404)
    request.app.state.pipeline.publish({"type": "meshes_changed", "recording": recording}, [])
    return JSONResponse({"ok": True})
