"""HTTP endpoints behind the dashboard's MESH METHODS and SAVED MESHES panels.

A mesh is built from the recording being replayed (`pipeline.player`) at its current or a
given frame, saved under Meshes/<recording>/ (library.py), and announced over the dashboard's
websocket via `pipeline.publish` (build progress, "meshes changed"). Prefix /api/surface:
/api/mesh/* is the SAM 3D Body mesh.
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


@router.post("/build")
async def build(request: Request, body: dict) -> Response:
    pipeline = request.app.state.pipeline
    try:
        result = await asyncio.to_thread(_build, pipeline, MeshLibrary(meshes_root(request)), body)
    except MeshError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001: show it in the panel rather than a bare 500
        log.exception("mesh build failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    return JSONResponse(result)


@router.get("")
def list_meshes(request: Request, recording: str | None = None) -> Response:
    root = meshes_root(request)
    return JSONResponse({"root": str(root), "items": MeshLibrary(root).list(recording)})


@router.get("/{recording}/{stem}.{ext}")
def mesh_file(request: Request, recording: str, stem: str, ext: str) -> Response:
    library = MeshLibrary(meshes_root(request))
    try:
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
