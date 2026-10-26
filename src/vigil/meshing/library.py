"""Saved meshes on disk, laid out so every mesh maps straight back to its recording and frame.

    Meshes/
      README.txt
      index.csv                               one row per mesh: recording, frame, time, files
      <recording name>/                       same folder name as in recordings/
        frame00600_t0021.88s_tsdf_20261002-143012.ply   the mesh (world frame, metres)
        frame00600_t0021.88s_tsdf_20261002-143012.json  recording, frame, seq, time, window, settings, stats
        frame00600_t0021.88s_tsdf_20261002-143012.jpg   camera 1 at that frame (thumbnail)
        frame00600_t0021.88s_flat-scaf_20261008-101500-123456.{ply,json,jpg,npz}
                                                FLATTEN MESH: that surface unrolled (SCAF / ARAP);
                                                the .npz maps every flat point back onto it

`frame` is the frame number the viewer shows (1-based, "FRAME 600 / 1317"); `index` in the
JSON is the same frame 0-based, `seq` the recorder's frame number (file names in recordings/).
"""

from __future__ import annotations

import bisect
import csv
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .tsdf import Mesh

if TYPE_CHECKING:
    from .flatten import FlatResult

log = logging.getLogger(__name__)

SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")

README = """Saved leg-surface meshes (vigil MESH METHODS)
=============================================

One folder per recording (same name as in recordings/). Every mesh has three files that
share a name:

  frameNNNNN_tSSSS.SSs_<method>_<created>.ply   the mesh: binary PLY, vertex normals + colours
  ....json                                      recording, frame, seq, time, fused time window,
                                                every setting used, and build stats
  ....jpg                                       camera 1's colour image at that frame

frameNNNNN is the frame number the viewer shows (1-based); tSSSS.SSs is the time into the
recording. In the viewer, SAVED MESHES opens a mesh together with its recording at that frame.

Coordinates: the recording's world frame (camera 1 colour optical frame: x right, y down,
z forward), metres -- the same frame as the recording's cloud/*.ply and frames.jsonl.

Flattened meshes (FLATTEN MESH) sit next to the surface they came from, named
frameNNNNN_tSSSS.SSs_flat-<scaf|arap>_<created>. Their .ply is the flat sheet (x, y in metres,
z = 0); the .npz holds the full correspondence (uv, the 3D vertices and triangles, per-triangle
stretch); the .json names the source mesh and holds the distortion stats.

index.csv lists every mesh in one table (rebuilt on every save / delete).
"""


class MeshLibrary:
    def __init__(self, root: Path):
        self.root = root

    # ------------------------------------------------------------------ paths
    def _dir(self, recording: str) -> Path:
        if not SAFE.match(recording):
            raise ValueError(f"bad recording name {recording!r}")
        return self.root / recording

    def path(self, recording: str, stem: str, ext: str) -> Path:
        if not SAFE.match(stem) or ext not in ("ply", "json", "jpg", "npz"):
            raise ValueError("bad mesh name")
        return self._dir(recording) / f"{stem}.{ext}"

    # ------------------------------------------------------------------ save
    def save(self, mesh: Mesh, recording_path: Path, lines: list, index: int, window_s: float) -> dict:
        """Write the mesh, its sidecar and a thumbnail; returns the sidecar entry."""
        line = lines[index]
        t0 = lines[0]["t"]
        t = line["t"] - t0
        created = datetime.now()
        stem = f"frame{index + 1:05d}_t{t:07.2f}s_{mesh.stats.get('method', 'mesh')}_{created:%Y%m%d-%H%M%S}"
        folder = self._dir(recording_path.name)
        folder.mkdir(parents=True, exist_ok=True)

        times = [ln["t"] for ln in lines]
        lo = bisect.bisect_left(times, line["t"] - window_s)
        hi = bisect.bisect_right(times, line["t"] + window_s) - 1
        thumb = None
        for f in line["files"]:
            if f.get("cam") == 0 and "color" in f:
                img = cv2.imread(str(recording_path / f["color"]), cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (320, round(320 * h / w)), interpolation=cv2.INTER_AREA)
                    (folder / f"{stem}.jpg").write_bytes(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
                    thumb = f"{stem}.jpg"
                break

        entry = {
            "recording": recording_path.name,
            "stem": stem,
            "frame": index + 1,
            "index": index,
            "seq": line["seq"],
            "time_s": round(t, 3),
            "window": {"seconds": window_s,
                       "from_frame": lo + 1, "to_frame": hi + 1,
                       "from_s": round(times[lo] - t0, 3), "to_s": round(times[hi] - t0, 3)},
            "method": mesh.stats.get("method"),
            "created": created.isoformat(timespec="seconds"),
            "files": {"mesh": f"{stem}.ply", "thumbnail": thumb, "sidecar": f"{stem}.json"},
            "frame_files": [f.get("color") or f.get("ultrasound") for f in line["files"]],
            "stats": {k: v for k, v in mesh.stats.items() if k != "recording"},
        }
        (folder / f"{stem}.ply").write_bytes(mesh.to_ply())
        (folder / f"{stem}.json").write_text(json.dumps(entry, indent=2))
        self._write_index()
        log.info("saved mesh %s/%s", recording_path.name, stem)
        return entry

    def save_flat(self, result: FlatResult, source: dict) -> dict:
        """Save a flattened mesh next to its source (whose sidecar entry is `source`)."""
        recording = source["recording"]
        folder = self._dir(recording)
        folder.mkdir(parents=True, exist_ok=True)
        created = datetime.now()
        stats = result.flat.stats
        stem = (f"frame{source['frame']:05d}_t{source['time_s']:07.2f}s_"
                f"flat-{stats['flatten_method']}_{created:%Y%m%d-%H%M%S-%f}")
        thumb = None
        src_jpg = self.path(recording, source["stem"], "jpg")
        if src_jpg.is_file():
            (folder / f"{stem}.jpg").write_bytes(src_jpg.read_bytes())
            thumb = f"{stem}.jpg"
        with open(folder / f"{stem}.npz", "wb") as f:
            np.savez_compressed(f, **result.arrays)
        (folder / f"{stem}.ply").write_bytes(result.flat.to_ply())
        entry = {k: source.get(k) for k in ("recording", "frame", "index", "seq", "time_s", "window", "frame_files")}
        entry.update({"stem": stem, "method": "flat", "flatten_method": stats["flatten_method"],
                      "created": created.isoformat(timespec="seconds"),
                      "source": {"recording": recording, "stem": source["stem"]},
                      "files": {"mesh": f"{stem}.ply", "thumbnail": thumb, "sidecar": f"{stem}.json",
                                "mapping": f"{stem}.npz"},
                      "stats": stats})
        (folder / f"{stem}.json").write_text(json.dumps(entry, indent=2))
        self._write_index()
        log.info("saved flat mesh %s/%s", recording, stem)
        return entry

    def load_flat(self, recording: str, stem: str) -> tuple[dict[str, np.ndarray], dict]:
        """A flattened mesh's correspondence arrays (the .npz) and its sidecar entry."""
        entry = json.loads(self.path(recording, stem, "json").read_text())
        if entry.get("method") != "flat":
            raise ValueError("not a flattened mesh")
        with np.load(self.path(recording, stem, "npz"), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}, entry

    # ------------------------------------------------------------------ read
    def list(self, recording: str | None = None) -> list[dict]:
        if not self.root.is_dir():
            return []
        pattern = f"{recording}/*.json" if recording else "*/*.json"
        if recording and not SAFE.match(recording):
            return []
        out = []
        for p in self.root.glob(pattern):
            try:
                e = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if not (p.with_suffix(".ply")).is_file():
                continue
            s = e.get("stats", {})
            out.append({k: e.get(k) for k in ("recording", "stem", "frame", "index", "seq", "time_s",
                                              "window", "method", "flatten_method", "source", "created")}
                       | {"thumbnail": bool(e.get("files", {}).get("thumbnail")),
                          "faces": s.get("faces"), "area_cm2": s.get("area_cm2"),
                          "vertices": s.get("vertices"),
                          "stretch_p95": (s.get("distortion") or {}).get("stretch_p95")})
        out.sort(key=lambda e: (e["recording"], e["index"], e["created"]))
        return out

    def load(self, recording: str, stem: str) -> tuple[Mesh, dict]:
        ply = self.path(recording, stem, "ply")
        side = self.path(recording, stem, "json")
        entry = json.loads(side.read_text()) if side.is_file() else {}
        stats = dict(entry.get("stats", {}), recording=recording)
        return Mesh.from_ply(ply.read_bytes(), stats), entry

    def delete(self, recording: str, stem: str) -> None:
        for ext in ("ply", "json", "jpg", "npz"):
            self.path(recording, stem, ext).unlink(missing_ok=True)
        folder = self._dir(recording)
        if folder.is_dir() and not any(folder.iterdir()):
            shutil.rmtree(folder)
        self._write_index()

    # ------------------------------------------------------------------ index
    def _write_index(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "README.txt").write_text(README)
        rows = self.list()
        with open(self.root / "index.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["recording", "frame", "time_s", "seq", "window_from_s", "window_to_s",
                        "method", "triangles", "area_cm2", "created", "mesh_file"])
            for e in rows:
                win = e.get("window") or {}
                w.writerow([e["recording"], e["frame"], e["time_s"], e["seq"], win.get("from_s"),
                            win.get("to_s"), e["method"] + (f"-{e['flatten_method']}" if e.get("flatten_method") else ""), e["faces"], e["area_cm2"], e["created"],
                            f"{e['recording']}/{e['stem']}.ply"])
