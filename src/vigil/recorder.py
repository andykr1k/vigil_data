"""Record a session to disk in formats any point cloud / RGB-D tool opens.

    <dir>/<YYYY-mm-dd_HH-MM-SS>_<procedure>/
      README.txt               what's here and how to load it
      meta.json                cameras (intrinsics, world pose), world frame, depth units
      frames.jsonl             one line per frame: timestamp, files written, and the dashboard's
                               frame message (probe pose/tip, body joints, tags, per-view keypoints)
      cam<i>/color/<seq>.jpg   RGB, as captured
      cam<i>/depth/<seq>.png   uint16 depth in millimetres, aligned to the colour image
      cloud/<seq>.ply          fused point cloud, world frame, metres, RGB (binary PLY)
      cam<i>/cloud/<seq>.ply   a not-yet-calibrated camera's own cloud (its camera frame)
      ultrasound/<seq>.jpg     Clarius B-mode frames

Files share the loop's frame number <seq>, so a frame's colour, depth, cloud, probe pose and
ultrasound image line up. meta.json also keeps the dashboard's `hello`, so the dashboard can
replay a recording exactly as it looked live (replay.py). Encoding and writing run on a thread pool; when the disk can't keep
up, frames are dropped (and counted) rather than slowing the 30 fps loop.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import RecordingConfig

log = logging.getLogger(__name__)

MAX_PENDING = 64  # queued writes before new frames are dropped

README = """VIGIL recording
===============

World frame: {world}. Units: metres. Depth PNGs: uint16 millimetres (0 = no depth),
aligned to the colour image of the same camera (same intrinsics).

meta.json     cameras: serial, intrinsics (fx, fy, cx, cy, distortion), T_world_camera
              (4x4, camera -> world; null = not calibrated when recorded)
frames.jsonl  per frame: seq, t (capture time, s), files written, and "frame": probe
              (position, rotation 3x3 row-major, tip), person (joints xyz+conf), views
cloud/*.ply   fused point cloud of all calibrated cameras, world frame, binary PLY with RGB
              (clipped to the dashboard's depth window: {depth_window})
cam<i>/       color/*.{color_ext}, depth/*.png per camera; cloud/*.ply if it was uncalibrated
ultrasound/   Clarius B-mode JPEGs

Point clouds open directly in CloudCompare, MeshLab, Blender, Open3D, PCL.

Open3D, rebuild a cloud from one camera's RGB-D frame:

    import json, open3d as o3d, numpy as np
    meta = json.load(open("meta.json")); cam = meta["cameras"][0]
    K = o3d.camera.PinholeCameraIntrinsic(cam["width"], cam["height"],
                                          cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.io.read_image("cam0/color/000001.{color_ext}"), o3d.io.read_image("cam0/depth/000001.png"),
        depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, K)
    if cam["T_world_camera"]: pcd.transform(np.array(cam["T_world_camera"]))
    o3d.visualization.draw_geometries([pcd, o3d.io.read_point_cloud("cloud/000001.ply")])
"""


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary little-endian PLY: float32 x y z + uchar red green blue."""
    pts = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    pts["x"], pts["y"], pts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    pts["red"], pts["green"], pts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(pts.tobytes())


class Recorder:
    def __init__(self, cfg: RecordingConfig, root: Path, cameras: list[dict], extra: dict):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = root / f"{stamp}_{extra.get('procedure') or 'session'}"
        self.cfg = cfg
        self.ext = "png" if cfg.color_format == "png" else "jpg"
        self._pool = ThreadPoolExecutor(max_workers=cfg.writers, thread_name_prefix="rec")
        self._lock = threading.Lock()
        self._pending = 0
        self.frames = self.clouds = self.dropped = self.bytes = 0
        self.t0 = time.monotonic()
        self.path.mkdir(parents=True)
        for cam in cameras:
            for sub in ("color", "depth", "cloud"):
                (self.path / f"cam{cam['index']}" / sub).mkdir(parents=True)
        (self.path / "cloud").mkdir()
        (self.path / "ultrasound").mkdir()
        meta = {"created": stamp, "world_frame": extra.pop("world_frame"), "units": "metres",
                "depth_png": {"dtype": "uint16", "scale": 0.001, "aligned_to": "color"},
                "color_format": self.ext, "cameras": cameras, **extra}
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2))
        (self.path / "README.txt").write_text(README.format(
            world=meta["world_frame"], color_ext=self.ext, depth_window=extra.get("depth_window")))
        self._log = open(self.path / "frames.jsonl", "w")

    # ------------------------------------------------------------------ writes
    def _submit(self, fn, *args) -> bool:
        with self._lock:
            if self._pending >= MAX_PENDING:
                self.dropped += 1
                return False
            self._pending += 1
        self._pool.submit(self._run, fn, *args)
        return True

    def _run(self, fn, *args) -> None:
        try:
            n = fn(*args)
            with self._lock:
                self.bytes += n
        except Exception:  # a failed write must not stop the recording
            log.exception("recording write failed")
        finally:
            with self._lock:
                self._pending -= 1

    def _image(self, path: Path, image: np.ndarray, params: list[int]) -> int:
        ok, buf = cv2.imencode(path.suffix, image, params)
        if ok:
            path.write_bytes(buf.tobytes())
        return len(buf) if ok else 0

    def _cloud(self, path: Path, xyz: np.ndarray, rgb: np.ndarray) -> int:
        write_ply(path, xyz, rgb)
        return path.stat().st_size

    def _bytes(self, path: Path, data: bytes) -> int:
        path.write_bytes(data)
        return len(data)

    # ------------------------------------------------------------------ frame API
    def add_frame(self, seq: int, frames: list, fresh: list[bool], record: dict,
                  ultrasound: bytes | None = None) -> None:
        """One loop iteration: every fresh camera frame, plus the per-frame record."""
        files = []
        for i, f in enumerate(frames):
            if f is None or not fresh[i]:
                continue
            color = self.path / f"cam{i}" / "color" / f"{seq:06d}.{self.ext}"
            depth = self.path / f"cam{i}" / "depth" / f"{seq:06d}.png"
            params = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality] if self.ext == "jpg" else []
            d_mm = np.clip(np.round(f.depth * 1000), 0, 65535).astype(np.uint16)
            if self._submit(self._image, color, f.color[:, :, ::-1], params):  # RGB → BGR for cv2
                self._submit(self._image, depth, d_mm, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                files.append({"cam": i, "frame_index": f.index, "timestamp": f.timestamp,
                              "color": str(color.relative_to(self.path)),
                              "depth": str(depth.relative_to(self.path))})
        if ultrasound is not None:
            us = self.path / "ultrasound" / f"{seq:06d}.jpg"
            if self._submit(self._bytes, us, ultrasound):
                files.append({"ultrasound": str(us.relative_to(self.path))})
        self.frames += 1
        self._log.write(json.dumps({"seq": seq, "files": files, **record},
                                   separators=(",", ":")) + "\n")

    def add_cloud(self, seq: int, cam: int | None, xyz: np.ndarray, rgb: np.ndarray) -> None:
        """A point cloud: fused (cam None, world frame) or one uncalibrated camera's own."""
        sub = "cloud" if cam is None else f"cam{cam}/cloud"
        if self._submit(self._cloud, self.path / sub / f"{seq:06d}.ply", xyz, rgb):
            self.clouds += 1

    def status(self) -> dict:
        return {"path": str(self.path), "seconds": round(time.monotonic() - self.t0, 1),
                "frames": self.frames, "clouds": self.clouds, "dropped": self.dropped,
                "mb": round(self.bytes / 1e6, 1)}

    def close(self) -> dict:
        """Finish every queued write, then return the final status."""
        self._pool.shutdown(wait=True)
        self._log.close()
        summary = self.status()
        meta_path = self.path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["summary"] = summary
        meta_path.write_text(json.dumps(meta, indent=2))
        return summary
