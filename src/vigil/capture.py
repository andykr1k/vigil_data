"""RealSense capture in a child process per camera, shared with the main process via shm.

librealsense's filters and align hold the Python GIL (~8 ms per frame per camera), which
starves inference threads. Each camera therefore gets its own process that captures,
filters and aligns, and publishes frames into a triple-buffered shared-memory block.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import queue
import time
from multiprocessing import shared_memory

import numpy as np

from .camera import Frame, Intrinsics
from .config import CamerasConfig

log = logging.getLogger(__name__)

SLOTS = 3
# Header (float64): [latest_slot, index, timestamp, width, height] + per-slot [index, ts, w, h]
_HDR = 5 + SLOTS * 4


def _max_pixels(cfg: CamerasConfig) -> int:
    return max(w * h for w, h, _ in [*cfg.modes, *cfg.usb2_modes])


class FrameBuffer:
    """Triple-buffered colour (RGB8) + aligned depth (uint16 mm-ish units) in shared memory."""

    def __init__(self, max_pixels: int, name: str | None = None):
        size = _HDR * 8 + SLOTS * max_pixels * 5
        self.shm = shared_memory.SharedMemory(name=name, create=name is None, size=size)
        self.max_pixels = max_pixels
        buf = self.shm.buf
        self.header = np.ndarray((_HDR,), np.float64, buf, 0)
        base = _HDR * 8
        self.color = [np.ndarray((max_pixels * 3,), np.uint8, buf, base + i * max_pixels * 5)
                      for i in range(SLOTS)]
        self.depth = [np.ndarray((max_pixels,), np.uint16, buf,
                                 base + i * max_pixels * 5 + max_pixels * 3)
                      for i in range(SLOTS)]
        if name is None:
            self.header[:] = -1

    @property
    def name(self) -> str:
        return self.shm.name

    def write(self, color: np.ndarray, depth: np.ndarray, index: int, ts: float) -> None:
        h, w = depth.shape
        slot = (int(self.header[0]) + 1) % SLOTS if self.header[0] >= 0 else 0
        self.color[slot][: h * w * 3] = color.reshape(-1)
        self.depth[slot][: h * w] = depth.reshape(-1)
        self.header[5 + slot * 4: 9 + slot * 4] = (index, ts, w, h)
        self.header[1:5] = (index, ts, w, h)
        self.header[0] = slot  # publish last

    def latest_index(self) -> int:
        return int(self.header[1])

    def read(self, depth_scale: float) -> Frame | None:
        slot = int(self.header[0])
        if slot < 0:
            return None
        index, ts, w, h = self.header[5 + slot * 4: 9 + slot * 4]
        w, h = int(w), int(h)
        color = self.color[slot][: h * w * 3].reshape(h, w, 3).copy()
        depth = self.depth[slot][: h * w].reshape(h, w).astype(np.float32) * depth_scale
        # The writer needs two more frames (~66 ms) to come back to this slot; a changed
        # index means we were far too slow and the copy may be torn.
        if int(self.header[5 + slot * 4]) != int(index):
            return None
        return Frame(color=color, depth=depth, timestamp=float(ts), index=int(index))

    def close(self, unlink: bool = False) -> None:
        # Views into the buffer must be released before the mapping can close.
        self.header = self.color = self.depth = None
        self.shm.close()
        if unlink:
            self.shm.unlink()


def _camera_main(cfg: CamerasConfig, serial: str, shm_name: str, max_pixels: int,
                 info_q: mp.Queue, stop: mp.Event) -> None:
    """Child process: capture → temporal filter → align → shared memory."""
    from .camera import RealSenseCamera

    # Ctrl+C reaches the whole process group; the parent owns shutdown and stops us
    # in order (dying here first would stall the parent's loops).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    buf = FrameBuffer(max_pixels, name=shm_name)
    try:
        cam = RealSenseCamera(cfg, serial)
    except RuntimeError as e:
        info_q.put({"serial": serial, "error": str(e)})
        return
    gen, state = -1, None
    frame = None
    try:
        while not stop.is_set():
            if cam.generation != gen or cam.state != state:
                gen, state = cam.generation, cam.state
                K = cam.intrinsics
                info_q.put({"serial": serial, "name": cam.name, "usb": cam.usb, "mode": cam.mode,
                            "depth_scale": cam.depth_scale, "state": cam.state,
                            "generation": gen, "intrinsics": K.__dict__})
            f = cam.latest(frame.index if frame else -1, timeout=0.5, raw_depth=True)
            if f is None:
                continue
            frame = f
            buf.write(f.color, f.depth, f.index, f.timestamp)
    finally:
        cam.close()
        buf.close()


class CameraClient:
    """Main-process handle to a camera process; mirrors RealSenseCamera's read API."""

    def __init__(self, cfg: CamerasConfig, serial: str, ctx=None):
        ctx = ctx or mp.get_context("spawn")
        self.serial = serial
        self.state = "starting"
        self.generation = 0
        self.max_pixels = _max_pixels(cfg)
        self.buffer = FrameBuffer(self.max_pixels)
        self._info_q: mp.Queue = ctx.Queue()
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=_camera_main, name=f"camera-{serial}", daemon=True,
            args=(cfg, serial, self.buffer.name, self.max_pixels, self._info_q, self._stop),
        )
        self._proc.start()
        # Wait until streaming (or failed); notice a crashed child immediately.
        deadline = time.monotonic() + 30
        while True:
            try:
                self._apply(self._info_q.get(timeout=0.2))
                break
            except queue.Empty:
                if not self._proc.is_alive() or time.monotonic() > deadline:
                    self.close()
                    raise RuntimeError(f"RealSense {serial}: capture process failed to start")

    def _apply(self, info: dict) -> None:
        if "error" in info:
            self.close()
            raise RuntimeError(info["error"])
        self.name, self.usb, self.mode = info["name"], info["usb"], tuple(info["mode"])
        self.depth_scale = info["depth_scale"]
        self.state = info["state"]
        self.generation = info["generation"]
        k = info["intrinsics"]
        self.intrinsics = Intrinsics(k["width"], k["height"], k["fx"], k["fy"], k["cx"], k["cy"],
                                     tuple(k["coeffs"]))

    def poll_info(self) -> bool:
        """Apply status updates from the camera process; True if intrinsics/mode changed."""
        changed = False
        while True:
            try:
                info = self._info_q.get_nowait()
            except queue.Empty:
                return changed
            changed |= info.get("generation") != self.generation
            self._apply(info)

    def latest(self, after_index: int = -1, timeout: float = 2.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        while True:
            if self.buffer.latest_index() > after_index:
                f = self.buffer.read(self.depth_scale)
                if f is not None and f.index > after_index:
                    return f
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.001)

    def close(self) -> None:
        self._stop.set()
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
        self.buffer.close(unlink=True)
