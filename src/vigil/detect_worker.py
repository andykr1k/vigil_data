"""Person detection in its own process.

RT-DETR costs ~45 ms per call and is launch-bound (lots of Python-side kernel launches
that hold the GIL), so running it in the main process would stall the 30 fps loops. This
process reads the newest frames straight from the cameras' shared memory and publishes
boxes back through a small shared block. The body loop never waits for it: it tracks the
subject with the most recent boxes.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from .capture import FrameBuffer
from .config import Config

MAX_BOXES = 16
# Per camera (float64): [seq, frame_index, frame_ts, n] + MAX_BOXES * (x1, y1, x2, y2, score)
_STRIDE = 4 + MAX_BOXES * 5


@dataclass
class Detections:
    frame_index: int
    timestamp: float  # capture time of the frame the boxes came from
    boxes: np.ndarray  # (N, 4) xyxy pixels, highest score first
    scores: np.ndarray  # (N,)


def _detector_main(cfg: Config, frame_shms: list[tuple[str, int]], out_name: str,
                   ready: mp.Event, stop: mp.Event, errors: mp.Queue) -> None:
    # Ctrl+C reaches the whole process group; the parent owns shutdown and stops us
    # in order (dying here first would stall the parent's loops).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        from .detector import PersonDetector

        detector = PersonDetector(cfg.detector, cfg.estimator.device)
        detector.detect_many([np.zeros((480, 640, 3), np.uint8)])  # warm up CUDA
    except Exception as e:  # report to the parent instead of dying silently
        errors.put(f"{type(e).__name__}: {e}")
        return
    frames = [FrameBuffer(max_px, name=name) for name, max_px in frame_shms]
    shm = shared_memory.SharedMemory(name=out_name)
    out = np.ndarray((len(frames), _STRIDE), np.float64, shm.buf)
    ready.set()
    last = [-1] * len(frames)
    seq = 0
    try:
        while not stop.is_set():
            batch, which = [], []
            for i, fb in enumerate(frames):
                if fb.latest_index() > last[i]:
                    f = fb.read(1.0)
                    if f is not None:
                        batch.append(f)
                        which.append(i)
            if not batch:
                time.sleep(0.002)
                continue
            results = detector.detect_many([f.color for f in batch])
            seq += 1
            for i, f, (boxes, scores) in zip(which, batch, results):
                last[i] = f.index
                n = min(len(boxes), MAX_BOXES)
                row = out[i]
                row[4: 4 + n * 5] = np.concatenate([boxes[:n], scores[:n, None]], 1).reshape(-1)
                row[1:4] = (f.index, f.timestamp, n)
                row[0] = seq  # publish last
    finally:
        out = None
        for fb in frames:
            fb.close()
        shm.close()


class DetectorProcess:
    def __init__(self, cfg: Config, frame_buffers: list[FrameBuffer], ctx=None):
        ctx = ctx or mp.get_context("spawn")
        self.n = len(frame_buffers)
        self.shm = shared_memory.SharedMemory(create=True, size=max(1, self.n) * _STRIDE * 8)
        self.table = np.ndarray((self.n, _STRIDE), np.float64, self.shm.buf)
        self.table[:] = -1
        self._ready = ctx.Event()
        self._stop = ctx.Event()
        self._errors: mp.Queue = ctx.Queue()
        self._proc = ctx.Process(
            target=_detector_main, name="detector", daemon=True,
            args=(cfg, [(fb.name, fb.max_pixels) for fb in frame_buffers], self.shm.name,
                  self._ready, self._stop, self._errors),
        )
        self._proc.start()

    def wait_ready(self, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        while not self._ready.wait(0.2):
            if not self._errors.empty():
                raise RuntimeError(f"detector process failed: {self._errors.get()}")
            if not self._proc.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("detector process failed to start")

    def latest(self, cam: int) -> Detections | None:
        for _ in range(3):  # retry if the detector wrote this row while we copied it
            row = self.table[cam].copy()
            if row[0] == self.table[cam, 0]:
                break
        if row[0] < 0:
            return None
        n = int(row[3])
        data = row[4: 4 + n * 5].reshape(n, 5)
        return Detections(int(row[1]), float(row[2]), data[:, :4].copy(), data[:, 4].copy())

    def close(self) -> None:
        self._stop.set()
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
        self.table = None
        self.shm.close()
        self.shm.unlink()
