"""Play a recording back through the dashboard, in place of live data.

The recording holds the dashboard's own `hello` and per-frame messages, so replay re-sends
them unchanged, with the camera images, point clouds and ultrasound read back from the
recording's files. Playback follows the recorded capture times (at any speed); seeking
re-sends the newest image / cloud / ultrasound at or before that frame.
"""

from __future__ import annotations

import bisect
import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """xyz (N, 3) float32 and rgb (N, 3) uint8 from a PLY written by recorder.write_ply."""
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    pts = np.frombuffer(data, PLY_DTYPE, offset=end)
    return (np.stack([pts["x"], pts["y"], pts["z"]], 1),
            np.stack([pts["red"], pts["green"], pts["blue"]], 1))


def list_recordings(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("*/meta.json"), reverse=True):
        try:
            meta = json.loads(d.read_text())
        except (OSError, ValueError):
            continue
        s = meta.get("summary") or {}
        out.append({"name": d.parent.name, "procedure": meta.get("procedure"),
                    "seconds": s.get("seconds"), "frames": s.get("frames"), "mb": s.get("mb")})
    return out


class Player:
    def __init__(self, path: Path, publish):
        from .pipeline import FUSED, MSG_CLOUD, MSG_JPEG, MSG_ULTRASOUND, _mm, _pack

        self._kinds = (MSG_JPEG, MSG_CLOUD, MSG_ULTRASOUND, FUSED)
        self._pack, self._mm = _pack, _mm
        self.path, self.publish = path, publish
        meta = json.loads((path / "meta.json").read_text())
        self.hello = meta["hello"]
        self.lines = [json.loads(line) for line in open(path / "frames.jsonl") if line.strip()]
        if not self.lines:
            raise ValueError("recording has no frames")
        self.t = [ln["t"] for ln in self.lines]
        # Per source: sorted line indices that carry a file, and those files.
        self.images: dict[int, tuple[list[int], list[str]]] = {}
        self.us: tuple[list[int], list[str]] = ([], [])
        for i, ln in enumerate(self.lines):
            for f in ln["files"]:
                if "cam" in f:
                    idx, files = self.images.setdefault(f["cam"], ([], []))
                    idx.append(i)
                    files.append(f["color"])
                elif "ultrasound" in f:
                    self.us[0].append(i)
                    self.us[1].append(f["ultrasound"])
        seq_of = [ln["seq"] for ln in self.lines]
        clouds = sorted((int(p.stem), p) for p in path.glob("cloud/*.ply"))
        if not clouds:  # nothing calibrated while recording: the cameras' own clouds
            clouds = sorted((int(p.stem), p) for p in path.glob("cam*/cloud/*.ply"))
        self.clouds = ([bisect.bisect_left(seq_of, s) for s, _ in clouds], [p for _, p in clouds])

        self.index, self.playing, self.speed = 0, True, 1.0
        self._sent: dict[str, object] = {}
        self._seek: int | None = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="replay", daemon=True)

    def start(self) -> None:
        """Begin playback (after the dashboard has been sent the recording's hello)."""
        self._thread.start()

    # ------------------------------------------------------------------ control
    def control(self, msg: dict) -> None:
        if "play" in msg:
            self.playing = bool(msg["play"])
            if self.playing and self.index >= len(self.lines) - 1:
                self._seek = 0  # play again from the start
        if "speed" in msg:
            self.speed = float(np.clip(float(msg["speed"]), 0.1, 8.0))
        if "seek" in msg:
            self._seek = int(np.clip(int(msg["seek"]), 0, len(self.lines) - 1))
        self._wake.set()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def info(self) -> dict:
        return {"name": self.path.name, "index": self.index, "count": len(self.lines),
                "time": round(self.t[self.index] - self.t[0], 2),
                "duration": round(self.t[-1] - self.t[0], 2),
                "playing": self.playing, "speed": self.speed}

    # ------------------------------------------------------------------ playback
    def _run(self) -> None:
        clock = None  # (wall time, recording time) at the last frame shown
        while not self._stop.is_set():
            if self._seek is not None:
                self.index, self._seek = self._seek, None
                self._sent.clear()  # show everything at the new position
                self._emit()
                clock = None
                continue
            if not self.playing or self.index >= len(self.lines) - 1:
                if self.playing:
                    self.playing = False
                    self._emit()  # tell the dashboard playback ended
                self._wake.wait(0.1)
                self._wake.clear()
                continue
            if clock is None:
                clock = (time.monotonic(), self.t[self.index])
            due = clock[0] + (self.t[self.index + 1] - clock[1]) / self.speed
            if self._wake.wait(max(0.0, due - time.monotonic())):
                self._wake.clear()
                clock = None  # speed / play / seek changed: restart the clock
                continue
            self.index += 1
            self._emit()

    def _latest(self, key: str, idx: list[int], files: list) -> object | None:
        """The newest file at or before the current frame, if not already sent."""
        k = bisect.bisect_right(idx, self.index) - 1
        if k < 0 or self._sent.get(key) == files[k]:
            return None
        self._sent[key] = files[k]
        return files[k]

    def _emit(self) -> None:
        MSG_JPEG, MSG_CLOUD, MSG_ULTRASOUND, FUSED = self._kinds
        line = self.lines[self.index]
        frame = dict(line["frame"], recording=None, replay=self.info())
        seq = frame["seq"]
        binaries = []
        try:
            for cam, (idx, files) in self.images.items():
                if (f := self._latest(f"cam{cam}", idx, files)) is not None:
                    data = (self.path / f).read_bytes()
                    if not f.endswith(".jpg"):  # lossless PNG recording: the dashboard wants JPEG
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                    binaries.append(self._pack(MSG_JPEG, cam, seq, data))
            if (p := self._latest("cloud", *self.clouds)) is not None:
                xyz, rgb = read_ply(p)
                cam = FUSED if p.parent.parent == self.path else int(p.parent.parent.name[3:])
                binaries.append(self._pack(MSG_CLOUD, cam, seq, len(xyz).to_bytes(4, "little")
                                           + self._mm(xyz) + rgb.tobytes()))
            if (f := self._latest("us", *self.us)) is not None:
                binaries.append(self._pack(MSG_ULTRASOUND, 0, seq, (self.path / f).read_bytes()))
        except OSError as e:  # a missing file (dropped write) just leaves the last one shown
            log.warning("replay: %s", e)
        self.publish(frame, binaries)
