"""Intel RealSense capture: color + depth aligned to color, on a background thread."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs

from .config import CamerasConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )

    def dist_coeffs(self) -> np.ndarray:
        return np.asarray(self.coeffs, dtype=np.float64)


@dataclass
class Frame:
    color: np.ndarray  # HxWx3 uint8 RGB
    depth: np.ndarray  # HxW float32 metres (0 = invalid), aligned to color
    timestamp: float  # seconds, host clock at capture
    index: int


def connected_serials() -> list[str]:
    return sorted(d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices())


class RealSenseCamera:
    """Streams aligned frames; `latest()` always returns the newest one (older ones are dropped)."""

    def __init__(self, cfg: CamerasConfig, serial: str):
        self.cfg = cfg
        self.serial = serial
        self.state = "ok"  # ok | reconnecting
        self.generation = 0  # bumps on every (re)start; intrinsics/mode may change
        self.pipeline = rs.pipeline()
        self._start()

        self.align = rs.align(rs.stream.color)
        self.filters: list = []
        if cfg.depth_filters.spatial:
            self.filters.append(rs.spatial_filter())
        if cfg.depth_filters.temporal:
            self.filters.append(rs.temporal_filter())
        if cfg.depth_filters.hole_filling:
            self.filters.append(rs.hole_filling_filter())

        self._latest: tuple[rs.composite_frame, float, int] | None = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"realsense-{serial}", daemon=True)
        self._thread.start()

    def _start(self) -> None:
        device = next((d for d in rs.context().query_devices()
                       if d.get_info(rs.camera_info.serial_number) == self.serial), None)
        if device is None:
            raise RuntimeError(f"RealSense {self.serial} not connected")
        usb = device.get_info(rs.camera_info.usb_type_descriptor)
        # A USB 2 link can't reliably carry the USB 3 modes.
        modes = self.cfg.usb2_modes if usb.startswith("2") else self.cfg.modes
        errors = []
        for width, height, fps in modes:
            rs_cfg = rs.config()
            rs_cfg.enable_device(self.serial)
            rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            rs_cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = self.pipeline.start(rs_cfg)
                break
            except RuntimeError as e:
                errors.append(f"{width}x{height}@{fps}: {e}")
        else:
            raise RuntimeError(f"RealSense {self.serial}: no configured mode works "
                               f"({'; '.join(errors)})")
        self.mode = (width, height, fps)

        device = profile.get_device()
        self.name = device.get_info(rs.camera_info.name)
        self.usb = device.get_info(rs.camera_info.usb_type_descriptor)
        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(
            color_intr.width, color_intr.height,
            color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy,
            tuple(float(c) for c in color_intr.coeffs),
        )
        self.generation += 1

    def _run(self) -> None:
        # Only grab raw framesets here: wait_for_frames releases the GIL, but librealsense
        # filters/align hold it and would starve the inference thread. They run in latest().
        index = 0
        misses = 0
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(1500)
            except RuntimeError:
                misses += 1
                if misses >= 2:  # ~3 s of silence: unplugged or the link reset
                    self._reconnect()
                    misses = 0
                continue
            misses = 0
            frames.keep()
            with self._cond:
                self._latest = (frames, time.time(), index)
                self._cond.notify_all()
            index += 1

    def _reconnect(self) -> None:
        self.state = "reconnecting"
        log.warning("RealSense %s stopped delivering frames; reconnecting", self.serial)
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
        while not self._stop.is_set():
            time.sleep(1.0)
            if self.serial not in connected_serials():
                continue
            try:
                self.pipeline = rs.pipeline()
                self._start()
            except RuntimeError as e:
                log.warning("RealSense %s restart failed: %s", self.serial, e)
                continue
            self.state = "ok"
            log.info("RealSense %s reconnected (USB %s, %dx%d@%d)", self.serial, self.usb, *self.mode)
            return

    def latest(self, after_index: int = -1, timeout: float = 2.0,
               raw_depth: bool = False) -> Frame | None:
        """Block until a frame newer than `after_index` is available, then filter + align it.

        raw_depth=True returns depth as the sensor's uint16 units instead of metres.
        """
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._latest is not None and self._latest[2] > after_index,
                timeout=timeout,
            )
            if not ok:
                return None
            frames, ts, index = self._latest

        # Filters run on the raw depth stream (they pass the color frame through);
        # alignment to the color viewpoint is done last.
        for f in self.filters:
            frames = f.process(frames).as_frameset()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return None
        return Frame(
            color=np.asanyarray(color.get_data()).copy(),
            depth=(np.asanyarray(depth.get_data()) if raw_depth
                   else np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale),
            timestamp=ts,
            index=index,
        )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass  # already stopped by a failed reconnect
