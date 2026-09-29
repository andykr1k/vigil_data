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

    def __init__(self, cfg: CamerasConfig, serial: str, sync_role: int = 0,
                 infrared: bool = False):
        self.cfg = cfg
        self.serial = serial
        # Infrared mode: also stream the left IR camera and alternate the depth projector
        # on/off per frame. Projector-on frames feed depth + colour; projector-off frames give
        # clean, global-shutter IR images for tag detection. Needs the 60 fps modes (USB 3).
        self.want_infrared = infrared
        self.infrared = False
        self.ir_intrinsics: Intrinsics | None = None
        self.T_color_ir: np.ndarray | None = None
        self._latest_ir: tuple[np.ndarray, float, int] | None = None
        self.sync_role = sync_role  # inter_cam_sync_mode: 0 default, 1 master, 2 slave
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
        if self.sync_role:
            sensor = device.first_depth_sensor()
            if sensor.supports(rs.option.inter_cam_sync_mode):
                sensor.set_option(rs.option.inter_cam_sync_mode, self.sync_role)
            else:
                log.warning("RealSense %s: hardware sync not supported", self.serial)
        # A USB 2 link can't reliably carry the USB 3 modes (or infrared mode).
        infrared = self.want_infrared and not usb.startswith("2")
        if self.want_infrared and not infrared:
            log.warning("RealSense %s: infrared tag detection needs USB 3; using colour", self.serial)
        modes = (self.cfg.infrared_modes if infrared
                 else self.cfg.usb2_modes if usb.startswith("2") else self.cfg.modes)
        errors = []
        for width, height, fps in modes:
            rs_cfg = rs.config()
            rs_cfg.enable_device(self.serial)
            rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            rs_cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            if infrared:
                rs_cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
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
        self.infrared = False
        if infrared:
            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.emitter_on_off):
                depth_sensor.set_option(rs.option.emitter_on_off, 1)
                ir = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
                ii = ir.get_intrinsics()
                self.ir_intrinsics = Intrinsics(ii.width, ii.height, ii.fx, ii.fy, ii.ppx, ii.ppy,
                                                tuple(float(c) for c in ii.coeffs))
                ex = ir.get_extrinsics_to(profile.get_stream(rs.stream.color))
                T = np.eye(4)
                T[:3, :3] = np.asarray(ex.rotation).reshape(3, 3).T  # librealsense: column-major
                T[:3, 3] = ex.translation
                self.T_color_ir = T
                self.infrared = True
            else:
                log.warning("RealSense %s: firmware can't alternate the emitter; using colour", self.serial)
        self.generation += 1

    def _emitter_off(self, frames) -> bool | None:
        """True for a projector-off frame (clean IR), None if the camera can't tell us."""
        ir = frames.get_infrared_frame(1)
        md = rs.frame_metadata_value.frame_laser_power_mode
        if not ir or not ir.supports_frame_metadata(md):
            return None
        return ir.get_frame_metadata(md) == 0

    def latest_ir(self) -> tuple[np.ndarray, float, int] | None:
        """Newest projector-off IR image (gray), its capture time, and index."""
        with self._cond:
            return self._latest_ir

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
            if self.infrared:
                off = self._emitter_off(frames)
                if off is None:
                    log.warning("RealSense %s: no emitter metadata; infrared detection disabled",
                                self.serial)
                    self.infrared = False
                elif off:
                    ir = np.asanyarray(frames.get_infrared_frame(1).get_data()).copy()
                    with self._cond:
                        self._latest_ir = (ir, self._timestamp(frames), index)
                    index += 1
                    continue  # projector off: no usable depth in this frame
            frames.keep()
            with self._cond:
                self._latest = (frames, self._timestamp(frames), index)
                self._cond.notify_all()
            index += 1

    def _timestamp(self, frames) -> float:
        """Capture time in seconds on the host clock.

        librealsense's global-time domain maps the camera's hardware clock onto the host
        clock, so frames from different cameras can be paired by when they were exposed
        rather than when they arrived. Falls back to arrival time if that's unavailable.
        """
        now = time.time()
        try:
            if frames.get_frame_timestamp_domain() == rs.timestamp_domain.global_time:
                ts = frames.get_timestamp() / 1000.0
                if abs(ts - now) < 1.0:  # sanity: must be close to the host clock
                    return ts
        except RuntimeError:
            pass
        if not getattr(self, "_warned_ts", False):
            self._warned_ts = True
            log.warning("RealSense %s: no global-time timestamps; using host arrival time", self.serial)
        return now

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
