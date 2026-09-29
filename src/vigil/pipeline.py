"""Fast loop (camera rate): probe, feeds, clouds, floor → publish. Runs on a worker thread.

Heavy work lives elsewhere so this loop keeps up with the cameras (≥30 fps):
  capture + align  → one process per camera (capture.py)
  person detection → its own process (detect_worker.py)
  body pose        → its own thread, compiled + batched across cameras (body.py)

World frame = the first camera's colour optical frame (x right, y down, z forward, metres).
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
import traceback
from typing import Callable

import cv2
import numpy as np

from .camera import Frame
from .config import Config
from .geometry import FloorEstimate, fit_floor, point_cloud
from .rig import Rig, RigCalibrator, RigCamera, RigHealth, transform
from .skeleton import BONES, JOINT_INDEX, JOINTS, LEG_JOINTS

log = logging.getLogger(__name__)

# Binary websocket messages: 8-byte header (u8 kind, u8 camera index, 2 pad, u32 seq).
MSG_MESH, MSG_JPEG, MSG_CLOUD = 1, 2, 3
FUSED = 255  # camera index of the fused (world-frame) cloud

# Leg segments the probe tip is measured against.
LEG_SEGMENTS = {
    f"{side}_{name}": (f"{side}_{a}", f"{side}_{b}")
    for side in ("left", "right")
    for name, a, b in (("thigh", "hip", "knee"), ("shin", "knee", "ankle"),
                       ("foot", "heel", "big_toe"))
}

Publish = Callable[[dict, list[bytes]], None]


def _pack(kind: int, cam: int, seq: int, payload: bytes) -> bytes:
    return struct.pack("<BBxxI", kind, cam, seq) + payload


def _mm(xyz: np.ndarray) -> bytes:
    return np.clip(np.round(xyz * 1000), -32768, 32767).astype("<i2").tobytes()


def _vec(v, nd: int = 4) -> list[float]:
    return [round(float(x), nd) for x in v]


class Pipeline:
    def __init__(self, cfg: Config, publish: Publish):
        self.cfg = cfg
        self.publish = publish
        self.hello: dict | None = None
        self.status: dict = {"type": "status", "state": "starting", "message": "Starting…"}
        self.faces: bytes | None = None
        self._commands: queue.SimpleQueue[dict] = queue.SimpleQueue()
        # Display depth window (m) set from the dashboard; clouds, and optionally feeds,
        # only show pixels whose depth — from their own camera — falls inside it.
        self.depth_range = (cfg.cameras.depth_min_m, cfg.cameras.depth_max_m)
        self.mask_feeds = False
        self._health = RigHealth()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._main, name="pipeline", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=15)

    def command(self, msg: dict) -> None:
        """Thread-safe: dashboard commands are applied at the top of the next iteration."""
        self._commands.put(msg)

    # ------------------------------------------------------------------ setup
    def _set_status(self, state: str, message: str) -> None:
        log.info("[%s] %s", state, message)
        self.status = {"type": "status", "state": state, "message": message}
        self.publish(self.status, [])

    def _main(self) -> None:
        rig = detector = body = None
        try:
            from .body import BodyWorker
            from .detect_worker import DetectorProcess
            from .estimators import build_estimator

            self._set_status("loading", "Starting camera processes…")
            rig = Rig(self.cfg.cameras, self.cfg.resolve(self.cfg.cameras.extrinsics_path))
            # The detector loads in its own process while we load/compile the pose model here.
            detector = DetectorProcess(self.cfg, [rc.camera.buffer for rc in rig.cameras])
            self._set_status("loading", f"Loading pose model ({self.cfg.estimator.backend})…")
            estimator = build_estimator(self.cfg)
            if estimator.faces is not None:
                self.faces = estimator.faces.astype("<u4").tobytes()
            tracker = None
            if self.cfg.probe.enabled:
                from .probe.tracker import ProbeTracker

                tracker = ProbeTracker(self.cfg)

            body = BodyWorker(self.cfg, rig, estimator, detector)
            body.start()
            self._set_status("loading", "Compiling pose model and loading detector "
                                        "(~30 s on first start)…")
            detector.wait_ready()
            body.ready.wait()
            if body.error:
                raise RuntimeError(f"pose model: {body.error}")

            self._send_hello(rig, estimator.name, tracker)
            names = ", ".join(f"{rc.camera.name} {rc.serial}" for rc in rig.cameras)
            msg = f"Streaming from {len(rig.cameras)} camera(s): {names}"
            if rig.warnings:
                msg += " — " + "; ".join(rig.warnings)
            self._set_status("running", msg)
            self._loop(rig, body, estimator.name, tracker)
        except Exception as e:  # surface every failure in the dashboard
            log.error("pipeline failed:\n%s", traceback.format_exc())
            self._set_status("error", _explain(e))
        finally:
            if body is not None:
                body.stop()
            if detector is not None:
                detector.close()
            if rig is not None:
                rig.close()

    def _send_hello(self, rig: Rig, backend: str, tracker) -> None:
        probe = None
        if tracker is not None:
            g = tracker.geometry
            probe = {
                "model_url": "/api/probe/model.glb",
                "tag_url": "/api/probe/tag/{id}.png",
                "marker_length": g.marker_length,
                # Tag PNGs carry a white margin: 600 px code + 2×50 px margin.
                "tag_size": g.marker_length * 700 / 600,
                "tip": list(g.tip),
                "mounts": [{"id": m.marker_id, "face": m.face, "center": list(m.center),
                            "x_axis": list(m.x_axis), "y_axis": list(m.y_axis)}
                           for m in g.mounts.values()],
                "method": tracker.filter.method,
            }
        self.hello = {
            "type": "hello",
            "backend": backend,
            "cameras": [rc.describe() for rc in rig.cameras],
            "joints": JOINTS,
            "leg_joints": LEG_JOINTS,
            "bones": BONES,
            "has_mesh": self.faces is not None,
            "probe": probe,
            "depth_limits": [self.cfg.cameras.depth_min_m, self.cfg.cameras.depth_max_m],
        }
        self.publish(self.hello, [])

    # ------------------------------------------------------------------ loop
    def _loop(self, rig: Rig, body, backend: str, tracker) -> None:
        cfg, sc = self.cfg, self.cfg.scene
        n = len(rig.cameras)
        frames: list[Frame | None] = [None] * n
        floor: tuple[np.ndarray, float] | None = None
        floor_estimate = FloorEstimate()
        calibrator: RigCalibrator | None = None
        body_seq_sent = -1
        fps, last_t, seq = 0.0, time.perf_counter(), 0

        while not self._stop.is_set():
            calibrator = self._handle_commands(rig, tracker, calibrator, backend)
            if any(rc.camera.poll_info() for rc in rig.cameras):
                self._send_hello(rig, backend, tracker)  # a camera reconnected in a new mode

            # The world camera paces the loop; every other camera contributes the frame it
            # captured closest in time to the world camera's (device timestamps).
            prev0 = frames[0]
            f0 = rig.world.camera.latest(prev0.index if prev0 else -1, timeout=0.1)
            if f0 is None:
                continue
            frames[0], fresh = f0, [True] + [False] * (n - 1)
            skew = 0.0
            for rc in rig.cameras[1:]:
                f = rc.camera.paired(f0.timestamp)
                if f is not None and (frames[rc.index] is None or f.index != frames[rc.index].index):
                    frames[rc.index], fresh[rc.index] = f, True
                    skew = max(skew, abs(f.timestamp - f0.timestamp))
            seq += 1

            state = body.latest()
            views: list[dict] = [{"cam": rc.index, "state": rc.camera.state,
                                  **state.views.get(rc.index, {})} for rc in rig.cameras]
            binaries: list[bytes] = []
            if state.vertices is not None and state.seq != body_seq_sent \
                    and cfg.estimator.sam3d_body.send_mesh:
                binaries.append(_pack(MSG_MESH, 0, seq, struct.pack("<I", len(state.vertices))
                                      + _mm(state.vertices)))
            body_seq_sent = state.seq

            t0 = time.perf_counter()
            probe = None
            if tracker is not None:
                pf = self._track_probe(rig, frames, fresh, tracker, f0.timestamp)
                probe = self._probe_json(pf, rig, frames, views, tracker,
                                         body.joints_at(f0.timestamp))
                self._health.add(rig.world.serial, pf.world_poses, f0.timestamp)
                if calibrator is not None:
                    try:
                        calibrator.add(rig.world.serial, pf.cam_poses)
                        calibrator = self._check_calibration(rig, calibrator, backend, tracker)
                    except Exception as e:  # a failed calibration must not stop streaming
                        log.error("rig calibration failed:\n%s", traceback.format_exc())
                        self._calibration_result("failed", f"Calibration error: {type(e).__name__}: {e}")
                        calibrator = None
            probe_ms = (time.perf_counter() - t0) * 1000

            for rc in rig.cameras:
                if fresh[rc.index]:
                    binaries.append(_pack(MSG_JPEG, rc.index, seq,
                                          self._preview(frames[rc.index])))
            # One fused cloud from every calibrated camera (world frame, de-duplicated);
            # uncalibrated cameras take turns sending their own until they're placed.
            if sc.point_cloud and seq % sc.point_cloud_every_n == 0:
                calibrated = [rc for rc in rig.cameras if rc.calibrated and frames[rc.index] is not None]
                loose = [rc for rc in rig.cameras if not rc.calibrated and frames[rc.index] is not None]
                tick = seq // sc.point_cloud_every_n
                if loose and tick % 2:
                    rc = loose[(tick // 2) % len(loose)]
                    xyz, rgb = self._cloud(frames[rc.index], rc)
                    binaries.append(_pack(MSG_CLOUD, rc.index, seq,
                                          struct.pack("<I", len(xyz)) + _mm(xyz) + rgb.tobytes()))
                elif calibrated:
                    xyz, rgb = self._fused_cloud(frames, calibrated)
                    binaries.append(_pack(MSG_CLOUD, FUSED, seq,
                                          struct.pack("<I", len(xyz)) + _mm(xyz) + rgb.tobytes()))
            if sc.floor_detection and (floor is None or seq % sc.floor_every_n == 0):
                xyz, _ = self._cloud(frames[0], rig.world, full_range=True)
                floor = floor_estimate.update(fit_floor(xyz, sc.floor_max_tilt_deg))

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / (now - last_t) if fps else 1.0 / (now - last_t)
            last_t = now
            msg = {
                "type": "frame",
                "seq": seq,
                "fps": round(fps, 1),
                "body_fps": round(state.fps, 1),
                "latency_ms": round((time.time() - frames[0].timestamp) * 1000, 1),
                "sync_ms": round(skew * 1000, 1),  # capture-time gap between paired camera frames
                "timings": {"pose_ms": round(state.pose_ms, 1), "probe_ms": round(probe_ms, 1),
                            "detection_age_ms": None if state.detection_age_ms is None
                            else round(state.detection_age_ms)},
                "people": state.people,
                "person": state.person,
                "probe": probe,
                "views": views,
                "depth_range": {"min": self.depth_range[0], "max": self.depth_range[1],
                                "mask_feeds": self.mask_feeds},
                "calibration": None if calibrator is None else {
                    "progress": calibrator.progress(), "target": calibrator.target},
                "rig_health": self._health.report(f0.timestamp),
                "floor": None if floor is None else {
                    "normal": [float(v) for v in floor[0]], "height": round(floor[1], 4)},
            }
            self.publish(msg, binaries)

    # ------------------------------------------------------------------ probe
    def _track_probe(self, rig: Rig, frames, fresh, tracker, timestamp: float):
        from .probe.tracker import ProbeInput

        inputs = []
        for rc in rig.cameras:
            if not fresh[rc.index]:
                continue
            f, K = frames[rc.index], rc.camera.intrinsics
            inputs.append(ProbeInput(rc.serial, rc.T_world_camera, K.matrix().astype(np.float64),
                                     K.dist_coeffs(), f.color, f.depth))
        return tracker.track(inputs, timestamp)

    def _probe_json(self, pf, rig: Rig, frames, views, tracker, joints) -> dict:
        by_serial = {rc.serial: rc for rc in rig.cameras}
        for serial, view in pf.views.items():
            rc = by_serial[serial]
            h, w = frames[rc.index].color.shape[:2]
            kept = {m.marker_id for m in view.markers}
            dets = views[rc.index].setdefault("markers", [])
            for m in view.markers:
                dets.append({"id": m.marker_id, "corners": [_vec(c / [w, h]) for c in m.image_corners]})
            views[rc.index]["rejected_tags"] = [i for i in view.rejected if i not in kept]
        est = pf.estimate
        probe = {"tracked": est is not None, "method": tracker.filter.method,
                 "cameras": len(pf.world_poses)}
        if pf.refined is not None:
            r = pf.refined
            probe["fit"] = {"rms_px": round(r.rms_px, 2), "corners": r.corners,
                            "depth_samples": r.depth_samples,
                            "depth_rms_mm": None if np.isnan(r.depth_rms_mm) else round(r.depth_rms_mm, 1)}
        if est is None:
            return probe
        tip = tracker.tip_world(est)
        probe.update({
            "marker_ids": list(est.marker_ids),
            "position": _vec(est.position, 5),
            "rotation": _vec(est.rotation_matrix.reshape(-1), 5),
            "tip": _vec(tip, 5),
            "nearest": _nearest_segment(tip, joints),
        })
        # Where the tip lands in each camera image (for the feed overlays).
        for rc in rig.cameras:
            if not rc.calibrated or frames[rc.index] is None:
                continue
            p_cam = transform(np.linalg.inv(rc.T_world_camera), tip[None])[0]
            if p_cam[2] <= 0.01:
                continue
            K = rc.camera.intrinsics
            uv, _ = cv2.projectPoints(p_cam[None], np.zeros(3), np.zeros(3),
                                      K.matrix().astype(np.float64), K.dist_coeffs())
            u, v = uv.reshape(2)
            views[rc.index]["tip"] = [round(float(u / K.width), 4), round(float(v / K.height), 4)]
        return probe

    # ------------------------------------------------------------------ commands
    def _handle_commands(self, rig, tracker, calibrator, backend):
        while True:
            try:
                msg = self._commands.get_nowait()
            except queue.Empty:
                return calibrator
            cmd = msg.get("cmd")
            try:
                if cmd == "probe_filter" and tracker is not None:
                    tracker.set_method(str(msg.get("method")))
                elif cmd == "probe_reset" and tracker is not None:
                    tracker.filter.reset()
                elif cmd == "depth_range":
                    self._set_depth_range(msg)
                elif cmd == "calibrate_rig":
                    calibrator = self._start_calibration(rig, tracker)
                elif cmd == "cancel_calibration" and calibrator is not None:
                    calibrator = None
                    self._calibration_result("cancelled", "Calibration cancelled.")
            except ValueError as e:
                log.warning("bad command %s: %s", msg, e)

    def _start_calibration(self, rig: Rig, tracker) -> RigCalibrator | None:
        if tracker is None:
            self._calibration_result("failed", "Probe tracking is disabled (probe.enabled).")
            return None
        if len(rig.cameras) < 2:
            self._calibration_result("failed", "Only one camera connected — nothing to calibrate.")
            return None
        self._calibration_result("collecting", "Hold the probe still where every camera sees "
                                 "its tags; move it to a new spot every few seconds.")
        return RigCalibrator(self.cfg.probe.calibration_samples)

    def _check_calibration(self, rig: Rig, cal: RigCalibrator, backend: str, tracker):
        others = [rc.serial for rc in rig.cameras[1:]]
        if not cal.finished(others):
            return cal
        solved, failed = [], []
        for serial in others:
            res = cal.solve(serial)
            if res is None:
                failed.append(serial)
                continue
            T, meta = res
            rig.set_extrinsic(serial, T, meta)
            solved.append(f"{serial}: {meta['samples']} samples, ±{meta['stderr_mm']} mm / "
                          f"±{meta['stderr_deg']}° (frame scatter {meta['spread_mm']} mm)")
        if solved:
            self._send_hello(rig, backend, tracker)
        text = "; ".join(solved)
        if failed:
            text += ("; " if text else "") + ("not enough steady views for " + ", ".join(failed))
        self._calibration_result("done" if solved else "failed", text)
        return None

    def _calibration_result(self, state: str, message: str) -> None:
        log.info("rig calibration %s: %s", state, message)
        self.publish({"type": "calibration", "state": state, "message": message}, [])

    # ------------------------------------------------------------------ helpers
    def _set_depth_range(self, msg: dict) -> None:
        c = self.cfg.cameras
        lo = float(msg.get("min", self.depth_range[0]))
        hi = float(msg.get("max", self.depth_range[1]))
        lo = min(max(lo, c.depth_min_m), c.depth_max_m)
        hi = min(max(hi, c.depth_min_m), c.depth_max_m)
        if hi - lo < 0.02:  # keep a sliver open rather than an empty scene
            hi = min(lo + 0.02, c.depth_max_m)
        self.depth_range = (round(lo, 3), round(hi, 3))
        if "mask_feeds" in msg:
            self.mask_feeds = bool(msg["mask_feeds"])

    def _cloud(self, frame: Frame, rc: RigCamera, full_range: bool = False):
        c, sc = self.cfg.cameras, self.cfg.scene
        lo, hi = (c.depth_min_m, c.depth_max_m) if full_range else self.depth_range
        return point_cloud(frame.color, frame.depth, rc.camera.intrinsics,
                           sc.point_cloud_stride, lo, hi)

    def _fused_cloud(self, frames, cams: list[RigCamera]) -> tuple[np.ndarray, np.ndarray]:
        """All calibrated cameras' clouds in the world frame, one point per voxel."""
        parts = [self._cloud(frames[rc.index], rc) for rc in cams]
        xyz = np.concatenate([transform(rc.T_world_camera, p[0]) for rc, p in zip(cams, parts)])
        rgb = np.concatenate([p[1] for p in parts])
        return voxel_dedupe(xyz.astype(np.float32), rgb, self.cfg.scene.fused_voxel_m)

    def _preview(self, frame: Frame) -> bytes:
        s = self.cfg.server
        h, w = frame.color.shape[:2]
        size = (s.preview_width, round(h * s.preview_width / w))
        small = cv2.resize(frame.color, size, interpolation=cv2.INTER_AREA)
        if self.mask_feeds:
            # Background removal: black out pixels outside the depth window (and no-depth pixels).
            d = cv2.resize(frame.depth, size, interpolation=cv2.INTER_NEAREST)
            lo, hi = self.depth_range
            small = small * ((d >= lo) & (d <= hi))[..., None].astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", small[:, :, ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, s.preview_jpeg_quality])
        return buf.tobytes() if ok else b""


def voxel_dedupe(xyz: np.ndarray, rgb: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep one point per voxel so overlapping cameras don't double the density."""
    if len(xyz) == 0:
        return xyz, rgb
    q = np.floor(xyz / voxel).astype(np.int64)
    q -= q.min(0)
    dims = q.max(0) + 1
    keys = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, first = np.unique(keys, return_index=True)
    return xyz[first], rgb[first]


def _nearest_segment(tip: np.ndarray, joints: np.ndarray | None) -> dict | None:
    """Closest leg bone to the probe tip, as distance to the bone's axis."""
    if joints is None:
        return None
    best = None
    for name, (a, b) in LEG_SEGMENTS.items():
        pa, pb = joints[JOINT_INDEX[a]], joints[JOINT_INDEX[b]]
        if pa[3] <= 0 or pb[3] <= 0 or not (np.isfinite(pa[:3]).all() and np.isfinite(pb[:3]).all()):
            continue
        d = pb[:3] - pa[:3]
        t = float(np.clip(np.dot(tip - pa[:3], d) / max(np.dot(d, d), 1e-9), 0, 1))
        closest = pa[:3] + t * d
        dist = float(np.linalg.norm(tip - closest))
        if best is None or dist < best["distance_mm"] / 1000:
            best = {"segment": name, "distance_mm": round(dist * 1000, 1),
                    "along": round(t, 3), "point": _vec(closest, 5)}
    return best


def _explain(e: Exception) -> str:
    text = f"{type(e).__name__}: {e}"
    if any(s in text for s in ("GatedRepo", "401", "403", "gated", "restricted")):
        return ("SAM 3D Body weights are gated. Request access at "
                "huggingface.co/facebook/sam-3d-body-dinov3, put HF_TOKEN in .env, "
                "run `uv run vigil setup` — or set estimator.backend: vitpose_depth.")
    if "No device connected" in text or "RealSense" in text:
        return f"RealSense error — {text}"
    return text
