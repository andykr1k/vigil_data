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
from .geometry import fit_floor, point_cloud
from .rig import Rig, RigCalibrator, RigCamera, to_h, transform
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
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._main, name="pipeline", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

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
        }
        self.publish(self.hello, [])

    # ------------------------------------------------------------------ loop
    def _loop(self, rig: Rig, body, backend: str, tracker) -> None:
        cfg, sc = self.cfg, self.cfg.scene
        n = len(rig.cameras)
        frames: list[Frame | None] = [None] * n
        floor: tuple[np.ndarray, float] | None = None
        calibrator: RigCalibrator | None = None
        body_seq_sent = -1
        fps, last_t, seq = 0.0, time.perf_counter(), 0

        while not self._stop.is_set():
            calibrator = self._handle_commands(rig, tracker, calibrator, backend)
            if any(rc.camera.poll_info() for rc in rig.cameras):
                self._send_hello(rig, backend, tracker)  # a camera reconnected in a new mode

            # The world camera paces the loop; the others contribute whatever is newest.
            fresh = [False] * n
            for rc in rig.cameras:
                i, prev = rc.index, frames[rc.index]
                f = rc.camera.latest(prev.index if prev else -1, timeout=0.1 if i == 0 else 0.0)
                if f is not None:
                    frames[i], fresh[i] = f, True
            if not fresh[0]:
                continue
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
                probe, cam_poses = self._track_probe(rig, frames, fresh, tracker, views,
                                                     time.monotonic(), state.joints)
                if calibrator is not None:
                    try:
                        calibrator.add(rig.world.serial, cam_poses)
                        calibrator = self._check_calibration(rig, calibrator, backend, tracker)
                    except Exception as e:  # a failed calibration must not stop streaming
                        log.error("rig calibration failed:\n%s", traceback.format_exc())
                        self._calibration_result("failed", f"Calibration error: {type(e).__name__}: {e}")
                        calibrator = None
            probe_ms = (time.perf_counter() - t0) * 1000

            for rc in rig.cameras:
                if fresh[rc.index]:
                    binaries.append(_pack(MSG_JPEG, rc.index, seq,
                                          self._preview(frames[rc.index].color)))
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
                xyz, _ = self._cloud(frames[0], rig.world)
                floor = _blend_floor(floor, fit_floor(xyz, sc.floor_max_tilt_deg))

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / (now - last_t) if fps else 1.0 / (now - last_t)
            last_t = now
            msg = {
                "type": "frame",
                "seq": seq,
                "fps": round(fps, 1),
                "body_fps": round(state.fps, 1),
                "latency_ms": round((time.time() - frames[0].timestamp) * 1000, 1),
                "timings": {"pose_ms": round(state.pose_ms, 1), "probe_ms": round(probe_ms, 1),
                            "detection_age_ms": None if state.detection_age_ms is None
                            else round(state.detection_age_ms)},
                "people": state.people,
                "person": state.person,
                "probe": probe,
                "views": views,
                "calibration": None if calibrator is None else {
                    "progress": calibrator.progress(), "target": calibrator.target},
                "floor": None if floor is None else {
                    "normal": [float(v) for v in floor[0]], "height": round(floor[1], 4)},
            }
            self.publish(msg, binaries)

    # ------------------------------------------------------------------ probe
    def _track_probe(self, rig: Rig, frames, fresh, tracker, views, now_mono, joints):
        world_poses = []
        cam_poses = {}
        for rc in rig.cameras:
            if not fresh[rc.index]:
                continue
            frame = frames[rc.index]
            K = rc.camera.intrinsics
            obs = tracker.observe(frame.color, K.matrix().astype(np.float64), K.dist_coeffs(),
                                  rc.serial)
            h, w = frame.color.shape[:2]
            views[rc.index]["markers"] = [
                {"id": m.marker_id, "corners": [_vec(c / [w, h]) for c in m.image_corners]}
                for m in obs.markers
            ]
            if obs.pose is None:
                continue
            cam_poses[rc.serial] = obs.pose
            if rc.calibrated:
                T = rc.T_world_camera @ to_h(obs.pose.rotation_matrix, obs.pose.position)
                world_poses.append(type(obs.pose)(obs.pose.marker_ids, T[:3, 3], T[:3, :3]))

        est = tracker.update(world_poses, now_mono)
        probe = {"tracked": est is not None, "method": tracker.filter.method,
                 "cameras": len(world_poses)}
        if est is None:
            return probe, cam_poses
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
        return probe, cam_poses

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
    def _cloud(self, frame: Frame, rc: RigCamera):
        c, sc = self.cfg.cameras, self.cfg.scene
        return point_cloud(frame.color, frame.depth, rc.camera.intrinsics,
                           sc.point_cloud_stride, c.depth_min_m, c.depth_max_m)

    def _fused_cloud(self, frames, cams: list[RigCamera]) -> tuple[np.ndarray, np.ndarray]:
        """All calibrated cameras' clouds in the world frame, one point per voxel."""
        parts = [self._cloud(frames[rc.index], rc) for rc in cams]
        xyz = np.concatenate([transform(rc.T_world_camera, p[0]) for rc, p in zip(cams, parts)])
        rgb = np.concatenate([p[1] for p in parts])
        return voxel_dedupe(xyz.astype(np.float32), rgb, self.cfg.scene.fused_voxel_m)

    def _preview(self, rgb: np.ndarray) -> bytes:
        s = self.cfg.server
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (s.preview_width, round(h * s.preview_width / w)),
                           interpolation=cv2.INTER_AREA)
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


def _blend_floor(prev, new, alpha: float = 0.3):
    if new is None:
        return prev
    if prev is None:
        return new
    n = (1 - alpha) * prev[0] + alpha * new[0]
    return n / np.linalg.norm(n), (1 - alpha) * prev[1] + alpha * new[1]


def _explain(e: Exception) -> str:
    text = f"{type(e).__name__}: {e}"
    if any(s in text for s in ("GatedRepo", "401", "403", "gated", "restricted")):
        return ("SAM 3D Body weights are gated. Request access at "
                "huggingface.co/facebook/sam-3d-body-dinov3, put HF_TOKEN in .env, "
                "run `uv run vigil setup` — or set estimator.backend: vitpose_depth.")
    if "No device connected" in text or "RealSense" in text:
        return f"RealSense error — {text}"
    return text
