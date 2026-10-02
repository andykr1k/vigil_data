"""Fast loop (camera rate): probe, feeds, clouds, floor → publish. Runs on a worker thread.

Heavy work lives elsewhere so this loop keeps up with the cameras (≥30 fps):
  capture + align  → one process per camera (capture.py)
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
from .config import PROJECT_ROOT, Config
from .geometry import FloorEstimate, fit_floor, point_cloud
from .rig import Rig, RigCalibrator, RigCamera, RigHealth, transform
from .skeleton import BONES, JOINT_INDEX, JOINTS, REGIONS

log = logging.getLogger(__name__)

# Binary websocket messages: 8-byte header (u8 kind, u8 camera index, 2 pad, u32 seq).
MSG_MESH, MSG_JPEG, MSG_CLOUD, MSG_ULTRASOUND = 1, 2, 3, 4
FUSED = 255  # camera index of the fused (world-frame) cloud

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
        self.clarius = None
        self.clarius_error: str | None = None
        self.procedure: str | None = None  # chosen in the dashboard before anything loads
        self.application: str | None = None  # its probe preset
        self.region = REGIONS["lower_limb"]  # its body target (leg or chest)
        self._chosen = threading.Event()
        self._us_seq = -1
        self.recorder = None  # recorder.Recorder while recording
        self.player = None  # replay.Player while replaying (live frames aren't sent)
        self._live_hello: dict | None = None
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
        if msg.get("cmd") == "start":
            if not self._chosen.is_set() and msg.get("procedure") in self.cfg.clarius.procedures:
                self.procedure = msg["procedure"]
                self._chosen.set()
            return
        self._commands.put(msg)

    # ------------------------------------------------------------------ setup
    def _set_status(self, state: str, message: str, **extra) -> None:
        log.info("[%s] %s", state, message)
        self._emit({"type": "status", "state": state, "message": message, **extra})

    def _emit(self, status: dict) -> None:
        self.status = status  # a dashboard that connects later gets the latest
        self.publish(status, [])

    def _choose_procedure(self) -> bool:
        """Wait for the dashboard to pick a procedure; nothing loads before that."""
        procs = [{"id": k, "label": k.replace("_", " ").title(), "application": v}
                 for k, v in self.cfg.clarius.procedures.items()]
        self._set_status("select", "Choose a procedure", procedures=procs)
        while not self._chosen.wait(0.2):
            if self._stop.is_set():
                return False
        return True

    def _main(self) -> None:
        rig = detector = body = startup = None
        try:
            
            if not self._choose_procedure():
                return
            
            c = self.cfg.clarius
            self.application = c.procedures[self.procedure]
            self.region = REGIONS[self.procedure]
            label = self.procedure.replace("_", " ").title()

            if self.cfg.testing.mode == 1:
                self._set_status("running", f"{label} · replay testing mode")
                self._testing_loop()
                return
            

            from .body import BodyWorker
            from .detect_worker import DetectorProcess
            from .estimators import build_estimator
            from .startup import Startup

           
            steps = [("cameras", "Cameras connected"), ("pose", "Pose model loaded"),
                    ("tracker", "Probe tracker ready"), ("detector", "Person detector loaded"),
                    ("compile", "Pose model compiled")]
            if c.enabled:
                steps.append(("clarius", "Ultrasound probe"))
            
            startup = Startup(steps, self._emit, PROJECT_ROOT / ".cache" / "startup.json",
                            {"procedure": label}, watch=lambda: self._watch_clarius(startup))
            startup.publish()

            if c.enabled:  # connects in the background while the models load
                try:
                    from .clarius import from_config

                    self.clarius = from_config(self.cfg, self.application)
                    startup.set("clarius", "active", f"{c.model} · {self.application}")
                except Exception as e:  # the cameras keep working without the ultrasound
                    log.warning("Clarius unavailable: %s", e)
                    self.clarius_error = str(e)
                    startup.done("clarius", str(e), state="warn")

            startup.begin("cameras")
            rig = Rig(self.cfg.cameras, self.cfg.resolve(self.cfg.cameras.extrinsics_path))
            names = ", ".join(f"{rc.camera.name} {rc.serial}" for rc in rig.cameras)
            startup.done("cameras", names, state="warn" if rig.warnings else "done")
            # The detector loads in its own process while we load/compile the pose model here.
            detector = DetectorProcess(self.cfg, [rc.camera.buffer for rc in rig.cameras])
            startup.begin("pose", self.cfg.estimator.backend)
            estimator = build_estimator(self.cfg)
            if estimator.faces is not None:
                self.faces = estimator.faces.astype("<u4").tobytes()
            startup.done("pose")
            tracker = None
            startup.begin("tracker")
            if self.cfg.probe.enabled:
                from .probe.tracker import ProbeTracker

                tracker = ProbeTracker(self.cfg)
            startup.done("tracker", "ArUco cube" if tracker else "disabled")

            body = BodyWorker(self.cfg, rig, estimator, detector, self.region)
            body.start()
            startup.begin("detector", self.cfg.detector.model_id)
            detector.wait_ready()
            startup.done("detector")
            startup.begin("compile", "~30 s on first start")
            body.ready.wait()
            if body.error:
                raise RuntimeError(f"pose model: {body.error}")
            startup.done("compile", "")
            if self.clarius is not None:
                self._wait_clarius(startup)
            startup.finish()

            self._send_hello(rig, estimator.name, tracker)
            msg = f"{label} · streaming from {len(rig.cameras)} camera(s): {names}"
            if rig.warnings:
                msg += " — " + "; ".join(rig.warnings)
            self._set_status("running", msg)
            self._loop(rig, body, estimator.name, tracker)
        except Exception as e:  # surface every failure in the dashboard
            log.error("pipeline failed:\n%s", traceback.format_exc())
            extra = {}
            if startup is not None:
                startup.fail(_explain(e))
                extra = {"checks": startup.status()["checks"]}
            self._set_status("error", _explain(e), **extra)
        finally:
            if startup is not None:
                startup.close()
            if self.player is not None:
                self.player.close()
            if self.recorder is not None:
                self.recorder.close()
            if getattr(self, "clarius", None) is not None:
                self.clarius.close()
            if body is not None:
                body.stop()
            if detector is not None:
                detector.close()
            if rig is not None:
                rig.close()

    def _watch_clarius(self, startup) -> None:
        """Live connection state on the probe's check while the models load."""
        if self.clarius is not None:
            startup.set("clarius", detail=self.clarius.snapshot().state)

    def _wait_clarius(self, startup, timeout: float = 30.0) -> None:
        """Give the probe time to finish connecting; never block startup on it."""
        startup.begin("clarius", self.clarius.snapshot().state)
        deadline = time.monotonic() + timeout
        while not self._stop.is_set():
            s = self.clarius.snapshot()
            if s.imaging:
                startup.done("clarius", f"{self.application} · battery {s.battery}%")
                return
            if s.state.startswith("not on the probe") or time.monotonic() > deadline:
                startup.done("clarius", s.error or s.state, state="warn")
                return
            self._stop.wait(0.25)

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
        hello = {
            "type": "hello",
            "backend": backend,
            "cameras": [rc.describe() for rc in rig.cameras],
            "joints": JOINTS,
            "procedure": self.procedure,
            "region": self.region.name,
            "focus_joints": self.region.joints,
            "bones": BONES,
            "has_mesh": self.faces is not None,
            "probe": probe,
            "depth_limits": [self.cfg.cameras.depth_min_m, self.cfg.cameras.depth_max_m],
            "clarius": {"model": self.cfg.clarius.model, "application": self.application}
            if self.cfg.clarius.enabled else None,
        }
        self._live_hello = hello
        if self.player is None:
            self.hello = hello
            self.publish(hello, [])


    def _testing_loop(self) -> None:
        """Keep recording discovery and replay working without live hardware."""
        self._list_recordings()

        while not self._stop.is_set():
            try:
                msg = self._commands.get(timeout=0.2)
            except queue.Empty:
                continue

            cmd = msg.get("cmd")
            try:
                if cmd == "recordings":
                    self._list_recordings()
                elif cmd == "replay":
                    self._replay(str(msg.get("name", "")))
                elif cmd == "replay_ctl" and self.player is not None:
                    self.player.control(msg)
                elif cmd == "replay_stop":
                    self._replay(None)
            except (ValueError, TypeError, KeyError, OSError) as e:
                log.warning("bad testing-mode command %s: %s", msg, e)

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
        clouds = _Worker(self._cloud_message, self._stop)

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
            clarius, us_new = None, None
            if self.clarius is not None:
                us = self.clarius.snapshot()
                clarius = us.json()
                if us.image is not None and us.image_seq != self._us_seq:
                    self._us_seq = us.image_seq
                    us_new = us.image
                    binaries.append(_pack(MSG_ULTRASOUND, 0, seq, us.image))
            elif self.clarius_error:
                clarius = {"state": "unavailable", "error": self.clarius_error}
            # One fused cloud from every calibrated camera (world frame, de-duplicated);
            # uncalibrated cameras take turns sending their own until they're placed.
            # Built on their own thread (the voxel sort takes tens of ms at full density), so
            # the loop hands over the newest frames when it's idle and sends what's finished.
            if sc.point_cloud and seq % sc.point_cloud_every_n == 0 and clouds.idle():
                calibrated = [rc for rc in rig.cameras if rc.calibrated and frames[rc.index] is not None]
                loose = [rc for rc in rig.cameras if not rc.calibrated and frames[rc.index] is not None]
                tick = seq // sc.point_cloud_every_n
                if loose and (tick % 2 or not calibrated):
                    rc = loose[(tick // 2) % len(loose)]
                    clouds.offer((rc.index, list(frames), [rc]))
                elif calibrated:
                    clouds.offer((FUSED, list(frames), calibrated))
            if (cloud := clouds.take()) is not None:
                binaries.append(_pack(MSG_CLOUD, cloud[0], seq, cloud[1]))
                if self.recorder is not None:
                    self.recorder.add_cloud(seq, None if cloud[0] == FUSED else cloud[0], *cloud[2:])
            if sc.floor_detection and (floor is None or seq % sc.floor_every_n == 0):
                xyz, _ = self._cloud(frames[0], rig.world, full_range=True, stride=4)
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
                "clarius": clarius,
                "views": views,
                "depth_range": {"min": self.depth_range[0], "max": self.depth_range[1],
                                "mask_feeds": self.mask_feeds},
                "calibration": None if calibrator is None else {
                    "progress": calibrator.progress(), "target": calibrator.target},
                "rig_health": self._health.report(f0.timestamp),
                "floor": None if floor is None else {
                    "normal": [float(v) for v in floor[0]], "height": round(floor[1], 4)},
            }
            if self.recorder is not None:
                self.recorder.add_frame(seq, frames, fresh, {"t": f0.timestamp, "frame": msg}, us_new)
                msg = dict(msg, recording=self.recorder.status())
            if self.player is None:  # while replaying, the dashboard shows the recording
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
                            "depth_rms_mm": None if np.isnan(r.depth_rms_mm) else round(r.depth_rms_mm, 1),
                            "rig_error_px": None if pf.rig_error_px is None else round(pf.rig_error_px, 1)}
        if est is None:
            return probe
        tip = tracker.tip_world(est)
        probe.update({
            "marker_ids": list(est.marker_ids),
            "position": _vec(est.position, 5),
            "rotation": _vec(est.rotation_matrix.reshape(-1), 5),
            "tip": _vec(tip, 5),
            "nearest": _nearest_segment(tip, joints, self.region.segments),
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
                elif cmd == "clarius_param" and self.clarius is not None:
                    self.clarius.set_param(str(msg.get("name")), float(msg.get("value")))
                elif cmd == "clarius_run" and self.clarius is not None:
                    self.clarius.set_running(bool(msg.get("run")))
                elif cmd == "depth_range":
                    self._set_depth_range(msg)
                elif cmd == "record":
                    self._record(rig, bool(msg.get("on")))
                elif cmd == "recordings":
                    self._list_recordings()
                elif cmd == "replay":
                    self._replay(str(msg.get("name", "")))
                elif cmd == "replay_ctl" and self.player is not None:
                    self.player.control(msg)
                elif cmd == "replay_stop":
                    self._replay(None)
                elif cmd == "calibrate_rig":
                    calibrator = self._start_calibration(rig, tracker)
                elif cmd == "cancel_calibration" and calibrator is not None:
                    calibrator = None
                    self._calibration_result("cancelled", "Calibration cancelled.")
            except (ValueError, TypeError, KeyError, OSError) as e:  # a bad message must not stop the loop
                log.warning("bad command %s: %s", msg, e)

    # ------------------------------------------------------------------ record / replay
    def _record(self, rig: Rig, on: bool) -> None:
        from .recorder import Recorder

        if on and self.recorder is None and self.player is None:
            cams = [dict(rc.describe(), dist=list(rc.camera.intrinsics.coeffs)) for rc in rig.cameras]
            self.recorder = Recorder(
                self.cfg.recording, self.cfg.resolve(self.cfg.recording.dir), cams,
                {"procedure": self.procedure, "application": self.application,
                 "world_frame": "camera 1 colour optical frame (x right, y down, z forward)",
                 "depth_window": list(self.depth_range), "hello": self._live_hello})
            self._recording_msg("recording", f"Recording to {self.recorder.path}")
        elif not on and self.recorder is not None:
            rec, self.recorder = self.recorder, None

            def finish():  # flushing queued writes can take a moment: off the loop
                s = rec.close()
                self._recording_msg("saved", f"Saved {s['seconds']:.0f} s, {s['frames']} frames, "
                                    f"{s['mb']:.0f} MB ({s['dropped']} dropped) to {s['path']}")
                self._list_recordings()

            threading.Thread(target=finish, name="rec-close", daemon=True).start()

    def _recording_msg(self, state: str, message: str) -> None:
        log.info("[recording] %s", message)
        self.publish({"type": "recording", "state": state, "message": message}, [])

    def _list_recordings(self) -> None:
        from .replay import list_recordings

        root = self.cfg.resolve(self.cfg.recording.dir)
        self.publish({"type": "recordings", "items": list_recordings(root) if root.is_dir() else []}, [])

    def _replay(self, name: str | None) -> None:
        """Start replaying a recording (by folder name), or go back to live (None)."""
        from .replay import Player

        if self.player is not None:
            self.player.close()
            self.player = None
        if name:
            if self.recorder is not None:
                self._record(None, False)
            root = self.cfg.resolve(self.cfg.recording.dir)
            path = (root / name).resolve()
            if path.parent != root.resolve() or not (path / "meta.json").is_file():
                raise ValueError(f"no recording named {name!r}")
            player = Player(path, self.publish)
            self.hello = dict(player.hello, replay=True)
            self.publish(self.hello, [])
            self.player = player
            player.start()
        elif self._live_hello is not None:
            self.hello = self._live_hello
            self.publish(self.hello, [])

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

    def _cloud(self, frame: Frame, rc: RigCamera, full_range: bool = False, stride: int | None = None):
        c, sc = self.cfg.cameras, self.cfg.scene
        lo, hi = (c.depth_min_m, c.depth_max_m) if full_range else self.depth_range
        return point_cloud(frame.color, frame.depth, rc.camera.intrinsics,
                           stride or sc.point_cloud_stride, lo, hi)

    def _cloud_message(self, job) -> tuple[int, bytes, np.ndarray, np.ndarray]:
        """(camera id, payload, xyz, rgb): one camera's own cloud, or the fused cloud
        (cam = FUSED)."""
        cam, frames, cams = job
        if cam == FUSED:
            xyz, rgb = self._fused_cloud(frames, cams)
        else:
            xyz, rgb = self._cloud(frames[cam], cams[0])
        return cam, struct.pack("<I", len(xyz)) + _mm(xyz) + rgb.tobytes(), xyz, rgb

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


class _Worker:
    """Runs fn(job) on its own thread, one job at a time; the result is picked up with take()."""

    def __init__(self, fn, stop: threading.Event):
        self._fn, self._stop = fn, stop
        self._job = self._out = None
        self._wake = threading.Event()
        threading.Thread(target=self._run, name="clouds", daemon=True).start()

    def idle(self) -> bool:
        return self._job is None

    def offer(self, job) -> None:
        self._job = job
        self._wake.set()

    def take(self):
        out, self._out = self._out, None
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(0.2):
                continue
            self._wake.clear()
            try:
                self._out = self._fn(self._job)
            except Exception:  # a bad frame must not kill the cloud thread
                log.exception("point cloud failed")
            self._job = None


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


def _nearest_segment(tip: np.ndarray, joints: np.ndarray | None,
                     segments: dict[str, tuple[str, str]]) -> dict | None:
    """Closest target segment (leg bone, chest line) to the probe tip, as distance to its axis."""
    if joints is None:
        return None
    best = None
    for name, (a, b) in segments.items():
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
