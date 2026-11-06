"""SCAN THIGH (pipeline.py): record for the chosen time, stop by itself, then mesh the scan
without clicks. The recorder and the meshing (meshing/auto.py) are stand-ins here."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vigil.config import Config
from vigil.pipeline import Pipeline


class FakeRecorder:
    def __init__(self, path: Path):
        self.path = path

    def close(self) -> dict:
        return {"seconds": 2.0, "frames": 60, "dropped": 0, "mb": 1.0, "path": str(self.path)}


def pipeline(tmp_path):
    msgs = []
    p = Pipeline(Config(recording={"dir": str(tmp_path / "recordings")}), lambda msg, binaries: msgs.append(msg))

    def record(rig, on):
        if on:
            p.recorder = FakeRecorder(tmp_path / "recordings" / "2026-01-01_00-00-00_lower_limb")

    p._record = record
    p._list_recordings = lambda: None
    return p, msgs


def finish(p):
    p.scan["until"] = 0
    p._finish_scan()
    for t in threading.enumerate():
        if t.name == "scan-mesh":
            t.join(10)
            assert not t.is_alive(), "scan worker did not finish"


def test_scan_captures_meshes_and_unfolds_without_replay(tmp_path):
    p, msgs = pipeline(tmp_path)
    p._replay = Mock(side_effect=AssertionError("live scan must not enter replay"))
    params = {"voxel_mm": 4, "max_frames": 12, "window_s": 0.2, "cameras": [0], "align": False}
    p._scan(None, {"cmd": "scan", "seconds": 2, "leg": "left", "params": params,
                   "unfold": {"method": "slim", "iterations": 23}})
    params["cameras"].append(1)  # settings are frozen at capture time
    assert p.scan["seconds"] == 2 and p.scan["leg"] == "left" and p.recorder is not None
    assert msgs[-1]["type"] == "scan" and msgs[-1]["state"] == "capturing"
    p._scan(None, {"cmd": "scan"})  # already scanning: ignored
    report = {"accepted": ["frame00030_t0001.00s_tsdf_x"],
              "attempts": [{"frame": 30, "stem": "frame00030_t0001.00s_tsdf_x", "area_cm2": 900.0, "side": "left"}]}
    source_mesh, source = object(), {"stem": report["accepted"][0]}
    flat_result = SimpleNamespace(flat=SimpleNamespace(stats={"warnings": []}))
    with patch("vigil.meshing.auto.run_recording", return_value=report) as run, \
            patch("vigil.meshing.library.MeshLibrary.load", return_value=(source_mesh, source)), \
            patch("vigil.meshing.flatten.flatten_mesh", return_value=flat_result) as unfold, \
            patch("vigil.meshing.library.MeshLibrary.save_flat", return_value={"stem": "flat_x"}) as save:
        finish(p)
    assert p.scan is None and p.recorder is None and p.player is None and not p._scan_busy
    kw = run.call_args.kwargs
    assert kw["side"] == "left" and kw["count"] == 1 and 0 < kw["window_s"] <= 0.5
    assert kw["window_s"] == 0.2
    assert kw["params"]["voxel_mm"] == 4 and kw["params"]["max_frames"] == 12
    assert kw["params"]["cameras"] == [0] and kw["params"]["align"] is False
    assert unfold.call_args.args[:3] == (source_mesh, "slim", 23)
    save.assert_called_once_with(flat_result, source)
    assert run.call_args.args[1].root == tmp_path / "Meshes"  # next to the recordings, like BUILD MESH
    done = [m for m in msgs if m.get("type") == "scan"][-1]
    assert done["state"] == "done" and done["stem"] == "frame00030_t0001.00s_tsdf_x"
    assert done["flat_stem"] == "flat_x"
    assert any(m.get("state") == "unfolding" and m.get("stem") == done["stem"] for m in msgs)
    assert p.scan_status == done  # reconnecting dashboards can retrieve the result
    p._replay.assert_not_called()
    assert any(m.get("type") == "meshes_changed" for m in msgs)


def test_scan_without_a_mesh_says_why(tmp_path):
    p, msgs = pipeline(tmp_path)
    p._scan(None, {"cmd": "scan", "seconds": 1})
    report = {"accepted": [], "attempts": [{"frame": 15, "reason": "The cameras disagree on the knee by 9.0 cm"}]}
    with patch("vigil.meshing.auto.run_recording", return_value=report):
        finish(p)
    last = [m for m in msgs if m.get("type") == "scan"][-1]
    assert last["state"] == "failed" and "disagree on the knee" in last["message"]


def test_no_scan_while_replaying(tmp_path):
    p, msgs = pipeline(tmp_path)
    p.player = object()
    p._scan(None, {"cmd": "scan"})
    assert p.scan is None and msgs[-1]["state"] == "error"


def test_unfolding_failure_keeps_mesh_and_releases_job(tmp_path):
    from vigil.meshing.routes import _building
    from vigil.meshing.tsdf import MeshError

    p, msgs = pipeline(tmp_path)
    p._scan(None, {"seconds": 0.5, "params": {"window_s": 5}})
    report = {"accepted": ["mesh_x"], "attempts": []}
    with patch("vigil.meshing.auto.run_recording", return_value=report) as run, \
            patch("vigil.meshing.library.MeshLibrary.load", return_value=(object(), {})), \
            patch("vigil.meshing.flatten.flatten_mesh", side_effect=MeshError("surface is not a disk")), \
            patch("vigil.meshing.library.MeshLibrary.save_flat") as save:
        finish(p)
    assert run.call_args.kwargs["window_s"] == pytest.approx(0.2)
    assert p.scan_status["state"] == "partial" and p.scan_status["stem"] == "mesh_x"
    assert "not a disk" in p.scan_status["message"]
    assert p.player is None and not p._scan_busy and not _building.locked()
    save.assert_not_called()


def test_scan_stays_busy_off_live_thread_until_unfolding_finishes(tmp_path):
    p, msgs = pipeline(tmp_path)
    p._scan(None, {})
    entered, release = threading.Event(), threading.Event()

    def unfold(*args):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(flat=SimpleNamespace(stats={}))

    with patch("vigil.meshing.auto.run_recording", return_value={"accepted": ["mesh"], "attempts": []}) as run, \
            patch("vigil.meshing.library.MeshLibrary.load", return_value=(object(), {})), \
            patch("vigil.meshing.flatten.flatten_mesh", side_effect=unfold), \
            patch("vigil.meshing.library.MeshLibrary.save_flat", return_value={"stem": "flat"}):
        try:
            p._finish_scan()
            assert entered.wait(5)
            assert p._scan_busy and p.scan is None and p.recorder is None and p.player is None
            p._scan(None, {})  # a second capture must not start during unfolding
            assert p.scan is None and p.recorder is None
            p.publish({"type": "frame", "seq": 99}, [])  # live delivery remains available
            assert msgs[-1]["seq"] == 99
        finally:
            release.set()
            for t in threading.enumerate():
                if t.name == "scan-mesh":
                    t.join(5)
        run.assert_called_once()
    assert not p._scan_busy and p.scan_status["state"] == "done"


@pytest.mark.parametrize("message", [
    {"unfold": {"method": "unknown"}}, {"unfold": {"iterations": 0}},
    {"seconds": float("nan")}, {"params": {"window_s": float("nan")}},
])
def test_invalid_scan_settings_do_not_start_capture(tmp_path, message):
    p, msgs = pipeline(tmp_path)
    p._scan(None, message)
    assert p.scan is None and p.recorder is None and not p._scan_busy
    assert msgs[-1]["state"] == "error"


def test_testing_mode_one_does_not_start_live_capture(tmp_path):
    p, msgs = pipeline(tmp_path)
    p.cfg.testing.mode = 1
    p._scan(None, {})
    assert p.recorder is None and p.scan is None and msgs[-1]["state"] == "error"


def test_capture_save_failure_releases_scan(tmp_path):
    p, msgs = pipeline(tmp_path)
    p._scan(None, {})
    p.recorder.close = Mock(side_effect=OSError("disk unavailable"))
    finish(p)
    assert not p._scan_busy and p.scan_status["state"] == "failed"
    assert "disk unavailable" in p.scan_status["message"]
