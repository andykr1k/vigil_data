"""Record → files any tool can open → replay through the dashboard protocol."""

import json
import time

import cv2
import numpy as np

from vigil.camera import Frame
from vigil.config import RecordingConfig
from vigil.pipeline import FUSED, MSG_CLOUD, MSG_JPEG, MSG_ULTRASOUND
from vigil.recorder import Recorder
from vigil.replay import Player, list_recordings, read_ply

CAMS = [{"index": 0, "serial": "A", "width": 64, "height": 48, "fx": 50.0, "fy": 50.0,
         "cx": 32.0, "cy": 24.0, "T_world_camera": None, "dist": [0.0] * 5}]
HELLO = {"type": "hello", "cameras": CAMS}


def record(tmp_path, n=5):
    rec = Recorder(RecordingConfig(), tmp_path, CAMS,
                   {"procedure": "lower_limb", "world_frame": "cam 1", "depth_window": [0.2, 6.0],
                    "hello": HELLO})
    rng = np.random.default_rng(0)
    for seq in range(1, n + 1):
        f = Frame(rng.integers(0, 255, (48, 64, 3), np.uint8),
                  np.full((48, 64), 1.234, np.float32), 100.0 + seq / 30, seq)
        us = cv2.imencode(".jpg", np.full((20, 20), seq * 10, np.uint8))[1].tobytes() if seq == 2 else None
        rec.add_frame(seq, [f], [True], {"t": f.timestamp, "frame": {"type": "frame", "seq": seq}}, us)
        if seq in (1, 4):
            rec.add_cloud(seq, None, rng.random((100, 3)).astype(np.float32),
                          rng.integers(0, 255, (100, 3), np.uint8))
    return rec, rec.close()


def test_recording_is_readable_by_other_tools(tmp_path):
    rec, summary = record(tmp_path)
    p = rec.path
    assert summary["frames"] == 5 and summary["clouds"] == 2 and summary["dropped"] == 0
    depth = cv2.imread(str(p / "cam0/depth/000003.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and depth[0, 0] == 1234  # millimetres
    assert cv2.imread(str(p / "cam0/color/000003.jpg")).shape == (48, 64, 3)
    xyz, rgb = read_ply(p / "cloud/000004.ply")
    assert xyz.shape == (100, 3) and rgb.dtype == np.uint8
    assert b"format binary_little_endian 1.0" in (p / "cloud/000004.ply").read_bytes()[:60]
    meta = json.loads((p / "meta.json").read_text())
    assert meta["cameras"][0]["fx"] == 50.0 and meta["summary"]["frames"] == 5
    lines = [json.loads(ln) for ln in open(p / "frames.jsonl")]
    assert [ln["seq"] for ln in lines] == [1, 2, 3, 4, 5]
    assert list_recordings(tmp_path)[0]["name"] == p.name


def test_replay_sends_the_recording_and_seeks(tmp_path):
    rec, _ = record(tmp_path)
    sent = []
    player = Player(rec.path, lambda msg, bins: sent.append((msg, bins)))
    assert player.hello == HELLO
    player.control({"play": False})
    player.start()
    deadline = time.monotonic() + 2
    while not sent and time.monotonic() < deadline:
        time.sleep(0.01)
    msg, bins = sent[-1]
    assert msg["seq"] == 1 and msg["replay"]["count"] == 5
    assert {b[0] for b in bins} == {MSG_JPEG, MSG_CLOUD}
    assert next(b for b in bins if b[0] == MSG_CLOUD)[1] == FUSED

    sent.clear()
    player.control({"seek": 3})  # frame seq 4: its own cloud; ultrasound from seq 2
    while not sent and time.monotonic() < deadline + 2:
        time.sleep(0.01)
    msg, bins = sent[-1]
    assert msg["seq"] == 4
    assert {b[0] for b in bins} == {MSG_JPEG, MSG_CLOUD, MSG_ULTRASOUND}

    sent.clear()
    player.control({"play": True, "speed": 8})  # plays to the end, then pauses
    time.sleep(0.3)
    assert sent[-1][0]["seq"] == 5 and not player.playing
    player.close()
