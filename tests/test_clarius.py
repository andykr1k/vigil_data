"""Clarius/Solum glue, exercised through the real SDK callbacks (no probe needed)."""

import ctypes as C
from pathlib import Path

import cv2
import numpy as np
import pytest

from vigil.clarius import (
    CONNECTED,
    ERROR_VERSION_MISMATCH,
    IMAGING_READY,
    ClariusProbe,
    _ProcessedImageInfo,
    on_probe_network,
)

SDK = Path(__file__).resolve().parents[1] / "third_party/solum/libsolum.so"
pytestmark = pytest.mark.skipif(not SDK.is_file(), reason="Solum SDK not installed (vigil setup)")


@pytest.fixture
def probe(tmp_path):
    # An address we're not on, so the supervisor idles instead of connecting.
    p = ClariusProbe(SDK, tmp_path / "keys", "10.254.254.1", 5000, "PALHD3", "msk", cert="CERT")
    yield p
    p.close()


def test_image_callback_delivers_jpeg_and_scale(probe):
    ok, jpg = cv2.imencode(".jpg", np.full((480, 640), 128, np.uint8))
    buf = C.create_string_buffer(jpg.tobytes(), len(jpg))
    info = _ProcessedImageInfo(width=640, height=480, imageSize=len(jpg), micronsPerPixel=120.0)
    probe._on_image(C.cast(buf, C.c_void_p), C.pointer(info), 0, None)
    s = probe.snapshot()
    assert s.image == jpg.tobytes() and s.image_seq == 1
    assert s.image_size == (640, 480) and s.microns_per_pixel == 120.0


def test_state_follows_callbacks(probe):
    probe._on_connect(CONNECTED, 5001, b"")
    probe._on_imaging(IMAGING_READY, 1)
    probe._on_cert(317)
    s = probe.snapshot()
    assert s.connected and s.imaging and "ready" in s.state and s.cert_days == 317
    assert probe._ev["connect"].is_set() and probe._ev["app"].is_set()
    assert "image" not in s.json()  # JSON status never carries the image bytes


def test_version_mismatch_releases_the_preset_wait(probe):
    # The probe reports this while it still verifies its firmware; startup retries on it.
    probe._ev["app"].clear()
    probe._on_error(ERROR_VERSION_MISMATCH, b"software versions do not match")
    assert probe._ev["app"].is_set() and probe._error_code == ERROR_VERSION_MISMATCH


def test_on_probe_network():
    assert on_probe_network("127.0.0.1")
    assert not on_probe_network("10.254.254.1")
