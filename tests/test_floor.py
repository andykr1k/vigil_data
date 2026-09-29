"""Floor detection: the lowest horizontal surface, stable over time, steep cameras."""

import cv2
import numpy as np

from vigil.geometry import FloorEstimate, fit_floor

rng = np.random.default_rng(2)


def scene(pitch_deg=60.0, cam_height=1.05):
    """Camera looking down steeply at a floor with a chair seat (more points than the floor)."""
    floor = np.c_[rng.uniform(-1.5, 1.5, 3000), np.zeros(3000), rng.uniform(0.3, 3.0, 3000)]
    seat = np.c_[rng.uniform(-0.25, 0.25, 5000), np.full(5000, 0.5), rng.uniform(0.6, 1.1, 5000)]
    pts = np.vstack([floor, seat]) + rng.normal(0, 0.004, (8000, 3))
    # world (y up, camera at height h) → camera optical frame (y down), pitched down
    R = cv2.Rodrigues(np.array([np.radians(pitch_deg), 0, 0]))[0]
    p = pts - [0, cam_height, 0]
    p[:, 1] *= -1
    return p @ R.T


def test_picks_the_floor_not_a_seat_even_looking_steeply_down():
    xyz = scene()
    for seed in range(10):
        n, h = fit_floor(xyz, 75, rng=np.random.default_rng(seed))
        assert abs(h - 1.05) < 0.02, h  # the seat is 0.55 m below the camera


def test_estimate_ignores_one_off_fits_and_follows_a_real_change():
    floor = (np.array([0.0, -1.0, 0.0]), 1.05)
    seat = (np.array([0.0, -1.0, 0.0]), 0.55)
    est = FloorEstimate()
    for fit in (floor, floor, seat, floor, seat, floor):
        est.update(fit)
    assert abs(est.plane[1] - 1.05) < 1e-6  # stray seat fits never averaged in
    moved = (np.array([0.0, -1.0, 0.0]), 1.30)  # camera actually raised
    for _ in range(3):
        est.update(moved)
    assert abs(est.plane[1] - 1.30) < 1e-6
