"""Linear constant-velocity Kalman filter for 3D position."""

import math

import cv2
import numpy as np

from .defaults import (
    LINEAR_ACCELERATION_NOISE_METERS_PER_SECOND2,
    POSITION_MEASUREMENT_NOISE_METERS,
)


class PositionKalmanFilter:
    """Track 3D position and velocity from noisy position measurements."""

    def __init__(
        self,
        initial_position: np.ndarray,
        timestamp: float,
        *,
        measurement_noise_meters: float = POSITION_MEASUREMENT_NOISE_METERS,
        acceleration_noise_meters_per_second2: float = (
            LINEAR_ACCELERATION_NOISE_METERS_PER_SECOND2
        ),
    ):
        # State: [x, y, z, velocity_x, velocity_y, velocity_z]
        # Measurement: [x, y, z]
        self.filter = cv2.KalmanFilter(6, 3, 0, cv2.CV_64F)
        self.filter.measurementMatrix = np.hstack(
            (np.eye(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64))
        )
        self.set_tuning(
            measurement_noise_meters,
            acceleration_noise_meters_per_second2,
        )
        self.filter.errorCovPost = np.diag(
            [measurement_noise_meters**2] * 3 + [1.0] * 3
        ).astype(np.float64)
        self.filter.statePost = np.vstack(
            (
                np.asarray(initial_position, dtype=np.float64).reshape(3, 1),
                np.zeros((3, 1), dtype=np.float64),
            )
        )
        self.last_timestamp = timestamp

    def set_tuning(
        self,
        measurement_noise_meters: float,
        acceleration_noise_meters_per_second2: float,
    ) -> None:
        """Change noise assumptions without discarding position/velocity history."""
        if not math.isfinite(measurement_noise_meters) or measurement_noise_meters <= 0:
            raise ValueError("measurement noise must be finite and positive")
        if (
            not math.isfinite(acceleration_noise_meters_per_second2)
            or acceleration_noise_meters_per_second2 < 0
        ):
            raise ValueError("acceleration noise must be finite and nonnegative")
        self.acceleration_noise_meters_per_second2 = (
            acceleration_noise_meters_per_second2
        )
        self.filter.measurementNoiseCov = (
            np.eye(3, dtype=np.float64) * measurement_noise_meters**2
        )

    def _set_motion_model(self, elapsed_seconds: float) -> None:
        dt = min(max(elapsed_seconds, 1 / 240), 0.25)
        self.filter.transitionMatrix = np.block(
            [
                [np.eye(3), np.eye(3) * dt],
                [np.zeros((3, 3)), np.eye(3)],
            ]
        ).astype(np.float64)

        acceleration_variance = self.acceleration_noise_meters_per_second2**2
        self.filter.processNoiseCov = acceleration_variance * np.block(
            [
                [np.eye(3) * dt**4 / 4, np.eye(3) * dt**3 / 2],
                [np.eye(3) * dt**3 / 2, np.eye(3) * dt**2],
            ]
        ).astype(np.float64)

    def update(self, measured_position: np.ndarray, timestamp: float) -> np.ndarray:
        """Predict forward, correct with a measurement, and return XYZ."""
        self._set_motion_model(timestamp - self.last_timestamp)
        self.filter.predict()
        corrected_state = self.filter.correct(
            np.asarray(measured_position, dtype=np.float64).reshape(3, 1)
        )
        self.last_timestamp = timestamp
        return corrected_state[:3, 0].copy()
