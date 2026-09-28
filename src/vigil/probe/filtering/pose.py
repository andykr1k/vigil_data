import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


#POS tells filters how noisy the PnP is assumed to be I thkn
#Linear 
#max how long filter's previous state may remain usefful
from .defaults import (
    LINEAR_ACCELERATION_NOISE_METERS_PER_SECOND2,
    MAX_MISSING_SECONDS,
    POSITION_MEASUREMENT_NOISE_METERS,
)
from ..object_pose import ObjectPose
from .position import PositionKalmanFilter



#JUST DIFFERENT FILTERS
FILTER_METHODS = {
    "raw": "Raw pose (no filter)",
    "kalman": "Linear Kalman (XYZ only)",
    "ekf": "SE(3) pose error-state EKF",
    "one_euro": 
    "One Euro filter (XYZ + rotation)",
}

#ig UI descriptions
FILTER_DESCRIPTIONS = {
    "raw": "Uses each joint-PnP measurement directly, with no temporal model.",
    "kalman": "Estimates XYZ with a linear Kalman filter; rotation stays raw.",
    "ekf": (
        "Estimates the SE(3) pose with position/velocity and quaternion/angular-rate "
        "states using a multiplicative error-state EKF."
    ),
    "one_euro": (
        "Adaptive low-pass on XYZ and orientation: more smoothing while "
        "still, less lag while moving. No motion model, unlike the EKF."
    ),
}


TRACKING_MODES = {
    "precision": "Precision tracking (SE(3) ESKF)",
    "guidance": "Responsive guidance (One Euro)",
    "position": "Position-only tracking (Kalman)",
    "raw": "Raw diagnostics (no filter)",
}
MODE_FILTER_METHODS = {
    "precision": "ekf",
    "guidance": "one_euro",
    "position": "kalman",
    "raw": "raw",
}
FILTER_METHOD_MODES = {
    method: mode for mode, method in MODE_FILTER_METHODS.items()
}


DEFAULT_TRACKING_MODE = "precision"
DEFAULT_FILTER_METHOD = MODE_FILTER_METHODS[DEFAULT_TRACKING_MODE]

ROTATION_MEASUREMENT_NOISE_DEGREES = 3.0
ANGULAR_ACCELERATION_NOISE_DEGREES_PER_SECOND2 = 45.0

POSITION_ONE_EURO_MIN_CUTOFF_HZ = 1.0
POSITION_ONE_EURO_BETA = 0.03
ROTATION_ONE_EURO_MIN_CUTOFF_HZ = 1.0
ROTATION_ONE_EURO_BETA = 0.03
ONE_EURO_DERIVATIVE_CUTOFF_HZ = 1.0


TUNING_BOUNDS = {
    "position_measurement_noise_meters": (0.000001, 1.0),
    "acceleration_noise_meters_per_second2": (0.0, 100.0),
    "rotation_measurement_noise_degrees": (0.001, 180.0),
    "angular_acceleration_noise_degrees_per_second2": (0.0, 3600.0),
    "position_min_cutoff_hz": (0.001, 1000.0),
    "position_beta": (0.0, 10000.0),
    "rotation_min_cutoff_hz": (0.001, 1000.0),
    "rotation_beta": (0.0, 10000.0),
    "derivative_cutoff_hz": (0.001, 1000.0),
}


@dataclass(frozen=True)
class FilterTuning:

    position_measurement_noise_meters: float = POSITION_MEASUREMENT_NOISE_METERS
    acceleration_noise_meters_per_second2: float = (
        LINEAR_ACCELERATION_NOISE_METERS_PER_SECOND2
    )
    rotation_measurement_noise_degrees: float = ROTATION_MEASUREMENT_NOISE_DEGREES
    angular_acceleration_noise_degrees_per_second2: float = (
        ANGULAR_ACCELERATION_NOISE_DEGREES_PER_SECOND2
    )
    position_min_cutoff_hz: float = POSITION_ONE_EURO_MIN_CUTOFF_HZ
    position_beta: float = POSITION_ONE_EURO_BETA
    rotation_min_cutoff_hz: float = ROTATION_ONE_EURO_MIN_CUTOFF_HZ
    rotation_beta: float = ROTATION_ONE_EURO_BETA
    derivative_cutoff_hz: float = ONE_EURO_DERIVATIVE_CUTOFF_HZ

    def __post_init__(self) -> None:
        for name, (minimum, maximum) in TUNING_BOUNDS.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{name} must be a finite number between {minimum} and {maximum}"
                )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "FilterTuning":
        """Reject unknown/missing fields instead of silently loading a partial preset."""
        if not isinstance(values, dict) or set(values) != set(TUNING_BOUNDS):
            raise ValueError("tuning must contain exactly the supported parameter names")
        return cls(**values)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product; composing R(left) @ R(right)."""
    return np.concatenate(
        (
            [left[0] * right[0] - np.dot(left[1:], right[1:])],
            left[0] * right[1:] + right[0] * left[1:] + np.cross(left[1:], right[1:]),
        )
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0])


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    """SO(3) exponential map, including a stable small-angle limit."""
    vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    scale = 0.5 - angle**2 / 48 if angle < 1e-8 else math.sin(angle / 2) / angle
    quaternion = np.concatenate(([math.cos(angle / 2)], vector * scale))
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """SO(3) logarithm using the shortest rotation; q and -q are equivalent."""
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-8:
        return 2 * quaternion[1:]
    angle = 2 * math.atan2(vector_norm, float(quaternion[0]))
    return quaternion[1:] * (angle / vector_norm)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return rotation_vector_to_quaternion(rotation_vector)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def right_rotation_jacobian(rotation_vector: np.ndarray) -> np.ndarray:
    """Map additive rotation-vector errors to local orientation errors."""
    angle = float(np.linalg.norm(rotation_vector))
    cross = skew(rotation_vector)
    if angle < 1e-5:
        return np.eye(3) - cross / 2 + cross @ cross / 6
    return (
        np.eye(3)
        - ((1 - math.cos(angle)) / angle**2) * cross
        + ((angle - math.sin(angle)) / angle**3) * (cross @ cross)
    )


class SE3ErrorStateEKF:
    """Multiplicative error-state EKF for an SE(3) pose and its velocity.

    Nominal state: p(3), v(3), q(4), omega(3).
    Covariance/error state: delta_p, delta_v, delta_theta, delta_omega (12).
    Linear velocity is camera-frame; angular velocity/errors are object-local.
    Measurements are the raw combined PnP position and orientation (6 DOF).

    The nominal rigid pose (R, p) lies on SE(3). Rotation corrections use the
    SO(3) exponential map in its tangent space, while translation errors are
    additive in camera coordinates. This is a pose error-state EKF, not an
    invariant/group-affine IEKF.
    """

    def __init__(
        self, initial_pose: ObjectPose, timestamp: float, *, tuning: FilterTuning | None = None
    ) -> None:
        self.position = initial_pose.position.astype(np.float64).reshape(3).copy()
        self.velocity = np.zeros(3)
        self.quaternion = rotation_matrix_to_quaternion(initial_pose.rotation_matrix)
        self.angular_velocity = np.zeros(3)
        self.last_timestamp = timestamp

        self.set_tuning(tuning if tuning is not None else FilterTuning())
        position_variance = self.tuning.position_measurement_noise_meters**2
        rotation_variance = math.radians(self.tuning.rotation_measurement_noise_degrees)**2
        self.covariance = np.diag(
            [position_variance] * 3 + [1.0] * 3
            + [rotation_variance] * 3 + [1.0] * 3
        )
        self.measurement_jacobian = np.zeros((6, 12))
        self.measurement_jacobian[:3, :3] = np.eye(3)
        self.measurement_jacobian[3:, 6:9] = np.eye(3)

    def set_tuning(self, tuning: FilterTuning) -> None:
        """Apply new measurement/process noise, retaining the estimated motion."""
        self.tuning = tuning
        self.measurement_covariance = np.diag(
            [tuning.position_measurement_noise_meters**2] * 3
            + [math.radians(tuning.rotation_measurement_noise_degrees)**2] * 3
        )

    def _transition_matrix(self, dt: float) -> np.ndarray:
        """Analytical Jacobian of the nonlinear prediction in error coordinates."""
        angular_step = self.angular_velocity * dt
        transition = np.eye(12)
        transition[:3, 3:6] = np.eye(3) * dt
        transition[6:9, 6:9] = quaternion_to_rotation_matrix(
            rotation_vector_to_quaternion(-angular_step)
        )
        transition[6:9, 9:12] = right_rotation_jacobian(angular_step) * dt
        return transition

    def _predict(self, dt: float) -> None:
        transition = self._transition_matrix(dt)
        angular_step = self.angular_velocity * dt
        self.position += self.velocity * dt
        self.quaternion = quaternion_multiply(
            self.quaternion, rotation_vector_to_quaternion(angular_step)
        )
        self.quaternion /= np.linalg.norm(self.quaternion)

        # Unknown linear/angular accelerations drive process uncertainty.
        noise_mapping = np.zeros((12, 6))
        noise_mapping[:3, :3] = np.eye(3) * dt**2 / 2
        noise_mapping[3:6, :3] = np.eye(3) * dt
        noise_mapping[6:9, 3:] = right_rotation_jacobian(angular_step) * dt**2 / 2
        noise_mapping[9:12, 3:] = np.eye(3) * dt
        acceleration_covariance = np.diag(
            [self.tuning.acceleration_noise_meters_per_second2**2] * 3
            + [math.radians(self.tuning.angular_acceleration_noise_degrees_per_second2)**2] * 3
        )
        self.covariance = (
            transition @ self.covariance @ transition.T
            + noise_mapping @ acceleration_covariance @ noise_mapping.T
        )

    def update(self, measured_pose: ObjectPose, timestamp: float) -> ObjectPose:
        dt = timestamp - self.last_timestamp
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("EKF timestamps must be finite and strictly increasing")
        self._predict(dt)

        measured_quaternion = rotation_matrix_to_quaternion(measured_pose.rotation_matrix)
        orientation_error = quaternion_to_rotation_vector(
            quaternion_multiply(quaternion_conjugate(self.quaternion), measured_quaternion)
        )
        innovation = np.concatenate(
            (measured_pose.position.reshape(3) - self.position, orientation_error)
        )
        h = self.measurement_jacobian
        r = self.measurement_covariance
        innovation_covariance = h @ self.covariance @ h.T + r
        gain = np.linalg.solve(innovation_covariance, h @ self.covariance).T
        correction = gain @ innovation

        # Joseph form maintains covariance symmetry/positive semidefiniteness.
        residual_mapping = np.eye(12) - gain @ h
        corrected_covariance = (
            residual_mapping @ self.covariance @ residual_mapping.T + gain @ r @ gain.T
        )
        self.position += correction[:3]
        self.velocity += correction[3:6]
        self.quaternion = quaternion_multiply(
            self.quaternion, rotation_vector_to_quaternion(correction[6:9])
        )
        self.quaternion /= np.linalg.norm(self.quaternion)
        self.angular_velocity += correction[9:12]

        # The local error frame changes when orientation is corrected.
        reset_jacobian = np.eye(12)
        reset_jacobian[6:9, 6:9] = right_rotation_jacobian(correction[6:9])
        self.covariance = reset_jacobian @ corrected_covariance @ reset_jacobian.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.last_timestamp = timestamp
        return ObjectPose(
            marker_ids=measured_pose.marker_ids,
            position=self.position.copy(),
            rotation_matrix=quaternion_to_rotation_matrix(self.quaternion),
        )


# Backward-compatible name for existing imports and saved notebooks.
QuaternionPoseEKF = SE3ErrorStateEKF


def _one_euro_alpha(cutoff: float, dt: float) -> float:
    """Exponential-smoothing weight for a low-pass at `cutoff` Hz over `dt` seconds."""
    time_constant = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + time_constant / dt)


class OneEuroFilter:
    """Scalar One Euro filter: https://cristal.univ-lille.fr/~casiez/1euro/

    A first-order low-pass whose cutoff frequency rises with the signal's
    estimated speed, so it smooths a nearly-still signal but avoids lagging
    behind a fast-changing one. `min_cutoff` sets the cutoff at zero speed;
    `beta` sets how much the cutoff opens up per unit of speed.
    """

    def __init__(
        self, value: float, min_cutoff: float, beta: float, d_cutoff: float
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.value = value
        self.derivative = 0.0

    def __call__(self, value: float, dt: float) -> float:
        derivative = (value - self.value) / dt
        derivative_alpha = _one_euro_alpha(self.d_cutoff, dt)
        self.derivative += derivative_alpha * (derivative - self.derivative)

        cutoff = self.min_cutoff + self.beta * abs(self.derivative)
        alpha = _one_euro_alpha(cutoff, dt)
        self.value += alpha * (value - self.value)
        return self.value


class OneEuroPoseFilter:
    """Adaptive low-pass filter for the box position and orientation."""

    def __init__(
        self, initial_pose: ObjectPose, timestamp: float, *, tuning: FilterTuning | None = None
    ) -> None:
        tuning = tuning if tuning is not None else FilterTuning()
        self.position_filters = [
            OneEuroFilter(
                float(value),
                tuning.position_min_cutoff_hz,
                tuning.position_beta,
                tuning.derivative_cutoff_hz,
            )
            for value in initial_pose.position.astype(np.float64).reshape(3)
        ]
        self.quaternion = rotation_matrix_to_quaternion(initial_pose.rotation_matrix)
        self.rotation_filters = [
            OneEuroFilter(
                float(value),
                tuning.rotation_min_cutoff_hz,
                tuning.rotation_beta,
                tuning.derivative_cutoff_hz,
            )
            for value in self.quaternion
        ]
        self.last_timestamp = timestamp

    def set_tuning(self, tuning: FilterTuning) -> None:
        """Keep smoothed values/derivatives while changing the adaptive cutoffs."""
        for scalar in self.position_filters:
            scalar.min_cutoff = tuning.position_min_cutoff_hz
            scalar.beta = tuning.position_beta
            scalar.d_cutoff = tuning.derivative_cutoff_hz
        for scalar in self.rotation_filters:
            scalar.min_cutoff = tuning.rotation_min_cutoff_hz
            scalar.beta = tuning.rotation_beta
            scalar.d_cutoff = tuning.derivative_cutoff_hz

    def update(self, measured_pose: ObjectPose, timestamp: float) -> ObjectPose:
        dt = timestamp - self.last_timestamp
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("filter timestamps must be finite and strictly increasing")
        self.last_timestamp = timestamp

        position = np.array(
            [
                filt(float(value), dt)
                for filt, value in zip(
                    self.position_filters, measured_pose.position.reshape(3)
                )
            ]
        )

        measured_quaternion = rotation_matrix_to_quaternion(measured_pose.rotation_matrix)
        # Match the running estimate's sign first: q and -q are the same
        # rotation, but componentwise smoothing needs a consistent sign to
        # avoid fighting itself across frames.
        if np.dot(measured_quaternion, self.quaternion) < 0:
            measured_quaternion = -measured_quaternion
        quaternion = np.array(
            [
                filt(float(value), dt)
                for filt, value in zip(self.rotation_filters, measured_quaternion)
            ]
        )
        quaternion /= np.linalg.norm(quaternion)
        self.quaternion = quaternion

        return ObjectPose(
            marker_ids=measured_pose.marker_ids,
            position=position,
            rotation_matrix=quaternion_to_rotation_matrix(quaternion),
        )


class ObjectPoseFilter:
    """One active estimator history, regardless of which tags are visible."""

    def __init__(
        self, method: str = DEFAULT_FILTER_METHOD, *, tuning: FilterTuning | None = None
    ) -> None:
        self.method = DEFAULT_FILTER_METHOD
        self.tuning = tuning if tuning is not None else FilterTuning()
        self.filter: (
            PositionKalmanFilter | SE3ErrorStateEKF | OneEuroPoseFilter | None
        ) = None
        self.last_timestamp: float | None = None
        self.set_method(method)

    def set_tuning(self, tuning: FilterTuning) -> None:
        """Update the active filter without resetting tracking history."""
        self.tuning = tuning
        if isinstance(self.filter, PositionKalmanFilter):
            self.filter.set_tuning(
                tuning.position_measurement_noise_meters,
                tuning.acceleration_noise_meters_per_second2,
            )
        elif self.filter is not None:
            self.filter.set_tuning(tuning)

    def set_method(self, method: str) -> None:
        if method not in FILTER_METHODS:
            raise ValueError(f"unknown pose filter: {method}")
        if method != self.method:
            self.method = method
            self.reset()

    def reset(self) -> None:
        self.filter = None
        self.last_timestamp = None

    def update(self, pose: ObjectPose | None, timestamp: float) -> ObjectPose | None:
        if not math.isfinite(timestamp):
            raise ValueError("pose timestamps must be finite")
        if self.last_timestamp is not None and (
            timestamp <= self.last_timestamp
            or timestamp - self.last_timestamp > MAX_MISSING_SECONDS
        ):
            self.reset()

        # Do not render dead-reckoned poses as if tags were still detected.
        if pose is None:
            return None
        if self.method == "raw":
            return pose

        if self.filter is None:
            if self.method == "kalman":
                self.filter = PositionKalmanFilter(
                    pose.position,
                    timestamp,
                    measurement_noise_meters=self.tuning.position_measurement_noise_meters,
                    acceleration_noise_meters_per_second2=(
                        self.tuning.acceleration_noise_meters_per_second2
                    ),
                )
            elif self.method == "ekf":
                self.filter = SE3ErrorStateEKF(pose, timestamp, tuning=self.tuning)
            else:
                self.filter = OneEuroPoseFilter(pose, timestamp, tuning=self.tuning)
            self.last_timestamp = timestamp
            return pose

        self.last_timestamp = timestamp
        if isinstance(self.filter, (SE3ErrorStateEKF, OneEuroPoseFilter)):
            return self.filter.update(pose, timestamp)
        return ObjectPose(
            marker_ids=pose.marker_ids,
            position=self.filter.update(pose.position, timestamp),
            rotation_matrix=pose.rotation_matrix.copy(),
        )
