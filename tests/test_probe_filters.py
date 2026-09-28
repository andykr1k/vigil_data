"""Ported from DataCollection/tests/test_pose_filters.py."""

import math
from itertools import count

import cv2
import numpy as np
import pytest

from vigil.probe.object_pose import ObjectPose
from vigil.probe.filtering.pose import (
    FILTER_METHODS,
    ObjectPoseFilter,
    OneEuroFilter,
    OneEuroPoseFilter,
    SE3ErrorStateEKF,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    quaternion_to_rotation_vector,
    right_rotation_jacobian,
    rotation_matrix_to_quaternion,
    rotation_vector_to_quaternion,
)


def make_pose(position=(0.0, 0.0, 0.6), rotation_vector=(0.0, 0.0, 0.0), ids=(0,)):
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    return ObjectPose(ids, np.asarray(position, dtype=np.float64), rotation)


def orientation_difference(left, right):
    return quaternion_to_rotation_vector(
        quaternion_multiply(
            quaternion_conjugate(rotation_matrix_to_quaternion(left)),
            rotation_matrix_to_quaternion(right),
        )
    )


@pytest.mark.parametrize(
    "vector",
    [(0.0, 0.0, 0.0), (1e-9, -2e-9, 3e-9), (0.4, -0.3, 0.2), (math.pi, 0.0, 0.0)],
)
def test_quaternion_round_trip_and_antipodal_equivalence(vector):
    vector = np.asarray(vector)
    quaternion = rotation_vector_to_quaternion(vector)
    expected_rotation, _ = cv2.Rodrigues(vector)

    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)
    np.testing.assert_allclose(quaternion_to_rotation_matrix(quaternion), expected_rotation, atol=1e-12)
    np.testing.assert_allclose(quaternion_to_rotation_matrix(-quaternion), expected_rotation, atol=1e-12)
    np.testing.assert_allclose(quaternion_to_rotation_vector(quaternion), vector, atol=1e-12)
    np.testing.assert_allclose(quaternion_to_rotation_vector(-quaternion), vector, atol=1e-12)


def test_quaternion_composition_matches_rotation_matrix_composition():
    left = rotation_vector_to_quaternion(np.array([0.2, -0.4, 0.3]))
    right = rotation_vector_to_quaternion(np.array([-0.5, 0.1, 0.2]))
    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(quaternion_multiply(left, right)),
        quaternion_to_rotation_matrix(left) @ quaternion_to_rotation_matrix(right),
        atol=1e-12,
    )


def test_ekf_prediction_jacobian_matches_numerical_derivatives():
    ekf = SE3ErrorStateEKF(make_pose(rotation_vector=(0.5, -0.2, 0.3)), 0.0)
    ekf.angular_velocity = np.array([0.8, -0.5, 0.4])
    dt = 0.08
    transition = ekf._transition_matrix(dt)
    nominal_prediction = quaternion_multiply(
        ekf.quaternion, rotation_vector_to_quaternion(ekf.angular_velocity * dt)
    )
    epsilon = 1e-6

    for column in range(6, 12):
        error = np.zeros(12)
        error[column] = epsilon
        true_quaternion = quaternion_multiply(
            ekf.quaternion, rotation_vector_to_quaternion(error[6:9])
        )
        true_prediction = quaternion_multiply(
            true_quaternion,
            rotation_vector_to_quaternion((ekf.angular_velocity + error[9:12]) * dt),
        )
        numerical_column = quaternion_to_rotation_vector(
            quaternion_multiply(quaternion_conjugate(nominal_prediction), true_prediction)
        ) / epsilon
        np.testing.assert_allclose(numerical_column, transition[6:9, column], atol=1e-7)


def test_rotation_reset_jacobian_matches_numerical_derivatives():
    correction = np.array([0.2, -0.3, 0.1])
    corrected_quaternion = rotation_vector_to_quaternion(correction)
    epsilon = 1e-6
    for column in range(3):
        perturbation = np.zeros(3)
        perturbation[column] = epsilon
        relative = quaternion_multiply(
            quaternion_conjugate(corrected_quaternion),
            rotation_vector_to_quaternion(correction + perturbation),
        )
        np.testing.assert_allclose(
            quaternion_to_rotation_vector(relative) / epsilon,
            right_rotation_jacobian(correction)[:, column],
            atol=1e-7,
        )


def test_ekf_reduces_stationary_position_and_rotation_noise():
    rng = np.random.default_rng(42)
    true_pose = make_pose(rotation_vector=(0.4, -0.3, 0.2))
    ekf = SE3ErrorStateEKF(true_pose, timestamp=0.0)
    raw_position_errors, filtered_position_errors = [], []
    raw_rotation_errors, filtered_rotation_errors = [], []

    for frame in range(1, 241):
        position_noise = rng.normal(0, 0.01, 3)
        rotation_noise = rng.normal(0, math.radians(3), 3)
        noise_rotation = quaternion_to_rotation_matrix(rotation_vector_to_quaternion(rotation_noise))
        measurement = ObjectPose(
            (0, 2), true_pose.position + position_noise, true_pose.rotation_matrix @ noise_rotation
        )
        filtered = ekf.update(measurement, timestamp=frame / 60)
        if frame > 30:
            raw_position_errors.append(np.linalg.norm(position_noise))
            filtered_position_errors.append(np.linalg.norm(filtered.position - true_pose.position))
            raw_rotation_errors.append(np.linalg.norm(rotation_noise))
            filtered_rotation_errors.append(
                np.linalg.norm(orientation_difference(true_pose.rotation_matrix, filtered.rotation_matrix))
            )
        assert filtered.marker_ids == (0, 2)
        assert np.linalg.norm(ekf.quaternion) == pytest.approx(1.0, abs=1e-12)

    assert np.mean(filtered_position_errors) < np.mean(raw_position_errors) * 0.65
    assert np.mean(filtered_rotation_errors) < np.mean(raw_rotation_errors) * 0.65
    np.testing.assert_allclose(ekf.covariance, ekf.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(ekf.covariance).min() >= -1e-12


def test_ekf_tracks_motion_across_180_degree_boundary():
    ekf = SE3ErrorStateEKF(make_pose(rotation_vector=(0, 0, math.radians(170))), 0.0)
    previous_rotation = ekf.quaternion.copy()
    for frame in range(1, 181):
        angle = math.radians(170 + frame * 0.5)
        measurement = make_pose(
            position=(frame / 1000, 0.0, 0.6), rotation_vector=(0, 0, angle)
        )
        filtered = ekf.update(measurement, frame / 60)
        relative_rotation = quaternion_to_rotation_vector(
            quaternion_multiply(quaternion_conjugate(previous_rotation), ekf.quaternion)
        )
        assert np.linalg.norm(relative_rotation) < math.radians(2)
        if frame > 60:
            assert np.linalg.norm(
                orientation_difference(measurement.rotation_matrix, filtered.rotation_matrix)
            ) < math.radians(1)
        previous_rotation = ekf.quaternion.copy()

    np.testing.assert_allclose(filtered.position, measurement.position, atol=1e-3)
    assert ekf.angular_velocity[2] == pytest.approx(math.radians(30), abs=math.radians(1))


def test_one_euro_filter_converges_monotonically_towards_step_input():
    filt = OneEuroFilter(0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    previous = 0.0
    value = previous
    for _ in range(120):
        value = filt(1.0, dt=1 / 60)
        assert value >= previous - 1e-12
        previous = value
    assert value == pytest.approx(1.0, abs=1e-3)


def test_one_euro_filter_beta_reduces_steady_state_lag_on_a_ramp():
    dt = 1 / 60
    speed = 0.5  # units per second
    lagging = OneEuroFilter(0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    responsive = OneEuroFilter(0.0, min_cutoff=1.0, beta=2.0, d_cutoff=1.0)
    true_value = lag_value = fast_value = 0.0
    for frame in range(1, 301):
        true_value = speed * frame * dt
        lag_value = lagging(true_value, dt)
        fast_value = responsive(true_value, dt)
    assert abs(true_value - fast_value) < abs(true_value - lag_value)


def test_one_euro_pose_filter_reduces_stationary_position_and_rotation_noise():
    rng = np.random.default_rng(11)
    true_pose = make_pose(rotation_vector=(0.4, -0.3, 0.2))
    filt = OneEuroPoseFilter(true_pose, timestamp=0.0)
    raw_position_errors, filtered_position_errors = [], []
    raw_rotation_errors, filtered_rotation_errors = [], []

    for frame in range(1, 241):
        position_noise = rng.normal(0, 0.01, 3)
        rotation_noise = rng.normal(0, math.radians(3), 3)
        noise_rotation = quaternion_to_rotation_matrix(rotation_vector_to_quaternion(rotation_noise))
        measurement = ObjectPose(
            (0, 2), true_pose.position + position_noise, true_pose.rotation_matrix @ noise_rotation
        )
        filtered = filt.update(measurement, timestamp=frame / 60)
        if frame > 30:
            raw_position_errors.append(np.linalg.norm(position_noise))
            filtered_position_errors.append(np.linalg.norm(filtered.position - true_pose.position))
            raw_rotation_errors.append(np.linalg.norm(rotation_noise))
            filtered_rotation_errors.append(
                np.linalg.norm(orientation_difference(true_pose.rotation_matrix, filtered.rotation_matrix))
            )
        assert filtered.marker_ids == (0, 2)
        assert np.linalg.norm(filt.quaternion) == pytest.approx(1.0, abs=1e-12)

    assert np.mean(filtered_position_errors) < np.mean(raw_position_errors) * 0.65
    assert np.mean(filtered_rotation_errors) < np.mean(raw_rotation_errors) * 0.65


def test_one_euro_pose_filter_tracks_motion_across_180_degree_boundary():
    """Antipodal sign correction must keep frame-to-frame quaternions close."""
    filt = OneEuroPoseFilter(make_pose(rotation_vector=(0, 0, math.radians(170))), 0.0)
    previous_rotation = filt.quaternion.copy()
    filtered = None
    for frame in range(1, 181):
        angle = math.radians(170 + frame * 0.5)
        measurement = make_pose(position=(frame / 1000, 0.0, 0.6), rotation_vector=(0, 0, angle))
        filtered = filt.update(measurement, frame / 60)
        relative_rotation = quaternion_to_rotation_vector(
            quaternion_multiply(quaternion_conjugate(previous_rotation), filt.quaternion)
        )
        assert np.linalg.norm(relative_rotation) < math.radians(5)
        previous_rotation = filt.quaternion.copy()

    np.testing.assert_allclose(filtered.position, measurement.position, atol=1e-2)


@pytest.mark.parametrize("method", FILTER_METHODS)
def test_filter_switching_reset_and_missing_tags(method):
    controller = ObjectPoseFilter(method)
    initial = make_pose()
    assert controller.update(initial, 0.0) is initial
    output = controller.update(make_pose(position=(0.02, 0.0, 0.6), ids=(0, 2, 4)), 1 / 30)
    assert output.marker_ids == (0, 2, 4)
    assert controller.update(None, 0.1) is None

    # Long gaps restart from the measurement, with no ghost/predicted output.
    assert controller.update(None, 1.0) is None
    reacquired = make_pose(position=(0.5, 0.0, 0.6), ids=(4,))
    assert controller.update(reacquired, 1.1) is reacquired
    controller.reset()
    assert controller.method == method
    assert controller.filter is None

    for next_method in FILTER_METHODS:
        controller.set_method(next_method)
        assert controller.update(initial, 2.0) is initial


def test_linear_and_ekf_use_the_same_xyz_model_for_regular_frames():
    linear = ObjectPoseFilter("kalman")
    ekf = ObjectPoseFilter("ekf")
    rng = np.random.default_rng(7)
    for frame in range(60):
        measurement = make_pose(
            position=np.array([0.0, 0.0, 0.6]) + rng.normal(0, 0.01, 3),
            rotation_vector=(0.0, 0.0, frame / 100),
        )
        linear_pose = linear.update(measurement, frame / 60)
        ekf_pose = ekf.update(measurement, frame / 60)
        np.testing.assert_allclose(linear_pose.position, ekf_pose.position, atol=1e-12)
        np.testing.assert_allclose(linear_pose.rotation_matrix, measurement.rotation_matrix)
