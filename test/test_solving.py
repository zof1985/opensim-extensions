"""Tests for opensim_models.operators.solving (solve_coordinates/solve_point_coincidence).

These are a crash-safe, scipy.optimize.least_squares-based alternative to
OpenSim's native Model.assemble() -- see solving.py's own module docstring
for the full rationale (confirmed native segfault on this package's models
whenever an unsatisfied constraint actually needs solving). The tests below
cover the generic N-dimensional core (solve_coordinates) on a small,
hand-built, fully-controlled model with an independently-computed
closed-form answer, the solve_point_coincidence convenience (both the
"(getter_a, getter_b)" and the "(frame_a, point_a, frame_b, point_b)" pair
shapes), a reported-not-crashed non-convergence case, and -- per this
package's own "verify empirically on the real model, not just a toy"
standard -- two scenarios on the real, bundled Rajagopal-based User model:
a hand-rolled brentq cross-check (mirroring the user's own analisi.py
pattern of solving a single posture angle against a target plane/position)
and a multi-coordinate inverse-kinematics-style recovery.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import OpenSimModel, User, operators

opensim = pytest.importorskip("opensim")


def make_model():
    return OpenSimModel(model_path=None)


def make_pinned_arm(model, length):
    """A single body, pinned to ground at the origin (rotation about Z),
    with a marker at ``(length, 0, 0)`` in its own local frame -- the
    smallest possible model with a non-trivial, independently-verifiable
    closed-form forward kinematics (marker ground position == (length *
    cos(theta), length * sin(theta), 0) for pin angle theta)."""
    with model.structural_change():
        body = operators.add_body(
            model, "arm", mass=1.0, inertia=(0.01, 0.01, 0.01, 0.0, 0.0, 0.0)
        )
        operators.add_pin_joint(model, "arm_pin", body)
        operators.add_marker(
            model, opensim.Marker("tip", body.raw, opensim.Vec3(length, 0.0, 0.0))
        )
    coordinate_name = list(model.joint("arm_pin").coordinates.keys())[0]
    return coordinate_name


# ---------------------------------------------------------------------------
# solve_coordinates: generic core, on a small hand-built model
# ---------------------------------------------------------------------------


def test_solve_coordinates_matches_closed_form_angle():
    model = make_model()
    length = 0.3
    coordinate_name = make_pinned_arm(model, length)

    theta_true = np.radians(35.0)
    target_y = length * np.sin(theta_true)

    def residual(x):
        return [model.markers["tip"].position_global[1] - target_y]

    result = model.solve_coordinates(residual, [coordinate_name], x0=[0.0])

    assert result.success
    assert result.x[0] == pytest.approx(theta_true, abs=1e-6)
    got = model.markers["tip"].position_global
    expected = (length * np.cos(theta_true), length * np.sin(theta_true), 0.0)
    assert got == pytest.approx(expected, abs=1e-6)


def test_solve_coordinates_leaves_model_exactly_at_reported_solution():
    model = make_model()
    length = 0.3
    coordinate_name = make_pinned_arm(model, length)
    target_y = length * np.sin(np.radians(20.0))

    def residual(x):
        return [model.markers["tip"].position_global[1] - target_y]

    result = model.solve_coordinates(residual, [coordinate_name], x0=[0.0])

    # The optimizer's last internal function evaluation (e.g. a
    # finite-difference Jacobian step) need not be at result.x itself --
    # solve_coordinates must explicitly re-set/re-realize at the end so the
    # model's own state matches the value it reports back.
    assert model.coordinate(coordinate_name).value == pytest.approx(result.x[0])


def test_solve_coordinates_x0_defaults_to_current_coordinate_value():
    model = make_model()
    length = 0.3
    coordinate_name = make_pinned_arm(model, length)
    model.coordinate(coordinate_name).set_value(np.radians(20.0), enforce_constraints=False)
    model.update_state()
    target_y = model.markers["tip"].position_global[1]  # already satisfied at current posture

    def residual(x):
        return [model.markers["tip"].position_global[1] - target_y]

    result = model.solve_coordinates(residual, [coordinate_name])  # x0 omitted

    assert result.success
    assert result.x[0] == pytest.approx(np.radians(20.0), abs=1e-9)
    assert result.nfev <= 2  # already at the solution; should take ~0 extra work


def test_solve_coordinates_rejects_empty_coordinate_names():
    model = make_model()
    with pytest.raises(ValueError, match="coordinate_names must not be empty"):
        model.solve_coordinates(lambda x: [0.0], [])


def test_solve_coordinates_rejects_wrong_length_x0():
    model = make_model()
    coordinate_name = make_pinned_arm(model, 0.3)
    with pytest.raises(ValueError, match="x0 must have exactly 1"):
        model.solve_coordinates(lambda x: [0.0], [coordinate_name], x0=[0.0, 0.0])


def test_solve_coordinates_reports_non_convergence_without_crashing():
    """A deliberately starved solve (max_nfev=1): confirms this reports
    failure as an ordinary, catchable exception (or a success=False result,
    with raise_on_failure=False) -- never the native assembler crash this
    module exists to avoid."""
    model = make_model()
    length = 0.3
    coordinate_name = make_pinned_arm(model, length)
    target_y = length * np.sin(np.radians(35.0))

    def residual(x):
        return [model.markers["tip"].position_global[1] - target_y]

    with pytest.raises(RuntimeError, match="did not converge"):
        model.solve_coordinates(residual, [coordinate_name], x0=[0.0], max_nfev=1)

    # Still usable afterward -- nothing crashed the process.
    result = model.solve_coordinates(
        residual, [coordinate_name], x0=[0.0], max_nfev=1, raise_on_failure=False
    )
    assert result.success is False


# ---------------------------------------------------------------------------
# solve_point_coincidence: (getter_a, getter_b) pair shape
# ---------------------------------------------------------------------------


def test_solve_point_coincidence_with_getter_pairs():
    model = make_model()
    length = 0.3
    coordinate_name = make_pinned_arm(model, length)

    theta_true = np.radians(-25.0)
    target_point = (length * np.cos(theta_true), length * np.sin(theta_true), 0.0)

    result = model.solve_point_coincidence(
        [(lambda: model.markers["tip"].position_global, lambda: target_point)],
        [coordinate_name],
        x0=[0.0],
    )

    assert result.success
    assert result.x[0] == pytest.approx(theta_true, abs=1e-6)
    got = model.markers["tip"].position_global
    assert operators.euclidean_distance(got, target_point) < 1e-9


# ---------------------------------------------------------------------------
# solve_point_coincidence: (frame_a, point_a, frame_b, point_b) pair shape
# ---------------------------------------------------------------------------


def test_solve_point_coincidence_with_frame_point_pairs():
    """Mirrors the shoulder-pad/heel-to-footrest shape from the user's own
    analisi.py: a body-fixed contact point on a moving body (``arm``) must
    coincide with a body-fixed contact point on a fixed piece of equipment
    (``target``, welded to ground)."""
    model = make_model()
    length = 0.3
    coordinate_name = make_pinned_arm(model, length)

    theta_true_deg = 40.0
    theta_true = np.radians(theta_true_deg)
    target_position = (length * np.cos(theta_true), length * np.sin(theta_true), 0.0)
    operators.add_box_body(
        model,
        "target",
        size=(0.05, 0.05, 0.05),
        density=500.0,
        position=target_position,
        orientation_deg=(0.0, 0.0, 0.0),
        reinitialize=True,
    )

    local_a = (length, 0.0, 0.0)
    local_b = (0.0, 0.0, 0.0)
    result = model.solve_point_coincidence(
        [(model.bodies["arm"], local_a, model.bodies["target"], local_b)],
        [coordinate_name],
        x0=[0.0],
    )

    assert result.success
    assert result.x[0] == pytest.approx(theta_true, abs=1e-6)
    point_a = operators.from_local_to_global(local_a, model.bodies["arm"])
    point_b = operators.from_local_to_global(local_b, model.bodies["target"])
    assert operators.euclidean_distance(point_a, point_b) < 1e-9


def test_solve_point_coincidence_rejects_empty_point_pairs():
    model = make_model()
    coordinate_name = make_pinned_arm(model, 0.3)
    with pytest.raises(ValueError, match="point_pairs must not be empty"):
        model.solve_point_coincidence([], [coordinate_name])


def test_solve_point_coincidence_rejects_bad_pair_shape():
    model = make_model()
    coordinate_name = make_pinned_arm(model, 0.3)
    bad_pair = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # 3 elements
    with pytest.raises(ValueError, match="2 elements .* or 4 elements"):
        model.solve_point_coincidence([bad_pair], [coordinate_name])


def test_solve_point_coincidence_rejects_non_callable_getter_pair():
    model = make_model()
    coordinate_name = make_pinned_arm(model, 0.3)
    bad_pair = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # not callables
    with pytest.raises(TypeError, match="both callable"):
        model.solve_point_coincidence([bad_pair], [coordinate_name])


# ---------------------------------------------------------------------------
# Real model (User): verified empirically, not just on the toy above
# ---------------------------------------------------------------------------


def make_user():
    return User(gender="M", height_cm=175.0)


def test_solve_coordinates_matches_hand_rolled_brentq_on_the_real_user_model():
    """Reproduces the user's own analisi.py pattern (a single posture angle
    solved against a target position via scipy.optimize.brentq, done by
    hand through the same enforce_constraints=False/realizePosition
    primitives) and confirms solve_coordinates converges to the identical
    answer. Uses the right shoulder's anterior-posterior (X) position as
    the target -- confirmed monotonic in lumbar_extension over its full
    +-90 degree range, unlike the joint centre itself (torso), which does
    not move at all under its own distal coordinate (a joint centre is a
    skeletal pivot, not a body-surface point -- see this package's own
    distinction between the two)."""
    user = make_user()
    coordinate = user.coordinate("lumbar_extension")

    true_deg = 10.0
    coordinate.set_value_degrees(true_deg, enforce_constraints=False)
    user.update_state()
    target_x = user.right_shoulder[0]

    def hand_rolled_residual_deg(theta_deg):
        coordinate.set_value_degrees(theta_deg, enforce_constraints=False)
        user.model.realizePosition(user.state)
        return user.right_shoulder[0] - target_x

    brentq_deg = brentq(hand_rolled_residual_deg, -45.0, 45.0)

    coordinate.set_value_degrees(0.0, enforce_constraints=False)

    def residual(x):
        return [user.right_shoulder[0] - target_x]

    result = user.solve_coordinates(residual, ["lumbar_extension"], x0=[0.0])

    assert result.success
    solved_deg = float(np.degrees(result.x[0]))
    assert solved_deg == pytest.approx(brentq_deg, abs=1e-4)
    assert solved_deg == pytest.approx(true_deg, abs=1e-4)


def test_solve_point_coincidence_recovers_known_posture_on_the_real_user_model():
    """Inverse-kinematics-style recovery on the real User model: pose three
    coordinates to a known value, record the resulting ankle position, reset
    the posture, then solve for those same three coordinates from a nearby
    guess using solve_point_coincidence -- the same category of problem as
    analisi.py's heel-to-footrest/shoulder-to-pad rigid attachments, just
    with the target expressed as a plain getter instead of a second body."""
    user = make_user()
    true_values_deg = {"hip_flexion_r": 30.0, "hip_adduction_r": -10.0, "knee_angle_r": 45.0}
    for name, degrees in true_values_deg.items():
        user.coordinate(name).set_value_degrees(degrees, enforce_constraints=False)
    user.update_state()
    target = user.right_ankle

    for name in true_values_deg:
        user.coordinate(name).set_value_degrees(0.0, enforce_constraints=False)
    user.update_state()

    x0_deg = {"hip_flexion_r": 15.0, "hip_adduction_r": -5.0, "knee_angle_r": 20.0}
    coordinate_names = list(true_values_deg.keys())
    x0 = [np.radians(x0_deg[name]) for name in coordinate_names]

    result = user.solve_point_coincidence(
        [(lambda: user.right_ankle, lambda: target)], coordinate_names, x0=x0
    )

    assert result.success
    solved_deg = dict(zip(coordinate_names, np.degrees(result.x)))
    for name, expected_deg in true_values_deg.items():
        assert solved_deg[name] == pytest.approx(expected_deg, abs=1e-3)
    assert operators.euclidean_distance(user.right_ankle, target) < 1e-9
