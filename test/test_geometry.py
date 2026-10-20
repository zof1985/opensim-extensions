"""Tests for opensim_models.operators' geometry module.

``euclidean_distance`` is a pure geometry helper with no OpenSim model
involved at all, so its own tests need no opensim bindings installed.
``from_global_to_local``/``from_local_to_global`` do read an object's
current pose, so their tests further down need a live OpenSimModel (and
therefore the opensim bindings) -- guarded with ``pytest.importorskip``,
same as every other OpenSim-dependent test file in this suite
(``test_operators.py``, ...)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import operators


def test_euclidean_distance_matches_a_known_3_4_5_triangle():
    assert operators.euclidean_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)


def test_euclidean_distance_is_zero_for_coincident_points():
    assert operators.euclidean_distance((1.0, -2.0, 3.0), (1.0, -2.0, 3.0)) == pytest.approx(0.0)


def test_euclidean_distance_is_symmetric():
    a = (1.0, 2.0, 3.0)
    b = (-4.0, 5.0, -6.0)
    assert operators.euclidean_distance(a, b) == pytest.approx(operators.euclidean_distance(b, a))


def test_euclidean_distance_accepts_lists_and_numpy_arrays():
    import numpy as np

    assert operators.euclidean_distance([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert operators.euclidean_distance(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    ) == pytest.approx(2.0)


def test_euclidean_distance_returns_a_plain_float():
    result = operators.euclidean_distance((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "position1,position2",
    [
        ((0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (1.0, 1.0)),
    ],
)
def test_euclidean_distance_rejects_wrong_sized_positions(position1, position2):
    with pytest.raises(ValueError, match="exactly 3 values"):
        operators.euclidean_distance(position1, position2)


def test_euclidean_distance_rejects_non_finite_position1():
    with pytest.raises(ValueError, match="finite"):
        operators.euclidean_distance((float("inf"), 0.0, 0.0), (0.0, 0.0, 0.0))


def test_euclidean_distance_rejects_non_finite_position2():
    with pytest.raises(ValueError, match="finite"):
        operators.euclidean_distance((0.0, 0.0, 0.0), (0.0, float("nan"), 0.0))


# ---------------------------------------------------------------------------
# from_global_to_local / from_local_to_global
# ---------------------------------------------------------------------------

opensim = pytest.importorskip("opensim")

from opensim_models import Box, OpenSimModel  # noqa: E402  (after importorskip)


def make_model():
    return OpenSimModel(model_path=None)


def add_rotated_body(model, name, position, angle_z_deg):
    """Add a body welded to ground at ``position``, rotated ``angle_z_deg`` about Z."""
    with model.structural_change():
        body = operators.add_body(model, name, mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(
            model,
            f"{name}_joint",
            body,
            position=position,
            orientation_deg=(0.0, 0.0, angle_z_deg),
        )
    return body


def test_from_global_to_local_matches_a_known_rotated_pose():
    # Body centred at (1, 0, 0), rotated 90deg about Z: its local X axis
    # points along the ground's +Y axis, and its local Y axis points along
    # the ground's -X axis.
    model = make_model()
    body = add_rotated_body(model, "b1", (1.0, 0.0, 0.0), 90.0)

    local = operators.from_global_to_local((1.0, 1.0, 0.0), body)

    assert local == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_from_local_to_global_matches_a_known_rotated_pose():
    model = make_model()
    body = add_rotated_body(model, "b1", (1.0, 0.0, 0.0), 90.0)

    global_point = operators.from_local_to_global((1.0, 0.0, 0.0), body)

    assert global_point == pytest.approx((1.0, 1.0, 0.0), abs=1e-9)


def test_from_global_to_local_matches_a_known_unrotated_offset():
    model = make_model()
    body = add_rotated_body(model, "b1", (2.0, 3.0, 4.0), 0.0)

    local = operators.from_global_to_local((5.0, 5.0, 5.0), body)

    assert local == pytest.approx((3.0, 2.0, 1.0), abs=1e-9)


def test_from_global_to_local_and_back_round_trips():
    model = make_model()
    body = add_rotated_body(model, "b1", (1.5, -2.0, 0.5), 37.0)
    point = (3.1, -4.2, 5.3)

    local = operators.from_global_to_local(point, body)
    back = operators.from_local_to_global(local, body)

    assert back == pytest.approx(point, abs=1e-9)


def test_from_local_to_global_and_back_round_trips():
    model = make_model()
    body = add_rotated_body(model, "b1", (1.5, -2.0, 0.5), 37.0)
    point = (0.4, 0.2, -0.6)

    global_point = operators.from_local_to_global(point, body)
    back = operators.from_global_to_local(global_point, body)

    assert back == pytest.approx(point, abs=1e-9)


def test_from_global_to_local_accepts_the_raw_opensim_body():
    model = make_model()
    body = add_rotated_body(model, "b1", (1.0, 0.0, 0.0), 90.0)

    local = operators.from_global_to_local((1.0, 1.0, 0.0), body.raw)

    assert local == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_from_global_to_local_accepts_a_physical_offset_frame():
    model = make_model()
    body = add_rotated_body(model, "b1", (1.0, 0.0, 0.0), 90.0)
    joint = model.joints.get("b1_joint")
    parent = joint.parent_frame  # an OffsetFrame, same pose as the body itself here

    local = operators.from_global_to_local((1.0, 1.0, 0.0), parent)

    assert local == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_from_global_to_local_accepts_a_box():
    # A Box keeps its own private container, rebuilt from scratch on every
    # pose change -- this exercises the owner-resolution path specific to
    # that (see _resolve_frame in operators/geometry.py), not just the
    # generic owner registry every plain Body already works through.
    box = Box(0.2, 0.2, 0.2, origin=(2.0, 0.0, 0.0), angle_deg=(0.0, 0.0, 90.0))

    local = operators.from_global_to_local((2.0, 1.0, 0.0), box)
    back = operators.from_local_to_global(local, box)

    assert local == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert back == pytest.approx((2.0, 1.0, 0.0), abs=1e-9)


def test_from_global_to_local_rejects_a_marker():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(0.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    with pytest.raises(TypeError, match="no ground-frame orientation"):
        operators.from_global_to_local((0.0, 0.0, 0.0), model.marker("mk1"))


def test_from_local_to_global_rejects_a_joint():
    model = make_model()
    add_rotated_body(model, "b1", (1.0, 0.0, 0.0), 0.0)
    joint = model.joints.get("b1_joint")

    with pytest.raises(TypeError, match="no ground-frame orientation"):
        operators.from_local_to_global((0.0, 0.0, 0.0), joint)


def test_from_global_to_local_rejects_an_object_with_no_resolvable_owner():
    with pytest.raises(TypeError, match="Cannot resolve the owning OpenSimModel"):
        operators.from_global_to_local((0.0, 0.0, 0.0), None)


@pytest.mark.parametrize(
    "coordinates",
    [(0.0, 0.0), (0.0, 0.0, 0.0, 0.0)],
)
def test_from_global_to_local_rejects_wrong_sized_coordinates(coordinates):
    model = make_model()
    body = add_rotated_body(model, "b1", (0.0, 0.0, 0.0), 0.0)

    with pytest.raises(ValueError, match="exactly 3 values"):
        operators.from_global_to_local(coordinates, body)


def test_from_local_to_global_rejects_non_finite_coordinates():
    model = make_model()
    body = add_rotated_body(model, "b1", (0.0, 0.0, 0.0), 0.0)

    with pytest.raises(ValueError, match="finite"):
        operators.from_local_to_global((0.0, float("nan"), 0.0), body)
