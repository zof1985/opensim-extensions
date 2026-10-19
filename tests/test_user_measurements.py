from __future__ import annotations

import math
from pathlib import Path

import pytest

opensim = pytest.importorskip("opensim")

from opensim_extensions import User


def make_user() -> User:
    return User("M", percentile=50.0)


def test_user_is_a_native_model_with_explicit_measurement_setters():
    user = make_user()

    assert issubclass(User, opensim.Model)
    assert isinstance(user, opensim.Model)
    assert user.set_left_arm_length.__doc__
    assert "Parameters" in user.set_left_arm_length.__doc__
    assert user.set_right_arm_length.__doc__


def test_left_and_right_arm_lengths_are_independent_and_rescale_their_side():
    user = make_user()
    right_initial = user.right_arm_length
    target_left = user.left_arm_length * 1.15

    user.set_left_arm_length(target_left)

    assert user.left_arm_length == pytest.approx(target_left, rel=1e-6)
    assert user.right_arm_length == pytest.approx(right_initial, rel=1e-6)
    assert user.getTotalMass(user.state) == pytest.approx(user.mass_kg, rel=1e-8)


def test_bilateral_ansur_measurement_override_does_not_change_the_other_side():
    user = make_user()
    right_initial = user.right_foot_length
    target_left = user.left_foot_length * 1.2

    user.set_left_foot_length(target_left)

    assert user.left_foot_length == pytest.approx(target_left)
    assert user.right_foot_length == pytest.approx(right_initial)
    assert user.getTotalMass(user.state) == pytest.approx(user.mass_kg, rel=1e-8)


def test_repeated_setter_does_not_accumulate_scaling():
    user = make_user()
    target = user.left_arm_length * 1.1

    user.set_left_arm_length(target)
    first_result = user.left_arm_length
    user.set_left_arm_length(target)

    assert first_result == pytest.approx(target, rel=1e-6)
    assert user.left_arm_length == pytest.approx(target, rel=1e-6)


def test_chest_width_setter_updates_measurement_and_model_scale():
    user = make_user()
    target = user.chest_width * 1.1
    torso = user.getBodySet().get("torso")
    original_mass = torso.getMass()

    user.set_chest_width(target)

    assert user.chest_width == pytest.approx(target)
    assert user.anthropometry.values["chestbreadth"] == pytest.approx(target * 1000.0)
    assert torso.getMass() > 0
    assert math.isfinite(original_mass)
    assert user.getTotalMass(user.state) == pytest.approx(user.mass_kg, rel=1e-8)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.inf, math.nan])
def test_measurement_setters_reject_non_positive_or_non_finite_values(bad_value):
    user = make_user()
    original = user.chest_width

    with pytest.raises(ValueError, match="positive finite"):
        user.set_chest_width(bad_value)

    assert user.chest_width == pytest.approx(original)


def test_to_osim_exports_scaled_model_at_current_posture_with_meshes(tmp_path: Path):
    user = make_user()
    user.set_right_hip_flexionextension(25.0)

    destination = user.to_osim(tmp_path / "user.osim")

    assert destination.is_file()
    assert (tmp_path / "Geometry" / "r_pelvis.vtp").is_file()
    exported = opensim.Model(str(destination))
    state = exported.initSystem()

    exported_coordinate = exported.getCoordinateSet().get("hip_flexion_r")
    assert exported_coordinate.get_default_value() == pytest.approx(math.radians(25.0))
    assert exported.getTotalMass(state) == pytest.approx(user.mass_kg, rel=1e-8)
    assert exported.getBodySet().get(0).getMass() == pytest.approx(
        user.getBodySet().get(0).getMass()
    )
