import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_extensions import Bike, User
from opensim_extensions.framework import VisualErgonomicsFramework
from opensim_extensions.framework import Framework
from opensim_extensions.viewer.viewer import _ViewerModel
from opensim_extensions.viewer.visualizer import VTKVisualizer


@pytest.fixture(scope="module")
def manager():
    user = User("M", height_cm=170.0, mass_kg=70.0)
    bike = Bike()
    bike.fit_user(user)
    return VisualErgonomicsFramework(user, bike, 45.0, 45.0)


def test_ergonomics_framework_is_renderable_and_reports_view_state(manager):
    assert isinstance(manager, Framework)
    assert manager["user"] is manager.user
    assert manager["equipment"] is manager.equipment
    assert {"vision_cone", "vision_cone_axis", "reach_r", "reach_l"} == {
        manager["visualization"].getBodySet().get(index).getName()
        for index in range(manager["visualization"].getBodySet().getSize())
    }
    assert manager.composite.getBodySet().getSize() > (
        manager.user.getBodySet().getSize() + manager.equipment.getBodySet().getSize()
    )
    visualizer = VTKVisualizer(_ViewerModel(manager.composite), offscreen=True)
    assert visualizer.has_vision_cone_and_axis
    assert visualizer.has_reachability_regions
    try:
        overlay_actors = {
            body.getName(): actor
            for body, actor in visualizer._actor_bodies
            if body.getName().startswith(("vision_cone", "reach_"))
        }
        assert len(overlay_actors) == 4
        assert overlay_actors["vision_cone_shape0"].GetProperty().GetOpacity() == pytest.approx(
            0.18
        )
        assert overlay_actors["vision_cone_axis_shape0"].GetProperty().GetColor() == pytest.approx(
            (1.0, 0.8, 0.0)
        )
        assert overlay_actors["reach_r_shape0"].GetProperty().GetOpacity() == pytest.approx(
            0.12
        )
        visualizer.set_vision_cone_visible(False)
        visualizer.set_reachability_visible(False)
        assert not overlay_actors["vision_cone_shape0"].GetVisibility()
        assert not overlay_actors["vision_cone_axis_shape0"].GetVisibility()
        assert not overlay_actors["reach_r_shape0"].GetVisibility()
        visualizer.set_vision_cone_visible(True)
        visualizer.set_reachability_visible(True)
        assert overlay_actors["vision_cone_shape0"].GetVisibility()
        assert overlay_actors["vision_cone_axis_shape0"].GetVisibility()
        assert overlay_actors["reach_r_shape0"].GetVisibility()
    finally:
        visualizer.close()

    view_state = manager.get_view_state()

    assert view_state["incidence_angle_deg"] >= 0.0
    assert 0.0 <= view_state["observed_screen_percent"] <= 100.0
    assert 0.0 <= view_state["reachable_screen_percent"] <= 100.0


def test_visual_cone_defaults_use_documented_ergonomic_values(manager):
    framework = VisualErgonomicsFramework(manager.user, manager.equipment)

    assert framework.cone_width_angle_deg == pytest.approx(30.0)
    assert framework.cone_height_angle_deg == pytest.approx(30.0)


def test_optimization_returns_independent_framework_and_respects_screen_bounds(manager):
    original_user = manager.user
    original_bike = manager.equipment
    original_neck = original_user.neck_flexion
    original_screen = original_bike.screen_inclination_deg

    optimized = manager.optimize_screen_position(
        incidence_weight=0.33,
        observed_screen_weight=0.33,
        reachability_weight=0.34,
        inplace=False,
        neck_flexionextension_range_deg=(-10.0, 10.0),
        head_extensionflexion_range_deg=(0.0, 0.0),
        screen_incline_range_deg=(40.0, 80.0),
        max_iterations=60,
    )

    assert isinstance(optimized, VisualErgonomicsFramework)
    assert isinstance(optimized.user, User)
    assert optimized.user is not original_user
    assert optimized.equipment is not original_bike
    assert original_user.neck_flexion == pytest.approx(original_neck)
    assert original_bike.screen_inclination_deg == pytest.approx(original_screen)
    optimized.user.set_neck_flexion(-5.0)
    assert optimized.user.neck_flexion == pytest.approx(-5.0)
    assert original_user.neck_flexion == pytest.approx(original_neck)
    assert 40.0 <= optimized.equipment.screen_inclination_deg <= 80.0
    assert min(optimized._screen_constraints(
        (
            optimized.equipment.screen_horizontal_offset_mm,
            optimized.equipment.screen_vertical_offset_mm,
            optimized.equipment.screen_inclination_deg,
        )
    )) >= -1e-6
    assert set(optimized.get_view_state()) == {
        "incidence_angle_deg",
        "observed_screen_percent",
        "reachable_screen_percent",
    }


def test_inplace_optimization_modifies_manager_and_returns_none(manager):
    original_screen = manager.equipment.screen_inclination_deg
    neck = manager.user.getCoordinateSet().get("neck_base_flexion")

    result = manager.optimize_screen_position(
        incidence_weight=1.0,
        observed_screen_weight=0.0,
        reachability_weight=0.0,
        inplace=True,
        neck_flexionextension_range_deg=(5.0, 5.0),
        head_extensionflexion_range_deg=(0.0, 0.0),
        screen_incline_range_deg=(original_screen, original_screen),
        screen_horizontal_offset_range_mm=(
            manager.equipment.screen_horizontal_offset_mm,
            manager.equipment.screen_horizontal_offset_mm,
        ),
        screen_vertical_offset_range_mm=(
            manager.equipment.screen_vertical_offset_mm,
            manager.equipment.screen_vertical_offset_mm,
        ),
    )

    assert result is None
    assert manager.equipment.screen_inclination_deg == pytest.approx(original_screen)
    assert neck.getValue(manager._user_state()) == pytest.approx(np.deg2rad(5.0))


def test_screen_pose_parameters_accept_fixed_values(manager):
    original = manager.equipment.screen
    horizontal = original.horizontal_offset_mm + 1.0
    vertical = original.vertical_offset_mm
    inclination = original.inclination_deg

    optimized = manager.optimize_screen_position(
        screen_horizontal_offset_mm=horizontal,
        screen_vertical_offset_mm=vertical,
        screen_inclination_deg=inclination,
        neck_flexionextension_range_deg=None,
        head_extensionflexion_range_deg=None,
    )

    assert optimized.equipment.screen_horizontal_offset_mm == pytest.approx(horizontal)
    assert optimized.equipment.screen_vertical_offset_mm == pytest.approx(vertical)
    assert optimized.equipment.screen_inclination_deg == pytest.approx(inclination)
    assert manager.equipment.screen_horizontal_offset_mm == pytest.approx(
        original.horizontal_offset_mm
    )


def test_screen_offset_range_expands_equipment_limit(manager):
    screen = manager.equipment.screen
    horizontal_limit = screen.horizontal_offset_min_mm - 0.1

    with pytest.warns(UserWarning, match="expanding the limits"):
        optimized = manager.optimize_screen_position(
            screen_horizontal_offset_mm=(
                horizontal_limit,
                screen.horizontal_offset_mm,
            ),
            neck_flexionextension_range_deg=None,
            head_extensionflexion_range_deg=None,
            max_iterations=60,
        )

    assert optimized.equipment.screen.horizontal_offset_min_mm == pytest.approx(
        horizontal_limit
    )
    assert optimized.equipment.screen_horizontal_offset_mm >= horizontal_limit


def test_optimize_user_position_fixes_float_and_expands_coordinate_limits(manager):
    original_screen = manager.equipment.screen
    target_degrees = -50.0
    arm_flexion_degrees = 25.0
    trunk_flexion_degrees = 10.0

    with pytest.warns(UserWarning, match="expanding the limits"):
        optimized = manager.optimize_user_position(
            neck_base_flexion_deg=target_degrees,
            arm_flex_r_deg=arm_flexion_degrees,
            lumbar_extension_deg=trunk_flexion_degrees,
        )

    coordinate = optimized.user.getCoordinateSet().get("neck_base_flexion")
    value = coordinate.getValue(optimized._user_state())
    assert value == pytest.approx(np.deg2rad(target_degrees))
    assert coordinate.getRangeMin() <= np.deg2rad(target_degrees)
    assert optimized.user.getCoordinateSet().get("arm_flex_r").getValue(
        optimized._user_state()
    ) == pytest.approx(np.deg2rad(arm_flexion_degrees))
    assert optimized.user.getCoordinateSet().get("lumbar_extension").getValue(
        optimized._user_state()
    ) == pytest.approx(np.deg2rad(trunk_flexion_degrees))
    assert optimized.equipment.screen_horizontal_offset_mm == pytest.approx(
        original_screen.horizontal_offset_mm
    )
    assert optimized.equipment.screen_vertical_offset_mm == pytest.approx(
        original_screen.vertical_offset_mm
    )
    assert optimized.equipment.screen_inclination_deg == pytest.approx(
        original_screen.inclination_deg
    )
    assert manager.user.getCoordinateSet().get(
        "neck_base_flexion"
    ).getRangeMin() > np.deg2rad(target_degrees)


def test_optimize_user_position_defaults_to_neck_range(manager):
    coordinates = manager.user.getCoordinateSet()
    original_state = manager._user_state()
    original_other_coordinate = coordinates.get("neck_rotation").getValue(
        original_state
    )

    optimized = manager.optimize_user_position(
        incidence_weight=1.0,
        observed_screen_weight=0.0,
        reachability_weight=0.0,
        max_iterations=20,
    )

    neck = optimized.user.getCoordinateSet().get("neck_base_flexion")
    neck_value_deg = math.degrees(neck.getValue(optimized._user_state()))
    assert -30.0 <= neck_value_deg <= 30.0
    assert optimized.user.getCoordinateSet().get("neck_rotation").getValue(
        optimized._user_state()
    ) == pytest.approx(original_other_coordinate)


def test_optimization_requires_valid_weights(manager):

    with pytest.raises(ValueError, match="sum to 1"):
        manager.optimize_screen_position(
            incidence_weight=0.4,
            observed_screen_weight=0.4,
            reachability_weight=0.4,
        )


def test_reachability_hull_uses_fully_extended_palm_markers_independent_of_grip(
    manager,
):
    user = manager.user
    state = manager._user_state()
    coordinates = user.getCoordinateSet()
    elbow = coordinates.get("elbow_flex_r")
    wrist = coordinates.get("wrist_flex_r")
    original = (elbow.getValue(state), wrist.getValue(state))
    try:
        elbow.setValue(state, elbow.getRangeMin(), False)
        wrist.setValue(state, wrist.getRangeMin(), False)
        user.realizePosition(state)
        manager._reachability_cache.clear()
        extended_points = manager._reachability_hull("r")[0].copy()

        elbow.setValue(state, elbow.getRangeMax(), False)
        wrist.setValue(state, wrist.getRangeMax(), False)
        user.realizePosition(state)
        manager._reachability_cache.clear()
        bent_grip_points = manager._reachability_hull("r")[0]

        assert bent_grip_points == pytest.approx(extended_points)
        assert elbow.getValue(state) == pytest.approx(elbow.getRangeMax())
        assert wrist.getValue(state) == pytest.approx(wrist.getRangeMax())
    finally:
        elbow.setValue(state, original[0], False)
        wrist.setValue(state, original[1], False)
        user.realizePosition(state)


def test_reachability_hull_includes_effective_shoulder_range_boundaries(manager):
    user = manager.user
    state = manager._user_state()
    coordinates = user.getCoordinateSet()
    names = [
        f"{name}_r"
        for name in (
            "arm_flex",
            "arm_add",
            "arm_rot",
            "pro_sup",
            "wrist_flex",
            "wrist_dev",
        )
    ]
    sampled = [coordinates.get(name) for name in names]
    elbow = coordinates.get("elbow_flex_r")
    original = [coordinate.getValue(state) for coordinate in sampled]
    original_elbow = elbow.getValue(state)
    points = manager._reachability_hull("r")[0]
    marker = user.getMarkerSet().get("RPALM")

    try:
        for boundary in ("min", "max"):
            for coordinate in sampled:
                value = (
                    coordinate.getRangeMin()
                    if boundary == "min"
                    else coordinate.getRangeMax()
                )
                coordinate.setValue(state, value, False)
            elbow.setValue(state, elbow.getRangeMin(), False)
            user.realizePosition(state)
            endpoint = manager._vector(marker.getLocationInGround(state))

            assert np.min(np.linalg.norm(points - endpoint, axis=1)) < 1e-8
    finally:
        for coordinate, value in zip(sampled, original):
            coordinate.setValue(state, value, False)
        elbow.setValue(state, original_elbow, False)
        user.realizePosition(state)
