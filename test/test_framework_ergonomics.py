import inspect
import math
import sys
import warnings
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


@pytest.fixture
def screen_ready_manager(manager):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return manager.optimize_user_position()


def _coordinate_snapshot(framework):
    coordinates = framework.user.getCoordinateSet()
    state = framework._user_state()
    return {
        coordinates.get(index).getName(): (
            coordinates.get(index).getValue(state),
            coordinates.get(index).getSpeedValue(state),
            coordinates.get(index).getDefaultValue(),
            coordinates.get(index).getDefaultSpeedValue(),
            coordinates.get(index).getRangeMin(),
            coordinates.get(index).getRangeMax(),
        )
        for index in range(coordinates.getSize())
    }


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

    assert set(view_state) == {
        "incidence_angle_deg",
        "observed_screen_percent",
        "reachable_screen_percent",
        "trunk_angle_from_horizontal_deg",
        "right_shoulder_flexion_deg",
        "left_shoulder_flexion_deg",
        "right_elbow_flexion_deg",
        "left_elbow_flexion_deg",
        "neck_base_flexion_deg",
        "head_flexion_deg",
    }
    assert view_state["incidence_angle_deg"] >= 0.0
    assert 0.0 <= view_state["observed_screen_percent"] <= 100.0
    assert 0.0 <= view_state["reachable_screen_percent"] <= 100.0
    assert view_state["right_shoulder_flexion_deg"] == pytest.approx(
        manager.user.right_shoulder_flexion
    )
    assert view_state["left_shoulder_flexion_deg"] == pytest.approx(
        manager.user.left_shoulder_flexion
    )
    assert view_state["right_elbow_flexion_deg"] == pytest.approx(
        manager.user.right_elbow_flexion
    )
    assert view_state["left_elbow_flexion_deg"] == pytest.approx(
        manager.user.left_elbow_flexion
    )
    assert view_state["neck_base_flexion_deg"] == pytest.approx(
        manager.user.neck_base_flexion
    )
    assert view_state["head_flexion_deg"] == pytest.approx(manager.user.neck_flexion)
    centers = manager.user.joint_centers
    hips = (
        np.asarray(centers["right_hip"]) + np.asarray(centers["left_hip"])
    ) / 2
    shoulders = (
        np.asarray(centers["right_shoulder"])
        + np.asarray(centers["left_shoulder"])
    ) / 2
    trunk_axis = shoulders - hips
    expected_trunk_angle = math.degrees(
        math.atan2(
            float(trunk_axis[1]),
            math.hypot(float(trunk_axis[0]), float(trunk_axis[2])),
        )
    )
    assert view_state["trunk_angle_from_horizontal_deg"] == pytest.approx(
        expected_trunk_angle
    )


def test_visual_cone_defaults_use_documented_ergonomic_values(manager):
    framework = VisualErgonomicsFramework(
        manager.user,
        manager.equipment,
        line_of_sight_offset_deg=-25.0,
    )

    assert framework.cone_width_angle_deg == pytest.approx(30.0)
    assert framework.cone_height_angle_deg == pytest.approx(30.0)
    assert framework.line_of_sight_offset_deg == pytest.approx(-25.0)
    _, configured_rotation = framework._eye_center_and_line_of_sight_rotation()
    _, head_rotation = framework._eye_center_and_head_rotation()
    offset = math.radians(-25.0)
    assert configured_rotation[:, 0] == pytest.approx(
        math.cos(offset) * head_rotation[:, 0]
        + math.sin(offset) * head_rotation[:, 1]
    )

def test_line_of_sight_is_tilted_down_from_head_plane(manager):
    _, head_rotation = manager._eye_center_and_head_rotation()
    _, visual_rotation = manager._eye_center_and_line_of_sight_rotation()
    downward = math.radians(abs(manager.line_of_sight_offset_deg))
    expected_forward = (
        math.cos(downward) * head_rotation[:, 0]
        - math.sin(downward) * head_rotation[:, 1]
    )

    assert visual_rotation[:, 0] == pytest.approx(expected_forward)


def test_visual_cone_starts_at_head_plane_independent_of_line_of_sight(manager):
    framework = VisualErgonomicsFramework(
        manager.user,
        manager.equipment,
        cone_height_angle_deg=45.0,
        line_of_sight_offset_deg=-15.0,
    )
    _, head_rotation = framework._eye_center_and_head_rotation()
    _, cone_rotation = framework._eye_center_and_cone_rotation()
    _, line_of_sight_rotation = framework._eye_center_and_line_of_sight_rotation()
    half_height = math.radians(framework.cone_height_angle_deg / 2)
    offset = math.radians(framework.line_of_sight_offset_deg)
    upper_edge_direction = (
        math.cos(half_height) * cone_rotation[:, 0]
        + math.sin(half_height) * cone_rotation[:, 1]
    )
    expected_line_of_sight = (
        math.cos(offset) * head_rotation[:, 0]
        + math.sin(offset) * head_rotation[:, 1]
    )

    assert upper_edge_direction == pytest.approx(head_rotation[:, 0])
    assert line_of_sight_rotation[:, 0] == pytest.approx(expected_line_of_sight)
    assert line_of_sight_rotation[:, 0] != pytest.approx(cone_rotation[:, 0])


def test_spherical_cap_mesh_uses_requested_axis_and_half_angle():
    from importlib import import_module

    ergonomics = import_module("opensim_extensions.framework.ergonomics")
    axis = np.array([1.0, 2.0, -3.0])
    original_axis = axis.copy()
    radius = 0.7
    points, faces = ergonomics._spherical_cap_mesh(
        radius, axis, half_angle_deg=45.0
    )
    unit_axis = axis / np.linalg.norm(axis)

    assert axis == pytest.approx(original_axis)
    assert np.linalg.norm(points, axis=1) == pytest.approx(
        np.full(len(points), radius)
    )
    assert np.all(
        points @ unit_axis / radius >= math.cos(math.radians(45.0)) - 1e-12
    )
    assert all(0 <= index < len(points) for face in faces for index in face)


def test_screen_contact_sphere_reaches_hand_distal_endpoint(manager):
    hand_length = manager.user.anthropometry.values["handlength"] * 1e-3
    for side, side_name in (("r", "right"), ("l", "left")):
        _, palm_center_radius = manager._reachability_sphere(side)
        _, contact_radius = manager._screen_contact_sphere(side)
        palm_length = getattr(manager.user, f"{side_name}_palm_length")

        assert contact_radius - palm_center_radius == pytest.approx(
            hand_length - palm_length / 2
        )


def test_reachability_objective_guides_screen_toward_distant_hull(manager, monkeypatch):
    position = (
        manager.equipment.screen_horizontal_offset_mm,
        manager.equipment.screen_vertical_offset_mm,
        manager.equipment.screen_inclination_deg,
    )
    screen = manager._screen_samples(position)

    def sphere_with_gap(gap):
        center = screen.mean(axis=0)
        center[0] -= 0.5
        radius = float(np.min(np.linalg.norm(screen - center, axis=1)) - gap)
        return center, radius

    gap = [0.05]
    monkeypatch.setattr(
        VisualErgonomicsFramework,
        "_reachability_sphere",
        lambda self, side: sphere_with_gap(gap[0]),
    )
    near_score = manager._optimization_scores(position)[2]
    gap[0] = 0.2
    far_score = manager._optimization_scores(position)[2]

    assert near_score > 0.1
    assert near_score > far_score


def test_reachability_hull_cache_tracks_small_trunk_changes(manager):
    state = manager._user_state()
    lumbar = manager.user.getCoordinateSet().get("lumbar_extension")
    original_value = lumbar.getValue(state)
    original_hull = manager._reachability_hull("r")[0].copy()
    changed_hull = None
    try:
        lumbar.setValue(state, original_value + math.radians(0.001), False)
        manager.user.realizePosition(state)
        changed_hull = manager._reachability_hull("r")[0].copy()
    finally:
        lumbar.setValue(state, original_value, False)
        manager.user.realizePosition(state)

    assert changed_hull is not None
    assert not np.array_equal(original_hull, changed_hull)


def test_optimization_returns_independent_framework_and_respects_screen_bounds(
    screen_ready_manager,
):
    original_user = screen_ready_manager.user
    original_equipment = screen_ready_manager.equipment
    user_before = _coordinate_snapshot(screen_ready_manager)
    screen_before = (
        original_equipment.screen_horizontal_offset_mm,
        original_equipment.screen_vertical_offset_mm,
        original_equipment.screen_inclination_deg,
    )
    optimized = screen_ready_manager.optimize_screen_position(
        screen_incline_range_deg=(40.0, 80.0)
    )

    assert isinstance(optimized, VisualErgonomicsFramework)
    assert isinstance(optimized.user, User)
    assert optimized.user is not original_user
    assert optimized.equipment is not original_equipment
    assert optimized.line_of_sight_offset_deg == pytest.approx(
        screen_ready_manager.line_of_sight_offset_deg
    )
    assert _coordinate_snapshot(screen_ready_manager) == user_before
    assert 40.0 <= optimized.equipment.screen_inclination_deg <= 80.0
    assert min(
        optimized._screen_constraints(
            (
                optimized.equipment.screen_horizontal_offset_mm,
                optimized.equipment.screen_vertical_offset_mm,
                optimized.equipment.screen_inclination_deg,
            )
        )
    ) >= -1e-6
    eye, gaze_rotation = optimized._eye_center_and_line_of_sight_rotation()
    gaze = gaze_rotation[:, 0]
    screen_corners = optimized.equipment.screen.corners(
        (
            optimized.equipment.screen_horizontal_offset_mm,
            optimized.equipment.screen_vertical_offset_mm,
            optimized.equipment.screen_inclination_deg,
        )
    )
    screen_center = screen_corners.mean(axis=0)
    assert screen_center - eye == pytest.approx(
        np.dot(screen_center - eye, gaze) * gaze, abs=1e-8
    )
    assert optimized.get_view_state()["incidence_angle_deg"] == pytest.approx(
        0.0, abs=1e-5
    )
    reach_spheres = [
        optimized._reachability_sphere(side) for side in ("r", "l")
    ]
    for corner in screen_corners:
        assert min(
            np.linalg.norm(corner - shoulder) - radius
            for shoulder, radius in reach_spheres
        ) <= 1e-8
    assert (
        optimized.equipment.screen_horizontal_offset_mm,
        optimized.equipment.screen_vertical_offset_mm,
        optimized.equipment.screen_inclination_deg,
    ) != pytest.approx(screen_before)
    assert set(optimized.get_view_state()) == {
        "incidence_angle_deg",
        "observed_screen_percent",
        "reachable_screen_percent",
        "trunk_angle_from_horizontal_deg",
        "right_shoulder_flexion_deg",
        "left_shoulder_flexion_deg",
        "right_elbow_flexion_deg",
        "left_elbow_flexion_deg",
        "neck_base_flexion_deg",
        "head_flexion_deg",
    }


def test_inplace_optimization_changes_only_the_screen(screen_ready_manager):
    user_before = _coordinate_snapshot(screen_ready_manager)
    result = screen_ready_manager.optimize_screen_position(inplace=True)

    assert result is None
    assert _coordinate_snapshot(screen_ready_manager) == user_before
    assert screen_ready_manager.get_view_state()["incidence_angle_deg"] == pytest.approx(
        0.0, abs=1e-5
    )


def test_screen_pose_parameters_accept_fixed_values(screen_ready_manager):
    screen_ready_manager.optimize_screen_position(inplace=True)
    original = screen_ready_manager.equipment.screen
    horizontal = original.horizontal_offset_mm
    vertical = original.vertical_offset_mm
    inclination = original.inclination_deg

    optimized = screen_ready_manager.optimize_screen_position(
        screen_horizontal_offset_mm=horizontal,
        screen_vertical_offset_mm=vertical,
        screen_inclination_deg=inclination,
    )

    assert optimized.equipment.screen_horizontal_offset_mm == pytest.approx(horizontal)
    assert optimized.equipment.screen_vertical_offset_mm == pytest.approx(vertical)
    assert optimized.equipment.screen_inclination_deg == pytest.approx(inclination)
    assert screen_ready_manager.equipment.screen_horizontal_offset_mm == pytest.approx(
        original.horizontal_offset_mm
    )


def test_screen_offset_range_expands_equipment_limit(screen_ready_manager):
    screen_ready_manager.optimize_screen_position(inplace=True)
    screen = screen_ready_manager.equipment.screen
    horizontal_limit = screen.horizontal_offset_min_mm - 0.1

    with pytest.warns(UserWarning, match="expanding the limits"):
        optimized = screen_ready_manager.optimize_screen_position(
            screen_horizontal_offset_mm=(
                horizontal_limit,
                screen.horizontal_offset_mm,
            ),
        )

    assert optimized.equipment.screen.horizontal_offset_min_mm == pytest.approx(
        horizontal_limit
    )
    assert optimized.equipment.screen_horizontal_offset_mm >= horizontal_limit


def test_optimize_user_position_fixes_float_and_expands_coordinate_limits(manager):
    original_screen = manager.equipment.screen
    target_degrees = -41.0
    state = manager._user_state()
    arm_flexion_degrees = math.degrees(
        manager.user.getCoordinateSet().get("arm_flex_r").getValue(state)
    )
    trunk_flexion_degrees = math.degrees(
        manager.user.getCoordinateSet().get("lumbar_extension").getValue(state)
    )

    with pytest.warns(UserWarning, match="expanding the limits"):
        optimized = manager.optimize_user_position(
            neck_base_flexion_deg=target_degrees,
            neck_rotation_deg=(-30.0, 30.0),
            neck_flexion_deg=(-45.0, 45.0),
            right_shoulder_flexion_deg=arm_flexion_degrees,
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


def test_user_position_joint_arguments_match_user_properties():
    parameters = inspect.signature(
        VisualErgonomicsFramework.optimize_user_position
    ).parameters
    joint_property_names = [
        name.removesuffix("_deg")
        for name in parameters
        if name.endswith("_deg")
    ]

    assert joint_property_names
    assert all(
        isinstance(getattr(User, name, None), property)
        for name in joint_property_names
    )


def test_optimize_user_position_defaults_to_neck_range(manager):
    coordinates = manager.user.getCoordinateSet()
    original_state = manager._user_state()
    neck_coordinate = coordinates.get("neck_base_flexion")
    original_neck = neck_coordinate.getValue(original_state)

    neck_coordinate.setValue(original_state, np.deg2rad(20.0), False)
    manager.user.realizePosition(original_state)
    try:
        optimized = manager.optimize_user_position(
            max_iterations=60,
        )
    finally:
        neck_coordinate.setValue(original_state, original_neck, False)
        manager.user.realizePosition(original_state)

    neck = optimized.user.getCoordinateSet().get("neck_base_flexion")
    _, gaze_angles = optimized._gaze_center_coordinates(
        (
            optimized.equipment.screen_horizontal_offset_mm,
            optimized.equipment.screen_vertical_offset_mm,
            optimized.equipment.screen_inclination_deg,
        )
    )
    assert gaze_angles == pytest.approx((0.0, 0.0), abs=1e-4)


def test_optimize_user_position_aligns_gaze_without_neck_optimization(
    manager, monkeypatch
):
    from importlib import import_module

    ergonomics_module = import_module("opensim_extensions.framework.ergonomics")

    def reject_optimization(*args, **kwargs):
        pytest.fail("neck alignment must not use the objective optimizer")

    monkeypatch.setattr(ergonomics_module, "minimize", reject_optimization)

    optimized = manager.optimize_user_position()
    _, gaze_angles = optimized._gaze_center_coordinates(
        (
            optimized.equipment.screen_horizontal_offset_mm,
            optimized.equipment.screen_vertical_offset_mm,
            optimized.equipment.screen_inclination_deg,
        )
    )

    assert gaze_angles == pytest.approx((0.0, 0.0), abs=1e-4)


def test_screen_optimization_rejects_incompatible_inclination_and_preserves_user(
    screen_ready_manager,
):
    user_before = _coordinate_snapshot(screen_ready_manager)

    with pytest.raises(ValueError, match="required screen inclination"):
        screen_ready_manager.optimize_screen_position(screen_inclination_deg=0.0)

    assert _coordinate_snapshot(screen_ready_manager) == user_before


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
    reachability_ranges = {
        "arm_flex_r": (0.0, math.pi),
        "arm_add_r": (math.radians(-30.0), math.radians(30.0)),
    }
    elbow = coordinates.get("elbow_flex_r")
    original = [coordinate.getValue(state) for coordinate in sampled]
    original_elbow = elbow.getValue(state)
    points = manager._reachability_hull("r")[0]
    marker = user.getMarkerSet().get("RPALM")

    try:
        for boundary in ("min", "max"):
            for name, coordinate in zip(names, sampled):
                lower, upper = reachability_ranges.get(
                    name,
                    (coordinate.getRangeMin(), coordinate.getRangeMax()),
                )
                value = lower if boundary == "min" else upper
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
