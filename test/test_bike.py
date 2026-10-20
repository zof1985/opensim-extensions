import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_extensions import Bike, User
from opensim_extensions.bodies import (
    Box,
    Cylinder,
    Disk,
    Ellipsoid,
    Sphere,
    Tube,
    cylinder_between,
    mass_properties,
)
from opensim_extensions.models.bike.bike import _horn_direction


def _forefoot_midpoint_pedal_coordinates_mm(bike, user, side):
    pedal = bike.getBodySet().get(f"pedal_{'right' if side == 'r' else 'left'}")
    transform = pedal.getTransformInGround(bike.state)
    rotation = np.array(
        [[transform.R().get(i, j) for j in range(3)] for i in range(3)]
    )
    position = np.array([transform.p().get(i) for i in range(3)])
    markers = user.getMarkerSet()
    toe = markers.get(f"{side.upper()}TOE")
    mt5 = markers.get(f"{side.upper()}MT5")
    toe_location = np.array(
        [toe.getLocationInGround(user.state).get(i) for i in range(3)]
    )
    mt5_location = np.array(
        [mt5.getLocationInGround(user.state).get(i) for i in range(3)]
    )
    midpoint = (toe_location + mt5_location) / 2
    return rotation.T @ (midpoint - position) * 1000


def _forefoot_midpoint_clearance_mm(bike, user, side):
    local_coordinates = _forefoot_midpoint_pedal_coordinates_mm(bike, user, side)
    pedal_half_thickness = bike._pedal_size_mm[2] / 2
    return local_coordinates[1] - pedal_half_thickness


def _foot_yaw_deg(user, side):
    markers = user.getMarkerSet()
    heel = markers.get(f"{side.upper()}CAL").getLocationInGround(user.state)
    toe = markers.get(f"{side.upper()}TOE").getLocationInGround(user.state)
    mt5 = markers.get(f"{side.upper()}MT5").getLocationInGround(user.state)
    midpoint = np.array([(toe.get(i) + mt5.get(i)) / 2 for i in range(3)])
    return math.degrees(
        math.atan2(
            midpoint[2] - heel.get(2),
            midpoint[0] - heel.get(0),
        )
    )


def _foot_pedal_orientation_errors_deg(bike, user, side):
    markers = user.getMarkerSet()
    heel, toe, mt5 = (
        np.array(
            [
                markers.get(f"{side.upper()}{name}")
                .getLocationInGround(user.state)
                .get(i)
                for i in range(3)
            ]
        )
        for name in ("CAL", "TOE", "MT5")
    )
    foot_axis = (toe + mt5) / 2 - heel
    foot_axis /= np.linalg.norm(foot_axis)
    sole_normal = np.cross(toe - heel, mt5 - heel)
    sole_normal /= np.linalg.norm(sole_normal)

    pedal_name = "pedal_right" if side == "r" else "pedal_left"
    transform = bike.getBodySet().get(pedal_name).getTransformInGround(bike.state)
    rotation = np.array(
        [[transform.R().get(i, j) for j in range(3)] for i in range(3)]
    )
    pedal_axis = rotation[:, 0]
    pedal_normal = rotation[:, 1]
    if np.dot(sole_normal, pedal_normal) < 0:
        sole_normal = -sole_normal

    plane_error = math.degrees(
        math.acos(np.clip(np.dot(sole_normal, pedal_normal), -1.0, 1.0))
    )
    axis_error = math.degrees(
        math.acos(np.clip(np.dot(foot_axis, pedal_axis), -1.0, 1.0))
    )
    return plane_error, axis_error


def test_primitive_volumes_and_inertia():
    mass, com, inertia = mass_properties([Sphere(radius=0.1)], 1000.0)
    assert mass == pytest.approx(1000 * 4 / 3 * math.pi * 1e-3)
    assert np.allclose(inertia, np.eye(3) * 0.4 * mass * 0.01)
    mass, _, inertia = mass_properties(
        [Cylinder(radius=0.1, length=1, axis="z")], 1000.0
    )
    assert inertia[2, 2] == pytest.approx(0.5 * mass * 0.01)
    mass, _, inertia = mass_properties([Box(size=(1, 2, 3))], 10.0)
    assert inertia[0, 0] == pytest.approx(mass * 13 / 12)
    assert Disk(radius=0.1, thickness=0.01).length == 0.01
    assert (
        Tube(radius=0.1, inner_radius=0.05, length=1).volume
        < Cylinder(radius=0.1, length=1).volume
    )
    ellipsoid = Ellipsoid(radii=(0.1, 0.2, 0.3))
    ellipsoid_mass = 1000.0 * ellipsoid.volume
    _, _, ellipsoid_inertia = mass_properties([ellipsoid], 1000.0)
    assert ellipsoid_inertia[0, 0] == pytest.approx(
        ellipsoid_mass * (0.2**2 + 0.3**2) / 5
    )


def test_bike_saddle_is_rounded_tapered_and_respects_configured_dimensions():
    bike = Bike(seat_length_mm=280.0, seat_width_mm=150.0, seat_height_mm=45.0)

    shapes = bike._seat_shapes()
    assert len(shapes) == 2
    assert all(isinstance(shape, Ellipsoid) for shape in shapes)
    x_min = min(shape.center[0] - shape.radii[0] for shape in shapes)
    x_max = max(shape.center[0] + shape.radii[0] for shape in shapes)
    assert x_min == pytest.approx(-140.0e-3)
    assert x_max == pytest.approx(140.0e-3)
    assert shapes[0].radii[2] == pytest.approx(75.0e-3)
    assert shapes[1].radii[2] < shapes[0].radii[2]
    assert bike.getBodySet().get("seat").getMass() > 0


def test_cylinder_between_aligns_axis():
    cyl = cylinder_between((0, 0, 0), (1, 2, 0.5), 0.02)
    direction = np.array([1, 2, 0.5]) / np.linalg.norm([1, 2, 0.5])
    assert np.allclose(cyl.rotation @ [0, 1, 0], direction)
    assert cyl.length == pytest.approx(np.linalg.norm([1, 2, 0.5]))


def test_bike_defaults_and_only_pedals_move():
    bike = Bike()
    assert bike.getCoordinateSet().getSize() == 3
    assert bike.total_mass_kg == pytest.approx(sum(bike.component_masses_kg.values()))
    state = bike.initSystem()
    crank = bike.getBodySet().get("crank_right")
    before = bike.getBodySet().get("seat").getPositionInGround(state)
    bike.crank_angle_deg = 90
    assert bike.getBodySet().get("seat").getPositionInGround(bike.state).get(
        1
    ) == pytest.approx(before.get(1))
    tip = bike.getBodySet().get("pedal_right").getPositionInGround(bike.state)
    assert tip.get(1) == pytest.approx(0.25 + 0.17)


def test_bike_public_positions_are_in_millimetres():
    bike = Bike()

    assert np.allclose(bike.crank_axis_position, [0.0, 250.0, 100.0])
    assert bike.pedal_lateral_offset == pytest.approx(185.0 / 2 + 70.0)
    positions = (
        (bike.seat_tube_top, bike._seat_tube_top_at(bike._seat_tube_step)),
        (
            bike.seat_position,
            bike._seat_position_at(bike._seat_tube_step, bike._seat_horizontal_step),
        ),
        (bike.stem_base, bike._stem_base_m),
        (bike.stem_top, bike._stem_top_at(bike._stem_step)),
        (bike.handlebar_position, bike._handlebar_position_m),
        (bike.screen_base, bike._screen_base_m),
        (bike.screen_center, bike._screen_center_m),
    )
    for position_mm, position_m in positions:
        assert np.allclose(position_mm, position_m * 1000.0)


def test_bike_setters_density_and_steps():
    bike = Bike(densities={"seat": 1000.0})
    iron = bike.component_masses_kg["frame"]
    bike.frame_density = 2 * bike.frame_density
    assert bike.component_masses_kg["frame"] == pytest.approx(2 * iron)
    bike.seat_tube_step = 0
    angle = math.radians(bike.seat_tube_angle_deg)
    assert np.allclose(
        bike.seat_tube_top - bike.crank_axis_position,
        250 * np.array([math.cos(angle), math.sin(angle), 0]),
    )
    with pytest.raises(ValueError):
        bike.seat_tube_step = 99
    with pytest.raises(ValueError):
        bike.set_brake_force_n(-1)
    bike.brake_force_n = 50.0
    assert bike.brake_force_n == 50.0


def test_pedal_spacing_is_numeric_independent_and_measures_the_inner_face_gap():
    bike = Bike(crank_spacing_mm=200.0)
    assert bike.pedal_spacing_mm == pytest.approx(185.0)
    assert Bike().pedal_spacing_mm == pytest.approx(185.0)

    bike.crank_spacing_mm = 180.0
    assert bike.pedal_spacing_mm == pytest.approx(185.0)

    bike.pedal_spacing_mm = 205.0
    pedal_width = bike._pedal_size_mm[1]
    right = bike.getBodySet().get("pedal_right").getPositionInGround(bike.state)
    left = bike.getBodySet().get("pedal_left").getPositionInGround(bike.state)
    inner_gap = abs(right.get(2) - left.get(2)) * 1000.0 - pedal_width
    assert inner_gap == pytest.approx(205.0)
    assert bike.pedal_lateral_offset == pytest.approx(
        205.0 / 2 + pedal_width
    )
    with pytest.raises(TypeError):
        Bike(pedal_spacing_mm=None)


def test_bike_screen_geometry_and_center_follow_configured_offsets():
    bike = Bike(
        screen_width_mm=180.0,
        screen_height_mm=120.0,
        screen_inclination_deg=120.0,
        screen_vertical_offset_mm=90.0,
        screen_horizontal_offset_mm=70.0,
    )

    angle = math.radians(120.0)
    expected_center = bike.handlebar_position + np.array([70.0, 90.0, 0.0])
    expected_center += 60.0 * np.array([math.cos(angle), math.sin(angle), 0.0])
    assert np.allclose(bike.screen_center, expected_center)
    screen = bike.getBodySet().get("screen")
    screen_position = screen.getPositionInGround(bike.state)
    assert np.allclose(
        [screen_position.get(i) * 1000.0 for i in range(3)],
        bike.screen_base,
    )
    screen_rotation = screen.getTransformInGround(bike.state).R()
    panel_up = np.array(
        [screen_rotation.get(i, 1) for i in range(3)]
    )
    actual_center_mm = (
        bike.screen_base + panel_up * bike._screen_height_mm / 2
    )
    assert np.allclose(actual_center_mm, bike.screen_center)
    screen_joint = bike.getJointSet().get("screen_weld")
    assert screen_joint.getConcreteClassName() == "WeldJoint"
    parent_frame = screen_joint.getParentFrame()
    assert parent_frame.getName() == "screen_mount_offset"
    assert bike.getBodySet().contains("screen_mount")

    bike.screen_horizontal_offset_mm = 80.0
    assert bike.screen_base[0] == pytest.approx(
        bike.handlebar_position[0] + 80.0
    )
    assert bike.getBodySet().get("screen") is not None


def test_bike_directed_angles_use_the_world_xy_reference():
    default_bike = Bike()
    assert default_bike.seat_tube_angle_deg == pytest.approx(104.0)
    assert default_bike.stem_angle_deg == pytest.approx(82.0)
    assert default_bike.screen_inclination_deg == pytest.approx(105.0)

    bike = Bike(
        seat_tube_angle_deg=120.0,
        stem_angle_deg=60.0,
        stem_step=10,
    )

    seat_vector = bike.seat_tube_top - bike.crank_axis_position
    stem_vector = bike.stem_top - bike.stem_base
    assert math.degrees(math.atan2(seat_vector[1], seat_vector[0])) == pytest.approx(
        120.0
    )
    assert math.degrees(math.atan2(stem_vector[1], stem_vector[0])) == pytest.approx(
        60.0
    )



def test_horn_pitch_raises_tips_and_positive_roll_turns_horns_medially():
    bike = Bike(horn_pitch_deg=90.0, horn_roll_deg=0.0)
    assert np.allclose(
        _horn_direction(bike.horn_pitch_deg, bike.horn_roll_deg, "r"),
        [0.0, 1.0, 0.0],
        atol=1e-7,
    )
    right = _horn_direction(0.0, 10.0, "r")
    left = _horn_direction(0.0, 10.0, "l")
    assert right[2] < 0.0
    assert left[2] > 0.0


def test_bike_screen_offsets_must_be_positive():
    for value in (-1.0, 0.0):
        with pytest.raises(ValueError):
            Bike(screen_horizontal_offset_mm=value)
        with pytest.raises(ValueError):
            Bike(screen_vertical_offset_mm=value)


def test_fit_user_places_user_on_bike():
    bike = Bike(stem_origin_mm=(600, 500, 0))
    user = User("M", height_cm=178)
    framework = bike.fit_user(user, trunk_angle_deg=None)

    report = bike.fit_report
    assert abs(report["knee_to_pedal_axis_mm"]) < 20
    assert abs(report["handlebar_height_error_mm"]) < 20
    assert framework.initSystem() is not None
    assert np.isfinite(bike.pedal_left_angle_deg)


@pytest.mark.parametrize("pedal_spacing_mm", [185.0, 205.0])
def test_fit_user_feet_align_forefoot_markers_with_pedals_and_hands_are_neutral(
    pedal_spacing_mm,
):
    bike = Bike(stem_origin_mm=(600, 500, 0), pedal_spacing_mm=pedal_spacing_mm)
    user = User("M", height_cm=178)
    framework = bike.fit_user(user)

    state = user.state
    subtalar_angles = [
        user.getCoordinateSet().get(f"subtalar_angle_{side}").getValue(state)
        for side in ("r", "l")
    ]
    assert any(abs(angle) > math.radians(1.0) for angle in subtalar_angles)
    pedal_width = bike._pedal_size_mm[1] * 1e-3
    for side, sign in (("R", 1.0), ("L", -1.0)):
        mt5 = user.getMarkerSet().get(f"{side}MT5").getLocationInGround(state)
        toe = user.getMarkerSet().get(f"{side}TOE").getLocationInGround(state)
        pedal_center_z = (
            bike.crank_axis_position[2] * 1e-3
            + sign * (bike.pedal_lateral_offset * 1e-3 - pedal_width / 2)
        )
        inner_edge_z = pedal_center_z - sign * pedal_width / 2
        assert toe.get(2) == pytest.approx(inner_edge_z, abs=5e-3)
        midpoint_coordinates = _forefoot_midpoint_pedal_coordinates_mm(
            bike,
            user,
            side.lower(),
        )
        assert midpoint_coordinates[0] == pytest.approx(0.0, abs=5.0)
        assert midpoint_coordinates[2] == pytest.approx(0.0, abs=5.0)
        assert np.linalg.norm(midpoint_coordinates[:2]) == pytest.approx(
            bike._pedal_size_mm[2] / 2 + 30.0,
            abs=5.0,
        )
        assert abs(_foot_yaw_deg(user, side.lower())) < 10.0
        plane_error, axis_error = _foot_pedal_orientation_errors_deg(
            bike,
            user,
            side.lower(),
        )
        assert plane_error < 5.0
        assert axis_error < 10.0
    assert framework.couplings["foot_right"].kind == "weld"
    assert framework.couplings["foot_left"].kind == "weld"
    for side in ("r", "l"):
        assert user.getCoordinateSet().get(f"pro_sup_{side}").getValue(state) == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("shoe_height_cm", [0.0, 3.0])
def test_fit_user_places_forefoot_midpoints_at_the_requested_shoe_height(
    shoe_height_cm,
):
    bike = Bike(stem_origin_mm=(600, 500, 0))
    user = User("M", height_cm=178)

    bike.fit_user(user, shoe_height_cm=shoe_height_cm)

    for side in ("r", "l"):
        assert _forefoot_midpoint_clearance_mm(bike, user, side) == pytest.approx(
            shoe_height_cm * 10,
            abs=1.0,
        )
    assert user.getCoordinateSet().get("ankle_angle_l").getValue(user.state) == pytest.approx(
        0.0,
        abs=1e-3,
    )
    assert abs(bike.pedal_left_angle_deg) > 1.0


def test_fit_user_trunk_angle_defines_handlebar():

    bike = Bike(
        stem_origin_mm=(500, 500, 0),
        handlebar_horizontal_range_mm=(-300, 300),
        handlebar_horizontal_steps=30,
        stem_excursion_mm=400,
        stem_steps=20,
        handlebar_width_mm=400,
    )
    user = User("M", height_cm=178)
    bike.fit_user(user, trunk_angle_deg=30.0)
    report = bike.fit_report
    assert report["trunk_angle_deg"] == pytest.approx(30.0, abs=1.0)
    assert report["hand_error_mm"] < 25
    assert abs(report["shoulder_flexion_deg"] - 75) < 5
    coordinates = user.getCoordinateSet()
    for right_name, left_name in (
        ("arm_flex_r", "arm_flex_l"),
        ("elbow_flex_r", "elbow_flex_l"),
    ):
        assert coordinates.get(right_name).getValue(user.state) == pytest.approx(
            coordinates.get(left_name).getValue(user.state), abs=1e-8
        )


def test_fit_user_adjusts_unspecified_trunk_to_reach_handlebar_horns():
    bike = Bike()
    user = User("M", height_cm=178, mass_kg=88)
    initial_lumbar = user.lumbar_extension

    bike.fit_user(
        user,
        shoe_height_cm=3.0,
        trunk_angle_deg=None,
        shoulder_flexion_deg=90.0,
        elbow_flexion_deg=15.0,
    )

    report = bike.fit_report
    assert user.lumbar_extension != pytest.approx(initial_lumbar, abs=1e-3)
    assert report["trunk_angle_deg"] > 20.0
    assert report["hand_error_mm"] < 5.0
    coordinates = user.getCoordinateSet()
    for right_name, left_name in (
        ("arm_flex_r", "arm_flex_l"),
        ("elbow_flex_r", "elbow_flex_l"),
    ):
        assert coordinates.get(right_name).getValue(user.state) == pytest.approx(
            coordinates.get(left_name).getValue(user.state), abs=1e-8
        )


def test_fit_user_can_keep_requested_trunk_shoulder_and_elbow_angles():
    bike = Bike(
        stem_origin_mm=(500, 500, 0),
        handlebar_horizontal_range_mm=(-300, 300),
        handlebar_horizontal_steps=30,
        stem_excursion_mm=400,
        stem_steps=20,
        handlebar_width_mm=400,
    )
    user = User("M", height_cm=178)

    bike.fit_user(
        user,
        trunk_angle_deg=30.0,
        shoulder_flexion_deg=60.0,
        elbow_flexion_deg=20.0,
        allow_ik_adjustment=False,
    )

    assert bike.fit_report["trunk_angle_deg"] == pytest.approx(30.0, abs=0.1)
    assert bike.fit_report["shoulder_flexion_deg"] == pytest.approx(60.0, abs=0.1)
    assert bike.fit_report["elbow_flexion_deg"] == pytest.approx(20.0, abs=0.1)
    coordinates = user.getCoordinateSet()
    for side in ("r", "l"):
        assert coordinates.get(f"arm_flex_{side}").getValue(user.state) == pytest.approx(
            math.radians(60.0), abs=1e-3
        )
        assert coordinates.get(f"elbow_flex_{side}").getValue(user.state) == pytest.approx(
            math.radians(20.0), abs=1e-3
        )
    assert any(
        abs(coordinates.get(f"arm_add_{side}").getValue(user.state)) > 1e-3
        for side in ("r", "l")
    )


def test_fit_user_preserves_existing_trunk_when_angle_is_none_and_ik_is_disabled():
    bike = Bike(stem_origin_mm=(600, 500, 0))
    user = User("M", height_cm=178)
    user.set_lumbar_extension(12.0)

    bike.fit_user(user, trunk_angle_deg=None, allow_ik_adjustment=False)

    assert user.lumbar_extension == pytest.approx(12.0, abs=1e-3)


def test_fit_user_preserves_wrist_and_neck_joint_angles():
    bike = Bike(stem_origin_mm=(600, 500, 0))
    user = User("M", height_cm=178)
    requested = {
        "neck_base_flexion": (user.set_neck_base_flexion, 12.0),
        "neck_base_lateral_flexion": (user.set_neck_base_lateral_flexion, -8.0),
        "neck_flexion": (user.set_neck_flexion, 20.0),
        "neck_lateral_flexion": (user.set_neck_lateral_flexion, 9.0),
        "neck_rotation": (user.set_neck_rotation, -7.0),
        "wrist_flex_r": (user.set_right_wrist_flexion, 8.0),
        "wrist_dev_r": (user.set_right_wrist_deviation, -4.0),
        "wrist_flex_l": (user.set_left_wrist_flexion, -6.0),
        "wrist_dev_l": (user.set_left_wrist_deviation, 5.0),
    }
    for setter, angle in requested.values():
        setter(angle)

    bike.fit_user(user, trunk_angle_deg=None)

    coordinates = user.getCoordinateSet()
    for name, (_, angle) in requested.items():
        assert coordinates.get(name).getValue(user.state) == pytest.approx(
            math.radians(angle)
        )


def test_neck_flexion_set_after_fit_is_included_in_framework_snapshot():
    bike = Bike(stem_origin_mm=(600, 500, 0))
    user = User("M", height_cm=178)
    framework = bike.fit_user(user, trunk_angle_deg=None)

    framework.initSystem()
    framework["user"].set_neck_flexion(30.0)
    snapshot = framework.merge_to_model()
    snapshot_state = snapshot.getWorkingState()

    assert snapshot.getCoordinateSet().get("neck_flexion").getValue(
        snapshot_state
    ) == pytest.approx(math.radians(30.0))


def test_fit_user_without_user_uses_the_default_benchmark_rider_and_pose():
    bike = Bike()

    framework = bike.fit_user()

    user = framework["user"]
    assert isinstance(user, User)
    assert user.gender == "M"
    assert user.height_cm == pytest.approx(170.0)
    assert user.mass_kg == pytest.approx(70.0)
    assert bike.fit_report["trunk_angle_deg"] == pytest.approx(45.0, abs=0.1)
    assert bike.fit_report["shoulder_flexion_deg"] == pytest.approx(75.0, abs=2.0)
    assert bike.fit_report["elbow_flexion_deg"] == pytest.approx(15.0, abs=2.0)
    assert bike.fit_report["hand_error_mm"] < 5.0


def test_fit_user_rejects_non_finite_angles():
    import pytest
    from opensim_extensions import User

    with pytest.raises(ValueError):
        Bike().fit_user(User("M", height_cm=178), shoulder_flexion_deg=float("nan"))
