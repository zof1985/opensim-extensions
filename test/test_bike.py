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
    Sphere,
    Tube,
    cylinder_between,
    mass_properties,
)


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


def test_bike_setters_density_and_steps():
    bike = Bike(densities={"seat": 1000.0})
    iron = bike.component_masses_kg["frame"]
    bike.frame_density = 2 * bike.frame_density
    assert bike.component_masses_kg["frame"] == pytest.approx(2 * iron)
    bike.seat_tube_step = 0
    assert np.allclose(
        bike.seat_tube_top - bike.crank_axis_position,
        0.25 * np.array([-math.cos(math.radians(73)), math.sin(math.radians(73)), 0]),
    )
    with pytest.raises(ValueError):
        bike.seat_tube_step = 99
    with pytest.raises(ValueError):
        bike.set_brake_force_n(-1)
    bike.brake_force_n = 50.0
    assert bike.brake_force_n == 50.0


def test_fit_user_places_user_on_bike():
    bike = Bike(stem_origin_mm=(600, 500, 0))
    user = User("M", height_cm=178)
    framework = bike.fit_user(user)

    report = bike.fit_report
    assert abs(report["knee_to_pedal_axis_mm"]) < 20
    assert abs(report["handlebar_height_error_mm"]) < 20
    assert framework.initSystem() is not None
    assert np.isfinite(bike.pedal_left_angle_deg)


def test_fit_user_trunk_angle_defines_handlebar():

    bike = Bike(
        stem_origin_mm=(500, 500, 0),
        handlebar_horizontal_range_mm=(-300, 300),
        handlebar_horizontal_steps=30,
        stem_excursion_mm=400,
        stem_steps=20,
        handlebar_width_mm=400,
    )
    bike.fit_user(User("M", height_cm=178), trunk_angle_deg=30.0)
    report = bike.fit_report
    assert report["trunk_angle_deg"] == pytest.approx(30.0, abs=0.1)
    assert report["hand_error_mm"] < 10
    assert abs(report["shoulder_flexion_deg"] - 90) < 5


def test_fit_user_rejects_non_finite_angles():
    import pytest
    from opensim_extensions import User

    with pytest.raises(ValueError):
        Bike().fit_user(User("M", height_cm=178), shoulder_flexion_deg=float("nan"))
