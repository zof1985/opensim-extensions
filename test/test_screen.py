import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import Screen, components
from opensim_models.model import OpenSimModel
from opensim_models.components.screen import (
    _MESH_FILENAME,
    _MESHES_DIR,
    _PLEXIGLASS_DENSITY_KG_M3,
    _THICKNESS_MM,
)

opensim = pytest.importorskip("opensim")


def joint_offset_frames(screen):
    joint = screen._container.model.getJointSet().get("screen_panel_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.getParentFrame())
    child = opensim.PhysicalOffsetFrame.safeDownCast(joint.getChildFrame())
    return parent, child


def expected_mass_kg(width_mm, height_mm):
    volume_m3 = (width_mm / 1000.0) * (height_mm / 1000.0) * (_THICKNESS_MM / 1000.0)
    return volume_m3 * _PLEXIGLASS_DENSITY_KG_M3


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_default_size_comes_from_22in_16_9_diagonal():
    screen = Screen()

    diagonal_mm = 22 * 25.4
    scale = diagonal_mm / math.hypot(16, 9)
    assert screen.width_mm is None
    assert screen.height_mm is None
    body = screen._container.model.getBodySet().get("screen_panel")
    assert body.get_mass() == pytest.approx(
        expected_mass_kg(16 * scale, 9 * scale), rel=1e-6
    )


def test_explicit_size_overrides_diagonal_and_ratio():
    screen = Screen(width_mm=500.0, height_mm=300.0)

    assert screen.width_mm == 500.0
    assert screen.height_mm == 300.0
    body = screen._container.model.getBodySet().get("screen_panel")
    assert body.get_mass() == pytest.approx(expected_mass_kg(500.0, 300.0), rel=1e-6)


def test_only_one_of_width_height_falls_back_to_diagonal():
    screen = Screen(width_mm=500.0)

    diagonal_mm = 22 * 25.4
    scale = diagonal_mm / math.hypot(16, 9)
    body = screen._container.model.getBodySet().get("screen_panel")
    assert body.get_mass() == pytest.approx(
        expected_mass_kg(16 * scale, 9 * scale), rel=1e-6
    )


def test_neither_sizing_method_usable_raises():
    with pytest.raises(ValueError, match="inches and ratio"):
        Screen(inches=None, ratio=None)


def test_invalid_ratio_string_raises():
    with pytest.raises(ValueError, match="aspect ratio"):
        Screen(ratio="16-9")
    with pytest.raises(ValueError, match="aspect ratio"):
        Screen(ratio="-16:9")


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


def test_center_and_angle_set_the_joint_offset():
    screen = Screen(center_x=1.0, center_y=2.0, center_z=3.0, angle_deg=45.0)

    parent, _ = joint_offset_frames(screen)
    translation = parent.get_translation()
    orientation = parent.get_orientation()
    assert (translation[0], translation[1], translation[2]) == pytest.approx(
        (1.0, 2.0, 3.0)
    )
    assert orientation[0] == pytest.approx(math.radians(45.0 - 90.0))


def test_default_angle_stands_the_panel_upright():
    screen = Screen()

    parent, _ = joint_offset_frames(screen)
    assert parent.get_orientation()[0] == pytest.approx(0.0)


def test_position_global_matches_center_coordinates():
    # The inherited Body.position_global (ground-frame body origin) is,
    # for a screen (a single uniform body with mass_center=(0, 0, 0)), the
    # same point as (center_x, center_y, center_z) -- and as origin, see
    # test_origin_matches_center_coordinates_and_position_global below.
    screen = Screen(center_x=1.0, center_y=2.0, center_z=3.0, angle_deg=45.0)

    assert screen.position_global == pytest.approx((1.0, 2.0, 3.0))


def test_position_local_is_always_the_origin():
    screen = Screen(center_x=1.0, center_y=2.0, center_z=3.0)

    assert screen.position_local == (0.0, 0.0, 0.0)


def test_origin_matches_center_coordinates_and_position_global():
    screen = Screen(center_x=1.0, center_y=2.0, center_z=3.0, angle_deg=45.0)

    assert screen.origin == pytest.approx((1.0, 2.0, 3.0))
    assert screen.origin == pytest.approx(screen.position_global)


def test_set_origin_moves_the_panel_and_keeps_its_angle():
    screen = Screen(center_x=1.0, center_y=1.0, center_z=1.0, angle_deg=30.0)

    screen.set_origin((5.0, 0.0, 0.0))

    assert screen.origin == pytest.approx((5.0, 0.0, 0.0))
    assert screen.center_x == pytest.approx(5.0)
    assert screen.center_y == pytest.approx(0.0)
    assert screen.center_z == pytest.approx(0.0)
    assert screen.angle_deg == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# rotate()/translate()
# ---------------------------------------------------------------------------


def test_rotate_about_its_own_com_updates_angle_but_not_origin():
    screen = Screen(center_x=1.0, center_y=2.0, center_z=0.0)
    original_com = screen.com

    screen.rotate(screen.com, (0.0, 1.0, 0.0), 30.0)

    assert screen.com == pytest.approx(original_com)


def test_translate_moves_the_panel():
    screen = Screen(center_x=0.0, center_y=0.0, center_z=0.0)

    screen.translate((5.0, 0.0, 0.0))

    assert screen.com == pytest.approx((5.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Model structure
# ---------------------------------------------------------------------------


def test_model_has_a_single_body_welded_to_ground():
    screen = Screen()

    assert screen._container.model.getBodySet().getSize() == 1
    assert screen._container.model.getJointSet().getSize() == 1
    joint = screen._container.model.getJointSet().get("screen_panel_joint")
    assert opensim.WeldJoint.safeDownCast(joint) is not None


def test_instances_are_backed_by_independent_opensim_models():
    small = Screen(width_mm=100.0, height_mm=100.0)
    large = Screen(width_mm=900.0, height_mm=900.0)

    assert small._container.model is not large._container.model
    assert small._container.model.getBodySet().get("screen_panel").get_mass() != pytest.approx(
        large._container.model.getBodySet().get("screen_panel").get_mass()
    )


def test_copy_preserves_type_and_parameters_without_a_screen_override():
    screen = Screen(width_mm=500.0, height_mm=300.0, center_x=1.0)

    duplicate = screen.copy()

    assert type(duplicate) is Screen
    assert duplicate.width_mm == 500.0
    assert duplicate.center_x == 1.0
    assert duplicate._container.model is not screen._container.model


# ---------------------------------------------------------------------------
# Screen is a component (Body), not a container (OpenSimModel)
# ---------------------------------------------------------------------------


def test_screen_is_a_body_component_not_a_container():
    screen = Screen()

    assert isinstance(screen, components.Body)
    assert not isinstance(screen, OpenSimModel)
    assert not hasattr(screen, "show")


def test_screen_is_addable_to_a_model_in_both_directions():
    screen = Screen(center_x=2.0)
    model = OpenSimModel(model_path=None)

    merged = model + screen
    assert isinstance(merged, OpenSimModel)
    assert "screen_panel" in merged.bodies
    assert "screen_panel" not in model.bodies

    merged_reflected = screen + model
    assert "screen_panel" in merged_reflected.bodies


# ---------------------------------------------------------------------------
# Setters rebuild the panel and its mesh
# ---------------------------------------------------------------------------


def test_setters_expose_getters_as_properties():
    screen = Screen()

    screen.set_width_mm(500.0)
    screen.set_height_mm(300.0)
    screen.set_inches(24.0)
    screen.set_ratio("4:3")
    screen.set_center_x(1.0)
    screen.set_center_y(2.0)
    screen.set_center_z(3.0)
    screen.set_angle_deg(30.0)

    assert screen.width_mm == 500.0
    assert screen.height_mm == 300.0
    assert screen.inches == 24.0
    assert screen.ratio == "4:3"
    assert screen.center_x == 1.0
    assert screen.center_y == 2.0
    assert screen.center_z == 3.0
    assert screen.angle_deg == 30.0


def test_switching_back_to_diagonal_sizing_after_explicit_size():
    screen = Screen(width_mm=500.0, height_mm=300.0)

    screen.set_width_mm(None)
    screen.set_height_mm(None)
    screen.set_inches(22.0)
    screen.set_ratio("16:9")

    diagonal_mm = 22 * 25.4
    scale = diagonal_mm / math.hypot(16, 9)
    body = screen._container.model.getBodySet().get("screen_panel")
    assert body.get_mass() == pytest.approx(
        expected_mass_kg(16 * scale, 9 * scale), rel=1e-6
    )


def test_size_setter_regenerates_the_mesh_on_disk():
    screen = Screen(width_mm=500.0, height_mm=300.0)
    mesh_path = _MESHES_DIR / _MESH_FILENAME
    first_contents = mesh_path.read_text()

    screen.set_width_mm(700.0)
    screen.set_height_mm(400.0)

    assert mesh_path.read_text() != first_contents
    assert mesh_path.read_text().count("facet normal") == 12


def test_screen_registers_its_own_mesh_directory():
    screen = Screen()

    assert _MESHES_DIR in screen._container.geometry_directories
