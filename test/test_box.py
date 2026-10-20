import itertools as it
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import Box, components
from opensim_models.model import OpenSimModel
from opensim_models.components.box import _MESH_FILENAME, _MESHES_DIR

opensim = pytest.importorskip("opensim")


def joint_offset_frames(box):
    joint = box._container.model.getJointSet().get("box_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.getParentFrame())
    child = opensim.PhysicalOffsetFrame.safeDownCast(joint.getChildFrame())
    return parent, child


def expected_inertia(mass, width, height, depth):
    return (
        mass * (height**2 + depth**2) / 12.0,
        mass * (width**2 + depth**2) / 12.0,
        mass * (width**2 + height**2) / 12.0,
    )


# ---------------------------------------------------------------------------
# Construction, mass/inertia, validation
# ---------------------------------------------------------------------------


def test_default_mass_is_one_kilogram():
    box = Box(width=0.2, height=0.3, depth=0.1)

    assert box.mass_kg == pytest.approx(1.0)
    body = box._container.model.getBodySet().get("box")
    assert body.get_mass() == pytest.approx(1.0)


def test_mass_and_inertia_match_a_solid_rectangular_prism():
    box = Box(width=0.2, height=0.3, depth=0.4, mass_kg=5.0)

    body = box._container.model.getBodySet().get("box")
    assert body.get_mass() == pytest.approx(5.0)
    moments = body.getInertia().getMoments()
    expected = expected_inertia(5.0, 0.2, 0.3, 0.4)
    assert moments.get(0) == pytest.approx(expected[0])
    assert moments.get(1) == pytest.approx(expected[1])
    assert moments.get(2) == pytest.approx(expected[2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0.0, "height": 1.0, "depth": 1.0},
        {"width": -1.0, "height": 1.0, "depth": 1.0},
        {"width": 1.0, "height": 0.0, "depth": 1.0},
        {"width": 1.0, "height": 1.0, "depth": -1.0},
        {"width": 1.0, "height": 1.0, "depth": 1.0, "mass_kg": 0.0},
    ],
)
def test_rejects_non_positive_dimensions_or_mass(kwargs):
    with pytest.raises(ValueError, match="strictly positive"):
        Box(**kwargs)


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


def test_origin_and_angle_set_the_joint_offset():
    box = Box(width=0.1, height=0.1, depth=0.1, origin=(1.0, 2.0, 3.0), angle_deg=(0.0, 0.0, 45.0))

    parent, _ = joint_offset_frames(box)
    translation = parent.get_translation()
    orientation = parent.get_orientation()
    assert (translation[0], translation[1], translation[2]) == pytest.approx((1.0, 2.0, 3.0))
    assert orientation[2] == pytest.approx(math.radians(45.0))


def test_origin_and_angle_properties_match_the_constructor_arguments():
    box = Box(width=0.1, height=0.1, depth=0.1, origin=(1.0, 2.0, 3.0), angle_deg=(10.0, 20.0, 30.0))

    assert box.origin == pytest.approx((1.0, 2.0, 3.0))
    assert box.angle_deg == pytest.approx((10.0, 20.0, 30.0))


def test_default_pose_is_the_origin_with_no_rotation():
    box = Box(width=0.1, height=0.1, depth=0.1)

    assert box.origin == pytest.approx((0.0, 0.0, 0.0))
    assert box.angle_deg == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_com_matches_origin_for_a_single_uniform_body():
    box = Box(width=0.2, height=0.3, depth=0.4, origin=(1.0, -2.0, 0.5))

    assert box.com == pytest.approx(box.origin)


def test_position_global_matches_origin():
    # Box.origin is now a thin alias of the inherited Body.position_global
    # (see components/box.py); this guards that equivalence directly.
    box = Box(width=0.2, height=0.3, depth=0.4, origin=(1.0, -2.0, 0.5), angle_deg=(0.0, 0.0, 30.0))

    assert box.position_global == pytest.approx(box.origin)
    assert box.position_global == pytest.approx((1.0, -2.0, 0.5))


def test_position_local_is_always_the_origin():
    box = Box(width=0.2, height=0.3, depth=0.4, origin=(1.0, -2.0, 0.5))

    assert box.position_local == (0.0, 0.0, 0.0)


def test_angle_deg_matches_inclination():
    # Box.angle_deg is now a thin alias of the inherited Body.inclination
    # (see components/box.py); this guards that equivalence directly.
    box = Box(width=0.1, height=0.1, depth=0.1, angle_deg=(10.0, 20.0, 30.0))

    assert box.angle_deg == pytest.approx(box.inclination)


def test_set_origin_moves_the_box_and_keeps_its_orientation():
    box = Box(width=0.1, height=0.1, depth=0.1, origin=(1.0, 1.0, 1.0), angle_deg=(10.0, 20.0, 30.0))

    box.set_origin((5.0, 0.0, 0.0))

    assert box.origin == pytest.approx((5.0, 0.0, 0.0))
    assert box.angle_deg == pytest.approx((10.0, 20.0, 30.0))


def test_set_angle_deg_reorients_the_box_and_keeps_its_centre():
    box = Box(width=0.1, height=0.1, depth=0.1, origin=(1.0, 1.0, 1.0), angle_deg=(10.0, 20.0, 30.0))

    box.set_angle_deg((0.0, 0.0, 0.0))

    assert box.angle_deg == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert box.origin == pytest.approx((1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# rotate()/translate() keep origin/angle_deg/corners live (no stale cache)
# ---------------------------------------------------------------------------


def test_rotate_about_its_own_com_updates_angle_but_not_origin():
    box = Box(width=0.2, height=0.2, depth=0.2, origin=(1.0, 2.0, 0.0))
    original_origin = box.origin

    box.rotate(box.com, (0.0, 0.0, 1.0), 90.0)

    assert box.origin == pytest.approx(original_origin)
    assert box.angle_deg[2] == pytest.approx(90.0)


def test_translate_updates_origin_and_corners():
    box = Box(width=0.2, height=0.2, depth=0.2, origin=(0.0, 0.0, 0.0))

    box.translate((5.0, 0.0, 0.0))

    assert box.origin == pytest.approx((5.0, 0.0, 0.0))
    corner_x_values = sorted(corner[0] for corner in box.corners)
    assert corner_x_values[0] == pytest.approx(4.9)
    assert corner_x_values[-1] == pytest.approx(5.1)


def test_dimension_setter_preserves_pose_set_by_rotate_and_translate():
    box = Box(width=0.2, height=0.2, depth=0.2)
    box.rotate(box.com, (0.0, 0.0, 1.0), 90.0)
    box.translate((5.0, 0.0, 0.0))

    box.set_width(0.5)

    assert box.origin == pytest.approx((5.0, 0.0, 0.0))
    assert box.angle_deg[2] == pytest.approx(90.0)
    assert box.width == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------


def test_corners_are_the_eight_combinations_of_half_extents():
    box = Box(width=0.2, height=0.4, depth=0.6, origin=(0.0, 0.0, 0.0))

    expected = {
        (sx * 0.1, sy * 0.2, sz * 0.3) for sx, sy, sz in it.product((-1, 1), repeat=3)
    }
    actual = {tuple(round(v, 9) for v in corner) for corner in box.corners}
    assert actual == {tuple(round(v, 9) for v in corner) for corner in expected}


def test_corners_translate_rigidly_with_the_box():
    box = Box(width=0.2, height=0.2, depth=0.2, origin=(0.0, 0.0, 0.0))
    original_corners = sorted(box.corners)

    box.translate((1.0, 2.0, 3.0))

    shifted = sorted(box.corners)
    for original, moved in zip(original_corners, shifted):
        assert moved == pytest.approx(
            (original[0] + 1.0, original[1] + 2.0, original[2] + 3.0)
        )


# ---------------------------------------------------------------------------
# Model structure, independence, copy
# ---------------------------------------------------------------------------


def test_model_has_a_single_body_welded_to_ground():
    box = Box(width=0.1, height=0.1, depth=0.1)

    assert box._container.model.getBodySet().getSize() == 1
    assert box._container.model.getJointSet().getSize() == 1
    joint = box._container.model.getJointSet().get("box_joint")
    assert opensim.WeldJoint.safeDownCast(joint) is not None


def test_instances_are_backed_by_independent_opensim_models():
    small = Box(width=0.1, height=0.1, depth=0.1)
    large = Box(width=0.9, height=0.9, depth=0.9, mass_kg=9.0)

    assert small._container.model is not large._container.model
    assert small._container.model.getBodySet().get("box").get_mass() != pytest.approx(
        large._container.model.getBodySet().get("box").get_mass()
    )


def test_copy_preserves_type_and_parameters():
    box = Box(width=0.2, height=0.3, depth=0.4, origin=(1.0, 0.0, 0.0), mass_kg=2.0)

    duplicate = box.copy()

    assert type(duplicate) is Box
    assert duplicate.width == 0.2
    assert duplicate.mass_kg == 2.0
    assert duplicate.origin == pytest.approx((1.0, 0.0, 0.0))
    assert duplicate._container.model is not box._container.model


# ---------------------------------------------------------------------------
# Box is a component (Body), not a container (OpenSimModel)
# ---------------------------------------------------------------------------


def test_box_is_a_body_component_not_a_container():
    box = Box(width=0.1, height=0.1, depth=0.1)

    assert isinstance(box, components.Body)
    assert not isinstance(box, OpenSimModel)
    assert not hasattr(box, "show")


def test_box_is_addable_to_a_model_in_both_directions():
    box = Box(width=0.1, height=0.1, depth=0.1, origin=(2.0, 0.0, 0.0))
    model = OpenSimModel(model_path=None)

    merged = model + box
    assert isinstance(merged, OpenSimModel)
    assert "box" in merged.bodies
    assert "box" not in model.bodies

    merged_reflected = box + model
    assert "box" in merged_reflected.bodies


# ---------------------------------------------------------------------------
# Setters rebuild the box and its mesh
# ---------------------------------------------------------------------------


def test_setters_expose_getters_as_properties():
    box = Box(width=0.1, height=0.1, depth=0.1)

    box.set_width(0.5)
    box.set_height(0.6)
    box.set_depth(0.7)
    box.set_mass_kg(3.0)

    assert box.width == pytest.approx(0.5)
    assert box.height == pytest.approx(0.6)
    assert box.depth == pytest.approx(0.7)
    assert box.mass_kg == pytest.approx(3.0)


def test_dimension_setter_regenerates_the_mesh_on_disk():
    box = Box(width=0.2, height=0.2, depth=0.2)
    mesh_path = _MESHES_DIR / _MESH_FILENAME
    first_contents = mesh_path.read_text()

    box.set_width(0.9)

    assert mesh_path.read_text() != first_contents
    assert mesh_path.read_text().count("facet normal") == 12


def test_box_registers_its_own_mesh_directory():
    box = Box(width=0.1, height=0.1, depth=0.1)

    assert _MESHES_DIR in box._container.geometry_directories
