import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import Cylinder, components
from opensim_models.model import OpenSimModel
from opensim_models.components.cylinder import _MESHES_DIR

opensim = pytest.importorskip("opensim")


def expected_inertia(mass, radius, height):
    transverse = mass * (3.0 * radius**2 + height**2) / 12.0
    axial = mass * radius**2 / 2.0
    return transverse, axial, transverse


def joint_offset_frames(cylinder):
    joint = cylinder._container.model.getJointSet().get("cylinder_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.getParentFrame())
    return parent


def test_constructor_properties_mass_and_inertia():
    cylinder = Cylinder(
        radius=0.2, height=0.8, mass_kg=4.0,
        center_x=1.0, center_y=2.0, center_z=3.0,
        angle_deg=(10.0, 20.0, 30.0),
    )

    assert cylinder.radius == pytest.approx(0.2)
    assert cylinder.height == pytest.approx(0.8)
    assert cylinder.mass_kg == pytest.approx(4.0)
    assert cylinder.origin == pytest.approx((1.0, 2.0, 3.0))
    assert cylinder.angle_deg == pytest.approx((10.0, 20.0, 30.0))

    body = cylinder._container.model.getBodySet().get("cylinder")
    assert body.get_mass() == pytest.approx(4.0)
    moments = body.getInertia().getMoments()
    assert tuple(moments.get(i) for i in range(3)) == pytest.approx(
        expected_inertia(4.0, 0.2, 0.8)
    )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("radius", 0.0),
        ("radius", -1.0),
        ("height", 0.0),
        ("height", -1.0),
        ("mass_kg", 0.0),
        ("mass_kg", -1.0),
    ],
)
def test_constructor_rejects_non_positive_dimensions_and_mass(parameter, value):
    kwargs = {"radius": 0.1, "height": 0.2}
    kwargs[parameter] = value
    with pytest.raises(ValueError, match="strictly positive"):
        Cylinder(**kwargs)


def test_specific_setters_update_properties_and_rebuild_inertia():
    cylinder = Cylinder(radius=0.1, height=0.3)

    cylinder.set_radius(0.2)
    cylinder.set_height(0.4)
    cylinder.set_mass_kg(3.0)

    assert cylinder.radius == pytest.approx(0.2)
    assert cylinder.height == pytest.approx(0.4)
    assert cylinder.mass_kg == pytest.approx(3.0)
    moments = cylinder._container.model.getBodySet().get("cylinder").getInertia().getMoments()
    assert tuple(moments.get(i) for i in range(3)) == pytest.approx(
        expected_inertia(3.0, 0.2, 0.4)
    )


def test_pose_setters_and_dimension_changes_preserve_other_pose_values():
    cylinder = Cylinder(
        0.1, 0.3, center_x=1.0, center_y=2.0, center_z=3.0,
        angle_deg=(10.0, 20.0, 30.0),
    )

    cylinder.set_center_x(5.0)
    assert cylinder.origin == pytest.approx((5.0, 2.0, 3.0))
    assert cylinder.angle_deg == pytest.approx((10.0, 20.0, 30.0))
    cylinder.set_center_y(6.0)
    cylinder.set_center_z(7.0)
    cylinder.set_angle_deg((0.0, 0.0, 45.0))
    cylinder.set_radius(0.25)

    assert cylinder.origin == pytest.approx((5.0, 6.0, 7.0))
    assert cylinder.angle_deg == pytest.approx((0.0, 0.0, 45.0))


def test_origin_setter_and_joint_offsets():
    cylinder = Cylinder(0.1, 0.4, angle_deg=(0.0, 0.0, 45.0))

    cylinder.set_origin((1.0, -2.0, 3.0))

    assert cylinder.origin == pytest.approx((1.0, -2.0, 3.0))
    assert cylinder.angle_deg == pytest.approx((0.0, 0.0, 45.0))
    parent = joint_offset_frames(cylinder)
    assert tuple(parent.get_translation().get(i) for i in range(3)) == pytest.approx(
        (1.0, -2.0, 3.0)
    )
    assert parent.get_orientation()[2] == pytest.approx(math.radians(45.0))


def test_rotate_translate_and_setter_keep_pose_live():
    cylinder = Cylinder(0.1, 0.3)
    cylinder.rotate(cylinder.com, (0.0, 0.0, 1.0), 90.0)
    cylinder.translate((1.0, 2.0, 3.0))
    pose = cylinder.origin
    angle = cylinder.angle_deg

    cylinder.set_height(0.5)

    assert cylinder.origin == pytest.approx(pose)
    assert cylinder.angle_deg == pytest.approx(angle)


def test_mesh_is_created_and_regenerated():
    cylinder = Cylinder(0.1, 0.2)
    mesh_path = _MESHES_DIR / "cylinder.stl"
    first_contents = mesh_path.read_text()
    assert first_contents.count("facet normal") == 128

    cylinder.set_radius(0.3)

    assert mesh_path.read_text() != first_contents
    assert mesh_path.read_text().count("facet normal") == 128


def test_custom_name_mesh_dir_copy_and_model_composition():
    with tempfile.TemporaryDirectory(prefix="cylinder-test-", dir=_MESHES_DIR) as path:
        mesh_dir = Path(path)
        cylinder = Cylinder(0.1, 0.2, mass_kg=2.0, name="roller", mesh_dir=mesh_dir)
        cylinder.set_radius(0.15)
        duplicate = cylinder.copy()

        assert isinstance(cylinder, components.Body)
        assert not isinstance(cylinder, OpenSimModel)
        assert not hasattr(cylinder, "show")
        assert type(duplicate) is Cylinder
        assert duplicate.name == "roller"
        assert duplicate.radius == pytest.approx(0.15)
        assert duplicate.mass_kg == pytest.approx(2.0)
        assert duplicate._container.model is not cylinder._container.model
        assert (mesh_dir / "roller.stl").is_file()

        model = OpenSimModel(model_path=None)
        combined = model + cylinder
        assert "roller" in combined.bodies
        assert "roller" not in model.bodies
