import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import OpenSimModel

pytest.importorskip("opensim")
OCC = pytest.importorskip("OCC")


def _make_box_step(directory: Path, name: str, dx: float, dy: float, dz: float) -> Path:
    """Write a single-solid STEP file, in millimetres, for a dx x dy x dz metre box.

    Declaring the file in millimetres (the typical unit for mechanical CAD
    exports) exercises OpenSimModel.from_step's unit conversion to metres.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    # Interface_Static only registers STEP parameters once a reader/writer
    # exists, so SetCVal must run after STEPControl_Writer() is constructed.
    Interface_Static.SetCVal("write.step.unit", "MM")
    box = BRepPrimAPI_MakeBox(dx * 1000.0, dy * 1000.0, dz * 1000.0).Shape()

    writer.Transfer(box, STEPControl_AsIs)
    step_path = directory / f"{name}.step"
    writer.Write(str(step_path))
    return step_path


def _make_two_box_step(
    directory: Path, dx: float, dy: float, dz: float, x_offset: float
) -> Path:
    """Write a two-solid STEP file: two identical dx x dy x dz boxes, the
    second translated by ``x_offset`` (metres) along X, both in millimetres.
    """
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Trsf, gp_Vec
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.unit", "MM")
    box1 = BRepPrimAPI_MakeBox(dx * 1000.0, dy * 1000.0, dz * 1000.0).Shape()
    box2 = BRepPrimAPI_MakeBox(dx * 1000.0, dy * 1000.0, dz * 1000.0).Shape()
    translation = gp_Trsf()
    translation.SetTranslation(gp_Vec(x_offset * 1000.0, 0.0, 0.0))
    box2 = BRepBuilderAPI_Transform(box2, translation, True).Shape()

    writer.Transfer(box1, STEPControl_AsIs)
    writer.Transfer(box2, STEPControl_AsIs)
    step_path = directory / "two_boxes.step"
    writer.Write(str(step_path))
    return step_path


def test_from_step_builds_one_body_with_analytic_mass_and_inertia():
    dx, dy, dz = 0.2, 0.3, 0.4
    density = 1200.0
    volume = dx * dy * dz
    expected_mass = volume * density
    # Analytic central inertia of a solid cuboid: I_xx = m/12 * (dy^2 + dz^2), etc.
    expected_ixx = expected_mass / 12.0 * (dy**2 + dz**2)
    expected_iyy = expected_mass / 12.0 * (dx**2 + dz**2)
    expected_izz = expected_mass / 12.0 * (dx**2 + dy**2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        step_path = _make_box_step(tmp_path, "box", dx, dy, dz)

        model = OpenSimModel.from_step(step_path, density=density)

        assert len(model.bodies) == 1
        body = model.model.getBodySet().get(0)
        assert body.getMass() == pytest.approx(expected_mass, rel=1e-3)

        inertia = body.getInertia().getMoments()
        assert inertia.get(0) == pytest.approx(expected_ixx, rel=1e-2)
        assert inertia.get(1) == pytest.approx(expected_iyy, rel=1e-2)
        assert inertia.get(2) == pytest.approx(expected_izz, rel=1e-2)

        assert len(model.joints) == 1
        mesh_files = list(model.geometry_directories[0].glob("*.stl"))
        assert len(mesh_files) == 1


def test_from_step_without_free_joints_adds_unconnected_bodies():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        step_path = _make_box_step(tmp_path, "box", 0.1, 0.1, 0.1)

        model = OpenSimModel.from_step(step_path, add_free_joints=False)

        assert len(model.bodies) == 1
        assert len(model.joints) == 0


def test_from_step_as_one_object_defaults_to_a_single_combined_body():
    dx, dy, dz, offset, density = 0.1, 0.1, 0.1, 1.0, 1000.0
    mass_each = dx * dy * dz * density

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        step_path = _make_two_box_step(tmp_path, dx, dy, dz, offset)

        model = OpenSimModel.from_step(step_path, density=density)

        assert len(model.bodies) == 1
        assert len(model.joints) == 1
        body = model.model.getBodySet().get(0)
        assert body.getMass() == pytest.approx(2 * mass_each, rel=1e-6)

        # Combined mass centre: both boxes' own centres, weighted by mass.
        com1 = dx / 2.0
        com2 = offset + dx / 2.0
        combined_x = (mass_each * com1 + mass_each * com2) / (2 * mass_each)
        joint = model.model.getJointSet().get(0)
        parent = model.opensim.PhysicalOffsetFrame.safeDownCast(joint.getParentFrame())
        assert parent.get_translation()[0] == pytest.approx(combined_x, rel=1e-6)

        # Two solids attached to the same combined body -> two mesh files.
        mesh_files = list(model.geometry_directories[0].glob("*.stl"))
        assert len(mesh_files) == 2


def test_from_step_as_one_object_false_keeps_one_body_per_solid():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        step_path = _make_two_box_step(tmp_path, 0.1, 0.1, 0.1, 1.0)

        model = OpenSimModel.from_step(step_path, as_one_object=False)

        assert len(model.bodies) == 2
        assert len(model.joints) == 2


def test_from_step_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        OpenSimModel.from_step("does/not/exist.step")


def test_from_step_rejects_step_file_without_solids():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        step_path = tmp_path / "empty.step"
        step_path.write_text(
            "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
        )

        with pytest.raises(ValueError):
            OpenSimModel.from_step(step_path)
