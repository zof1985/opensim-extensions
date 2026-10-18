"""Focused tests for ``OpenSimModel.from_osim`` and ``to_osim``."""

import shutil
from pathlib import Path

import opensim
import pytest

from opensim_models import OpenSimModel


def _write_model(path: Path, mesh_name: str | None = None) -> None:
    model = opensim.Model()
    body = opensim.Body("test_body", 1.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1))
    if mesh_name is not None:
        body.attachGeometry(opensim.Mesh(mesh_name))
    model.addBody(body)
    model.addJoint(opensim.WeldJoint("test_joint", model.getGround(), body))
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(str(path))


def test_from_osim_loads_model_and_records_default_geometry_paths(tmp_path):
    source = tmp_path / "source.osim"
    _write_model(source)

    model = OpenSimModel.from_osim(source)

    assert isinstance(model, OpenSimModel)
    assert model.getBodySet().get("test_body").getName() == "test_body"
    assert tmp_path.resolve() in model.geometry_directories
    assert (tmp_path / "Geometry").resolve() in model.geometry_directories


def test_from_osim_registers_explicit_mesh_directory_before_loading(tmp_path):
    mesh_dir = tmp_path / "custom_meshes"
    mesh_dir.mkdir()
    source_mesh = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "opensim_models"
        / "components"
        / "assets"
        / "meshes"
        / "box.stl"
    )
    shutil.copy2(source_mesh, mesh_dir / "shape.stl")
    source = tmp_path / "mesh_model.osim"
    _write_model(source, "shape.stl")

    model = OpenSimModel.from_osim(source, meshes=mesh_dir)

    assert mesh_dir.resolve() in model.geometry_directories
    attached = model.getBodySet().get("test_body").get_attached_geometry(0)
    assert opensim.Mesh.safeDownCast(attached) is not None


def test_to_osim_saves_model_and_referenced_meshes(tmp_path):
    mesh_dir = tmp_path / "input_meshes"
    mesh_dir.mkdir()
    source_mesh = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "opensim_models"
        / "components"
        / "assets"
        / "meshes"
        / "box.stl"
    )
    shutil.copy2(source_mesh, mesh_dir / "shape.stl")
    source = tmp_path / "input.osim"
    _write_model(source, "shape.stl")
    model = OpenSimModel.from_osim(source, meshes=mesh_dir)

    destination = tmp_path / "output" / "combined.osim"
    destination.parent.mkdir()
    written = model.to_osim(destination)

    assert written == destination
    assert destination.is_file()
    assert (destination.parent / "Geometry" / "shape.stl").is_file()
    reloaded = OpenSimModel.from_osim(destination)
    assert reloaded.getBodySet().get("test_body").get_attached_geometry(0) is not None


def test_from_osim_reports_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        OpenSimModel.from_osim(tmp_path / "missing.osim")

    source = tmp_path / "source.osim"
    _write_model(source)
    with pytest.raises(FileNotFoundError):
        OpenSimModel.from_osim(source, meshes=tmp_path / "missing-meshes")

    with pytest.raises(TypeError, match="str or pathlib.Path"):
        OpenSimModel.from_osim(source, meshes=[object()])


def test_to_osim_can_skip_mesh_copy(tmp_path):
    source = tmp_path / "source.osim"
    _write_model(source)
    model = OpenSimModel.from_osim(source)
    destination = tmp_path / "without_meshes.osim"

    assert model.to_osim(destination, with_meshes=False) == destination
    assert destination.is_file()
