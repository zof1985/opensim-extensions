"""Turning an OpenSim ``DecorativeGeometry`` into a VTK source, shared.

Both :class:`~opensim_models._gui.visualizer.VTKVisualizer` (building a
renderable actor) and :class:`~opensim_models.components.Body` (computing
local-frame bounds for :attr:`~opensim_models.components.Body.corners`)
need the exact same ``Mesh``/``Brick``/``Cylinder``/``Sphere`` dispatch;
this module is that one shared implementation instead of two copies that
could drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

__all__ = ["build_source", "local_bounds"]

READER_BY_SUFFIX = {
    ".vtp": "vtkXMLPolyDataReader",
    ".stl": "vtkSTLReader",
    ".obj": "vtkOBJReader",
}


def build_source(
    vtk: Any, opensim: Any, geometry: Any, resolve_file: Callable[[str], Path | None]
) -> Any | None:
    """Return a VTK source/reader for one attached ``DecorativeGeometry``.

    ``None`` for a mesh whose file can't be resolved (via ``resolve_file``,
    e.g. :meth:`~opensim_models.model.OpenSimModel._resolve_geometry_file`),
    or a geometry type not handled (only ``Mesh``/``Brick``/``Cylinder``/
    ``Sphere`` -- what :mod:`opensim_models.operators`'s
    ``add_box_body``/``add_cylinder_body``/``add_sphere_body`` and every
    bundled model's own mesh attach -- are).

    Applies ``geometry``'s own ``scale_factors`` (every ``opensim.Geometry``
    has one, default ``(1, 1, 1)``) on top of the source -- this is what
    makes a scaled mesh (e.g. every :class:`~opensim_models.models.User`
    body not at the base model's own size) render at its actual scaled
    size instead of the generic unscaled one baked into the mesh file,
    which would otherwise visibly detach a long bone's rendered geometry
    from its (correctly scaled) neighboring joints.
    """
    mesh = opensim.Mesh.safeDownCast(geometry)
    if mesh is not None:
        source = _reader_for_file(vtk, mesh.get_mesh_file(), resolve_file)
    else:
        brick = opensim.Brick.safeDownCast(geometry)
        if brick is not None:
            half_lengths = brick.get_half_lengths()
            source = vtk.vtkCubeSource()
            source.SetXLength(half_lengths.get(0) * 2.0)
            source.SetYLength(half_lengths.get(1) * 2.0)
            source.SetZLength(half_lengths.get(2) * 2.0)
        else:
            cylinder = opensim.Cylinder.safeDownCast(geometry)
            if cylinder is not None:
                source = vtk.vtkCylinderSource()
                source.SetRadius(cylinder.get_radius())
                source.SetHeight(cylinder.get_half_height() * 2.0)
                source.SetResolution(32)
            else:
                sphere = opensim.Sphere.safeDownCast(geometry)
                if sphere is not None:
                    source = vtk.vtkSphereSource()
                    source.SetRadius(sphere.get_radius())
                    source.SetThetaResolution(32)
                    source.SetPhiResolution(32)
                else:
                    source = None

    if source is None:
        return None
    return _apply_scale(vtk, source, geometry)


def _apply_scale(vtk: Any, source: Any, geometry: Any) -> Any:
    scale_factors = geometry.get_scale_factors()
    sx, sy, sz = scale_factors.get(0), scale_factors.get(1), scale_factors.get(2)
    if (sx, sy, sz) == (1.0, 1.0, 1.0):
        return source
    transform = vtk.vtkTransform()
    transform.Scale(sx, sy, sz)
    scaler = vtk.vtkTransformPolyDataFilter()
    scaler.SetTransform(transform)
    scaler.SetInputConnection(source.GetOutputPort())
    return scaler


def _reader_for_file(
    vtk: Any, filename: str, resolve_file: Callable[[str], Path | None]
) -> Any | None:
    path = resolve_file(filename)
    if path is None:
        return None
    reader_class_name = READER_BY_SUFFIX.get(Path(filename).suffix.lower())
    if reader_class_name is None:
        return None
    reader = getattr(vtk, reader_class_name)()
    reader.SetFileName(str(path))
    reader.Update()
    return reader


def local_bounds(
    vtk: Any, opensim: Any, geometry: Any, resolve_file: Callable[[str], Path | None]
) -> tuple[float, float, float, float, float, float] | None:
    """Return ``geometry``'s local-frame axis-aligned bounding box.

    ``(xmin, xmax, ymin, ymax, zmin, zmax)``, in metres, in the geometry's
    own (attached body's) frame -- or ``None`` if :func:`build_source`
    can't resolve a source for it. Works uniformly across every geometry
    type :func:`build_source` handles, procedural or file-based: all of
    them produce an actual ``vtkPolyDataAlgorithm``, whose own
    ``GetOutput().GetBounds()`` is the real, exact bounding box either way.
    """
    source = build_source(vtk, opensim, geometry, resolve_file)
    if source is None:
        return None
    source.Update()
    return source.GetOutput().GetBounds()
