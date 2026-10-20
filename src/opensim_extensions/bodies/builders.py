"""Helpers that turn shapes into OpenSim bodies and attach them to a model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..opensim import import_opensim
from .shapes import (
    Box,
    Cylinder,
    Disk,
    Shape,
    Sphere,
    cylinder_between,
    matrix_to_euler_xyz,
)

__all__ = [
    "mass_properties",
    "make_body",
    "attach_shapes",
    "add_welded_body",
    "add_pinned_body",
    "box_body",
    "cylinder_body",
    "sphere_body",
    "disk_body",
    "tube_body",
]

opensim = import_opensim()

_ZERO = (0.0, 0.0, 0.0)


def mass_properties(
    shapes: Sequence[Shape], density: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Total mass (kg), centre of mass and inertia about it (body-frame axes)."""
    if not shapes:
        raise ValueError("at least one shape is required")
    if density <= 0:
        raise ValueError("density must be positive")
    masses = [density * shape.volume for shape in shapes]
    total = sum(masses)
    centers = [np.asarray(shape.center, float) for shape in shapes]
    com = sum(m * c for m, c in zip(masses, centers)) / total
    inertia = np.zeros((3, 3))
    for m, shape, center in zip(masses, shapes, centers):
        d = center - com
        inertia += shape.inertia_about_center(m) + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return total, com, inertia


def attach_shapes(body, shapes: Sequence[Shape], prefix: str | None = None) -> None:
    """Attach each shape to ``body`` as visible geometry on its own offset frame."""
    prefix = prefix or body.getName()
    for index, shape in enumerate(shapes):
        frame = opensim.PhysicalOffsetFrame()
        frame.setName(f"{prefix}_shape{index}")
        frame.setParentFrame(body)
        frame.set_translation(opensim.Vec3(*shape.center))
        frame.set_orientation(opensim.Vec3(*matrix_to_euler_xyz(shape.rotation)))
        geometry = shape._geometry()
        if shape.color is not None:
            geometry.setColor(opensim.Vec3(*shape.color))
        frame.attachGeometry(geometry)
        body.addComponent(frame)


def make_body(name: str, shapes: Sequence[Shape], density: float):
    """Create an ``opensim.Body`` whose mass properties come from ``shapes`` and ``density``.

    Returns ``(body, mass_kg)``. The body is not yet added to a model.
    """
    mass, com, i = mass_properties(shapes, density)
    inertia = opensim.Inertia(i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])
    body = opensim.Body(name, mass, opensim.Vec3(*com), inertia)
    attach_shapes(body, shapes)
    return body, mass


def add_welded_body(
    model,
    name: str,
    shapes: Sequence[Shape],
    density: float,
    parent=None,
    location: Sequence[float] = _ZERO,
    orientation: Sequence[float] = _ZERO,
) -> float:
    """Add a body welded to ``parent`` (default ground) and return its mass (kg).

    ``location``/``orientation`` (m, XYZ Euler rad) place the body frame in the parent frame.
    """
    body, mass = make_body(name, shapes, density)
    model.addBody(body)
    model.addJoint(
        opensim.WeldJoint(
            f"{name}_weld",
            parent or model.getGround(),
            opensim.Vec3(*location),
            opensim.Vec3(*orientation),
            body,
            opensim.Vec3(*_ZERO),
            opensim.Vec3(*_ZERO),
        )
    )
    return mass


def add_pinned_body(
    model,
    name: str,
    shapes: Sequence[Shape],
    density: float,
    coordinate_name: str,
    parent=None,
    location: Sequence[float] = _ZERO,
    orientation: Sequence[float] = _ZERO,
):
    """Add a body hinged to ``parent`` about the z axis of the joint frames.

    Returns ``(body, mass_kg)``; the joint coordinate is named ``coordinate_name``.
    """
    body, mass = make_body(name, shapes, density)
    model.addBody(body)
    joint = opensim.PinJoint(
        f"{name}_joint",
        parent or model.getGround(),
        opensim.Vec3(*location),
        opensim.Vec3(*orientation),
        body,
        opensim.Vec3(*_ZERO),
        opensim.Vec3(*_ZERO),
    )
    joint.updCoordinate().setName(coordinate_name)
    model.addJoint(joint)
    return body, mass


def box_body(name: str, size: Sequence[float], density: float):
    """Body made of a single box of full ``size`` (m)."""
    return make_body(name, [Box(size=size)], density)


def cylinder_body(name: str, radius: float, length: float, density: float, axis: str = "y"):
    """Body made of a single solid cylinder along ``axis``."""
    return make_body(name, [Cylinder(radius=radius, length=length, axis=axis)], density)


def sphere_body(name: str, radius: float, density: float):
    """Body made of a single solid sphere."""
    return make_body(name, [Sphere(radius=radius)], density)


def disk_body(name: str, radius: float, thickness: float, density: float, axis: str = "z"):
    """Body made of a single thin disk whose normal is ``axis``."""
    return make_body(name, [Disk(radius=radius, thickness=thickness, axis=axis)], density)


def tube_body(
    name: str, start: Sequence[float], end: Sequence[float], radius: float, density: float
):
    """Body made of a cylinder joining two points (body-frame coordinates, m)."""
    return make_body(name, [cylinder_between(start, end, radius)], density)
