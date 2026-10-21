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
    """Calculate combined mass properties for a set of shapes.

    The component masses are density times analytic shape volume. Their
    centres are combined by mass-weighted averaging, and the parallel-axis
    theorem translates each shape's centre inertia to the combined centre.

    Parameters
    ----------
    shapes : sequence of Shape
        Non-empty sequence of primitives expressed in the body's local frame.
    density : float
        Uniform material density in kilograms per cubic metre.

    Returns
    -------
    mass_kg : float
        Sum of the primitive masses.
    center_of_mass : numpy.ndarray, shape (3,)
        Combined centre of mass in metres in the body frame.
    inertia : numpy.ndarray, shape (3, 3)
        Combined inertia tensor in kg m², about the combined centre and
        expressed in body-frame axes.

    Raises
    ------
    ValueError
        If ``shapes`` is empty or ``density`` is not positive.
    """
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
    """Attach shapes as visible geometry on separate body-fixed offset frames.

    Parameters
    ----------
    body : opensim.Body
        Body receiving the geometry frames.
    shapes : sequence of Shape
        Geometric primitives whose centres and orientations are expressed in
        the body frame.
    prefix : str or None, optional
        Prefix used for generated frame names. Defaults to the body's name.

    Returns
    -------
    None
    """
    prefix = prefix or body.getName()
    previous_level = opensim.Logger.getLevelString()
    # OpenSim logs a harmless "Mesh ... not connected to a model" error while a
    # mesh is attached to a body that is not yet part of a model; it loads later.
    opensim.Logger.setLevelString("Off")
    try:
        _attach_shapes(body, shapes, prefix)
    finally:
        opensim.Logger.setLevelString(previous_level)


def _attach_shapes(body, shapes: Sequence[Shape], prefix: str) -> None:
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
    """Create an OpenSim body with geometry and mass properties from shapes.

    The combined centre of mass and inertia tensor are computed by
    :func:`mass_properties`; the body is returned unattached to any model.

    Parameters
    ----------
    name : str
        Name assigned to the new OpenSim body.
    shapes : sequence of Shape
        Non-empty primitives attached to the body as visible geometry.
    density : float
        Uniform material density in kilograms per cubic metre.

    Returns
    -------
    body : opensim.Body
        Constructed body with calculated mass, centre, inertia, and geometry.
    mass_kg : float
        Total mass represented by ``shapes``.
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
    """Add a body rigidly welded to a parent frame.

    Parameters
    ----------
    model : opensim.Model
        Model that receives the body and weld joint.
    name : str
        Name assigned to the body and its weld joint.
    shapes : sequence of Shape
        Non-empty primitives used for body mass properties and display.
    density : float
        Uniform material density in kilograms per cubic metre.
    parent : opensim.PhysicalFrame or None, optional
        Parent frame for the weld. Defaults to the model's ground frame.
    location : sequence of float, default=(0, 0, 0)
        Body-frame origin in the parent frame, in metres.
    orientation : sequence of float, default=(0, 0, 0)
        Body orientation in the parent frame as body-fixed XYZ Euler angles,
        in radians.

    Returns
    -------
    float
        Mass of the created body in kilograms.
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
    """Add a body connected to its parent by a pin joint about the z axis.

    Parameters
    ----------
    model : opensim.Model
        Model that receives the body and pin joint.
    name : str
        Name assigned to the body; the joint is named ``"{name}_joint"``.
    shapes : sequence of Shape
        Non-empty primitives used for body mass properties and display.
    density : float
        Uniform material density in kilograms per cubic metre.
    coordinate_name : str
        Name assigned to the pin-joint rotation coordinate.
    parent : opensim.PhysicalFrame or None, optional
        Parent frame for the joint. Defaults to the model's ground frame.
    location : sequence of float, default=(0, 0, 0)
        Joint origin in the parent frame, in metres.
    orientation : sequence of float, default=(0, 0, 0)
        Joint-frame orientation in the parent frame as body-fixed XYZ Euler
        angles, in radians.

    Returns
    -------
    body : opensim.Body
        Newly created body attached to ``model``.
    mass_kg : float
        Mass of the body in kilograms.
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
    """Create a body from one cuboid.

    Parameters
    ----------
    name : str
        Name assigned to the body.
    size : sequence of float
        Full cuboid dimensions ``(x, y, z)`` in metres.
    density : float
        Material density in kilograms per cubic metre.

    Returns
    -------
    tuple of opensim.Body and float
        The unattached body and its mass in kilograms.
    """
    return make_body(name, [Box(size=size)], density)


def cylinder_body(name: str, radius: float, length: float, density: float, axis: str = "y"):
    """Create a body from one solid cylinder.

    Parameters
    ----------
    name : str
        Name assigned to the body.
    radius : float
        Cylinder radius in metres.
    length : float
        Cylinder length in metres.
    density : float
        Material density in kilograms per cubic metre.
    axis : {'x', 'y', 'z'}, default='y'
        Body-frame axis along which the cylinder is oriented.

    Returns
    -------
    tuple of opensim.Body and float
        The unattached body and its mass in kilograms.
    """
    return make_body(name, [Cylinder(radius=radius, length=length, axis=axis)], density)


def sphere_body(name: str, radius: float, density: float):
    """Create a body from one solid sphere.

    Parameters
    ----------
    name : str
        Name assigned to the body.
    radius : float
        Sphere radius in metres.
    density : float
        Material density in kilograms per cubic metre.

    Returns
    -------
    tuple of opensim.Body and float
        The unattached body and its mass in kilograms.
    """
    return make_body(name, [Sphere(radius=radius)], density)


def disk_body(name: str, radius: float, thickness: float, density: float, axis: str = "z"):
    """Create a body from one thin cylindrical disk.

    Parameters
    ----------
    name : str
        Name assigned to the body.
    radius : float
        Disk radius in metres.
    thickness : float
        Disk thickness in metres.
    density : float
        Material density in kilograms per cubic metre.
    axis : {'x', 'y', 'z'}, default='z'
        Disk normal in the body frame.

    Returns
    -------
    tuple of opensim.Body and float
        The unattached body and its mass in kilograms.
    """
    return make_body(name, [Disk(radius=radius, thickness=thickness, axis=axis)], density)


def tube_body(
    name: str, start: Sequence[float], end: Sequence[float], radius: float, density: float
):
    """Create a body from a tube whose axis joins two body-frame points.

    Parameters
    ----------
    name : str
        Name assigned to the body.
    start, end : sequence of float
        Distinct ``(x, y, z)`` cylinder end points in metres in the body frame.
    radius : float
        Outer tube radius in metres. The default inner radius is zero, so this
        convenience function creates a solid cylinder.
    density : float
        Material density in kilograms per cubic metre.

    Returns
    -------
    tuple of opensim.Body and float
        The unattached body and its mass in kilograms.
    """
    return make_body(name, [cylinder_between(start, end, radius)], density)
