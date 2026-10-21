"""Geometric primitives with analytic volume and inertia, in SI units (m, kg)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..opensim import import_opensim, register_geometry_search_path

__all__ = [
    "MESHES_DIR",
    "Shape",
    "Box",
    "Cylinder",
    "Tube",
    "Disk",
    "Sphere",
    "Ellipsoid",
    "cylinder_between",
    "euler_xyz_to_matrix",
    "matrix_to_euler_xyz",
]

opensim = import_opensim()

MESHES_DIR = Path(__file__).resolve().parent / "assets" / "meshes"
register_geometry_search_path(opensim, MESHES_DIR)


def _mesh(file_name: str, scale: Sequence[float]):
    """Bundled unit mesh (unit size, centred) scaled to the requested size."""
    mesh = opensim.Mesh(file_name)
    mesh.set_scale_factors(opensim.Vec3(*(float(v) for v in scale)))
    return mesh

_AXIS_EULER = {
    "x": (0.0, 0.0, -math.pi / 2),
    "y": (0.0, 0.0, 0.0),
    "z": (math.pi / 2, 0.0, 0.0),
}


def euler_xyz_to_matrix(angles: Sequence[float]) -> np.ndarray:
    """Convert body-fixed XYZ Euler angles to a rotation matrix.

    Parameters
    ----------
    angles : sequence of float
        Three rotations ``(x, y, z)`` in radians, using the body-fixed
        convention expected by OpenSim.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix mapping vectors from the shape's natural frame to its
        parent frame.
    """
    a, b, c = angles
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz


def matrix_to_euler_xyz(matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a rotation matrix to body-fixed XYZ Euler angles.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (3, 3)
        Rotation matrix expressed using the convention of
        :func:`euler_xyz_to_matrix`.

    Returns
    -------
    tuple of float
        The ``(x, y, z)`` body-fixed Euler angles in radians, with the middle
        angle in ``[-pi/2, pi/2]``.
    """
    b = math.asin(max(-1.0, min(1.0, matrix[0, 2])))
    if abs(matrix[0, 2]) < 1 - 1e-12:
        a = math.atan2(-matrix[1, 2], matrix[2, 2])
        c = math.atan2(-matrix[0, 1], matrix[0, 0])
    else:
        a = math.atan2(matrix[2, 1], matrix[1, 1])
        c = 0.0
    return a, b, c


@dataclass
class Shape:
    """Base geometric primitive expressed in a body's local frame.

    Parameters
    ----------
    center : sequence of float, default=(0, 0, 0)
        Primitive centre ``(x, y, z)`` in metres in the body frame.
    orientation : sequence of float, default=(0, 0, 0)
        Body-fixed XYZ Euler rotation from the primitive's natural axes to the
        body frame, in radians.
    color : sequence of float or None, optional
        Optional RGB display color with components in ``[0, 1]``.
    """

    center: Sequence[float] = (0.0, 0.0, 0.0)
    orientation: Sequence[float] = (0.0, 0.0, 0.0)
    color: Sequence[float] | None = field(default=None, kw_only=True)

    @property
    def volume(self) -> float:
        """Return the analytic primitive volume in cubic metres.

        Returns
        -------
        float
            Geometric volume. Concrete subclasses provide the formula.
        """
        raise NotImplementedError

    def _local_inertia(self, mass: float) -> np.ndarray:
        """Inertia tensor about the shape centre in the shape's own axes."""
        raise NotImplementedError

    def _axis_rotation(self) -> np.ndarray:
        return np.eye(3)

    @property
    def rotation(self) -> np.ndarray:
        """Return the shape-to-body rotation matrix.

        Returns
        -------
        numpy.ndarray, shape (3, 3)
            Matrix mapping vectors from the primitive's natural axes to the
            body frame.
        """
        return euler_xyz_to_matrix(self.orientation) @ self._axis_rotation()

    def inertia_about_center(self, mass: float) -> np.ndarray:
        """Rotate the primitive's local inertia tensor into the body frame.

        Parameters
        ----------
        mass : float
            Primitive mass in kilograms.

        Returns
        -------
        numpy.ndarray, shape (3, 3)
            Inertia tensor in kg m² about the primitive centre, expressed in
            body-frame axes.
        """
        rotation = self.rotation
        return rotation @ self._local_inertia(mass) @ rotation.T

    def _geometry(self):
        raise NotImplementedError


def _validate(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value")


@dataclass
class Box(Shape):
    """Cuboid primitive.

    Parameters
    ----------
    size : sequence of float, default=(1, 1, 1)
        Full side lengths ``(x, y, z)`` in metres.
    center, orientation, color
        See :class:`Shape`.
    """

    size: Sequence[float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        _validate(**{f"size[{i}]": v for i, v in enumerate(self.size)})

    @property
    def volume(self) -> float:
        """Return the cuboid volume in cubic metres.

        Returns
        -------
        float
            Product of the three full side lengths.
        """
        return float(np.prod(self.size))

    def _local_inertia(self, mass: float) -> np.ndarray:
        a, b, c = self.size
        return np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2]) * mass / 12

    def _geometry(self):
        return _mesh("unit_box.vtp", self.size)


@dataclass
class Cylinder(Shape):
    """Solid cylinder primitive.

    Parameters
    ----------
    radius : float, default=0.5
        Cylinder radius in metres.
    length : float, default=1.0
        Full cylinder length in metres.
    axis : {'x', 'y', 'z'}, default='y'
        Cylinder axis in the body frame.
    center, orientation, color
        See :class:`Shape`.
    """

    radius: float = 0.5
    length: float = 1.0
    axis: str = "y"

    def __post_init__(self):
        _validate(radius=self.radius, length=self.length)
        if self.axis not in _AXIS_EULER:
            raise ValueError("axis must be 'x', 'y' or 'z'")

    @property
    def volume(self) -> float:
        """Return the solid cylinder volume in cubic metres.

        Returns
        -------
        float
            ``pi * radius**2 * length``.
        """
        return math.pi * self.radius**2 * self.length

    def _axis_rotation(self) -> np.ndarray:
        return euler_xyz_to_matrix(_AXIS_EULER[self.axis])

    def _local_inertia(self, mass: float) -> np.ndarray:
        radial = mass * (3 * self.radius**2 + self.length**2) / 12
        return np.diag([radial, 0.5 * mass * self.radius**2, radial])

    def _geometry(self):
        return _mesh("unit_cylinder.vtp", (2 * self.radius, self.length, 2 * self.radius))


@dataclass
class Disk(Cylinder):
    """Thin cylinder whose axis is the disk normal.

    Parameters
    ----------
    radius : float, default=0.5
        Disk radius in metres.
    thickness : float, default=0.005
        Disk thickness in metres.
    axis : {'x', 'y', 'z'}, default='z'
        Disk normal in the body frame.
    center, orientation, color
        See :class:`Shape`.
    """

    thickness: float = 0.005
    length: float = field(init=False, default=0.005)

    def __post_init__(self):
        self.length = self.thickness
        super().__post_init__()


@dataclass
class Tube(Cylinder):
    """Hollow cylindrical primitive.

    Parameters
    ----------
    radius : float, default=0.5
        Outer radius in metres.
    inner_radius : float, default=0.0
        Inner radius in metres; must be smaller than ``radius``.
    length : float, default=1.0
        Full tube length in metres.
    axis : {'x', 'y', 'z'}, default='y'
        Tube axis in the body frame.
    center, orientation, color
        See :class:`Shape`.
    """

    inner_radius: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.inner_radius < self.radius:
            raise ValueError("inner_radius must be in [0, radius)")

    @property
    def volume(self) -> float:
        """Return the hollow-cylinder volume in cubic metres.

        Returns
        -------
        float
            ``pi * (radius**2 - inner_radius**2) * length``.
        """
        return math.pi * (self.radius**2 - self.inner_radius**2) * self.length

    def _local_inertia(self, mass: float) -> np.ndarray:
        r2 = self.radius**2 + self.inner_radius**2
        radial = mass * (3 * r2 + self.length**2) / 12
        return np.diag([radial, 0.5 * mass * r2, radial])


@dataclass
class Sphere(Shape):
    """Solid sphere primitive.

    Parameters
    ----------
    radius : float, default=0.5
        Sphere radius in metres.
    center, orientation, color
        See :class:`Shape`.
    """

    radius: float = 0.5

    def __post_init__(self):
        _validate(radius=self.radius)

    @property
    def volume(self) -> float:
        """Return the sphere volume in cubic metres.

        Returns
        -------
        float
            ``4 * pi * radius**3 / 3``.
        """
        return 4.0 / 3.0 * math.pi * self.radius**3

    def _local_inertia(self, mass: float) -> np.ndarray:
        return np.eye(3) * 0.4 * mass * self.radius**2

    def _geometry(self):
        return _mesh("unit_sphere.vtp", (2 * self.radius,) * 3)


@dataclass
class Ellipsoid(Shape):
    """Ellipsoid primitive.

    Parameters
    ----------
    radii : sequence of float, default=(0.5, 0.5, 0.5)
        Semi-axis lengths ``(x, y, z)`` in metres.
    center, orientation, color
        See :class:`Shape`.
    """

    radii: Sequence[float] = (0.5, 0.5, 0.5)

    def __post_init__(self):
        _validate(**{f"radii[{i}]": value for i, value in enumerate(self.radii)})

    @property
    def volume(self) -> float:
        """Return the ellipsoid volume in cubic metres.

        Returns
        -------
        float
            ``4 * pi * rx * ry * rz / 3``.
        """
        return float(4.0 / 3.0 * math.pi * np.prod(self.radii))

    def _local_inertia(self, mass: float) -> np.ndarray:
        rx, ry, rz = self.radii
        return np.diag(
            [
                ry**2 + rz**2,
                rx**2 + rz**2,
                rx**2 + ry**2,
            ]
        ) * mass / 5

    def _geometry(self):
        return _mesh("unit_sphere.vtp", tuple(2 * radius for radius in self.radii))


def cylinder_between(
    start: Sequence[float], end: Sequence[float], radius: float, inner_radius: float = 0.0
) -> Cylinder:
    """Create a cylinder or hollow tube aligned between two points.

    The helper computes the segment length and rotation from the body-frame
    y axis to the direction from ``start`` to ``end``. A positive
    ``inner_radius`` creates a ``Tube``; zero creates a solid ``Cylinder``.

    Parameters
    ----------
    start, end : sequence of float
        Distinct endpoint coordinates ``(x, y, z)`` in metres in the body
        frame.
    radius : float
        Outer radius in metres.
    inner_radius : float, default=0.0
        Inner radius in metres; zero denotes a solid cylinder.

    Returns
    -------
    Cylinder or Tube
        Primitive centred at the midpoint and oriented along the endpoint
        segment.

    Raises
    ------
    ValueError
        If the endpoints coincide.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0:
        raise ValueError("start and end must be distinct points")
    direction = delta / length
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, direction)
    c = float(np.dot(y, direction))
    if np.linalg.norm(v) < 1e-12:
        rotation = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation = np.eye(3) + vx + vx @ vx / (1 + c)
    kwargs = dict(
        center=tuple((start + end) / 2),
        orientation=matrix_to_euler_xyz(rotation),
        radius=radius,
        length=length,
        axis="y",
    )
    return Tube(inner_radius=inner_radius, **kwargs) if inner_radius else Cylinder(**kwargs)
