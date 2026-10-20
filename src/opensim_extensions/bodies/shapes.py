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
    """Rotation matrix of body-fixed X-Y-Z Euler angles (rad), as used by OpenSim."""
    a, b, c = angles
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz


def matrix_to_euler_xyz(matrix: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`euler_xyz_to_matrix` (``|b| <= pi/2``)."""
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
    """Base primitive: ``center`` (m) and ``orientation`` (XYZ Euler, rad) in the body frame."""

    center: Sequence[float] = (0.0, 0.0, 0.0)
    orientation: Sequence[float] = (0.0, 0.0, 0.0)
    color: Sequence[float] | None = field(default=None, kw_only=True)

    @property
    def volume(self) -> float:
        raise NotImplementedError

    def _local_inertia(self, mass: float) -> np.ndarray:
        """Inertia tensor about the shape centre in the shape's own axes."""
        raise NotImplementedError

    def _axis_rotation(self) -> np.ndarray:
        return np.eye(3)

    @property
    def rotation(self) -> np.ndarray:
        """Rotation from the shape's natural axes to the body frame."""
        return euler_xyz_to_matrix(self.orientation) @ self._axis_rotation()

    def inertia_about_center(self, mass: float) -> np.ndarray:
        """Inertia tensor about the centre, expressed in body-frame axes."""
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
    """Cuboid with full ``size`` (x, y, z) in metres."""

    size: Sequence[float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        _validate(**{f"size[{i}]": v for i, v in enumerate(self.size)})

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def _local_inertia(self, mass: float) -> np.ndarray:
        a, b, c = self.size
        return np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2]) * mass / 12

    def _geometry(self):
        return _mesh("unit_box.vtp", self.size)


@dataclass
class Cylinder(Shape):
    """Solid cylinder of ``radius`` and ``length`` (m) along ``axis`` ('x', 'y' or 'z')."""

    radius: float = 0.5
    length: float = 1.0
    axis: str = "y"

    def __post_init__(self):
        _validate(radius=self.radius, length=self.length)
        if self.axis not in _AXIS_EULER:
            raise ValueError("axis must be 'x', 'y' or 'z'")

    @property
    def volume(self) -> float:
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
    """Thin cylinder: ``radius`` and ``thickness`` (m); the disk normal is ``axis``."""

    thickness: float = 0.005
    length: float = field(init=False, default=0.005)

    def __post_init__(self):
        self.length = self.thickness
        super().__post_init__()


@dataclass
class Tube(Cylinder):
    """Hollow cylinder with ``radius`` (outer), ``inner_radius`` and ``length`` (m)."""

    inner_radius: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.inner_radius < self.radius:
            raise ValueError("inner_radius must be in [0, radius)")

    @property
    def volume(self) -> float:
        return math.pi * (self.radius**2 - self.inner_radius**2) * self.length

    def _local_inertia(self, mass: float) -> np.ndarray:
        r2 = self.radius**2 + self.inner_radius**2
        radial = mass * (3 * r2 + self.length**2) / 12
        return np.diag([radial, 0.5 * mass * r2, radial])


@dataclass
class Sphere(Shape):
    """Solid sphere of ``radius`` (m)."""

    radius: float = 0.5

    def __post_init__(self):
        _validate(radius=self.radius)

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius**3

    def _local_inertia(self, mass: float) -> np.ndarray:
        return np.eye(3) * 0.4 * mass * self.radius**2

    def _geometry(self):
        return _mesh("unit_sphere.vtp", (2 * self.radius,) * 3)


@dataclass
class Ellipsoid(Shape):
    """Ellipsoid with semi-axes ``radii`` (x, y, z) in metres."""

    radii: Sequence[float] = (0.5, 0.5, 0.5)

    def __post_init__(self):
        _validate(**{f"radii[{i}]": value for i, value in enumerate(self.radii)})

    @property
    def volume(self) -> float:
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
    """Cylinder (or tube, if ``inner_radius`` > 0) whose axis joins two points (m)."""
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
