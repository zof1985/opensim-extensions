"""Visual ergonomics analysis and optimization for a user and equipment model."""

from __future__ import annotations

import math
import tempfile
import warnings
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Self

import numpy as np
from scipy.optimize import minimize, root, root_scalar
from scipy.special import expit
from scipy.spatial import ConvexHull, QhullError

from ..bodies import Cylinder, Shape, add_welded_body, matrix_to_euler_xyz
from ..models.equipment import Equipment, Screen
from ..models.user import User
from ..opensim import import_opensim
from .framework import Framework

__all__ = ["VisualErgonomicsFramework"]

if TYPE_CHECKING:
    import opensim
else:
    opensim = import_opensim()
_OPENSIM = opensim
_MESH_DENSITY = 1000.0
_SCREEN_GRID_SIZE = 11
_REACHABILITY_SAMPLES = 256
_REACHABILITY_SCORE_SCALE_M = 0.1
_REACH_COORDINATES = (
    "arm_flex",
    "arm_add",
    "arm_rot",
    "pro_sup",
    "wrist_flex",
    "wrist_dev",
)
_REACH_FIXED_COORDINATES = ("elbow_flex",)
OptimizationParameter = float | tuple[float, float] | None


class _MeshShape(Shape):
    def __init__(
        self,
        path: Path,
        *,
        color: Sequence[float],
        opacity: float,
    ) -> None:
        super().__init__(color=color)
        self.path = path
        self.opacity = opacity

    @property
    def volume(self) -> float:
        return 1e-6

    def _local_inertia(self, mass: float) -> np.ndarray:
        return np.eye(3) * mass * 1e-6

    def _geometry(self):
        geometry = _OPENSIM.Mesh(str(self.path))
        geometry.setColor(_OPENSIM.Vec3(*self.color))
        geometry.upd_Appearance().set_opacity(self.opacity)
        return geometry


def _write_polydata(
    path: Path, points: np.ndarray, faces: Sequence[Sequence[int]]
) -> None:
    connectivity = " ".join(str(index) for face in faces for index in face)
    offsets = " ".join(str(3 * (index + 1)) for index in range(len(faces)))
    coordinates = " ".join(f"{value:.9g}" for value in points.ravel())
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{len(points)}" NumberOfPolys="{len(faces)}">\n'
        '      <Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">'
        f"{coordinates}</DataArray></Points>\n"
        "      <Polys>\n"
        f'        <DataArray type="Int32" Name="connectivity" format="ascii">{connectivity}</DataArray>\n'
        f'        <DataArray type="Int32" Name="offsets" format="ascii">{offsets}</DataArray>\n'
        "      </Polys>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n",
        encoding="ascii",
    )


def _pyramid_mesh(
    width_angle: float, height_angle: float, length: float
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    half_width = length * math.tan(math.radians(width_angle) / 2)
    vertical_half_extent = length * math.tan(math.radians(height_angle) / 2)
    points = np.array(
        [
            (0.0, 0.0, 0.0),
            (length, vertical_half_extent, -half_width),
            (length, vertical_half_extent, half_width),
            (length, -vertical_half_extent, half_width),
            (length, -vertical_half_extent, -half_width),
        ],
        dtype=float,
    )
    return points, [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 1),
        (1, 4, 3),
        (1, 3, 2),
    ]


def _sphere_mesh(
    radius: float, latitude_steps: int = 12, longitude_steps: int = 24
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    points = [(0.0, radius, 0.0)]
    for latitude in range(1, latitude_steps):
        polar = math.pi * latitude / latitude_steps
        for longitude in range(longitude_steps):
            azimuth = 2 * math.pi * longitude / longitude_steps
            points.append(
                (
                    radius * math.sin(polar) * math.cos(azimuth),
                    radius * math.cos(polar),
                    radius * math.sin(polar) * math.sin(azimuth),
                )
            )
    south_pole = len(points)
    points.append((0.0, -radius, 0.0))
    faces = []
    first_ring = 1
    for longitude in range(longitude_steps):
        next_longitude = (longitude + 1) % longitude_steps
        faces.append(
            (0, first_ring + longitude, first_ring + next_longitude)
        )
    for latitude in range(latitude_steps - 2):
        ring = 1 + latitude * longitude_steps
        next_ring = ring + longitude_steps
        for longitude in range(longitude_steps):
            next_longitude = (longitude + 1) % longitude_steps
            faces.extend(
                (
                    (ring + longitude, next_ring + longitude, next_ring + next_longitude),
                    (ring + longitude, next_ring + next_longitude, ring + next_longitude),
                )
            )
    last_ring = 1 + (latitude_steps - 2) * longitude_steps
    for longitude in range(longitude_steps):
        next_longitude = (longitude + 1) % longitude_steps
        faces.append(
            (south_pole, last_ring + next_longitude, last_ring + longitude)
        )
    return np.asarray(points, dtype=float), faces


def _rotation_in_ground(frame: Any, state: Any) -> np.ndarray:
    transform = frame.getTransformInGround(state)
    return np.array(
        [[transform.R().get(row, column) for column in range(3)] for row in range(3)]
    )


def _position_in_ground(frame: Any, state: Any) -> np.ndarray:
    point = frame.getPositionInGround(state)
    return np.array([point.get(index) for index in range(3)])


class VisualErgonomicsFramework(Framework):
    """Framework with a visual-cone overlay and screen ergonomics metrics.

    The framework stores its user and equipment under ``"user"`` and
    ``"equipment"``. All public angle values are in degrees; screen dimensions
    and offsets are in millimetres.

    The default visual cone uses a 30-degree full horizontal width and a
    30-degree vertical height extending downward from the head plane through
    the eyes. The line of sight is separately offset 15 degrees below that
    plane, configurable through ``line_of_sight_offset_deg``. The literature
    does not prescribe a matching numeric horizontal cone width; 30 degrees
    total (15 degrees to either side) is a conservative symmetric modelling
    choice, not a separate standard limit.

    References
    ----------
    Turville, K. L., Psihogios, J. P., Ulmer, T. R., & Mirka, G. A. (1998).
    The effects of video display terminal height on the operator: a
    comparison of the 15 degree and 40 degree recommendations. *Applied
    Ergonomics*, 29(4), 239-246.
    DOI: 10.1016/S0003-6870(97)00048-3
    PubMed: https://pubmed.ncbi.nlm.nih.gov/9701537/

    Burgess-Limerick, R., Mon-Williams, M., & Coppard, V. L. (2000). Visual
    display height. *Human Factors*, 42(1), 140-150.
    DOI: 10.1518/001872000779656598
    PubMed: https://pubmed.ncbi.nlm.nih.gov/10917150/

    Canadian Centre for Occupational Health and Safety. Monitor Positioning.
    OSH Answers. The guidance describes a preferred 30-degree visual zone
    around the normal line of sight, approximately 15 degrees below
    horizontal, and notes that recommendations are guidelines rather than
    fixed limits.
    https://www.ccohs.ca/oshanswers/ergonomics/office/monitor_positioning.html
    """

    def __init__(
        self,
        user: User | opensim.Model,
        equipment: Equipment,
        cone_width_angle_deg: float = 30.0,
        cone_height_angle_deg: float = 30.0,
        line_of_sight_offset_deg: float = -15.0,
    ) -> None:
        """Create a visual-ergonomics framework for a user and equipment.

        The constructor adds a visual-cone overlay and palm-marker reachability
        surfaces to a framework containing the supplied models. The equipment
        must provide a screen.

        Parameters
        ----------
        user : User or opensim.Model
            Rider model. Its eye and palm markers and current trunk/pelvis
            posture are used by the ergonomics calculations.
        equipment : Equipment
            Equipment model whose screen is evaluated.
        cone_width_angle_deg : float, default=30.0
            Full horizontal visual-cone angle in degrees (15 degrees on each
            side of the forward direction). The cited ergonomic guidance is
            vertical; this symmetric horizontal width is a conservative
            modelling choice.
        cone_height_angle_deg : float, default=30.0
            Full vertical visual-cone angle in degrees, extending downward
            from the head plane through the eyes.
        line_of_sight_offset_deg : float, default=-15.0
            Signed angle between the head plane through the eyes and the line
            of sight, in degrees. Negative values point downward. This is
            independent of the visual-cone height.

        Raises
        ------
        TypeError
            If ``user`` is not an OpenSim model or ``equipment`` is not an
            :class:`Equipment`.
        ValueError
            If the equipment has no screen, or an angle is not finite or falls
            outside its supported interval.

        References
        ----------
        See the :class:`VisualErgonomicsFramework` reference list. In
        particular, Turville et al. (1998) compared 15- and 40-degree
        downward display locations and found increased head tilt and muscle
        activity at 40 degrees; CCOHS describes the preferred 30-degree
        visual zone and cautions that the numerical guidance is not an
        absolute limit.
        """
        if not isinstance(user, _OPENSIM.Model):
            raise TypeError("user must be an OpenSim Model")
        if not isinstance(equipment, Equipment):
            raise TypeError("equipment must be an Equipment")
        if not equipment.has_screen():
            raise ValueError("equipment must provide a screen for visual ergonomics")
        self._validate_angle("cone_width_angle_deg", cone_width_angle_deg, 0.0, 180.0)
        self._validate_angle("cone_height_angle_deg", cone_height_angle_deg, 0.0, 89.0)
        self._validate_angle("line_of_sight_offset_deg", line_of_sight_offset_deg, -89.0, 89.0)
        object.__setattr__(self, "cone_width_angle_deg", float(cone_width_angle_deg))
        object.__setattr__(self, "cone_height_angle_deg", float(cone_height_angle_deg))
        object.__setattr__(
            self, "line_of_sight_offset_deg", float(line_of_sight_offset_deg)
        )
        object.__setattr__(
            self,
            "_geometry_directory",
            tempfile.TemporaryDirectory(prefix="opensim-visual-ergonomics-"),
        )
        object.__setattr__(self, "_reachability_cache", {})
        super().__init__({"user": user, "equipment": equipment})
        self._refresh_visualization()

    @staticmethod
    def _validate_angle(name: str, value: float, low: float, high: float) -> None:
        if not math.isfinite(float(value)) or not low < float(value) < high:
            raise ValueError(f"{name} must be finite and in ({low}, {high})")

    @property
    def user(self) -> opensim.Model:
        """Return the OpenSim rider model used by this analysis.

        Returns
        -------
        opensim.Model
            Rider model supplying posture, eye markers, and palm markers.
        """
        return self["user"]

    @property
    def equipment(self) -> Equipment:
        """Return the equipment analyzed by this framework.

        Returns
        -------
        Equipment
            Equipment model supplying the screen and its pose.
        """
        return self["equipment"]

    @property
    def screen(self) -> Screen:
        """Return the current screen description after validating availability.

        Returns
        -------
        Screen
            Current screen dimensions, mounting offsets, and world-frame pose.

        Raises
        ------
        ValueError
            If the equipment no longer provides a screen.
        """
        screen = self.equipment.screen
        if screen is None:
            raise ValueError("equipment must provide a screen for visual ergonomics")
        return screen

    def _user_state(self) -> Any:
        state = getattr(self.user, "state", None)
        if state is not None:
            return state
        state = self.user.getWorkingState()
        if state is None:
            state = self.user.initSystem()
            self.user.state = state
        return state

    @staticmethod
    def _vector(value: Any) -> np.ndarray:
        return np.array([value.get(index) for index in range(3)], dtype=float)

    def _eye_center_and_head_rotation(self) -> tuple[np.ndarray, np.ndarray]:
        state = self._user_state()
        self.user.realizePosition(state)
        markers = self.user.getMarkerSet()
        eyes = np.array(
            [
                self._vector(markers.get(name).getLocationInGround(state))
                for name in ("REYE", "LEYE")
            ]
        )
        skull = self.user.getBodySet().get("skull")
        return eyes.mean(axis=0), _rotation_in_ground(skull, state)

    def _eye_center_and_line_of_sight_rotation(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return eye midpoint and configured line-of-sight frame."""
        return self._eye_center_and_vertical_offset_rotation(
            self.line_of_sight_offset_deg
        )

    def _eye_center_and_cone_rotation(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the cone frame with its upper edge parallel to the head plane."""
        return self._eye_center_and_vertical_offset_rotation(
            -self.cone_height_angle_deg / 2
        )

    def _eye_center_and_vertical_offset_rotation(
        self, offset_deg: float
    ) -> tuple[np.ndarray, np.ndarray]:
        eye, head_rotation = self._eye_center_and_head_rotation()
        offset = math.radians(offset_deg)
        cosine, sine = math.cos(offset), math.sin(offset)
        offset_rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        return eye, head_rotation @ offset_rotation

    def _gaze_center_coordinates(
        self, position: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        eye, visual_rotation = self._eye_center_and_line_of_sight_rotation()
        screen_center = self._screen_points(position).mean(axis=0)
        local = (screen_center - eye) @ visual_rotation
        angles = np.array(
            [
                math.atan2(local[1], local[0]),
                math.atan2(local[2], local[0]),
            ]
        )
        return local, angles

    def _palm_marker_positions(self) -> np.ndarray:
        state = self._user_state()
        self.user.realizePosition(state)
        markers = self.user.getMarkerSet()
        return np.array(
            [
                self._vector(markers.get(name).getLocationInGround(state))
                for name in ("RPALM", "LPALM")
            ]
        )

    def _screen_points(self, position: Sequence[float] | None = None) -> np.ndarray:
        candidate = None if position is None else tuple(map(float, position))
        return self.screen.corners(candidate)

    def _screen_samples(self, position: Sequence[float] | None = None) -> np.ndarray:
        candidate = None if position is None else tuple(map(float, position))
        return self.screen.samples(_SCREEN_GRID_SIZE, candidate)

    def _metrics(self, position: Sequence[float] | None = None) -> dict[str, float]:
        """Evaluate incidence, visual coverage, and reachable screen coverage.

        Incidence is computed from the eye midpoint, screen centre, and screen
        normal. Visibility uses the rectangular visual cone in cone-local
        coordinates. Visibility and reachability are
        reported as the proportions of the uniform screen samples that satisfy
        their respective predicates.

        Parameters
        ----------
        position : sequence of float or None, optional
            Candidate screen position ``(horizontal_offset_mm,
            vertical_offset_mm, inclination_deg)``. ``None`` evaluates the
            bike's currently configured screen.

        Returns
        -------
        dict[str, float]
            Incidence in degrees and visible/reachable screen fractions as
            percentages.

        Raises
        ------
        ValueError
            If the screen centre coincides with the eye midpoint.
        """
        eye, cone_rotation = self._eye_center_and_cone_rotation()
        screen = self._screen_points(position)
        screen_center = screen.mean(axis=0)
        inclination = math.radians(
            self.screen.inclination_deg if position is None else position[2]
        )
        normal = np.array([math.sin(inclination), -math.cos(inclination), 0.0])
        sight = screen_center - eye
        sight_length = float(np.linalg.norm(sight))
        if sight_length <= 1e-12:
            raise ValueError("the screen centre and eye midpoint must be distinct")
        incidence = math.degrees(
            math.acos(
                float(np.clip(abs(np.dot(sight / sight_length, normal)), 0.0, 1.0))
            )
        )

        points = self._screen_samples(position)
        local = (points - eye) @ cone_rotation
        forward = local[:, 0]
        horizontal_angle = np.arctan2(np.abs(local[:, 2]), np.maximum(forward, 1e-12))
        downward_angle = np.arctan2(-local[:, 1], np.maximum(forward, 1e-12))
        vertical_half_angle = math.radians(self.cone_height_angle_deg / 2)
        visible = (
            (forward > 0.0)
            & (horizontal_angle <= math.radians(self.cone_width_angle_deg / 2))
            & (np.abs(downward_angle) <= vertical_half_angle)
        )
        reachable = self._reachable_screen_samples(points)
        return {
            "incidence_angle_deg": incidence,
            "observed_screen_percent": float(np.mean(visible) * 100.0),
            "reachable_screen_percent": float(np.mean(reachable) * 100.0),
        }

    def _optimization_scores(self, position: Sequence[float]) -> np.ndarray:
        """Compute smooth objective scores for one candidate screen pose.

        Incidence is normalized to a score in ``[0, 1]``. Visual-cone margins
        and signed distances from each shoulder-centred reach sphere are converted
        through logistic functions, giving the optimizer smooth
        approximations to the visible and reachable sample fractions.

        Parameters
        ----------
        position : sequence of float
            Candidate screen position ``(horizontal_offset_mm,
            vertical_offset_mm, inclination_deg)``.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Scores ordered as incidence, visibility, and reachability. Each
            component is maximized by the optimizer.
        """
        metrics = self._metrics(position)
        points = self._screen_samples(position)
        eye, cone_rotation = self._eye_center_and_cone_rotation()
        local = (points - eye) @ cone_rotation
        forward = local[:, 0]
        horizontal = np.arctan2(np.abs(local[:, 2]), np.maximum(forward, 1e-12))
        downward = np.arctan2(-local[:, 1], np.maximum(forward, 1e-12))
        angular_scale = math.radians(2.0)
        vertical_half_angle = math.radians(self.cone_height_angle_deg / 2)
        cone_margin = np.minimum.reduce(
            (
                forward / 0.02,
                (math.radians(self.cone_width_angle_deg / 2) - horizontal)
                / angular_scale,
                (vertical_half_angle - np.abs(downward)) / angular_scale,
            )
        )
        visible_score = float(np.mean(expit(cone_margin)))

        reachable_probabilities = []
        for side in ("r", "l"):
            center, radius = self._reachability_sphere(side)
            violation = np.linalg.norm(points - center, axis=1) - radius
            reachable_probabilities.append(
                expit(-violation / _REACHABILITY_SCORE_SCALE_M)
            )
        reachable_score = float(
            np.mean(
                1.0
                - (1.0 - reachable_probabilities[0])
                * (1.0 - reachable_probabilities[1])
            )
        )
        return np.array(
            [
                1.0 - metrics["incidence_angle_deg"] / 90.0,
                visible_score,
                reachable_score,
            ]
        )

    def _reachability_hull(
        self, side: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample palm-centre positions and construct their convex hull.

        Shoulder flexion is sampled from 0 to 180 degrees and shoulder
        adduction/abduction from -30 to 30 degrees. Forearm and wrist
        coordinates use their effective OpenSim ranges. In addition to
        deterministic combinations of all range endpoints, seeded random
        samples cover the interior. The elbow is fixed at its minimum
        coordinate value, which represents full extension in the bundled
        model. Trunk and pelvis coordinates remain at the current posture. The
        method restores all sampled coordinates after realization and caches
        the points, hull plane equations, and triangular surface facets by
        posture, anthropometric dimensions, and coordinate ranges.

        Parameters
        ----------
        side : {'r', 'l'}
            Side whose palm marker and arm coordinates define the hull.

        Returns
        -------
        points : numpy.ndarray, shape (n_samples, 3)
            Sampled palm-centre marker positions in metres in the ground frame.
        equations : numpy.ndarray, shape (n_facets, 4)
            Convex-hull plane equations. Each row contains a normal vector and
            offset; a point is inside when the plane expression is non-positive.
            The array is empty when a three-dimensional hull cannot be formed.
        simplices : numpy.ndarray, shape (n_facets, 3)
            Point indices for triangular hull facets, or an empty array when
            the hull is degenerate.
        """
        if side not in {"r", "l"}:
            raise ValueError("side must be 'r' or 'l'")
        state = self._user_state()
        coordinates = self.user.getCoordinateSet()
        key = (
            0.0 if side == "r" else 1.0,
            round(
                float(
                    getattr(
                        self.user,
                        f"{'right' if side == 'r' else 'left'}_arm_length",
                    )
                ),
                6,
            ),
            round(
                float(
                    getattr(
                        self.user,
                        f"{'right' if side == 'r' else 'left'}_forearm_length",
                    )
                ),
                6,
            ),
            round(
                float(
                    getattr(
                        self.user,
                        f"{'right' if side == 'r' else 'left'}_palm_length",
                    )
                ),
                6,
            ),
        )
        names = [f"{name}_{side}" for name in _REACH_COORDINATES]
        coordinate_values = [coordinates.get(name) for name in names]
        fixed_coordinates = [
            coordinates.get(f"{name}_{side}") for name in _REACH_FIXED_COORDINATES
        ]
        shoulder_ranges = {
            "arm_flex": (0.0, math.pi),
            "arm_add": (math.radians(-30.0), math.radians(30.0)),
        }
        lower = np.array(
            [
                shoulder_ranges[name][0]
                if name in shoulder_ranges
                else coordinate.getRangeMin()
                for name, coordinate in zip(_REACH_COORDINATES, coordinate_values)
            ]
        )
        upper = np.array(
            [
                shoulder_ranges[name][1]
                if name in shoulder_ranges
                else coordinate.getRangeMax()
                for name, coordinate in zip(_REACH_COORDINATES, coordinate_values)
            ]
        )
        key += tuple(
            round(float(value), 6)
            for bounds in zip(lower, upper)
            for value in bounds
        ) + tuple(
            round(float(value), 6)
            for coordinate in fixed_coordinates
            for value in (coordinate.getRangeMin(), coordinate.getRangeMax())
        )
        cached = self._reachability_cache.get(key)
        if cached is None:
            original_values = [
                coordinate.getValue(state) for coordinate in coordinate_values
            ]
            original_fixed_values = [
                coordinate.getValue(state) for coordinate in fixed_coordinates
            ]
            random = np.random.default_rng(1729 if side == "r" else 1730)
            random_samples = random.uniform(
                lower, upper, (_REACHABILITY_SAMPLES, len(names))
            )
            boundary_samples = np.asarray(
                tuple(product(*zip(lower, upper))), dtype=float
            )
            samples = np.vstack((random_samples, boundary_samples))
            marker = self.user.getMarkerSet().get(
                "RPALM" if side == "r" else "LPALM"
            )
            endpoints = []
            try:
                for values in samples:
                    for coordinate, value in zip(coordinate_values, values):
                        coordinate.setValue(state, float(value), False)
                    for coordinate in fixed_coordinates:
                        coordinate.setValue(state, coordinate.getRangeMin(), False)
                    self.user.realizePosition(state)
                    endpoints.append(
                        self._vector(marker.getLocationInGround(state))
                    )
            finally:
                for coordinate, value in zip(coordinate_values, original_values):
                    coordinate.setValue(state, value, False)
                for coordinate, value in zip(
                    fixed_coordinates, original_fixed_values
                ):
                    coordinate.setValue(state, value, False)
                self.user.realizePosition(state)
            torso = self.user.getBodySet().get("torso")
            torso_position = _position_in_ground(torso, state)
            torso_rotation = _rotation_in_ground(torso, state)
            local_points = (np.asarray(endpoints) - torso_position) @ torso_rotation
            try:
                hull = ConvexHull(local_points)
                cached = (
                    local_points,
                    hull.equations.copy(),
                    hull.simplices.copy(),
                )
            except QhullError:
                cached = (
                    local_points,
                    np.empty((0, 4)),
                    np.empty((0, 3), dtype=int),
                )
            if len(self._reachability_cache) > 24:
                self._reachability_cache.clear()
            self._reachability_cache[key] = cached

        local_points, local_equations, faces = cached
        torso = self.user.getBodySet().get("torso")
        torso_position = _position_in_ground(torso, state)
        torso_rotation = _rotation_in_ground(torso, state)
        points = local_points @ torso_rotation.T + torso_position
        if local_equations.size:
            normals = local_equations[:, :3] @ torso_rotation.T
            offsets = local_equations[:, 3] - normals @ torso_position
            equations = np.column_stack((normals, offsets))
        else:
            equations = local_equations
        return points, equations, faces

    def _reachable_screen_samples(self, screen_points: np.ndarray) -> np.ndarray:
        reachable = []
        for side in ("r", "l"):
            center, radius = self._reachability_sphere(side)
            reachable.append(
                np.linalg.norm(screen_points - center, axis=1) <= radius + 1e-8
            )
        return np.logical_or.reduce(reachable)

    def _reachability_sphere(self, side: str) -> tuple[np.ndarray, float]:
        points, _, _ = self._reachability_hull(side)
        shoulder = _position_in_ground(
            self.user.getJointSet().get(f"acromial_{side}").getChildFrame(),
            self._user_state(),
        )
        radius = float(np.max(np.linalg.norm(points - shoulder, axis=1)))
        return shoulder, radius

    def get_view_state(self) -> dict[str, float]:
        """Measure screen incidence, visual coverage, and hand reachability.

        Incidence is the angle between the screen normal and the line from the
        midpoint of the eye markers to the screen centre. Visibility and
        reachability percentages are sampled on a uniform screen grid.
        Reachability is the union of spheres centred at the shoulder joints.
        Each sphere radius is the greatest shoulder-to-palm distance sampled
        with the elbow fully extended. Trunk and pelvis coordinates remain at
        their current values.

        Returns
        -------
        dict[str, float]
            Mapping with ``incidence_angle_deg`` (degrees),
            ``observed_screen_percent`` (visible grid fraction), and
            ``reachable_screen_percent`` (fraction inside either shoulder-
            centred reach sphere). Percentages are in ``[0, 100]``.
        """
        return self._metrics()

    def _screen_constraints(self, position: Sequence[float]) -> np.ndarray:
        return self.equipment.screen_position_constraints(
            tuple(map(float, position))
        )

    @staticmethod
    def _range(name: str, value: Sequence[float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        low, high = float(value[0]), float(value[1])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"{name} must be a finite increasing range")
        return low, high

    @staticmethod
    def _parameter_bounds(
        name: str,
        value: float | tuple[float, float] | None,
        current: float,
        allowed: tuple[float, float],
    ) -> tuple[float, float]:
        if value is None:
            return current, current
        if isinstance(value, bool):
            raise TypeError(f"{name} must be None, a float, or a tuple of two floats")
        if isinstance(value, (int, float, np.integer, np.floating)):
            low = high = float(value)
        elif isinstance(value, tuple):
            low, high = VisualErgonomicsFramework._range(name, value)
        else:
            raise TypeError(f"{name} must be None, a float, or a tuple of two floats")
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(f"{name} values must be finite")
        if low < allowed[0] or high > allowed[1]:
            warnings.warn(
                f"{name} bounds {(low, high)} exceed the current limits "
                f"{allowed}; expanding the limits to include them",
                UserWarning,
                stacklevel=3,
            )
        return low, high

    def _clone_models(self) -> tuple[User, Equipment]:
        original_user = self.user
        state = self._user_state()
        user = User.__new__(User)
        _OPENSIM.Model.__init__(user, original_user)
        user._reference = original_user._reference
        user._baseline = original_user._baseline
        user.opensim = original_user.opensim
        user._geometry_dirs = list(original_user._geometry_dirs)
        user._side_measurements = dict(original_user._side_measurements)
        user._visualizer = None
        user._mass_kg = original_user._mass_kg
        reach_lengths = getattr(original_user, "_visual_arm_reach_lengths", None)
        if reach_lengths is None:
            reach_lengths = {
                "right": (
                    float(original_user.right_arm_length),
                    float(original_user.right_forearm_length),
                ),
                "left": (
                    float(original_user.left_arm_length),
                    float(original_user.left_forearm_length),
                ),
            }
        user._visual_arm_reach_lengths = dict(reach_lengths)
        source_coordinates = original_user.getCoordinateSet()
        target_coordinates = user.getCoordinateSet()
        coordinate_values = {
            source_coordinates.get(index).getName(): (
                source_coordinates.get(index).getValue(state),
                source_coordinates.get(index).getSpeedValue(state),
            )
            for index in range(source_coordinates.getSize())
        }
        for name, (value, speed) in coordinate_values.items():
            target = target_coordinates.get(name)
            target.setDefaultValue(value)
            target.setDefaultSpeedValue(speed)
        user.state = user.initSystem()
        for name, (value, speed) in coordinate_values.items():
            target = user.getCoordinateSet().get(name)
            target.setValue(user.state, value, False)
            target.setSpeedValue(user.state, speed)
        user.realizePosition(user.state)

        return user, self.equipment.clone()

    def _coordinate_ranges(
        self,
        coordinate_ranges_deg: Mapping[str, Sequence[float]] | None,
        neck_flexionextension_range_deg: Sequence[float] | None,
        head_extensionflexion_range_deg: Sequence[float] | None,
        trunk_flexionextension_range_deg: Sequence[float] | None,
        shoulder_flexion_range_deg: Sequence[float] | None,
        elbow_flexion_range_deg: Sequence[float] | None,
    ) -> dict[str, tuple[float, float]]:
        result: dict[str, tuple[float, float]] = {}
        aliases = (
            (neck_flexionextension_range_deg, "neck_base_flexion"),
            (head_extensionflexion_range_deg, "neck_flexion"),
            (trunk_flexionextension_range_deg, "lumbar_extension"),
        )
        for bounds, name in aliases:
            if bounds is not None:
                result[name] = self._range(f"{name} range", bounds)
        if shoulder_flexion_range_deg is not None:
            result.update(
                {
                    f"arm_flex_{side}": self._range(
                        "shoulder_flexion_range_deg", shoulder_flexion_range_deg
                    )
                    for side in ("r", "l")
                }
            )
        if elbow_flexion_range_deg is not None:
            result.update(
                {
                    f"elbow_flex_{side}": self._range(
                        "elbow_flexion_range_deg", elbow_flexion_range_deg
                    )
                    for side in ("r", "l")
                }
            )
        for name, bounds in (coordinate_ranges_deg or {}).items():
            self._range(f"coordinate_ranges_deg[{name!r}]", bounds)
            result[name] = tuple(map(float, bounds))
        coordinates = self.user.getCoordinateSet()
        for name, (low, high) in result.items():
            try:
                coordinate = coordinates.get(name)
            except RuntimeError as error:
                raise ValueError(f"unknown user coordinate {name!r}") from error
            allowed = (
                math.degrees(coordinate.getRangeMin()),
                math.degrees(coordinate.getRangeMax()),
            )
            if low < allowed[0] or high > allowed[1]:
                warnings.warn(
                    f"{name} range {(low, high)} exceeds model limits {allowed}; "
                    "expanding the model limits to include it",
                    UserWarning,
                    stacklevel=2,
                )
                coordinate.setRangeMin(math.radians(min(low, allowed[0])))
                coordinate.setRangeMax(math.radians(max(high, allowed[1])))
        return result

    def optimize_screen_position(
        self,
        *,
        incidence_weight: float = 0.34,
        observed_screen_weight: float = 0.33,
        reachability_weight: float = 0.33,
        inplace: bool = False,
        neck_flexionextension_range_deg: Sequence[float] | None = (-30.0, 30.0),
        head_extensionflexion_range_deg: Sequence[float] | None = (0.0, 0.0),
        trunk_flexionextension_range_deg: Sequence[float] | None = None,
        shoulder_flexion_range_deg: Sequence[float] | None = None,
        elbow_flexion_range_deg: Sequence[float] | None = None,
        screen_incline_range_deg: Sequence[float] | None = None,
        screen_horizontal_offset_range_mm: Sequence[float] | None = None,
        screen_vertical_offset_range_mm: Sequence[float] | None = None,
        coordinate_ranges_deg: Mapping[str, Sequence[float]] | None = None,
        screen_horizontal_offset_mm: float | tuple[float, float] | None = None,
        screen_vertical_offset_mm: float | tuple[float, float] | None = None,
        screen_inclination_deg: float | tuple[float, float] | None = None,
        coordinate_parameters_deg: Mapping[
            str, OptimizationParameter
        ]
        | None = None,
        _neck_flexion_weight: float | None = None,
        max_iterations: int = 100,
    ) -> Self | None:
        """Optimize screen position and selected rider coordinates.

        The objective combines screen incidence, visibility within the visual
        cone, and palm-marker reachability. The incidence score also penalizes
        misalignment between the gaze axis and the eye-to-screen-centre line.
        A bounded numerical optimization evaluates the screen on a regular
        grid; the palm reachability objective uses the extended-arm hulls
        independently of any fitted grip posture.
        By default only neck-base flexion/extension is free to change; the
        trunk, shoulders, and elbows remain at their current posture unless
        corresponding ranges are supplied.
        If ``inplace`` is false, model copies are optimized and the original
        framework remains unchanged.

        Parameters
        ----------
        incidence_weight : float, default=0.34
            Weight of perpendicular screen incidence in the objective.
        observed_screen_weight : float, default=0.33
            Weight of the screen fraction inside the visual cone.
        reachability_weight : float, default=0.33
            Weight of the screen fraction inside either palm-marker reach hull.
        inplace : bool, default=False
            If true, modify this framework and return ``None``; otherwise
            optimize independent copies and return the new framework.
        neck_flexionextension_range_deg : sequence of float or None, optional
            Inclusive lower and upper bounds for neck-base flexion/extension.
            Defaults to ``(-30, 30)`` degrees; ``None`` leaves the coordinate
            at its current value.
        head_extensionflexion_range_deg : sequence of float or None, optional
            Bounds for head flexion/extension in degrees. Defaults to
            ``(0, 0)``, keeping the head coordinate neutral.
        trunk_flexionextension_range_deg : sequence of float or None, optional
            Bounds for lumbar flexion/extension in degrees; ``None`` leaves
            the coordinate at its current value.
        shoulder_flexion_range_deg : sequence of float or None, optional
            Shared bounds for both shoulder flexion coordinates, in degrees;
            ``None`` leaves them at their current values.
        elbow_flexion_range_deg : sequence of float or None, optional
            Shared bounds for both elbow flexion coordinates, in degrees.
            These bounds affect the fitted rider posture, not the separate
            reachability envelope, which always uses fully extended elbows.
        screen_incline_range_deg : sequence of float or None, optional
            Legacy inclusive bounds for screen inclination in degrees.
        screen_horizontal_offset_range_mm : sequence of float or None, optional
            Legacy bounds for the screen's forward offset from its equipment
            reference point, in millimetres. If omitted, the current value is
            fixed unless ``screen_horizontal_offset_mm`` is supplied.
        screen_vertical_offset_range_mm : sequence of float or None, optional
            Legacy bounds for the screen's vertical offset from its equipment
            reference point, in millimetres. If omitted, the current value is
            fixed unless ``screen_vertical_offset_mm`` is supplied.
        coordinate_ranges_deg : mapping of str to sequence of float, optional
            Legacy coordinate-name to ``(minimum, maximum)`` bounds, in
            degrees, for user coordinates not covered by the dedicated range
            arguments.
        screen_horizontal_offset_mm, screen_vertical_offset_mm : float, tuple,
        or None, optional
            Screen offset values in millimetres. ``None`` keeps the current
            value, a float fixes the value, and a tuple sets the inclusive
            optimization bounds.
        screen_inclination_deg : float, tuple, or None, optional
            Screen inclination in degrees. ``None`` keeps the current value,
            a float fixes the value, and a tuple sets the inclusive
            optimization bounds.
        coordinate_parameters_deg : mapping, optional
            Mapping of user coordinate names to ``None``, a fixed value in
            degrees, or an inclusive optimization range in degrees. Omitted
            coordinates remain fixed. Values outside model limits expand the
            coordinate limits with a warning.
        max_iterations : int, default=100
            Maximum number of iterations for the numerical optimizer.

        Returns
        -------
        VisualErgonomicsFramework or None
            The optimized framework when ``inplace=False``; ``None`` when
            ``inplace=True``.

        Raises
        ------
        TypeError
            If ``inplace`` is not boolean.
        ValueError
            If weights are invalid, parameter values or bounds are invalid,
            the screen is unavailable, or no feasible position exists.
        """
        weight_values = [
            incidence_weight,
            observed_screen_weight,
            reachability_weight,
        ]
        if _neck_flexion_weight is not None:
            weight_values.append(_neck_flexion_weight)
        weights = np.array(weight_values, dtype=float)
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.any(weights > 1.0)
            or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("objective weights must be in [0, 1] and sum to 1")
        screen_weights = weights[:3]
        neck_weight = weights[3] if len(weights) == 4 else 0.0
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be bool")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")

        target: VisualErgonomicsFramework
        if inplace:
            target = self
        else:
            cloned_user, cloned_bike = self._clone_models()
            target = type(self)(
                cloned_user,
                cloned_bike,
                self.cone_width_angle_deg,
                self.cone_height_angle_deg,
                self.line_of_sight_offset_deg,
            )
        palm_reference = target._palm_marker_positions()

        coordinate_ranges = target._coordinate_ranges(
            coordinate_ranges_deg,
            neck_flexionextension_range_deg,
            head_extensionflexion_range_deg,
            trunk_flexionextension_range_deg,
            shoulder_flexion_range_deg,
            elbow_flexion_range_deg,
        )
        for name, value in (coordinate_parameters_deg or {}).items():
            try:
                coordinate = target.user.getCoordinateSet().get(name)
            except RuntimeError as error:
                raise ValueError(f"unknown user coordinate {name!r}") from error
            if value is None:
                coordinate_ranges.pop(name, None)
                continue
            current = math.degrees(
                coordinate.getValue(target._user_state())
            )
            allowed = (
                math.degrees(coordinate.getRangeMin()),
                math.degrees(coordinate.getRangeMax()),
            )
            bounds = target._parameter_bounds(
                f"coordinate_parameters_deg[{name!r}]",
                value,
                current,
                allowed,
            )
            if bounds[0] < allowed[0]:
                coordinate.setRangeMin(math.radians(bounds[0]))
            if bounds[1] > allowed[1]:
                coordinate.setRangeMax(math.radians(bounds[1]))
            coordinate_ranges[name] = bounds

        screen = target.screen
        screen_inputs = (
            (
                "screen_horizontal_offset_mm",
                screen_horizontal_offset_mm,
                screen_horizontal_offset_range_mm,
                screen.horizontal_offset_mm,
                screen.horizontal_offset_min_mm,
            ),
            (
                "screen_vertical_offset_mm",
                screen_vertical_offset_mm,
                screen_vertical_offset_range_mm,
                screen.vertical_offset_mm,
                screen.vertical_offset_min_mm,
            ),
            (
                "screen_inclination_deg",
                screen_inclination_deg,
                screen_incline_range_deg,
                screen.inclination_deg,
                -math.inf,
            ),
        )
        screen_ranges: dict[str, tuple[float, float]] = {}
        for name, value, legacy_range, current, minimum in screen_inputs:
            if value is not None and legacy_range is not None:
                raise ValueError(
                    f"specify either {name} or its legacy range, not both"
                )
            allowed = (
                minimum if minimum is not None else -math.inf,
                math.inf,
            )
            if value is not None:
                bounds = target._parameter_bounds(name, value, current, allowed)
            elif legacy_range is not None:
                bounds = target._range(name, legacy_range)
                if bounds[0] < allowed[0]:
                    warnings.warn(
                        f"{name} range {bounds} is below the current limit "
                        f"{allowed[0]}; expanding the limit to include it",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                bounds = (current, current)
            screen_ranges[name] = bounds

        target.equipment.expand_screen_offset_limits(
            horizontal_min_mm=(
                screen_ranges["screen_horizontal_offset_mm"][0]
                if screen.horizontal_offset_min_mm is not None
                and screen_ranges["screen_horizontal_offset_mm"][0]
                < screen.horizontal_offset_min_mm
                else None
            ),
            vertical_min_mm=(
                screen_ranges["screen_vertical_offset_mm"][0]
                if screen.vertical_offset_min_mm is not None
                and screen_ranges["screen_vertical_offset_mm"][0]
                < screen.vertical_offset_min_mm
                else None
            ),
        )

        user = target.user
        state = target._user_state()
        coordinates = user.getCoordinateSet()
        coordinate_state = {
            name: math.degrees(coordinates.get(name).getValue(state))
            for name in coordinate_ranges
        }
        coordinate_bounds: dict[str, tuple[float, float]] = {}
        for name, bounds in coordinate_ranges.items():
            coordinate = coordinates.get(name)
            allowed = (
                math.degrees(coordinate.getRangeMin()),
                math.degrees(coordinate.getRangeMax()),
            )
            if bounds[0] < allowed[0]:
                coordinate.setRangeMin(math.radians(bounds[0]))
            if bounds[1] > allowed[1]:
                coordinate.setRangeMax(math.radians(bounds[1]))
            coordinate_bounds[name] = bounds
        neck_coordinate = coordinates.get("neck_base_flexion")
        neck_range = coordinate_bounds.get("neck_base_flexion")
        if neck_range is None:
            neck_range = (
                math.degrees(neck_coordinate.getRangeMin()),
                math.degrees(neck_coordinate.getRangeMax()),
            )
        neck_scale = max(abs(neck_range[0]), abs(neck_range[1]), 1.0)

        parameter_specs: list[tuple[str, float, float, float]] = []
        screen_values = {
            "screen_horizontal_offset_mm": screen.horizontal_offset_mm,
            "screen_vertical_offset_mm": screen.vertical_offset_mm,
            "screen_inclination_deg": screen.inclination_deg,
        }
        for name, bounds in screen_ranges.items():
            parameter_specs.append((name, screen_values[name], bounds[0], bounds[1]))
        for name, bounds in coordinate_bounds.items():
            parameter_specs.append((name, coordinate_state[name], bounds[0], bounds[1]))
        free = [
            index
            for index, (_, _, low, high) in enumerate(parameter_specs)
            if high > low
        ]
        baseline = np.array([item[1] for item in parameter_specs], dtype=float)
        lower = np.array([item[2] for item in parameter_specs], dtype=float)
        upper = np.array([item[3] for item in parameter_specs], dtype=float)
        start = np.clip(baseline, lower, upper)

        def unpack(free_values: np.ndarray) -> np.ndarray:
            values = start.copy()
            values[free] = free_values
            return values

        def apply(values: np.ndarray) -> None:
            for index, (name, _, _, _) in enumerate(parameter_specs):
                if name in coordinate_bounds:
                    coordinates.get(name).setValue(
                        state, math.radians(float(values[index])), False
                    )
            user.realizePosition(state)

        screen_indices = {
            name: next(
                index for index, item in enumerate(parameter_specs) if item[0] == name
            )
            for name in screen_ranges
        }

        def screen_position(values: np.ndarray) -> tuple[float, float, float]:
            return tuple(
                float(values[screen_indices[name]])
                for name in (
                    "screen_horizontal_offset_mm",
                    "screen_vertical_offset_mm",
                    "screen_inclination_deg",
                )
            )

        def hard_constraint_values(free_values: np.ndarray) -> np.ndarray:
            values = unpack(free_values)
            apply(values)
            _, gaze_angles = target._gaze_center_coordinates(
                screen_position(values)
            )
            palm_displacements = (
                target._palm_marker_positions() - palm_reference
            ) / 0.01
            return np.concatenate((gaze_angles, palm_displacements.ravel()))

        def gaze_forward_distance(free_values: np.ndarray) -> float:
            values = unpack(free_values)
            apply(values)
            local, _ = target._gaze_center_coordinates(screen_position(values))
            return float(local[0] - 1e-6)

        def restore_coordinates() -> None:
            for name, value in coordinate_state.items():
                coordinates.get(name).setValue(state, value, False)
            user.realizePosition(state)

        def objective(free_values: np.ndarray) -> float:
            values = unpack(free_values)
            apply(values)
            scores = target._optimization_scores(screen_position(values))
            violation = np.minimum(
                target._screen_constraints(screen_position(values)), 0.0
            )
            neck_flexion_deg = math.degrees(
                neck_coordinate.getValue(state)
            )
            neck_score = 1.0 - min(abs(neck_flexion_deg) / neck_scale, 1.0)
            return float(
                -np.dot(screen_weights, scores)
                - neck_weight * neck_score
                + 1000.0 * np.dot(violation, violation)
            )

        def screen_feasibility(free_values: np.ndarray) -> np.ndarray:
            return target._screen_constraints(screen_position(unpack(free_values)))

        hard_constraint_tolerance = 1e-4
        if free:
            initial_free = start[free]
            initial_constraints = hard_constraint_values(initial_free)
            constraint_jacobian = np.zeros((len(initial_constraints), len(free)))
            for column, parameter_index in enumerate(free):
                step = max(
                    (upper[parameter_index] - lower[parameter_index]) * 1e-5,
                    1e-4,
                )
                left = max(initial_free[column] - step, lower[parameter_index])
                right = min(initial_free[column] + step, upper[parameter_index])
                if right <= left:
                    continue
                left_values = initial_free.copy()
                right_values = initial_free.copy()
                left_values[column] = left
                right_values[column] = right
                constraint_jacobian[:, column] = (
                    hard_constraint_values(right_values)
                    - hard_constraint_values(left_values)
                ) / (right - left)

            independent_rows: list[int] = []
            for row in range(len(initial_constraints)):
                candidate_rows = independent_rows + [row]
                rank = np.linalg.matrix_rank(
                    constraint_jacobian[candidate_rows], tol=1e-8
                )
                if rank > len(independent_rows):
                    independent_rows.append(row)
            if (
                not independent_rows
                and np.max(np.abs(initial_constraints))
                > hard_constraint_tolerance
            ):
                restore_coordinates()
                raise ValueError(
                    "no adjustable parameters can align the line of sight "
                    "with the screen centre while preserving palm positions"
                )

            constraints = [
                {"type": "ineq", "fun": screen_feasibility},
                {"type": "ineq", "fun": gaze_forward_distance},
            ]
            if independent_rows:
                constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda values: hard_constraint_values(values)[
                            independent_rows
                        ],
                    }
                )
            result = minimize(
                objective,
                initial_free,
                method="SLSQP",
                bounds=list(zip(lower[free], upper[free])),
                constraints=constraints,
                options={"maxiter": max_iterations, "ftol": 1e-7, "eps": 1e-3},
            )
            if not result.success:
                final_constraints = hard_constraint_values(result.x)
                final_forward_distance = gaze_forward_distance(result.x)
                restore_coordinates()
                if (
                    np.max(np.abs(final_constraints)) > hard_constraint_tolerance
                    or final_forward_distance < 0.0
                ):
                    raise ValueError(
                        "no feasible posture satisfies the screen-centre "
                        "gaze and fixed-palm constraints "
                        f"(optimizer: {result.message}; maximum normalized "
                        f"constraint error: {np.max(np.abs(final_constraints)):g})"
                    )
                if np.min(screen_feasibility(result.x)) < -1e-6:
                    raise ValueError(
                        "no feasible screen position satisfies the handlebar constraints"
                    )
                raise RuntimeError(
                    f"screen optimization did not converge: {result.message}"
                )
            values = unpack(result.x)
            final_constraints = hard_constraint_values(result.x)
            final_forward_distance = gaze_forward_distance(result.x)
            if (
                np.max(np.abs(final_constraints)) > hard_constraint_tolerance
                or final_forward_distance < 0.0
            ):
                restore_coordinates()
                raise ValueError(
                    "optimization did not satisfy the screen-centre gaze "
                    "and fixed-palm constraints"
                )
        else:
            values = start
            final_constraints = hard_constraint_values(np.empty(0))
            final_forward_distance = gaze_forward_distance(np.empty(0))
            if (
                np.max(np.abs(final_constraints)) > hard_constraint_tolerance
                or final_forward_distance < 0.0
            ):
                restore_coordinates()
                raise ValueError(
                    "the fixed parameters do not satisfy the screen-centre "
                    "gaze and fixed-palm constraints"
                )
            if np.min(target._screen_constraints(screen_position(values))) < -1e-6:
                raise ValueError(
                    "the current screen position violates the handlebar constraints"
                )

        apply(values)
        target.equipment.set_screen_position(
            float(values[screen_indices["screen_horizontal_offset_mm"]]),
            float(values[screen_indices["screen_vertical_offset_mm"]]),
            float(values[screen_indices["screen_inclination_deg"]])
        )
        target._reachability_cache.clear()
        target._refresh_visualization()

        if inplace:
            return None
        return target

    def optimize_user_position(
        self,
        *,
        inplace: bool = False,
        lumbar_extension_deg: OptimizationParameter = None,
        right_shoulder_flexion_deg: OptimizationParameter = None,
        right_shoulder_adduction_deg: OptimizationParameter = None,
        right_shoulder_rotation_deg: OptimizationParameter = None,
        right_elbow_flexion_deg: OptimizationParameter = None,
        right_forearm_pronation_deg: OptimizationParameter = None,
        right_wrist_flexion_deg: OptimizationParameter = None,
        right_wrist_deviation_deg: OptimizationParameter = None,
        left_shoulder_flexion_deg: OptimizationParameter = None,
        left_shoulder_adduction_deg: OptimizationParameter = None,
        left_shoulder_rotation_deg: OptimizationParameter = None,
        left_elbow_flexion_deg: OptimizationParameter = None,
        left_forearm_pronation_deg: OptimizationParameter = None,
        left_wrist_flexion_deg: OptimizationParameter = None,
        left_wrist_deviation_deg: OptimizationParameter = None,
        neck_base_flexion_deg: OptimizationParameter = (-45.0, 45.0),
        neck_base_lateral_flexion_deg: OptimizationParameter = None,
        neck_lateral_flexion_deg: OptimizationParameter = None,
        neck_rotation_deg: OptimizationParameter = (-30.0, 30.0),
        neck_flexion_deg: OptimizationParameter = (-45.0, 45.0),
        max_iterations: int = 100,
    ) -> Self | None:
        """Optimize the user's posture in ordered reach, grip, and gaze stages.

        First, lumbar flexion advances toward the most-forward allowed posture
        until the screen is inside either fully-extended arm's reach sphere, or
        the lumbar bound is reached. Next, the arm coordinates are solved to
        restore both palm markers to their original positions. Finally, the
        neck coordinates are solved geometrically so the configured line of
        sight passes through the screen centre. The screen is kept fixed.
        Neck alignment uses the pair of free coordinates with the most
        independent effects on gaze; any other free neck coordinates stay at
        their current values.

        Each upper-body joint coordinate and lumbar flexion/extension is
        exposed as a named argument. ``None`` keeps its current value, a number
        fixes it, and a tuple sets its inclusive optimization range. By
        default lumbar and arm coordinates remain fixed while neck-base
        flexion and neck flexion are optimized over ``(-45, 45)`` degrees,
        and neck rotation over ``(-30, 30)`` degrees.

        Parameters
        ----------
        inplace : bool, default=False
            If true, modify this framework and return ``None``; otherwise
            optimize independent model copies and return the new framework.
        *_deg : float, tuple, or None
            Named upper-body and lumbar flexion/extension joint coordinates
            in degrees. ``None`` leaves a coordinate unchanged, a float fixes
            its value, and a tuple sets its inclusive optimization range.
            Neck-base flexion and neck flexion default to ``(-45, 45)``
            degrees, and neck rotation defaults to ``(-30, 30)`` degrees;
            other coordinates default to ``None``. The lumbar range bounds
            the reachability-first trunk stage; arm ranges bound palm
            restoration; neck ranges bound gaze alignment.
        max_iterations : int, default=100
            Maximum iterations for arm inverse kinematics and neck alignment.

        Returns
        -------
        VisualErgonomicsFramework or None
            The optimized framework when ``inplace=False``; ``None`` when
            ``inplace=True``.

        Raises
        ------
        ValueError
            If the requested bounds are invalid, the palms cannot be restored
            within the arm ranges, or the screen centre cannot be aligned with
            the line of sight within the neck ranges.
        RuntimeError
            If either inverse-kinematics solve fails to converge.
        """
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be bool")
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")

        coordinate_parameters = {
            "lumbar_extension": lumbar_extension_deg,
            "arm_flex_r": right_shoulder_flexion_deg,
            "arm_add_r": right_shoulder_adduction_deg,
            "arm_rot_r": right_shoulder_rotation_deg,
            "elbow_flex_r": right_elbow_flexion_deg,
            "pro_sup_r": right_forearm_pronation_deg,
            "wrist_flex_r": right_wrist_flexion_deg,
            "wrist_dev_r": right_wrist_deviation_deg,
            "arm_flex_l": left_shoulder_flexion_deg,
            "arm_add_l": left_shoulder_adduction_deg,
            "arm_rot_l": left_shoulder_rotation_deg,
            "elbow_flex_l": left_elbow_flexion_deg,
            "pro_sup_l": left_forearm_pronation_deg,
            "wrist_flex_l": left_wrist_flexion_deg,
            "wrist_dev_l": left_wrist_deviation_deg,
            "neck_base_flexion": neck_base_flexion_deg,
            "neck_base_lateral_flexion": neck_base_lateral_flexion_deg,
            "neck_lateral_flexion": neck_lateral_flexion_deg,
            "neck_rotation": neck_rotation_deg,
            "neck_flexion": neck_flexion_deg,
        }
        arm_names = (
            "arm_flex_r",
            "arm_add_r",
            "arm_rot_r",
            "elbow_flex_r",
            "pro_sup_r",
            "wrist_flex_r",
            "wrist_dev_r",
            "arm_flex_l",
            "arm_add_l",
            "arm_rot_l",
            "elbow_flex_l",
            "pro_sup_l",
            "wrist_flex_l",
            "wrist_dev_l",
        )
        neck_names = (
            "neck_base_flexion",
            "neck_base_lateral_flexion",
            "neck_lateral_flexion",
            "neck_rotation",
            "neck_flexion",
        )

        target: VisualErgonomicsFramework
        if inplace:
            target = self
        else:
            cloned_user, cloned_bike = self._clone_models()
            target = type(self)(
                cloned_user,
                cloned_bike,
                self.cone_width_angle_deg,
                self.cone_height_angle_deg,
                self.line_of_sight_offset_deg,
            )
        user = target.user
        state = target._user_state()
        coordinates = user.getCoordinateSet()
        original_values = {
            coordinates.get(index).getName(): coordinates.get(index).getValue(state)
            for index in range(coordinates.getSize())
        }
        coordinate_bounds: dict[str, tuple[float, float]] = {}
        for name, value in coordinate_parameters.items():
            coordinate = coordinates.get(name)
            current = math.degrees(coordinate.getValue(state))
            if value is None:
                bounds = (current, current)
            else:
                allowed = (
                    math.degrees(coordinate.getRangeMin()),
                    math.degrees(coordinate.getRangeMax()),
                )
                bounds = target._parameter_bounds(name, value, current, allowed)
                if bounds[0] < allowed[0]:
                    coordinate.setRangeMin(math.radians(bounds[0]))
                if bounds[1] > allowed[1]:
                    coordinate.setRangeMax(math.radians(bounds[1]))
            coordinate_bounds[name] = bounds

        palm_reference = target._palm_marker_positions()
        screen_position = (
            target.equipment.screen_horizontal_offset_mm,
            target.equipment.screen_vertical_offset_mm,
            target.equipment.screen_inclination_deg,
        )
        if np.min(target._screen_constraints(screen_position)) < -1e-6:
            raise ValueError("the current screen position violates the handlebar constraints")

        completed = False
        try:
            lumbar = coordinates.get("lumbar_extension")
            lumbar_low, lumbar_high = coordinate_bounds["lumbar_extension"]
            current_lumbar = math.degrees(lumbar.getValue(state))
            lumbar_candidates = np.unique(
                np.concatenate(
                    (
                        np.linspace(
                            lumbar_low,
                            lumbar_high,
                            max(41, min(201, 2 * max_iterations + 1)),
                        ),
                        [float(np.clip(current_lumbar, lumbar_low, lumbar_high))],
                    )
                )
            )

            reach_radii: dict[str, float] = {}
            for side in ("r", "l"):
                _, reach_radii[side] = target._reachability_sphere(side)

            def forward_trunk_angle() -> float:
                centers = user.joint_centers
                hips = (
                    np.asarray(centers["right_hip"])
                    + np.asarray(centers["left_hip"])
                ) / 2
                shoulders = (
                    np.asarray(centers["right_shoulder"])
                    + np.asarray(centers["left_shoulder"])
                ) / 2
                line = shoulders - hips
                return math.atan2(float(line[0]), float(line[1]))

            initial_forward_angle = forward_trunk_angle()
            lumbar_angles = []
            for value in lumbar_candidates:
                lumbar.setValue(state, math.radians(float(value)), False)
                user.realizePosition(state)
                lumbar_angles.append((forward_trunk_angle(), float(value)))

            screen_corners = target._screen_points()

            def screen_fits_reach_sphere() -> bool:
                for side in ("r", "l"):
                    shoulder = _position_in_ground(
                        user.getJointSet().get(f"acromial_{side}").getChildFrame(),
                        state,
                    )
                    if np.all(
                        np.linalg.norm(screen_corners - shoulder, axis=1)
                        <= reach_radii[side] + 1e-8
                    ):
                        return True
                return False

            most_flexed = max(lumbar_angles, key=lambda item: item[0])
            selected_lumbar = most_flexed[1]
            for angle, value in sorted(lumbar_angles):
                if angle < initial_forward_angle - 1e-8:
                    continue
                lumbar.setValue(state, math.radians(value), False)
                user.realizePosition(state)
                if screen_fits_reach_sphere():
                    selected_lumbar = value
                    break
            lumbar.setValue(state, math.radians(selected_lumbar), False)
            user.realizePosition(state)

            arm_free_names = []
            arm_initial = []
            arm_lower = []
            arm_upper = []
            for name in arm_names:
                low, high = coordinate_bounds[name]
                coordinate = coordinates.get(name)
                current = math.degrees(coordinate.getValue(state))
                if high > low:
                    start = float(np.clip(current, low, high))
                    arm_free_names.append(name)
                    arm_initial.append(start)
                    arm_lower.append(low)
                    arm_upper.append(high)
                else:
                    coordinate.setValue(state, math.radians(low), False)
            user.realizePosition(state)
            arm_initial_array = np.asarray(arm_initial, dtype=float)

            def apply_arm_values(values: np.ndarray) -> None:
                for name, value in zip(arm_free_names, values):
                    coordinates.get(name).setValue(
                        state, math.radians(float(value)), False
                    )
                user.realizePosition(state)

            def palm_residual(values: np.ndarray) -> np.ndarray:
                apply_arm_values(values)
                return (
                    target._palm_marker_positions() - palm_reference
                ).ravel() / 0.01

            if arm_free_names:
                arm_scale = np.maximum(
                    np.asarray(arm_upper) - np.asarray(arm_lower), 1.0
                )
                arm_result = minimize(
                    lambda values: float(
                        np.sum(((values - arm_initial_array) / arm_scale) ** 2)
                    ),
                    arm_initial_array,
                    method="SLSQP",
                    bounds=list(zip(arm_lower, arm_upper)),
                    constraints=[{"type": "eq", "fun": palm_residual}],
                    options={"maxiter": max_iterations, "ftol": 1e-10, "eps": 1e-3},
                )
                final_palm_residual = palm_residual(arm_result.x)
                if np.max(np.abs(final_palm_residual)) > 1e-4:
                    raise ValueError(
                        "no arm posture within the requested ranges can restore "
                        "the original palm positions"
                    )
                if not arm_result.success:
                    raise RuntimeError(
                        f"palm-restoration optimization did not converge: "
                        f"{arm_result.message}"
                    )
            else:
                final_palm_residual = palm_residual(np.empty(0))
                if np.max(np.abs(final_palm_residual)) > 1e-4:
                    raise ValueError(
                        "the fixed arm coordinates cannot restore the original "
                        "palm positions"
                    )

            neck_free_names = []
            neck_initial: dict[str, float] = {}
            for name in neck_names:
                low, high = coordinate_bounds[name]
                coordinate = coordinates.get(name)
                current = math.degrees(coordinate.getValue(state))
                if high > low:
                    start = float(np.clip(current, low, high))
                    neck_free_names.append(name)
                    neck_initial[name] = start
                    coordinate.setValue(state, math.radians(start), False)
                else:
                    coordinate.setValue(state, math.radians(low), False)
            user.realizePosition(state)

            def gaze_residual() -> np.ndarray:
                user.realizePosition(state)
                _, angles = target._gaze_center_coordinates(screen_position)
                return angles

            if len(neck_free_names) >= 2:
                neck_sensitivities: dict[str, np.ndarray] = {}
                for name in neck_free_names:
                    low, high = coordinate_bounds[name]
                    current = neck_initial[name]
                    step = min(0.1, (high - low) / 2)
                    lower_sample = max(low, current - step)
                    upper_sample = min(high, current + step)
                    if upper_sample <= lower_sample:
                        continue
                    coordinate = coordinates.get(name)
                    coordinate.setValue(
                        state, math.radians(lower_sample), False
                    )
                    lower_residual = gaze_residual()
                    coordinate.setValue(
                        state, math.radians(upper_sample), False
                    )
                    upper_residual = gaze_residual()
                    coordinate.setValue(
                        state, math.radians(current), False
                    )
                    neck_sensitivities[name] = (
                        upper_residual - lower_residual
                    ) / (upper_sample - lower_sample)

                neck_pairs = [
                    (first, second)
                    for index, first in enumerate(neck_free_names)
                    for second in neck_free_names[index + 1 :]
                    if first in neck_sensitivities
                    and second in neck_sensitivities
                ]
                if not neck_pairs:
                    raise ValueError(
                        "the requested neck ranges do not allow gaze alignment"
                    )

                neck_pair = max(
                    neck_pairs,
                    key=lambda pair: abs(
                        np.linalg.det(
                            np.column_stack(
                                (
                                    neck_sensitivities[pair[0]],
                                    neck_sensitivities[pair[1]],
                                )
                            )
                        )
                    ),
                )
                pair_sensitivity = np.column_stack(
                    (
                        neck_sensitivities[neck_pair[0]],
                        neck_sensitivities[neck_pair[1]],
                    )
                )
                if abs(float(np.linalg.det(pair_sensitivity))) < 1e-10:
                    raise ValueError(
                        "the requested neck ranges do not provide independent "
                        "gaze adjustments"
                    )

                neck_initial_pair = np.asarray(
                    [neck_initial[name] for name in neck_pair], dtype=float
                )

                def apply_neck_pair(values: np.ndarray) -> None:
                    for name, value in zip(neck_pair, values):
                        coordinates.get(name).setValue(
                            state, math.radians(float(value)), False
                        )
                    user.realizePosition(state)

                def neck_equations(values: np.ndarray) -> np.ndarray:
                    apply_neck_pair(values)
                    return gaze_residual()

                neck_result = root(
                    neck_equations,
                    neck_initial_pair,
                    options={"maxfev": max_iterations * 10, "xtol": 1e-10},
                )
                apply_neck_pair(neck_result.x)
                final_gaze_residual = gaze_residual()
                outside_neck_bounds = any(
                    value < coordinate_bounds[name][0] - 1e-7
                    or value > coordinate_bounds[name][1] + 1e-7
                    for name, value in zip(neck_pair, neck_result.x)
                )
                if outside_neck_bounds or np.max(np.abs(final_gaze_residual)) > 1e-4:
                    raise ValueError(
                        "the requested neck ranges cannot align the line of "
                        "sight with the screen centre"
                    )
                if not neck_result.success and np.max(
                    np.abs(final_gaze_residual)
                ) > 1e-8:
                    raise RuntimeError(
                        f"geometric gaze alignment did not converge: "
                        f"{neck_result.message}"
                    )
            elif len(neck_free_names) == 1:
                name = neck_free_names[0]
                low, high = coordinate_bounds[name]
                coordinate = coordinates.get(name)

                def single_coordinate_residual(value: float) -> np.ndarray:
                    coordinate.setValue(state, math.radians(value), False)
                    return gaze_residual()

                lower_residual = single_coordinate_residual(low)
                upper_residual = single_coordinate_residual(high)
                component = int(
                    np.argmax(np.abs(upper_residual - lower_residual))
                )
                lower_value = float(lower_residual[component])
                upper_value = float(upper_residual[component])
                if lower_value * upper_value > 0.0:
                    raise ValueError(
                        "the requested neck range cannot align the line of "
                        "sight with the screen centre"
                    )
                neck_result = root_scalar(
                    lambda value: float(
                        single_coordinate_residual(value)[component]
                    ),
                    bracket=(low, high),
                    method="brentq",
                    xtol=1e-10,
                    maxiter=max_iterations,
                )
                final_gaze_residual = single_coordinate_residual(
                    float(neck_result.root)
                )
                if np.max(np.abs(final_gaze_residual)) > 1e-4:
                    raise ValueError(
                        "the requested neck range cannot align the line of "
                        "sight with the screen centre"
                    )
            else:
                final_gaze_residual = gaze_residual()
                if np.max(np.abs(final_gaze_residual)) > 1e-4:
                    raise ValueError(
                        "the fixed neck coordinates cannot align the line of "
                        "sight with the screen centre"
                    )

            target._refresh_visualization()
            target.get_view_state()
            completed = True
            if inplace:
                return None
            return target
        finally:
            if not completed and inplace:
                for name, value in original_values.items():
                    coordinates.get(name).setValue(state, value, False)
                user.realizePosition(state)
                target._reachability_cache.clear()

    def _refresh_visualization(self) -> None:
        """Rebuild the visual cone and palm-marker reachability surfaces.

        The cone extends downward from the head plane through the eye midpoint;
        its central axis is positioned independently by the configured
        line-of-sight offset. Each reach sphere is centred on a shoulder joint
        and uses the same radius as the quantitative screen-area analysis.
        """
        user_state = self._user_state()
        eye, cone_rotation = self._eye_center_and_cone_rotation()
        _, line_of_sight_rotation = self._eye_center_and_line_of_sight_rotation()
        screen_corners = self._screen_points()
        forward_distance = max(
            0.5,
            float(np.max((screen_corners - eye) @ cone_rotation[:, 0])) + 0.15,
        )
        pyramid_points, pyramid_faces = _pyramid_mesh(
            self.cone_width_angle_deg,
            self.cone_height_angle_deg,
            forward_distance,
        )
        directory = Path(self._geometry_directory.name)
        pyramid_path = directory / "vision_cone.vtp"
        _write_polydata(pyramid_path, pyramid_points, pyramid_faces)

        overlay = _OPENSIM.Model()
        overlay.setName("visual_ergonomics")
        cone = _MeshShape(pyramid_path, color=(0.2, 0.65, 0.95), opacity=0.18)
        add_welded_body(
            overlay,
            "vision_cone",
            [cone],
            _MESH_DENSITY,
            location=eye,
            orientation=matrix_to_euler_xyz(cone_rotation),
        )
        axis_center = eye + line_of_sight_rotation[:, 0] * forward_distance / 2
        axis = Cylinder(
            radius=0.0015,
            length=forward_distance,
            axis="x",
            color=(1.0, 0.8, 0.0),
        )
        add_welded_body(
            overlay,
            "vision_cone_axis",
            [axis],
            _MESH_DENSITY,
            location=axis_center,
            orientation=matrix_to_euler_xyz(line_of_sight_rotation),
        )

        for side, color in (
            ("r", (0.95, 0.3, 0.25)),
            ("l", (0.3, 0.8, 0.45)),
        ):
            shoulder, radius = self._reachability_sphere(side)
            points, faces = _sphere_mesh(radius)
            mesh_path = directory / f"reach_{side}.vtp"
            _write_polydata(mesh_path, points, faces)
            shape = _MeshShape(mesh_path, color=color, opacity=0.12)
            add_welded_body(
                overlay,
                f"reach_{side}",
                [shape],
                _MESH_DENSITY,
                location=shoulder,
            )
        overlay.finalizeConnections()
        overlay.geometry_directories = (directory,)
        overlay.initSystem()
        self["visualization"] = overlay

    def show(self, motion: Any = None, *, loop: bool = False, fps: float = 30.0) -> Any:
        """Refresh reachability overlays and display the composed models.

        Parameters
        ----------
        motion : object or None, optional
            Optional motion data or file path accepted by the viewer.
        loop : bool, default=False
            Whether motion playback repeats after its final sample.
        fps : float, default=30.0
            Playback and rendering frame rate.

        Returns
        -------
        object
            Viewer window returned by :meth:`Framework.show`.
        """
        self._reachability_cache.clear()
        self._refresh_visualization()
        return super().show(motion, loop=loop, fps=fps)

    def copy(self) -> Self:
        """Create a shallow framework copy that shares its user and bike models.

        Returns
        -------
        VisualErgonomicsFramework
            New framework instance with the same model objects and visual-cone
            angle settings. Use :meth:`__deepcopy__` to copy the models too.
        """
        result = type(self)(
            self.user,
            self.equipment,
            self.cone_width_angle_deg,
            self.cone_height_angle_deg,
            self.line_of_sight_offset_deg,
        )
        return result

    def __copy__(self) -> Self:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        user, equipment = self._clone_models()
        result = type(self)(
            user,
            equipment,
            self.cone_width_angle_deg,
            self.cone_height_angle_deg,
            self.line_of_sight_offset_deg,
        )
        memo[id(self)] = result
        return result
