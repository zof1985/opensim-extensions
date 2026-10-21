"""Visual ergonomics analysis and optimization for a user and equipment model."""

from __future__ import annotations

import math
import tempfile
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Self

import numpy as np
from scipy.optimize import minimize
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
_REACH_COORDINATES = (
    "arm_flex",
    "arm_add",
    "arm_rot",
    "pro_sup",
    "wrist_flex",
    "wrist_dev",
)
_REACH_FIXED_COORDINATES = ("elbow_flex",)


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
    drop = length * math.tan(math.radians(height_angle))
    points = np.array(
        [
            (0.0, 0.0, 0.0),
            (length, -drop, -half_width),
            (length, -drop, half_width),
            (length, 0.0, half_width),
            (length, 0.0, -half_width),
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
    30-degree downward extent. The vertical extent represents the preferred
    display zone from eye level to 30 degrees below it, centred on the
    approximately 15-degree downward line of sight recommended for display
    work. The literature does not prescribe a matching numeric horizontal
    cone width; 30 degrees total (15 degrees to either side) is a conservative
    symmetric modelling choice, not a separate standard limit.

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
            Maximum downward angle from horizontal, in degrees, representing
            the preferred vertical display zone centred about 15 degrees
            below eye level.

        Raises
        ------
        TypeError
            If ``user`` is not an OpenSim model or ``equipment`` is not an
            :class:`Equipment`.
        ValueError
            If the equipment has no screen, or either cone angle is not finite
            or falls outside its supported interval.

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
        object.__setattr__(self, "cone_width_angle_deg", float(cone_width_angle_deg))
        object.__setattr__(self, "cone_height_angle_deg", float(cone_height_angle_deg))
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

    def _screen_points(self, position: Sequence[float] | None = None) -> np.ndarray:
        candidate = None if position is None else tuple(map(float, position))
        return self.screen.corners(candidate)

    def _screen_samples(self, position: Sequence[float] | None = None) -> np.ndarray:
        candidate = None if position is None else tuple(map(float, position))
        return self.screen.samples(_SCREEN_GRID_SIZE, candidate)

    def _metrics(self, position: Sequence[float] | None = None) -> dict[str, float]:
        """Evaluate incidence, visual coverage, and reachable screen coverage.

        Incidence is computed from the eye midpoint, screen centre, and screen
        normal. Visibility uses the rectangular visual cone in head-local
        coordinates. Visibility and reachability are reported as the
        proportions of the uniform screen samples that satisfy their
        respective predicates.

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
        eye, head_rotation = self._eye_center_and_head_rotation()
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
        local = (points - eye) @ head_rotation
        forward = local[:, 0]
        horizontal_angle = np.arctan2(np.abs(local[:, 2]), np.maximum(forward, 1e-12))
        downward_angle = np.arctan2(-local[:, 1], np.maximum(forward, 1e-12))
        visible = (
            (forward > 0.0)
            & (horizontal_angle <= math.radians(self.cone_width_angle_deg / 2))
            & (downward_angle >= 0.0)
            & (downward_angle <= math.radians(self.cone_height_angle_deg))
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
        and signed distances from each palm reachability hull are converted
        through logistic functions, giving the optimizer smooth approximations
        to the visible and reachable sample fractions.

        Parameters
        ----------
        position : sequence of float
            Candidate screen position ``(horizontal_offset_mm,
            vertical_offset_mm, inclination_deg)``.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Scores ordered as incidence, visibility, and reachability. Each
            component is maximized by the screen-position optimizer.
        """
        metrics = self._metrics(position)
        eye, head_rotation = self._eye_center_and_head_rotation()
        points = self._screen_samples(position)
        local = (points - eye) @ head_rotation
        forward = local[:, 0]
        horizontal = np.arctan2(np.abs(local[:, 2]), np.maximum(forward, 1e-12))
        downward = np.arctan2(-local[:, 1], np.maximum(forward, 1e-12))
        angular_scale = math.radians(2.0)
        cone_margin = np.minimum.reduce(
            (
                forward / 0.02,
                (math.radians(self.cone_width_angle_deg / 2) - horizontal)
                / angular_scale,
                downward / angular_scale,
                (math.radians(self.cone_height_angle_deg) - downward) / angular_scale,
            )
        )
        visible_score = float(np.mean(expit(cone_margin)))

        reachable_probabilities = []
        for side in ("r", "l"):
            _, equations, _ = self._reachability_hull(side)
            if equations.size:
                violation = np.max(
                    points @ equations[:, :3].T + equations[:, 3], axis=1
                )
                reachable_probabilities.append(expit(-violation / 0.01))
            else:
                reachable_probabilities.append(np.zeros(len(points)))
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

        Shoulder flexion/extension, shoulder adduction/abduction, forearm, and
        wrist coordinates are sampled across their effective OpenSim
        coordinate ranges. In addition to deterministic combinations of all
        range endpoints, seeded random samples cover the interior. The elbow
        is fixed at its minimum coordinate value, which represents full
        extension in the bundled model. Trunk and pelvis coordinates remain
        at the current posture. The method restores all sampled coordinates
        after realization and caches the points, hull plane equations, and
        triangular surface facets by posture, anthropometric dimensions, and
        coordinate ranges.

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
        trunk_names = (
            "lumbar_extension",
            "lumbar_bending",
            "lumbar_rotation",
            "pelvis_tx",
            "pelvis_ty",
            "pelvis_tz",
        )
        key = tuple(
            round(coordinates.get(name).getValue(state), 3) for name in trunk_names
        ) + (
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
        lower = np.array([coordinate.getRangeMin() for coordinate in coordinate_values])
        upper = np.array([coordinate.getRangeMax() for coordinate in coordinate_values])
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
        if cached is not None:
            return cached

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
        boundary_samples = np.asarray(tuple(product(*zip(lower, upper))), dtype=float)
        samples = np.vstack((random_samples, boundary_samples))
        marker = self.user.getMarkerSet().get("RPALM" if side == "r" else "LPALM")
        endpoints = []
        try:
            for values in samples:
                for coordinate, value in zip(coordinate_values, values):
                    coordinate.setValue(state, float(value), False)
                for coordinate in fixed_coordinates:
                    coordinate.setValue(state, coordinate.getRangeMin(), False)
                self.user.realizePosition(state)
                endpoints.append(self._vector(marker.getLocationInGround(state)))
        finally:
            for coordinate, value in zip(coordinate_values, original_values):
                coordinate.setValue(state, value, False)
            for coordinate, value in zip(fixed_coordinates, original_fixed_values):
                coordinate.setValue(state, value, False)
            self.user.realizePosition(state)
        points = np.asarray(endpoints)
        try:
            hull = ConvexHull(points)
            equations = hull.equations.copy()
            result = (points, equations, hull.simplices.copy())
        except QhullError:
            result = (points, np.empty((0, 4)), np.empty((0, 3), dtype=int))
        if len(self._reachability_cache) > 24:
            self._reachability_cache.clear()
        self._reachability_cache[key] = result
        return result

    def _reachable_screen_samples(self, screen_points: np.ndarray) -> np.ndarray:
        results = []
        for side in ("r", "l"):
            _, equations, _ = self._reachability_hull(side)
            if equations.size:
                distances = screen_points @ equations[:, :3].T + equations[:, 3]
                results.append(np.all(distances <= 1e-8, axis=1))
            else:
                results.append(np.zeros(len(screen_points), dtype=bool))
        return np.logical_or.reduce(results)

    def get_view_state(self) -> dict[str, float]:
        """Measure screen incidence, visual coverage, and hand reachability.

        Incidence is the angle between the screen normal and the line from the
        midpoint of the eye markers to the screen centre. Visibility and
        reachability percentages are sampled on a uniform screen grid.
        Reachability is the union of the convex hulls of the palm-centre
        marker positions sampled over shoulder, forearm, and wrist motion.
        The elbow is held at its full-extension limit, independently of its
        current grip posture or any optimization coordinate bounds; trunk and
        pelvis coordinates remain at their current values.

        Returns
        -------
        dict[str, float]
            Mapping with ``incidence_angle_deg`` (degrees),
            ``observed_screen_percent`` (visible grid fraction), and
            ``reachable_screen_percent`` (fraction inside either palm-marker
            hull). Percentages are in ``[0, 100]``.
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
                raise ValueError(
                    f"{name} range {low, high} exceeds model limits {allowed}"
                )
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
        screen_incline_range_deg: Sequence[float] = (10.0, 90.0),
        screen_horizontal_offset_range_mm: Sequence[float] | None = None,
        screen_vertical_offset_range_mm: Sequence[float] | None = None,
        coordinate_ranges_deg: Mapping[str, Sequence[float]] | None = None,
        max_iterations: int = 100,
    ) -> Self | None:
        """Optimize screen position and selected rider coordinates.

        The objective combines screen incidence, visibility within the visual
        cone, and palm-marker reachability. A bounded numerical optimization
        evaluates the screen on a regular grid; the palm reachability objective
        uses the extended-arm hulls independently of any fitted grip posture.
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
        screen_incline_range_deg : sequence of float, default=(10, 90)
            Inclusive bounds for screen inclination in degrees.
        screen_horizontal_offset_range_mm : sequence of float or None, optional
            Bounds for the screen's forward offset from its equipment
            reference point, in millimetres. If omitted, a 200 mm interval
            around its current value is used, clipped to any configured lower
            limit.
        screen_vertical_offset_range_mm : sequence of float or None, optional
            Bounds for the screen's vertical offset from its equipment
            reference point, in millimetres. If omitted, a 200 mm interval
            around its current value is used, clipped to any configured lower
            limit.
        coordinate_ranges_deg : mapping of str to sequence of float, optional
            Additional coordinate-name to ``(minimum, maximum)`` bounds, in
            degrees, for user coordinates not covered by the dedicated range
            arguments.
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
            If weights are invalid, ranges exceed model limits, screen bounds
            are invalid, the screen is unavailable, or no feasible position
            exists.
        """
        weights = np.array(
            [incidence_weight, observed_screen_weight, reachability_weight],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.any(weights > 1.0)
            or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("objective weights must be in [0, 1] and sum to 1")
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
            )

        coordinate_ranges = target._coordinate_ranges(
            coordinate_ranges_deg,
            neck_flexionextension_range_deg,
            head_extensionflexion_range_deg,
            trunk_flexionextension_range_deg,
            shoulder_flexion_range_deg,
            elbow_flexion_range_deg,
        )
        screen = target.screen
        screen_ranges = {
            "screen_horizontal_offset_mm": (
                screen_horizontal_offset_range_mm
                if screen_horizontal_offset_range_mm is not None
                else (
                    max(
                        screen.horizontal_offset_min_mm
                        if screen.horizontal_offset_min_mm is not None
                        else -math.inf,
                        screen.horizontal_offset_mm - 100.0,
                    ),
                    screen.horizontal_offset_mm + 100.0,
                )
            ),
            "screen_vertical_offset_mm": (
                screen_vertical_offset_range_mm
                if screen_vertical_offset_range_mm is not None
                else (
                    max(
                        screen.vertical_offset_min_mm
                        if screen.vertical_offset_min_mm is not None
                        else -math.inf,
                        screen.vertical_offset_mm - 100.0,
                    ),
                    screen.vertical_offset_mm + 100.0,
                )
            ),
            "screen_inclination_deg": screen_incline_range_deg,
        }
        for name, bounds in screen_ranges.items():
            screen_ranges[name] = target._range(name, bounds)
        if (
            screen.horizontal_offset_min_mm is not None
            and screen_ranges["screen_horizontal_offset_mm"][0]
            < screen.horizontal_offset_min_mm
        ):
            raise ValueError("screen horizontal offset is below the equipment limit")
        if (
            screen.vertical_offset_min_mm is not None
            and screen_ranges["screen_vertical_offset_mm"][0]
            < screen.vertical_offset_min_mm
        ):
            raise ValueError("screen vertical offset is below the equipment limit")

        user = target.user
        state = target._user_state()
        coordinates = user.getCoordinateSet()
        coordinate_state = {
            name: coordinates.get(name).getValue(state) for name in coordinate_ranges
        }
        coordinate_bounds: dict[str, tuple[float, float]] = {}
        for name, bounds in coordinate_ranges.items():
            coordinate = coordinates.get(name)
            allowed = (
                math.degrees(coordinate.getRangeMin()),
                math.degrees(coordinate.getRangeMax()),
            )
            coordinate_bounds[name] = (
                max(bounds[0], allowed[0]),
                min(bounds[1], allowed[1]),
            )
            if coordinate_bounds[name][0] > coordinate_bounds[name][1]:
                raise ValueError(f"{name} range does not overlap model limits")

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

        def objective(free_values: np.ndarray) -> float:
            values = unpack(free_values)
            apply(values)
            scores = target._optimization_scores(screen_position(values))
            violation = np.minimum(
                target._screen_constraints(screen_position(values)), 0.0
            )
            return float(
                -np.dot(weights, scores) + 1000.0 * np.dot(violation, violation)
            )

        def screen_feasibility(free_values: np.ndarray) -> np.ndarray:
            return target._screen_constraints(screen_position(unpack(free_values)))

        if free:
            result = minimize(
                objective,
                start[free],
                method="SLSQP",
                bounds=list(zip(lower[free], upper[free])),
                constraints=({"type": "ineq", "fun": screen_feasibility},),
                options={"maxiter": max_iterations, "ftol": 1e-7},
            )
            if not result.success:
                for name, value in coordinate_state.items():
                    coordinates.get(name).setValue(state, value, False)
                user.realizePosition(state)
                if np.min(screen_feasibility(result.x)) < -1e-6:
                    raise ValueError(
                        "no feasible screen position satisfies the handlebar constraints"
                    )
                raise RuntimeError(
                    f"screen optimization did not converge: {result.message}"
                )
            values = unpack(result.x)
        else:
            values = start
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

    def _refresh_visualization(self) -> None:
        """Rebuild the visual cone and palm-marker reachability surfaces.

        The cone and its central axis are generated from the eye midpoint and
        head orientation. For each side, the reach surface reuses the same
        sampled palm-marker hull as the quantitative screen-area analysis,
        transformed into a shoulder-centred body frame for display.
        """
        user_state = self._user_state()
        eye, head_rotation = self._eye_center_and_head_rotation()
        screen_corners = self._screen_points()
        forward_distance = max(
            0.5,
            float(np.max((screen_corners - eye) @ head_rotation[:, 0])) + 0.15,
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
        torso_rotation = _rotation_in_ground(
            self.user.getBodySet().get("torso"), user_state
        )
        cone = _MeshShape(pyramid_path, color=(0.2, 0.65, 0.95), opacity=0.18)
        add_welded_body(
            overlay,
            "vision_cone",
            [cone],
            _MESH_DENSITY,
            location=eye,
            orientation=matrix_to_euler_xyz(head_rotation),
        )
        axis_center = eye + head_rotation[:, 0] * forward_distance / 2
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
            orientation=matrix_to_euler_xyz(head_rotation),
        )

        for side, color in (
            ("r", (0.95, 0.3, 0.25)),
            ("l", (0.3, 0.8, 0.45)),
        ):
            points, equations, faces = self._reachability_hull(side)
            if not equations.size:
                continue
            mesh_path = directory / f"reach_{side}.vtp"
            joint = self.user.getJointSet().get(f"acromial_{side}")
            shoulder = _position_in_ground(joint.getChildFrame(), user_state)
            local_points = (points - shoulder) @ torso_rotation
            _write_polydata(mesh_path, local_points, faces)
            shape = _MeshShape(mesh_path, color=color, opacity=0.12)
            add_welded_body(
                overlay,
                f"reach_{side}",
                [shape],
                _MESH_DENSITY,
                location=shoulder,
                orientation=matrix_to_euler_xyz(torso_rotation),
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
        )
        memo[id(self)] = result
        return result
