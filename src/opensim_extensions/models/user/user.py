from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...bodies.shapes import MESHES_DIR
from ...opensim import import_opensim, register_geometry_search_path
from .data import DEFAULT_DATASET, resolve_reference
from .joint_center_table import JOINT_CENTER_NAMES as _JOINT_CENTER_NAMES
from .joint_centers_generated import _JointCenterMixin
from .mapping import segment_scale_factors
from .measurement_generated import _MeasurementMixin
from .posture_generated import _PostureMixin

__all__ = ["User"]

opensim = import_opensim()

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_MODEL_PATH = _ASSETS_DIR / "rajagopalaiulrich2023.osim"
DEFAULT_MESHES_DIR = _ASSETS_DIR / "meshes"

_PELVIS_TRANSLATION_COORDINATES = ("pelvis_tx", "pelvis_ty", "pelvis_tz")

# Friendly name -> OpenSim marker name. The foot has no 1st-metatarsal
# marker in this model (only the 5th): no landmark is exposed for it rather
# than guessing one.
_FOOT_MARKER_NAMES = {
    "left_heel": "LCAL",
    "right_heel": "RCAL",
    "left_toe": "LTOE",
    "right_toe": "RTOE",
    "left_mt5": "LMT5",
    "right_mt5": "RMT5",
}


def _validate_mass_kg(mass_kg: float) -> None:
    """Raise ``ValueError`` unless ``mass_kg`` is a finite, strictly positive number.

    Shared by the constructor's ``mass_kg`` parameter and
    :meth:`User.set_mass_kg`, so both reject the same values with the same
    message.
    """
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass_kg must be a positive finite value in kilograms")


def _axis_scale_factors(
    ratio: float, axes: tuple[str, ...]
) -> tuple[float, float, float]:
    return (
        ratio if "x" in axes else 1.0,
        ratio if "y" in axes else 1.0,
        ratio if "z" in axes else 1.0,
    )


def _remove_segment_mass(body: Any, mass: float, center: np.ndarray) -> None:
    """Split a rigid segment's mass into a new body while preserving its first moment."""
    original_mass = body.getMass()
    remaining_mass = original_mass - mass
    if remaining_mass <= 0:
        raise ValueError(f"cannot split {mass:g} kg from body {body.getName()!r}")
    original_center = np.array([body.getMassCenter().get(i) for i in range(3)])
    remaining_center = (original_mass * original_center - mass * center) / remaining_mass
    body.scaleMass(remaining_mass / original_mass)
    body.setMassCenter(opensim.Vec3(*remaining_center))


class User(opensim.Model, _MeasurementMixin, _PostureMixin, _JointCenterMixin):
    """An ANSUR-based anthropometric user backed by an OpenSim model.

    Built from the Rajagopal-Lai-Uhlrich full-body OpenSim model
    (``rajagopalaiulrich2023.osim``, bundled under ``assets/``), scaled
    per-body to the requested anthropometry using ANSUR II reference data
    (see "Scaling antropometrico" in the package README), then calibrated
    so the model's total mass matches a real anthropometric mass reference
    (``mass_kg``, if given, otherwise the resolved ANSUR ``weight_kg`` --
    see :attr:`mass_kg`) rather than whatever the geometric scaling alone
    produces volumetrically; unlike every other anthropometric parameter
    here, total mass can also be recalibrated after construction, via
    :meth:`set_mass_kg`. Every joint angle is exposed through a
    ``set_<joint>_<motion>``/``<joint>_<motion>`` setter/getter pair.
    Anthropometric dimensions have explicit measurement setters and
    properties. Bilateral setters (for example ``set_left_arm_length`` and
    ``set_right_arm_length``) scale only their corresponding side; ANSUR's
    shared measurements provide equal initial values until one side is
    overridden. Each hand body is connected to its radius by a universal
    wrist joint with independent flexion/extension and radial/ulnar-deviation
    coordinates. ``RPALM`` and ``LPALM`` are hand-attached markers at the
    anthropometric palm centres, repositioned after body scaling.

    Parameters
    ----------
    gender : str | Literal['M', 'F']
        Sex code used to select the ANSUR II reference population. Must be
        exactly ``"M"`` or ``"F"`` (case-sensitive, no other values
        accepted).
    height_cm : float or None, optional
        Requested stature, in centimetres. If provided, every anthropometric
        measurement (stature included) is resolved directly from this
        height via a per-measurement PCHIP regression against the ANSUR
        subjects of the requested ``gender``, and takes precedence over
        ``percentile`` (which is still computed afterwards, purely for
        information -- see :attr:`percentile`). A height outside the ANSUR
        range observed for that gender is not rejected: it is extrapolated
        (PCHIP with linear tails) and raises a :class:`UserWarning`, with
        extrapolation reliability dropping the further the requested height
        is from the observed range.
    percentile : float, optional
        Common ANSUR percentile applied to every numeric measurement when
        ``height_cm`` is not provided. Defaults to ``50.0``. Must lie in the
        inclusive interval ``[0.1, 99.9]``; this applies ``numpy.percentile``
        independently per measurement (no single reference subject is
        selected, and no interpolation is performed across different
        measurements).
    mass_kg : float or None, optional
        Requested total body mass, in kilograms. If provided, it is used
        as an *independent* scaling input, exactly like ``height_cm``: once
        every body has been scaled geometrically (see "Scaling
        antropometrico" in the README), every body's mass *and* inertia
        tensor are uniformly rescaled (a single scalar factor, via
        ``opensim.Body.scaleMass``) so the whole model's total mass
        (``self.getTotalMass(self.state)``) matches ``mass_kg`` exactly --
        independent of whatever the height-driven geometric scaling alone
        would have produced volumetrically. If not provided (the default),
        the same rescaling is still applied, but targeting the ANSUR
        ``weight_kg`` resolved for this user's height/percentile (see
        :attr:`anthropometry`) -- i.e. mass is always resolved "like the
        other measurements" (from height when given, via
        :func:`~opensim_extensions.models.user.data.resolve_reference`),
        never left as a pure side effect of geometric scaling. See
        :attr:`mass_kg` to read back the resolved value either way.
    dataset : str or pathlib.Path, optional
        Path to the ANSUR II reference CSV to resolve measurements from.
        Defaults to the dataset bundled with this package.
    Raises
    ------
    ValueError
        If ``gender`` is not ``"M"``/``"F"``, if ``height_cm`` is given but is
        not a positive finite number, if ``percentile`` is not finite or
        falls outside ``[0.1, 99.9]``, or if ``mass_kg`` is given but is not
        a positive finite number.
    """

    def __init__(
        self,
        gender: str | Literal["M", "F"],
        height_cm: float | None = None,
        percentile: float = 50.0,
        mass_kg: float | None = None,
        *,
        dataset: str | Path = DEFAULT_DATASET,
    ):
        """Resolve anthropometry, load the base model, scale it, and calibrate its total mass.

        See the class docstring for the full description of ``gender``,
        ``height_cm``, ``percentile``, ``mass_kg``, and ``dataset``, and of
        the conditions under which a
        :class:`ValueError` or :class:`UserWarning` is raised. After
        resolving the anthropometric reference, this loads the bundled model
        (unlocking every coordinate -- see ``User._unlock_coordinates``),
        registers the bundled mesh directory for rendering, scales every
        body via :meth:`scale_bodies` using per-segment factors derived by
        comparing the resolved reference against the 50th-percentile
        baseline for the same ``gender`` (see "Scaling antropometrico" in
        the README), and finally calibrates the whole model's total mass
        (see :meth:`_rescale_total_mass`) to ``mass_kg`` if given, or to the
        resolved ANSUR ``weight_kg`` otherwise.
        """
        if mass_kg is not None:
            _validate_mass_kg(mass_kg)
        self._reference = resolve_reference(gender, height_cm, percentile, dataset)
        self.opensim = opensim
        self._geometry_dirs: list[Path] = []
        self._side_measurements: dict[str, float] = {}
        self._visualizer = None
        # Meshes are resolved while the native model file is loaded.
        register_geometry_search_path(opensim, DEFAULT_MESHES_DIR)
        opensim.Model.__init__(self, str(DEFAULT_MODEL_PATH))
        self._unlock_coordinates()
        self._baseline = resolve_reference(self.gender, percentile=50.0, dataset=dataset)
        self._add_anatomical_bodies()
        self.finalizeConnections()
        self.state = self.initSystem()
        self.add_geometry_directory(DEFAULT_MESHES_DIR)
        factors = segment_scale_factors(self._reference, self._baseline)
        self.scale_bodies(self._expand_bilateral_bodies(factors))
        # Geometric scaling alone only moves mass as a volumetric side
        # effect (OpenSim's own preserveMassDist=True, driven purely by the
        # height-derived per-body scale factors above) -- not calibrated
        # against any real mass reference. This always (re)calibrates the
        # model's total mass against either the explicit mass_kg or, if not
        # given, the ANSUR weight_kg resolved for this user, exactly as
        # every other anthropometric measurement is already resolved "from
        # height when given, else from percentile".
        target_mass_kg = (
            mass_kg if mass_kg is not None else self._reference.values["weight_kg"]
        )
        self._rescale_total_mass(target_mass_kg)
        self._mass_kg = target_mass_kg

    def _add_anatomical_bodies(self) -> None:
        """Create articulated head segments and anatomical eye/palm markers.

        The skull and neck are separated from the torso, their mesh geometry
        is reassigned, and cervical joints are added. Eye markers are attached
        to the skull. Palm-centre markers are attached to the existing
        articulated hand bodies; their final local offsets are updated after
        anthropometric scaling.
        """
        torso = self.getBodySet().get("torso")
        head_meshes = []
        spine_mesh = None
        for index in range(torso.getPropertyByName("attached_geometry").size()):
            mesh = opensim.Mesh.safeDownCast(torso.get_attached_geometry(index))
            if mesh is None:
                continue
            if mesh.get_mesh_file() in {"hat_skull.vtp", "hat_jaw.vtp"}:
                mesh.upd_Appearance().set_visible(False)
                head_meshes.append(mesh)
            elif mesh.get_mesh_file() == "hat_spine.vtp":
                mesh.set_mesh_file("hat_spine_torso.vtp")
                spine_mesh = mesh

        neck_center = np.array([0.0, 0.445, 0.0])
        neck_mass = 1000.0 * math.pi * 0.045**2 * 0.09
        neck_radial_inertia = neck_mass * (
            3 * 0.045**2 + 0.09**2
        ) / 12
        neck_axial_inertia = 0.5 * neck_mass * 0.045**2
        if spine_mesh is None:
            raise RuntimeError("The bundled model does not contain hat_spine.vtp")
        neck = opensim.Body(
            "neck",
            neck_mass,
            opensim.Vec3(*neck_center),
            opensim.Inertia(
                neck_radial_inertia,
                neck_axial_inertia,
                neck_radial_inertia,
            ),
        )
        neck_mesh = opensim.Mesh("hat_neck.vtp")
        neck_mesh.setName("neck_spine")
        neck_mesh.set_scale_factors(spine_mesh.get_scale_factors())
        neck_mesh.setColor(spine_mesh.getColor())

        skull_mass = 4.5
        skull_center = np.array([0.0, 0.57, 0.0])
        skull_inertia = opensim.Inertia(0.0103, 0.0104, 0.0124)
        skull = opensim.Body(
            "skull",
            skull_mass,
            opensim.Vec3(*skull_center),
            skull_inertia,
        )

        _remove_segment_mass(
            torso,
            skull_mass + neck_mass,
            (skull_mass * skull_center + neck_mass * neck_center)
            / (skull_mass + neck_mass),
        )
        self.addBody(neck)
        self.addBody(skull)
        neck.attachGeometry(neck_mesh)
        base_neck_joint = opensim.UniversalJoint(
            "torso_neck",
            torso,
            opensim.Vec3(0.0, 0.445, 0.0),
            opensim.Vec3(0),
            neck,
            opensim.Vec3(0.0, 0.445, 0.0),
            opensim.Vec3(0),
        )
        base_neck_coordinates = (
            ("neck_base_flexion", -0.7, 0.85),
            ("neck_base_lateral_flexion", -0.5, 0.5),
        )
        for index, (name, low, high) in enumerate(base_neck_coordinates):
            coordinate = base_neck_joint.getCoordinate(index)
            coordinate.setName(name)
            coordinate.setRangeMin(low)
            coordinate.setRangeMax(high)
            coordinate.set_clamped(True)
            coordinate.set_locked(False)
        self.addJoint(base_neck_joint)
        head_joint = opensim.BallJoint(
            "neck_skull",
            neck,
            opensim.Vec3(0.0, 0.49, 0.0),
            opensim.Vec3(0),
            skull,
            opensim.Vec3(0.0, 0.49, 0.0),
            opensim.Vec3(0),
        )
        neck_coordinates = (
            ("neck_lateral_flexion", -0.5, 0.5),
            ("neck_rotation", -1.22, 1.22),
            ("neck_flexion", -0.7, 0.85),
        )
        for index, (name, low, high) in enumerate(neck_coordinates):
            coordinate = head_joint.getCoordinate(index)
            coordinate.setName(name)
            coordinate.setRangeMin(low)
            coordinate.setRangeMax(high)
            coordinate.set_clamped(True)
            coordinate.set_locked(False)
        self.addJoint(head_joint)

        for mesh in head_meshes:
            head_mesh = opensim.Mesh(mesh.get_mesh_file())
            head_mesh.setName(f"skull_{mesh.getName()}")
            head_mesh.set_scale_factors(mesh.get_scale_factors())
            head_mesh.setColor(mesh.getColor())
            skull.attachGeometry(head_mesh)

        for marker_name, location in (
            ("REYE", (0.075, 0.56, 0.03)),
            ("LEYE", (0.075, 0.56, -0.03)),
        ):
            self.addMarker(
                opensim.Marker(marker_name, skull, opensim.Vec3(*location))
            )

        baseline_palm_length = self._baseline.values["palmlength"] * 1e-3
        for side, marker_name in (("r", "RPALM"), ("l", "LPALM")):
            hand = self.getBodySet().get(f"hand_{side}")
            self.addMarker(
                opensim.Marker(
                    marker_name,
                    hand,
                    opensim.Vec3(0.0, -baseline_palm_length / 2, 0.0),
                )
            )

    def _unlock_coordinates(self) -> None:
        """Unlock the coordinates that the bundled model locks by default."""
        coordinates = self.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinates.get(index).set_locked(False)

    def update_state(self) -> None:
        """Realize the model through position, velocity, and dynamics stages.

        The method first realizes position and velocity, equilibrates muscles
        when the model contains any, and then realizes dynamics.
        """
        self.realizePosition(self.state)
        self.realizeVelocity(self.state)
        if self.getMuscles().getSize():
            self.equilibrateMuscles(self.state)
        self.realizeDynamics(self.state)

    def _sync_coordinate_defaults(self) -> None:
        coordinates = self.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            coordinate.setDefaultValue(coordinate.getValue(self.state))
            coordinate.setDefaultSpeedValue(coordinate.getSpeedValue(self.state))

        constraints = self.getConstraintSet()
        for index in range(constraints.getSize()):
            coupler = self.opensim.CoordinateCouplerConstraint.safeDownCast(
                constraints.get(index)
            )
            if coupler is None:
                continue
            independent_names = coupler.getIndependentCoordinateNames()
            independent_values = self.opensim.Vector(independent_names.getSize(), 0.0)
            for coordinate_index in range(independent_names.getSize()):
                independent = coordinates.get(independent_names.get(coordinate_index))
                independent_values.set(
                    coordinate_index, independent.getDefaultValue()
                )
            dependent = coordinates.get(coupler.getDependentCoordinateName())
            dependent.setDefaultValue(
                coupler.getFunction().calcValue(independent_values)
            )

    def reinitialize(self) -> None:
        """Rebuild the OpenSim system while preserving current coordinates.

        Current coordinate values are copied to their defaults, dependent
        coordinate defaults are recalculated from their coupler functions,
        model connections are finalized, and a fresh system state is created.
        """
        self._sync_coordinate_defaults()
        self.finalizeConnections()
        self.state = self.initSystem()

    def scale_bodies(self, factors: dict[str, tuple[float, float, float]]) -> None:
        """Apply OpenSim's native body scaling and preserve the current state.

        After scaling, each palm marker is repositioned at half the resolved
        palm length from the wrist along the hand's local negative y axis.

        Parameters
        ----------
        factors : dict[str, tuple[float, float, float]]
            Body names mapped to positive, unitless ``(x, y, z)`` scale
            factors. Names not present in this model are ignored.

        Raises
        ------
        ValueError
            If a scale factor for an existing body is not positive and finite.
        """
        body_set = self.getBodySet()
        scale_set = self.opensim.ScaleSet()
        for body_name, axes in factors.items():
            if not body_set.contains(body_name):
                continue
            if len(axes) != 3 or any(
                not math.isfinite(value) or value <= 0 for value in axes
            ):
                raise ValueError(f"Invalid scale factor for body {body_name!r}: {axes}")
            scale = self.opensim.Scale()
            scale.setSegmentName(body_name)
            scale.setScaleFactors(self.opensim.Vec3(*axes))
            scale.setApply(True)
            scale_set.adoptAndAppend(scale)
        self.scale(self.state, scale_set, True)
        for side in ("left", "right"):
            marker = self.getMarkerSet().get(
                "LPALM" if side == "left" else "RPALM"
            )
            marker.set_location(
                self.opensim.Vec3(
                    0.0,
                    -float(getattr(self, f"{side}_palm_length")) / 2,
                    0.0,
                )
            )
        self._sync_coordinate_defaults()
        self.state = self.initSystem()

    def add_geometry_directory(self, directory: str | Path) -> None:
        """Register a directory containing geometry files for this model.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory to resolve and register with OpenSim's geometry search
            path.
        """
        path = Path(directory).expanduser().resolve()
        register_geometry_search_path(self.opensim, path)
        self._geometry_dirs.append(path)

    @property
    def geometry_directories(self) -> tuple[Path, ...]:
        """Return the geometry directories registered for this user.

        Returns
        -------
        tuple of pathlib.Path
            Resolved directories registered for mesh lookup, in registration
            order.
        """
        return tuple(self._geometry_dirs)

    @property
    def visualizer(self) -> Any | None:
        """Return the active VTK visualizer, if one is displaying this user.

        Returns
        -------
        VTKVisualizer or None
            Active visualizer, or ``None`` when the user is not shown.
        """
        return self._visualizer

    def show(
        self,
        motion: Any = None,
        *,
        loop: bool = False,
        fps: float = 30.0,
    ) -> Any:
        """Open the interactive viewer for this user.

        Parameters
        ----------
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, default=None
            Optional motion table or path to a ``.mot``/``.sto`` file.
        loop : bool, default=False
            Whether playback repeats after the final sample.
        fps : float, default=30.0
            Playback refresh and animation-export frame rate.

        Returns
        -------
        PlayerWindow
            Handle used to inspect playback state or close the viewer.
        """
        from ...viewer import VTKViewer

        return VTKViewer().show(self, motion, loop=loop, fps=fps)

    def to_osim(
        self,
        file: str | Path,
        *,
        with_meshes: bool = True,
        mesh_dir: str = "Geometry",
        overwrite: bool = True,
    ) -> Path:
        """Export the scaled model in its current posture to an OpenSim file.

        OpenSim stores coordinate defaults in the model, but live coordinate
        values in its ``State``. This method synchronizes the current values
        before serialization, so the exported model opens in the current
        posture. When requested, referenced meshes are copied beside the
        model and their exported paths are made relative to ``mesh_dir``.

        Parameters
        ----------
        file : str or pathlib.Path
            Destination ``.osim`` file.
        with_meshes : bool, default=True
            Copy referenced mesh files beside the model.
        mesh_dir : str, default="Geometry"
            Relative folder name for copied mesh files.
        overwrite : bool, default=True
            Whether an existing destination file may be replaced.

        Returns
        -------
        pathlib.Path
            The destination path.

        Raises
        ------
        FileExistsError
            If the destination exists and ``overwrite`` is false.
        ValueError
            If ``mesh_dir`` is not a safe relative path or mesh filenames
            collide in the destination folder.
        FileNotFoundError
            If any referenced mesh cannot be located.
        RuntimeError
            If OpenSim cannot serialize the model.
        """
        destination = Path(file).expanduser()
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        mesh_subdirectory = Path(mesh_dir)
        if with_meshes and (
            not mesh_dir
            or mesh_subdirectory.is_absolute()
            or ".." in mesh_subdirectory.parts
        ):
            raise ValueError("mesh_dir must be a safe relative directory path")

        self._sync_coordinate_defaults()
        mesh_files = self._referenced_mesh_files() if with_meshes else []
        if with_meshes:
            self._export_mesh_files(
                destination.parent / mesh_subdirectory, mesh_files
            )

        try:
            if with_meshes:
                for mesh, filename in mesh_files:
                    mesh.set_mesh_file(str(mesh_subdirectory / Path(filename).name))
            self.printToXML(str(destination))
        finally:
            for mesh, filename in mesh_files:
                mesh.set_mesh_file(filename)
        return destination

    def _export_mesh_files(
        self, destination_dir: Path, mesh_files: list[tuple[Any, str]]
    ) -> None:
        sources: dict[str, Path] = {}
        missing: list[str] = []
        search_directories = (
            *self._geometry_dirs,
            DEFAULT_MODEL_PATH.parent,
            MESHES_DIR,
        )
        for _, filename in mesh_files:
            source = next(
                (
                    directory / filename
                    for directory in search_directories
                    if (directory / filename).is_file()
                ),
                None,
            )
            if source is None:
                missing.append(filename)
                continue
            destination_name = Path(filename).name
            existing = sources.get(destination_name)
            if existing is not None and existing.resolve() != source.resolve():
                raise ValueError(
                    f"Mesh filename collision for {destination_name!r}: "
                    f"{existing} and {source}"
                )
            sources[destination_name] = source
        if missing:
            raise FileNotFoundError(
                "Could not locate referenced mesh file(s): "
                + ", ".join(sorted(missing))
            )
        if not sources:
            return
        destination_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in sources.items():
            destination = destination_dir / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

    def _referenced_mesh_files(self) -> list[tuple[Any, str]]:
        meshes: list[tuple[Any, str]] = []
        for component in self.getComponentsList():
            frame = opensim.PhysicalFrame.safeDownCast(component)
            if frame is None:
                continue
            geometry_property = frame.getPropertyByName("attached_geometry")
            for geometry_index in range(geometry_property.size()):
                mesh = opensim.Mesh.safeDownCast(
                    frame.get_attached_geometry(geometry_index)
                )
                if mesh is not None:
                    meshes.append((mesh, mesh.get_mesh_file()))
        return meshes

    def _set_coordinate_degrees(self, name: str, degrees: float) -> None:
        if not math.isfinite(degrees):
            raise ValueError("coordinate angle must be finite")
        self.getCoordinateSet().get(name).setValue(
            self.state, math.radians(degrees), False
        )

    def _coordinate_degrees(self, name: str) -> float:
        value = self.getCoordinateSet().get(name).getValue(self.state)
        return math.degrees(value)

    def set_neck_flexion(self, degrees: float) -> None:
        """Set head flexion/extension relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target angle in degrees. Positive values follow the OpenSim
            coordinate's flexion direction.
        """
        self._set_coordinate_degrees("neck_flexion", degrees)

    @property
    def neck_flexion(self) -> float:
        """Return the current head flexion/extension angle.

        Returns
        -------
        float
            ``neck_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_flexion")

    def set_neck_lateral_flexion(self, degrees: float) -> None:
        """Set lateral flexion of the head relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target lateral-flexion angle in degrees.
        """
        self._set_coordinate_degrees("neck_lateral_flexion", degrees)

    @property
    def neck_lateral_flexion(self) -> float:
        """Return the current head lateral-flexion angle.

        Returns
        -------
        float
            ``neck_lateral_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_lateral_flexion")

    def set_neck_rotation(self, degrees: float) -> None:
        """Set axial rotation of the head relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target axial-rotation angle in degrees.
        """
        self._set_coordinate_degrees("neck_rotation", degrees)

    @property
    def neck_rotation(self) -> float:
        """Return the current head axial-rotation angle.

        Returns
        -------
        float
            ``neck_rotation`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_rotation")

    def set_neck_base_flexion(self, degrees: float) -> None:
        """Set flexion/extension of the cervical spine relative to the torso.

        Parameters
        ----------
        degrees : float
            Target angle in degrees.
        """
        self._set_coordinate_degrees("neck_base_flexion", degrees)

    @property
    def neck_base_flexion(self) -> float:
        """Return cervical flexion/extension relative to the torso.

        Returns
        -------
        float
            ``neck_base_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_base_flexion")

    def set_neck_base_lateral_flexion(self, degrees: float) -> None:
        """Set cervical lateral flexion relative to the torso.

        Parameters
        ----------
        degrees : float
            Target lateral-flexion angle in degrees.
        """
        self._set_coordinate_degrees("neck_base_lateral_flexion", degrees)

    @property
    def neck_base_lateral_flexion(self) -> float:
        """Return cervical lateral flexion relative to the torso.

        Returns
        -------
        float
            ``neck_base_lateral_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_base_lateral_flexion")

    def _rescale_total_mass(self, mass_kg: float) -> None:
        """Uniformly rescale every body's mass and inertia so the model's total mass matches ``mass_kg`` exactly.

        Applies one scalar correction factor (``mass_kg`` divided by the
        model's current ``getTotalMass(self.state)``, i.e. whatever the geometric,
        height-driven scaling in :meth:`scale_bodies` produced
        volumetrically) to every body via ``opensim.Body.scaleMass`` --
        confirmed directly against a hand-built body that this native
        OpenSim method scales both ``mass`` *and* the full inertia tensor
        (moments and products alike) by the same factor, leaving
        ``mass_center`` untouched. A single uniform factor across every
        body keeps each body's relative share of the total mass (and each
        one's own radius of gyration/shape) exactly as the geometric
        scaling left it; only the overall scale of "how heavy" changes.

        Mass feeds into the multibody system's mass matrix, built once in
        ``initSystem()`` -- the same reason
        :meth:`~opensim_extensions.components.Body.set_mass` calls
        :meth:`User.reinitialize` after
        changing a single body's mass -- so this does the same once every
        body has been rescaled, preserving whatever posture is current when
        it runs (the model's still-default posture, the first time this
        runs inside the constructor; otherwise whatever :meth:`set_mass_kg`
        found current when it was called).

        Parameters
        ----------
        mass_kg : float
            Target total body mass, in kilograms. Expected to already be
            validated (finite, strictly positive) by the caller -- see
            the constructor's own ``mass_kg`` validation.
        """
        scale_factor = mass_kg / self.getTotalMass(self.state)
        body_set = self.getBodySet()
        for index in range(body_set.getSize()):
            body_set.get(index).scaleMass(scale_factor)
        self.reinitialize()

    @staticmethod
    def _validate_measurement(name: str, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value")
        return float(value)

    def _apply_reference_scale(self, reference) -> None:
        current_factors = self._expand_bilateral_bodies(
            segment_scale_factors(self._reference, self._baseline)
        )
        target_factors = self._expand_bilateral_bodies(
            segment_scale_factors(reference, self._baseline)
        )
        incremental_factors: dict[str, tuple[float, float, float]] = {}
        for body_name, target in target_factors.items():
            current = current_factors[body_name]
            ratios = (
                target[0] / current[0],
                target[1] / current[1],
                target[2] / current[2],
            )
            if any(not math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12) for ratio in ratios):
                incremental_factors[body_name] = ratios
        if incremental_factors:
            self.scale_bodies(incremental_factors)
            self._rescale_total_mass(self._mass_kg)

    def _set_reference_measurement(
        self,
        property_name: str,
        source_measure: str,
        value: float,
        *,
        source_multiplier: float,
    ) -> None:
        """Update one symmetric ANSUR input and apply its scale delta."""
        measurement = self._validate_measurement(property_name, value)
        updated_values = self._reference.values.copy()
        updated_values[source_measure] = measurement * source_multiplier
        updated_reference = replace(self._reference, values=updated_values)
        self._apply_reference_scale(updated_reference)
        self._reference = updated_reference

    def _set_side_measurement(
        self,
        property_name: str,
        value: float,
        *,
        body_axes: tuple[tuple[str, tuple[str, ...]], ...],
        store_override: bool,
    ) -> None:
        """Apply a side-specific measurement without changing its opposite."""
        measurement = self._validate_measurement(property_name, value)
        body_set = self.getBodySet()
        for body_name, _ in body_axes:
            if not body_set.contains(body_name):
                raise RuntimeError(
                    f"Cannot apply {property_name!r}: body {body_name!r} is absent"
                )

        current = float(getattr(self, property_name))
        if store_override:
            ratio = measurement / current
            if not math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
                scale_factors = {
                    body_name: _axis_scale_factors(ratio, axes)
                    for body_name, axes in body_axes
                }
                self.scale_bodies(scale_factors)
                self._rescale_total_mass(self._mass_kg)
            self._side_measurements[property_name] = measurement
            return

        for _ in range(8):
            current = float(getattr(self, property_name))
            ratio = measurement / current
            if math.isclose(ratio, 1.0, rel_tol=1e-6, abs_tol=1e-9):
                return
            scale_factors = {
                body_name: _axis_scale_factors(ratio, axes)
                for body_name, axes in body_axes
            }
            self.scale_bodies(scale_factors)
            self._rescale_total_mass(self._mass_kg)

        current = float(getattr(self, property_name))
        if not math.isclose(current, measurement, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"Scaling {property_name!r} did not reach the requested value "
                f"{measurement:g} m; the resulting value is {current:g} m"
            )

    def _side_measurement_value(self, property_name: str, source_measure: str) -> float:
        override = self._side_measurements.get(property_name)
        if override is not None:
            return override
        return self._reference.values[source_measure] / 1000.0

    @property
    def gender(self):
        """Return the normalized sex code used to resolve this user's anthropometry.

        Returns
        -------
        str
            ``"M"`` or ``"F"``, matching the ``gender`` passed to the
            constructor (normalization only affects case/whitespace, not
            the two accepted values themselves).
        """
        return self._reference.gender

    @property
    def height_cm(self):
        """Return this user's resolved stature, in centimetres.

        If the constructor was called with ``height_cm=...``, this is exactly
        that value (even if it was outside the ANSUR range and triggered
        extrapolation). Otherwise it is the ANSUR stature at the resolved
        ``percentile`` for this user's ``gender``.

        Returns
        -------
        float
            Stature, in centimetres.
        """
        return self._reference.height_cm

    @property
    def percentile(self):
        """Return the ANSUR percentile associated with this user.

        When the constructor was called with ``percentile=...`` (the
        default path), this is exactly that value, applied independently to
        every ANSUR measurement. When it was called with ``height_cm=...``
        instead, every measurement is resolved directly from the requested
        height (not from this percentile), and this property instead
        reports the *empirical* percentile of that height within the ANSUR
        population of this user's ``gender`` -- informational only, see the
        constructor's ``height_cm`` parameter.

        Returns
        -------
        float
            Percentile in the inclusive range ``[0.1, 99.9]``.
        """
        return self._reference.percentile

    @property
    def mass_kg(self):
        """Return this user's resolved and calibrated total body mass, in kilograms.

        If the constructor was called with ``mass_kg=...`` (or
        :meth:`set_mass_kg` was called afterward), this is exactly that
        value. Otherwise, it is the ANSUR ``weight_kg`` resolved for this
        user's height/percentile (see :attr:`anthropometry`). Either way,
        this is not just a label: ``self.getTotalMass(self.state)``
        is calibrated to match it exactly (see :meth:`_rescale_total_mass`),
        unlike the purely volumetric, height-driven mass that the
        geometric per-body scaling alone would otherwise leave in place.

        Returns
        -------
        float
            Total body mass, in kilograms.
        """
        return self._mass_kg

    def set_mass_kg(self, mass_kg: float) -> None:
        """Recalibrate this user's total body mass to ``mass_kg``, in kilograms.

        This re-applies the same independent
        mass correction the constructor itself uses (see
        :meth:`_rescale_total_mass`) directly to the already geometrically
        scaled model, rather than rebuilding it from scratch. Safe to call
        more than once, and safe to call regardless of whether the
        constructor was given an explicit ``mass_kg`` or not -- it always
        rescales from whatever the model's current total mass happens to be
        right now to the new target.

        Parameters
        ----------
        mass_kg : float
            New target total body mass, in kilograms. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``mass_kg`` is not finite or not strictly positive.
        """
        _validate_mass_kg(mass_kg)
        self._rescale_total_mass(mass_kg)
        self._mass_kg = mass_kg

    @property
    def anthropometry(self):
        """Return every anthropometric measurement resolved for this user.

        Returns
        -------
        opensim_extensions.models.user.data.AnthropometricReference
            Internal reference object exposing ``gender``, ``percentile``,
            ``values`` (a ``dict`` mapping every numeric ANSUR II column
            resolved for this user to its value in the dataset's original
            units -- predominantly millimetres, with ``values["stature_m"]``
            in metres; see "Dati ANSUR risolti" in the README), and the
            convenience properties ``height_m``/``height_cm`` (stature only,
            derived from ``values["stature_m"]``). This type is an internal
            implementation detail (not part of the package's re-exported
            public surface), but the object itself and its attributes
            remain freely readable once a :class:`User` has been built.
        """
        return self._reference

    def _joint_center(self, joint_name: str) -> tuple[float, float, float]:
        self.realizePosition(self.state)
        frame = self.getJointSet().get(joint_name).getChildFrame()
        position = frame.getPositionInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def joint_centers(self) -> dict[str, tuple[float, float, float]]:
        """Return every named joint centre at once, in the ground frame.

        Equivalent to calling every individual joint-centre property below
        (``pelvis``, ``left_hip``, ``right_knee``, ...) and collecting the
        results into one dictionary, keyed by the same friendly names.
        Reflects the model's current posture: each position is computed via
        ``getPositionInGround`` with an automatic ``realizePosition``, so it
        is correct immediately after a posture setter, with no need to call
        :meth:`update_state` first.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from friendly joint name (``"pelvis"``, ``"left_hip"``,
            ``"right_ankle"``, ...) to that joint's centre as an ``(x, y,
            z)`` position in the ground frame, in metres.
        """
        return {
            name: self._joint_center(joint)
            for name, joint in _JOINT_CENTER_NAMES.items()
        }

    def _marker_location(self, marker_name: str) -> tuple[float, float, float]:
        self.realizePosition(self.state)
        position = self.getMarkerSet().get(marker_name).getLocationInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def foot_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return every foot landmark marker at once, in the ground frame.

        Equivalent to calling each of ``left_heel``/``right_heel``,
        ``left_toe``/``right_toe``, and ``left_mt5``/``right_mt5`` and
        collecting the results into one dictionary. Reflects the model's
        current posture (``realizePosition`` is called automatically; no
        need to call :meth:`update_state` first). There is no 1st-metatarsal
        marker in this model (only the 5th, see :attr:`left_mt5`), so none
        is exposed here either.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from friendly marker name (``"left_heel"``,
            ``"right_toe"``, ``"left_mt5"``, ...) to that marker's location
            as an ``(x, y, z)`` position in the ground frame, in metres.
        """
        return {
            name: self._marker_location(marker)
            for name, marker in _FOOT_MARKER_NAMES.items()
        }

    @property
    def eye_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return the right and left orbital marker positions.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from ``"right"`` and ``"left"`` to the corresponding eye
            marker's ``(x, y, z)`` position in metres in the ground frame.
        """
        return {
            side: self._marker_location(marker)
            for side, marker in (("right", "REYE"), ("left", "LEYE"))
        }

    @property
    def palm_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return the articulated right and left palm-centre marker positions.

        Each marker is attached to its corresponding hand body at the centre
        of the resolved palm, halfway from the wrist along the hand's local
        negative y axis. The hand bodies are connected to the radii by
        universal wrist joints with flexion/extension and radial/ulnar
        deviation coordinates. Marker positions therefore follow the current
        wrist posture and are expressed in the ground frame.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from ``"right"`` and ``"left"`` to the corresponding
            palm-centre ``(x, y, z)`` positions in metres in the ground frame.
        """
        return {
            side: self._marker_location(marker)
            for side, marker in (("right", "RPALM"), ("left", "LPALM"))
        }

    @property
    def left_heel(self) -> tuple[float, float, float]:
        """Return the left heel marker location, in the ground frame.

        Reads the OpenSim marker ``LCAL``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LCAL")

    @property
    def right_heel(self) -> tuple[float, float, float]:
        """Return the right heel marker location, in the ground frame.

        Reads the OpenSim marker ``RCAL``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RCAL")

    @property
    def left_toe(self) -> tuple[float, float, float]:
        """Return the left toe marker location, in the ground frame.

        Reads the OpenSim marker ``LTOE``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LTOE")

    @property
    def right_toe(self) -> tuple[float, float, float]:
        """Return the right toe marker location, in the ground frame.

        Reads the OpenSim marker ``RTOE``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RTOE")

    @property
    def left_mt5(self) -> tuple[float, float, float]:
        """Return the left 5th-metatarsal marker location, in the ground frame.

        Reads the OpenSim marker ``LMT5``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first). There is no 1st-metatarsal marker in
        this model, so this is the only metatarsal landmark available.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LMT5")

    @property
    def right_mt5(self) -> tuple[float, float, float]:
        """Return the right 5th-metatarsal marker location, in the ground frame.

        Reads the OpenSim marker ``RMT5``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first). There is no 1st-metatarsal marker in
        this model, so this is the only metatarsal landmark available.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RMT5")

    @property
    def com(self) -> tuple[float, float, float]:
        """Return the whole-body centre of mass, in the ground frame.

        Computed natively by OpenSim (``Model.calcMassCenterPosition``),
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need to call :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        self.realizePosition(self.state)
        position = self.calcMassCenterPosition(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def cop(self) -> tuple[float, float, float]:
        """Return the ground projection of the centre of mass.

        This is a purely kinematic projection of :attr:`com` straight down
        onto the ground plane (``y = 0``): the ``x``/``z`` coordinates are
        copied unchanged from :attr:`com` and ``y`` is forced to ``0.0``.
        Despite the name, this is **not** a dynamically computed centre of
        pressure derived from contact/ground-reaction forces.

        Returns
        -------
        tuple[float, float, float]
            ``(x, 0.0, z)`` position in metres, in the ground frame, where
            ``x``/``z`` match :attr:`com`.
        """
        x, _, z = self.com
        return (x, 0.0, z)

    def set_position(
        self, reference: tuple[float, float, float], x: float, y: float, z: float
    ) -> None:
        """Translate the whole model so that ``reference`` ends up at ``(x, y, z)``.

        ``reference`` is any ground-frame point belonging to this model --
        e.g. :attr:`com`, :attr:`cop`, a :attr:`joint_centers` entry, or one
        of the named joint-centre properties (``left_ankle``, ``pelvis``,
        ...). The whole model is translated rigidly through its root
        (``pelvis_tx``/``pelvis_ty``/``pelvis_tz``): relative posture and
        joint angles are unaffected. Calls :meth:`update_state` internally,
        so derived quantities are immediately up to date.

        Parameters
        ----------
        reference : tuple[float, float, float]
            Current ground-frame position, in metres, of the point to move
            (read it right before calling, since it depends on the current
            posture).
        x, y, z : float
            Target ground-frame coordinates for that point, in metres.

        Raises
        ------
        ValueError
            If any coordinate is not finite.
        """
        if not all(math.isfinite(value) for value in (x, y, z, *reference)):
            raise ValueError("position must be finite")
        targets = (x, y, z)
        for name, target, current in zip(
            _PELVIS_TRANSLATION_COORDINATES, targets, reference
        ):
            coordinate = self.getCoordinateSet().get(name)
            coordinate.setValue(
                self.state,
                coordinate.getValue(self.state) + (target - current),
                False,
            )
        self.update_state()

    @property
    def left_foot_length(self) -> float:
        """Return left foot length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``footlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres;
        does not depend on the model's current posture or on any joint
        centre.

        Returns
        -------
        float
            Foot length, in metres.
        """
        return self._side_measurement_value("left_foot_length", "footlength")

    @property
    def right_foot_length(self) -> float:
        """Return right foot length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``footlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres;
        does not depend on the model's current posture or on any joint
        centre. ANSUR does not record left/right foot length separately, so
        this is identical to :attr:`left_foot_length`.

        Returns
        -------
        float
            Foot length, in metres.
        """
        return self._side_measurement_value("right_foot_length", "footlength")

    @property
    def left_foot_height(self) -> float:
        """Return the left foot's height above the ground.

        Approximated as the ``y`` coordinate of :attr:`left_ankle` (the
        left ankle joint centre), at the model's current posture -- this is
        an approximation of the sole's height, not a direct measurement of
        the foot's own geometry.

        Returns
        -------
        float
            Height above the ground, in metres.
        """
        return self._joint_center("ankle_l")[1]

    @property
    def right_foot_height(self) -> float:
        """Return the right foot's height above the ground.

        Approximated as the ``y`` coordinate of :attr:`right_ankle` (the
        right ankle joint centre), at the model's current posture -- this
        is an approximation of the sole's height, not a direct measurement
        of the foot's own geometry.

        Returns
        -------
        float
            Height above the ground, in metres.
        """
        return self._joint_center("ankle_r")[1]

    @property
    def left_thigh_length(self) -> float:
        """Return the left thigh length, at the model's current posture.

        Euclidean distance between :attr:`left_hip` and :attr:`left_knee`
        in the ground frame; since both depend on posture, this changes as
        the hip/knee angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-to-knee distance, in metres.
        """
        return math.dist(
            self._joint_center("hip_l"), self._joint_center("walker_knee_l")
        )

    @property
    def right_thigh_length(self) -> float:
        """Return the right thigh length, at the model's current posture.

        Euclidean distance between :attr:`right_hip` and :attr:`right_knee`
        in the ground frame; since both depend on posture, this changes as
        the hip/knee angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-to-knee distance, in metres.
        """
        return math.dist(
            self._joint_center("hip_r"), self._joint_center("walker_knee_r")
        )

    @property
    def left_shank_length(self) -> float:
        """Return the left shank length, at the model's current posture.

        Euclidean distance between :attr:`left_knee` and :attr:`left_ankle`
        in the ground frame; since both depend on posture, this changes as
        the knee/ankle angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Knee-to-ankle distance, in metres.
        """
        return math.dist(
            self._joint_center("walker_knee_l"), self._joint_center("ankle_l")
        )

    @property
    def right_shank_length(self) -> float:
        """Return the right shank length, at the model's current posture.

        Euclidean distance between :attr:`right_knee` and
        :attr:`right_ankle` in the ground frame; since both depend on
        posture, this changes as the knee/ankle angles change (it is not a
        fixed ANSUR measurement).

        Returns
        -------
        float
            Knee-to-ankle distance, in metres.
        """
        return math.dist(
            self._joint_center("walker_knee_r"), self._joint_center("ankle_r")
        )

    @property
    def torso_height(self) -> float:
        """Return the trunk height, at the model's current posture.

        Euclidean distance between a "hip centre" and a "shoulder centre",
        each the midpoint between the corresponding left/right joint
        centres (:attr:`left_hip`/:attr:`right_hip` and
        :attr:`left_shoulder`/:attr:`right_shoulder`). Since all four
        inputs depend on posture, this changes with the current hip/spine
        posture (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-centre-to-shoulder-centre distance, in metres.
        """
        left_hip, right_hip = self._joint_center("hip_l"), self._joint_center("hip_r")
        left_shoulder, right_shoulder = (
            self._joint_center("acromial_l"),
            self._joint_center("acromial_r"),
        )
        hip_center = tuple((a + b) / 2 for a, b in zip(left_hip, right_hip))
        shoulder_center = tuple(
            (a + b) / 2 for a, b in zip(left_shoulder, right_shoulder)
        )
        return math.dist(hip_center, shoulder_center)

    @property
    def shoulder_width(self) -> float:
        """Return the left-to-right shoulder joint centre distance.

        Euclidean distance between :attr:`left_shoulder` and
        :attr:`right_shoulder`, at the model's current posture -- a
        geometric measurement, not read from ANSUR. See
        :attr:`biacromial_breadth` for the ANSUR-measured equivalent; the
        two are independent and generally do not coincide numerically (see
        the README's "Centri articolari e misure derivate" section for why).

        Returns
        -------
        float
            Shoulder-to-shoulder distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_l"), self._joint_center("acromial_r")
        )

    @property
    def biacromial_breadth(self) -> float:
        """Return the biacromial breadth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``biacromialbreadth``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. See :attr:`shoulder_width` for the
        posture-dependent geometric equivalent derived from the model's
        shoulder joint centres.

        Returns
        -------
        float
            Biacromial breadth, in metres.
        """
        return self._reference.values["biacromialbreadth"] / 1000.0

    @property
    def left_arm_length(self) -> float:
        """Return the left upper-arm length, at the model's current posture.

        Euclidean distance between :attr:`left_shoulder` and
        :attr:`left_elbow` in the ground frame; since both depend on
        posture, this changes as the shoulder/elbow angles change (it is
        not a fixed ANSUR measurement).

        Returns
        -------
        float
            Shoulder-to-elbow distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_l"), self._joint_center("elbow_l")
        )

    @property
    def right_arm_length(self) -> float:
        """Return the right upper-arm length, at the model's current posture.

        Euclidean distance between :attr:`right_shoulder` and
        :attr:`right_elbow` in the ground frame; since both depend on
        posture, this changes as the shoulder/elbow angles change (it is
        not a fixed ANSUR measurement).

        Returns
        -------
        float
            Shoulder-to-elbow distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_r"), self._joint_center("elbow_r")
        )

    @property
    def left_forearm_length(self) -> float:
        """Return the left forearm length, at the model's current posture.

        Euclidean distance between :attr:`left_elbow` and :attr:`left_wrist`
        in the ground frame; since both depend on posture, this changes as
        the elbow/wrist/forearm angles change (it is not a fixed ANSUR
        measurement).

        Returns
        -------
        float
            Elbow-to-wrist distance, in metres.
        """
        return math.dist(
            self._joint_center("elbow_l"), self._joint_center("radius_hand_l")
        )

    @property
    def right_forearm_length(self) -> float:
        """Return the right forearm length, at the model's current posture.

        Euclidean distance between :attr:`right_elbow` and
        :attr:`right_wrist` in the ground frame; since both depend on
        posture, this changes as the elbow/wrist/forearm angles change (it
        is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Elbow-to-wrist distance, in metres.
        """
        return math.dist(
            self._joint_center("elbow_r"), self._joint_center("radius_hand_r")
        )

    @property
    def left_palm_length(self) -> float:
        """Return left palm length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``palmlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.
        This is the palm only, not including the fingers (ANSUR II and the
        underlying OpenSim model do not provide individual finger lengths,
        see the README).

        Returns
        -------
        float
            Palm length, in metres.
        """
        return self._side_measurement_value("left_palm_length", "palmlength")

    @property
    def right_palm_length(self) -> float:
        """Return right palm length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``palmlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.
        This is the palm only, not including the fingers. ANSUR does not
        record left/right palm length separately, so this is identical to
        :attr:`left_palm_length`.

        Returns
        -------
        float
            Palm length, in metres.
        """
        return self._side_measurement_value("right_palm_length", "palmlength")

    @property
    def left_arm_circumference(self) -> float:
        """Return left flexed-biceps circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``bicepscircumferenceflexed``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Flexed biceps circumference, in metres.
        """
        return self._side_measurement_value(
            "left_arm_circumference", "bicepscircumferenceflexed"
        )

    @property
    def right_arm_circumference(self) -> float:
        """Return right flexed-biceps circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``bicepscircumferenceflexed``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_arm_circumference`.

        Returns
        -------
        float
            Flexed biceps circumference, in metres.
        """
        return self._side_measurement_value(
            "right_arm_circumference", "bicepscircumferenceflexed"
        )

    @property
    def left_forearm_circumference(self) -> float:
        """Return left flexed-forearm circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR
        ``forearmcircumferenceflexed`` measurement (see
        :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Flexed forearm circumference, in metres.
        """
        return self._side_measurement_value(
            "left_forearm_circumference", "forearmcircumferenceflexed"
        )

    @property
    def right_forearm_circumference(self) -> float:
        """Return right flexed-forearm circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR
        ``forearmcircumferenceflexed`` measurement (see
        :attr:`anthropometry`), converted from millimetres to metres.
        ANSUR does not record left/right circumference separately, so this
        is identical to :attr:`left_forearm_circumference`.

        Returns
        -------
        float
            Flexed forearm circumference, in metres.
        """
        return self._side_measurement_value(
            "right_forearm_circumference", "forearmcircumferenceflexed"
        )

    @property
    def neck_circumference(self) -> float:
        """Return neck circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``neckcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Neck circumference, in metres.
        """
        return self._reference.values["neckcircumference"] / 1000.0

    @property
    def chest_circumference(self) -> float:
        """Return chest circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Chest circumference, in metres.
        """
        return self._reference.values["chestcircumference"] / 1000.0

    @property
    def chest_depth(self) -> float:
        """Return chest (sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Chest depth (front-to-back), in metres.
        """
        return self._reference.values["chestdepth"] / 1000.0

    @property
    def chest_width(self) -> float:
        """Return chest width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Chest width (side-to-side), in metres.
        """
        return self._reference.values["chestbreadth"] / 1000.0

    @property
    def waist_circumference(self) -> float:
        """Return waist circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Waist circumference, in metres.
        """
        return self._reference.values["waistcircumference"] / 1000.0

    @property
    def waist_depth(self) -> float:
        """Return waist (sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Waist depth (front-to-back), in metres.
        """
        return self._reference.values["waistdepth"] / 1000.0

    @property
    def waist_width(self) -> float:
        """Return waist width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Waist width (side-to-side), in metres.
        """
        return self._reference.values["waistbreadth"] / 1000.0

    @property
    def hip_circumference(self) -> float:
        """Return hip (buttock) circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``buttockcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Hip/buttock circumference, in metres.
        """
        return self._reference.values["buttockcircumference"] / 1000.0

    @property
    def hip_depth(self) -> float:
        """Return hip (buttock, sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``buttockdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Hip/buttock depth (front-to-back), in metres.
        """
        return self._reference.values["buttockdepth"] / 1000.0

    @property
    def hip_width(self) -> float:
        """Return hip width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``hipbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Hip width (side-to-side), in metres.
        """
        return self._reference.values["hipbreadth"] / 1000.0

    @property
    def left_thigh_circumference(self) -> float:
        """Return left thigh circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``thighcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Thigh circumference, in metres.
        """
        return self._side_measurement_value(
            "left_thigh_circumference", "thighcircumference"
        )

    @property
    def right_thigh_circumference(self) -> float:
        """Return right thigh circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``thighcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_thigh_circumference`.

        Returns
        -------
        float
            Thigh circumference, in metres.
        """
        return self._side_measurement_value(
            "right_thigh_circumference", "thighcircumference"
        )

    @property
    def left_calf_circumference(self) -> float:
        """Return left calf circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``calfcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Calf circumference, in metres.
        """
        return self._side_measurement_value(
            "left_calf_circumference", "calfcircumference"
        )

    @property
    def right_calf_circumference(self) -> float:
        """Return right calf circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``calfcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_calf_circumference`.

        Returns
        -------
        float
            Calf circumference, in metres.
        """
        return self._side_measurement_value(
            "right_calf_circumference", "calfcircumference"
        )

    @property
    def left_thigh_depth(self) -> float:
        """Return left thigh depth, assuming a circular cross-section.

        ANSUR has no thigh breadth measurement to fit an ellipse against,
        so this derives a diameter from the thigh circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh depth (front-to-back), in metres.
        """
        return self.left_thigh_circumference / math.pi

    @property
    def right_thigh_depth(self) -> float:
        """Return right thigh depth, assuming a circular cross-section.

        ANSUR has no thigh breadth measurement to fit an ellipse against,
        so this derives a diameter from the thigh circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh depth (front-to-back), in metres.
        """
        return self.right_thigh_circumference / math.pi

    @property
    def left_thigh_width(self) -> float:
        """Return left thigh width, assuming a circular cross-section.

        Equal to :attr:`left_thigh_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh width (side-to-side), in metres.
        """
        return self.left_thigh_depth

    @property
    def right_thigh_width(self) -> float:
        """Return right thigh width, assuming a circular cross-section.

        Equal to :attr:`right_thigh_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh width (side-to-side), in metres.
        """
        return self.right_thigh_depth

    @property
    def left_calf_depth(self) -> float:
        """Return left calf depth, assuming a circular cross-section.

        ANSUR has no calf breadth measurement to fit an ellipse against, so
        this derives a diameter from the calf circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf depth (front-to-back), in metres.
        """
        return self.left_calf_circumference / math.pi

    @property
    def right_calf_depth(self) -> float:
        """Return right calf depth, assuming a circular cross-section.

        ANSUR has no calf breadth measurement to fit an ellipse against, so
        this derives a diameter from the calf circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf depth (front-to-back), in metres.
        """
        return self.right_calf_circumference / math.pi

    @property
    def left_calf_width(self) -> float:
        """Return left calf width, assuming a circular cross-section.

        Equal to :attr:`left_calf_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf width (side-to-side), in metres.
        """
        return self.left_calf_depth

    @property
    def right_calf_width(self) -> float:
        """Return right calf width, assuming a circular cross-section.

        Equal to :attr:`right_calf_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf width (side-to-side), in metres.
        """
        return self.right_calf_depth

    def _expand_bilateral_bodies(self, factors: dict[str, tuple[float, float, float]]):
        expanded: dict[str, tuple[float, float, float]] = {}
        body_set = self.getBodySet()
        for body, values in factors.items():
            for candidate in (body, f"{body}_r", f"{body}_l"):
                if body_set.contains(candidate):
                    expanded[candidate] = values
        return expanded
