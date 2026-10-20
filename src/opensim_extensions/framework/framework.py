"""Dictionary-like container for independent native OpenSim models."""

from __future__ import annotations

import keyword
from collections.abc import Iterable, Mapping, Sequence
from math import isclose, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self, TypeGuard

from ..bodies.shapes import MESHES_DIR
from ..opensim import import_opensim
from .couplings import ComponentRef, Coupling, normalize_reference

if TYPE_CHECKING:
    import opensim
else:
    opensim = import_opensim()

__all__ = ["Framework"]

_MODEL_SETS = (
    ("bodyset", "updBodySet"),
    ("jointset", "updJointSet"),
    ("forceset", "updForceSet"),
    ("markerset", "updMarkerSet"),
    ("constraintset", "updConstraintSet"),
    ("controllerset", "updControllerSet"),
    ("contactgeometryset", "updContactGeometrySet"),
    ("probeset", "updProbeSet"),
)

_COMPONENT_SETS = {
    "force": ("forceset", "addForce"),
    "constraint": ("constraintset", "addConstraint"),
    "controller": ("controllerset", "addController"),
}


def _set_items(component_set: Any) -> list[Any]:
    return [component_set.get(index) for index in range(component_set.getSize())]


def _unique_name(prefix: str, name: str, existing: set[str]) -> str:
    candidate = f"{prefix}_{name}"
    suffix = 2
    while candidate in existing:
        candidate = f"{prefix}_{name}_{suffix}"
        suffix += 1
    return candidate


def _patch_socket_paths(root: Any, renamed_paths: dict[str, str]) -> None:
    for component in (root, *root.getComponentsList()):
        for socket_name in component.getSocketNames():
            socket = component.updSocket(socket_name)
            path = socket.getConnecteePath()
            for old_path, new_path in renamed_paths.items():
                if path == old_path or path.startswith(old_path + "/"):
                    socket.setConnecteePath(new_path + path[len(old_path) :])
                    break


def _current_coordinate_values(
    model: opensim.Model,
) -> dict[str, tuple[float, float]]:
    state = getattr(model, "state", None)
    if state is None:
        try:
            state = model.getWorkingState()
        except RuntimeError:
            return {}
    coordinates = model.getCoordinateSet()
    return {
        coordinates.get(index).getName(): (
            coordinates.get(index).getValue(state),
            coordinates.get(index).getSpeedValue(state),
        )
        for index in range(coordinates.getSize())
    }


def _apply_coordinate_defaults(
    model: opensim.Model, values: dict[str, tuple[float, float]]
) -> None:
    coordinates = model.getCoordinateSet()
    for name, (value, speed) in values.items():
        coordinate = coordinates.get(name)
        coordinate.setDefaultValue(value)
        coordinate.setDefaultSpeedValue(speed)

    # The coordinate set is a cache rebuilt from the joints on initSystem,
    # so the defaults must also be stored on the joint-owned coordinates.
    joints = model.getJointSet()
    for joint_index in range(joints.getSize()):
        joint = joints.get(joint_index)
        for index in range(joint.numCoordinates()):
            owned = joint.upd_coordinates(index)
            if owned.getName() in values:
                value, speed = values[owned.getName()]
                owned.setDefaultValue(value)
                owned.setDefaultSpeedValue(speed)

    constraints = model.getConstraintSet()
    for index in range(constraints.getSize()):
        coupler = opensim.CoordinateCouplerConstraint.safeDownCast(
            constraints.get(index)
        )
        if coupler is None:
            continue
        independent_names = coupler.getIndependentCoordinateNames()
        independent_values = opensim.Vector(independent_names.getSize(), 0.0)
        for coordinate_index in range(independent_names.getSize()):
            independent = coordinates.get(independent_names.get(coordinate_index))
            independent_values.set(
                coordinate_index, independent.getDefaultValue()
            )
        dependent = coordinates.get(coupler.getDependentCoordinateName())
        dependent.setDefaultValue(coupler.getFunction().calcValue(independent_values))


def _component_path(component: Any) -> str:
    return component.getAbsolutePathString()


def _remap_component_path(path: str, renamed_paths: Mapping[str, str]) -> str:
    for old_path, new_path in renamed_paths.items():
        if path == old_path or path.startswith(old_path + "/"):
            return new_path + path[len(old_path) :]
    return path


class Framework(dict[str, opensim.Model]):
    """Store named OpenSim models while keeping each model independent.

    Parameters
    ----------
    models : Mapping[str, opensim.Model] or Iterable[tuple[str, opensim.Model]], optional
        Initial models. Keys must be valid, non-reserved Python identifiers;
        values must be distinct ``opensim.Model`` instances.
    **named_models : opensim.Model
        Additional models keyed by their keyword names.

    Raises
    ------
    TypeError
        If a key is not a string or a value is not an ``opensim.Model``.
    ValueError
        If a name is invalid, reserved, or the same model instance is stored
        under more than one name.

    Notes
    -----
    Model names can be accessed using either dictionary syntax
    (``framework["user"]``) or attribute syntax (``framework.user``). Names
    that conflict with dictionary or framework attributes are reserved.
    """

    _active_composite: opensim.Model | None
    _active_state: Any | None
    _active_path_map: dict[tuple[str, str], str]
    _couplings: dict[str, Coupling]

    def __init__(
        self,
        models: Mapping[str, opensim.Model]
        | Iterable[tuple[str, opensim.Model]] = (),
        /,
        **named_models: opensim.Model,
    ) -> None:
        dict.__init__(self)
        object.__setattr__(self, "_active_composite", None)
        object.__setattr__(self, "_active_state", None)
        object.__setattr__(self, "_active_path_map", {})
        object.__setattr__(self, "_couplings", {})
        self.update(models, **named_models)

    @staticmethod
    def _reserved_names() -> set[str]:
        return set(dir(dict)) | set(dir(Framework))

    @classmethod
    def _validate_key(cls, key: object) -> str:
        if not isinstance(key, str):
            raise TypeError(f"model name must be str, got {type(key).__name__}")
        if not key.isidentifier() or keyword.iskeyword(key):
            raise ValueError(
                f"model name must be a valid Python identifier, got {key!r}"
            )
        if key.startswith("_") or key in cls._reserved_names():
            raise ValueError(f"model name is reserved, got {key!r}")
        return key

    @staticmethod
    def _is_opensim_model(model: object) -> TypeGuard[opensim.Model]:
        return isinstance(model, import_opensim().Model)

    @staticmethod
    def _validate_model(model: object) -> opensim.Model:
        if not Framework._is_opensim_model(model):
            raise TypeError(
                "Framework values must be opensim.Model instances; "
                f"got {type(model).__name__}"
            )
        return model

    @classmethod
    def _validate_pair(
        cls,
        target: dict[str, opensim.Model],
        key: object,
        value: object,
    ) -> tuple[str, opensim.Model]:
        name = cls._validate_key(key)
        model = cls._validate_model(value)
        if any(other_name != name and other is model for other_name, other in target.items()):
            raise ValueError("the same model instance cannot be stored under multiple names")
        return name, model

    def _invalidate_snapshot(self) -> None:
        object.__setattr__(self, "_active_composite", None)
        object.__setattr__(self, "_active_state", None)
        object.__setattr__(self, "_active_path_map", {})

    def __setitem__(self, key: str, value: opensim.Model) -> None:
        name, model = self._validate_pair(dict(self), key, value)
        dict.__setitem__(self, name, model)
        self._invalidate_snapshot()

    def update(
        self,
        models: Any = (),
        /,
        **named_models: opensim.Model,
    ) -> None:
        """Validate and insert models from a mapping or iterable of pairs.

        Parameters
        ----------
        models : Mapping or iterable of pairs, optional
            Models to add or replace.
        **named_models : opensim.Model
            Additional models to add by keyword.

        Raises
        ------
        TypeError
            If any key or value has an unsupported type.
        ValueError
            If any name is invalid/reserved or a model instance would be
            stored under multiple names.
        """
        if isinstance(models, Mapping):
            pairs = list(models.items())
        else:
            pairs = list(models)
        pairs.extend(named_models.items())

        staged = dict(self)
        for pair in pairs:
            try:
                key, value = pair
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "update() expects an iterable of (name, model) pairs"
                ) from error
            name, model = self._validate_pair(staged, key, value)
            staged[name] = model

        dict.clear(self)
        dict.update(self, staged)
        self._invalidate_snapshot()

    def setdefault(self, key: str, default: opensim.Model | None = None) -> opensim.Model:
        """Return a model by name, inserting ``default`` when absent.

        Parameters
        ----------
        key : str
            Model name.
        default : opensim.Model
            Model to insert if ``key`` is absent.

        Returns
        -------
        opensim.Model
            Existing or newly inserted model.

        Raises
        ------
        TypeError
            If the default is not an ``opensim.Model``.
        ValueError
            If the name is invalid or reserved.
        """
        name = self._validate_key(key)
        if dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        model = self._validate_model(default)
        self[name] = model
        return model

    def __ior__(self, other: Any) -> Self:
        self.update(other)
        return self

    @classmethod
    def fromkeys(
        cls,
        keys: Iterable[Any],
        value: Any = None,
    ) -> Self:
        """Create a framework with each key assigned to ``value``.

        As in normal framework insertion, one mutable model instance cannot
        be stored under more than one key.
        """
        result = cls()
        for key in keys:
            result[key] = cls._validate_model(value)
        return result

    def add(self, model: opensim.Model, name: str) -> opensim.Model:
        """Add one native model and return it.

        Parameters
        ----------
        model : opensim.Model
            Model instance to store.
        name : str
            Unique identifier used as the framework key.

        Returns
        -------
        opensim.Model
            The inserted model.
        """
        self[name] = model
        return model

    def __getattr__(self, name: str) -> opensim.Model:
        try:
            return dict.__getitem__(self, name)
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: opensim.Model) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dict.keys(self)))

    def pop(self, key: str, *default: Any) -> Any:
        """Remove and return a named model, optionally returning a default."""
        name = self._validate_key(key)
        if len(default) > 1:
            raise TypeError(f"pop expected at most 2 arguments, got {len(default) + 1}")
        try:
            result = dict.pop(self, name)
            self._invalidate_snapshot()
            return result
        except KeyError:
            if default:
                return default[0]
            raise

    def popitem(self) -> tuple[str, opensim.Model]:
        """Remove and return the most recently inserted model."""
        result = dict.popitem(self)
        self._invalidate_snapshot()
        return result

    def __delitem__(self, key: str) -> None:
        name = self._validate_key(key)
        dict.__delitem__(self, name)
        self._invalidate_snapshot()

    def clear(self) -> None:
        """Remove every model from the framework."""
        dict.clear(self)
        self._invalidate_snapshot()

    def copy(self) -> Self:
        """Return a shallow copy with the same model objects and couplings."""
        result = type(self)(self)
        object.__setattr__(result, "_couplings", dict(self._couplings))
        return result

    def __copy__(self) -> Self:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Clone every native model into an independent framework.

        OpenSim's model copy constructor returns a native ``opensim.Model``;
        specialized Python subclasses are therefore copied as native models.
        """
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        result = type(self)()
        memo[id(self)] = result
        opensim = import_opensim()
        for name, model in dict.items(self):
            result[name] = opensim.Model(model)
        object.__setattr__(result, "_couplings", dict(self._couplings))
        return result

    @property
    def composite(self) -> opensim.Model:
        """Return the current native snapshot, building it if necessary.

        Call :meth:`initSystem` to explicitly rebuild after modifying a member
        model directly; OpenSim does not provide mutation notifications.
        """
        composite = self._active_composite
        if composite is None:
            self.initSystem()
            composite = self._active_composite
        if composite is None:
            raise RuntimeError("failed to initialize the composite model")
        return composite

    @property
    def path_map(self) -> Mapping[tuple[str, str], str]:
        """Map ``(member_name, member_component_path)`` to composite paths.

        The mapping describes the current composite snapshot.
        """
        if self._active_composite is None:
            self.initSystem()
        return MappingProxyType(self._active_path_map.copy())

    @property
    def reverse_path_map(self) -> Mapping[str, tuple[str, str]]:
        """Map composite component paths back to their member and source path."""
        return MappingProxyType(
            {
                composite_path: source
                for source, composite_path in self.path_map.items()
            }
        )

    @property
    def state(self) -> Any:
        """Return the state associated with the active composite snapshot."""
        if self._active_state is None:
            return self.initSystem()
        return self._active_state

    @property
    def gravity(self) -> tuple[float, float, float]:
        """Return the common gravity vector used by the active composite."""
        vector = self.composite.getGravity()
        return (
            float(vector.get(0)),
            float(vector.get(1)),
            float(vector.get(2)),
        )

    def set_gravity(self, gravity: Any) -> None:
        """Set a common finite gravity vector on every member model.

        Parameters
        ----------
        gravity : opensim.Vec3 or iterable of float
            Three-dimensional gravity vector in metres per second squared.

        Raises
        ------
        ValueError
            If the vector does not contain exactly three finite values or the
            framework has no member models.
        """
        if isinstance(gravity, opensim.Vec3):
            values = tuple(float(gravity.get(index)) for index in range(3))
        else:
            if isinstance(gravity, (str, bytes)):
                raise ValueError("gravity must contain three finite values")
            try:
                values = tuple(float(value) for value in gravity)
            except (TypeError, ValueError) as error:
                raise ValueError("gravity must contain three finite values") from error
        if (
            len(values) != 3
            or any(not isfinite(value) for value in values)
            or not self
        ):
            raise ValueError(
                "gravity must contain three finite values and the framework "
                "must contain at least one model"
            )
        vector = opensim.Vec3(values[0], values[1], values[2])
        for model in self.values():
            model.setGravity(vector)
        self._invalidate_snapshot()

    def initSystem(self) -> Any:
        """Build and initialize a composite snapshot from the current members.

        Returns
        -------
        opensim.State
            The initialized state of the newly built composite model.
        """
        composite, path_map = self._build_merge(
            None, list(dict.keys(self)), list(self._couplings.values())
        )
        state = composite.initSystem()
        self._push_member_speeds(composite, state, path_map)
        object.__setattr__(self, "_active_composite", composite)
        object.__setattr__(self, "_active_state", state)
        object.__setattr__(self, "_active_path_map", path_map)
        return state

    def update_state(self) -> Any:
        """Copy the active composite's coordinates and speeds back to members.

        This performs the reverse synchronization after changing or simulating
        the active composite. Call :meth:`initSystem` first if member models
        were changed directly since the active snapshot was built.

        Returns
        -------
        opensim.State
            The synchronized composite state.
        """
        composite = self._active_composite
        state = self._active_state
        if composite is None or state is None:
            state = self.initSystem()
            composite = self._active_composite
        if composite is None:
            raise RuntimeError("failed to initialize the composite model")

        paths_by_member: dict[str, list[tuple[str, str]]] = {}
        for (member_name, source_path), composite_path in self._active_path_map.items():
            paths_by_member.setdefault(member_name, []).append(
                (source_path, composite_path)
            )

        for member_name, paths in paths_by_member.items():
            member = dict.__getitem__(self, member_name)
            coordinates = []
            for source_path, composite_path in paths:
                source_component = member.getComponent(source_path)
                composite_component = composite.getComponent(composite_path)
                source_coordinate = opensim.Coordinate.safeDownCast(source_component)
                composite_coordinate = opensim.Coordinate.safeDownCast(
                    composite_component
                )
                if source_coordinate is None and composite_coordinate is None:
                    continue
                if source_coordinate is None or composite_coordinate is None:
                    raise RuntimeError(
                        "Coordinate path mapping no longer resolves: "
                        f"{member_name!r} {source_path!r} -> {composite_path!r}"
                    )
                coordinates.append((source_coordinate, composite_coordinate))
            if not coordinates:
                continue

            member_state = getattr(member, "state", None)
            if member_state is None:
                try:
                    member_state = member.updWorkingState()
                except RuntimeError:
                    member_state = member.initSystem()
            for source_coordinate, composite_coordinate in coordinates:
                source_coordinate.setValue(
                    member_state, composite_coordinate.getValue(state), False
                )
                source_coordinate.setSpeedValue(
                    member_state, composite_coordinate.getSpeedValue(state)
                )
            member.realizePosition(member_state)
            member.realizeVelocity(member_state)
            if hasattr(member, "state"):
                member.state = member_state

        return state

    def _push_member_speeds(
        self,
        composite: opensim.Model,
        state: Any,
        path_map: Mapping[tuple[str, str], str],
    ) -> None:
        # OpenSim ignores a coordinate's default speed in initSystem().
        for (member_name, source_path), composite_path in path_map.items():
            member = dict.__getitem__(self, member_name)
            member_state = getattr(member, "state", None)
            if member_state is None:
                try:
                    member_state = member.getWorkingState()
                except RuntimeError:
                    continue
            source = opensim.Coordinate.safeDownCast(member.getComponent(source_path))
            target = opensim.Coordinate.safeDownCast(
                composite.getComponent(composite_path)
            )
            if source is not None and target is not None:
                target.setSpeedValue(state, source.getSpeedValue(member_state))

    def realize(self, stage: Any = None) -> Any:
        """Realize the active composite through ``stage`` (Position by default).

        Parameters
        ----------
        stage : opensim.Stage constant or None, default=None
            One of ``opensim.Stage.Time``, ``Position``, ``Velocity``,
            ``Dynamics``, ``Acceleration`` or ``Report``.

        Returns
        -------
        opensim.State
            The realized composite state.

        Raises
        ------
        ValueError
            If ``stage`` is not one of the supported stages.
        """
        composite = self._active_composite
        if composite is None:
            self.initSystem()
            composite = self._active_composite
        if composite is None:
            raise RuntimeError("failed to initialize the composite model")
        if stage is None:
            stage = opensim.Stage.Position
        methods = {
            opensim.Stage.Time: "realizeTime",
            opensim.Stage.Position: "realizePosition",
            opensim.Stage.Velocity: "realizeVelocity",
            opensim.Stage.Dynamics: "realizeDynamics",
            opensim.Stage.Acceleration: "realizeAcceleration",
            opensim.Stage.Report: "realizeReport",
        }
        if stage not in methods:
            raise ValueError("unsupported realization stage")
        getattr(composite, methods[stage])(self.state)
        return self.state

    def forward(self, final_time: float) -> Any:
        """Integrate the composite to ``final_time`` and synchronize members.

        Parameters
        ----------
        final_time : float
            Absolute simulation time, in seconds. Must be later than the
            current composite state time.

        Returns
        -------
        opensim.State
            The integrated composite state.
        """
        if (
            isinstance(final_time, bool)
            or not isinstance(final_time, (int, float))
            or not isfinite(final_time)
        ):
            raise ValueError("final_time must be a finite number")
        composite = self._active_composite
        if composite is None:
            self.initSystem()
            composite = self._active_composite
        if composite is None:
            raise RuntimeError("failed to initialize the composite model")
        state = self.state
        if final_time <= state.getTime():
            raise ValueError("final_time must be later than the current state time")

        self._check_integrable_joints(composite)
        manager = opensim.Manager(composite)
        manager.initialize(state)
        state = opensim.State(manager.integrate(float(final_time)))
        object.__setattr__(self, "_active_state", state)
        self.update_state()
        return state

    def simulate(self, final_time: float) -> Any:
        """Alias for :meth:`forward`."""
        return self.forward(final_time)

    @staticmethod
    def _check_integrable_joints(model: opensim.Model) -> None:
        for index in range(model.getJointSet().getSize()):
            joint = model.getJointSet().get(index)
            if opensim.FreeJoint.safeDownCast(joint) is not None:
                raise RuntimeError(
                    "OpenSim Manager integration is unsafe for "
                    f"opensim.FreeJoint {joint.getName()!r}; replace it with "
                    "a BallJoint+SliderJoint pair or a CustomJoint."
                )

    def settle_under_gravity(
        self,
        duration_s: float,
        *,
        accuracy: float = 1e-4,
        max_step_size_s: float = 0.01,
    ) -> Any:
        """Integrate the active composite forward for ``duration_s`` seconds.

        The integration includes gravity and every force present in the
        composite; it does not test whether the system has reached equilibrium.
        OpenSim Manager may fail natively on some muscle-driven models, so test
        with a short duration on the target model first.
        """
        values = (duration_s, accuracy, max_step_size_s)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError(
                "duration_s, accuracy, and max_step_size_s must be "
                "positive finite numbers"
            )
        composite = self._active_composite
        if composite is None:
            self.initSystem()
            composite = self._active_composite
        if composite is None:
            raise RuntimeError("failed to initialize the composite model")
        self._check_integrable_joints(composite)
        state = self.state
        manager = opensim.Manager(composite)
        manager.setIntegratorAccuracy(float(accuracy))
        manager.setIntegratorMaximumStepSize(float(max_step_size_s))
        manager.initialize(state)
        state = opensim.State(
            manager.integrate(state.getTime() + float(duration_s))
        )
        object.__setattr__(self, "_active_state", state)
        self.update_state()
        return state

    def merge_to_model(
        self,
        name: str | None = None,
        only: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        include_couplings: bool = True,
        return_mapping: bool = False,
    ) -> opensim.Model | tuple[opensim.Model, dict[tuple[str, str], str]]:
        """Create an independent native model containing framework models.

        Components are cloned into a new ``opensim.Model``. The source
        models and their states are not modified; current coordinate values
        and speeds become defaults in the merged snapshot. Name collisions
        are resolved by prefixing the colliding component with its framework
        key, and joints attached to a member model's ground remain attached
        to the merged model's shared ground. The result is independent of the
        framework: later changes to either side do not affect the other.

        Parameters
        ----------
        name : str or None, default=None
            Name of the merged model; ``"framework"`` when omitted.
        only : iterable of str or None, default=None
            Member names to include. All members when omitted.
        exclude : iterable of str or None, default=None
            Member names to leave out, applied after ``only``.
        include_couplings : bool, default=True
            Whether to recreate the framework's couplings. A coupling that
            references a member left out of the selection is skipped.
        return_mapping : bool, default=False
            Also return the ``(member, source_path) -> merged_path`` mapping.

        Returns
        -------
        opensim.Model or tuple[opensim.Model, dict[tuple[str, str], str]]
            A new, initialized model, optionally with the path mapping.

        Raises
        ------
        KeyError
            If ``only`` or ``exclude`` names a model that does not exist.
        ValueError
            If gravities differ, no model is selected, or a coupling no longer
            resolves.
        """
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a non-empty string")
        members = list(dict.keys(self))
        for label, names in (("only", only), ("exclude", exclude)):
            if names is None:
                continue
            if isinstance(names, str):
                names = [names]
            names = list(names)
            missing = [item for item in names if item not in self]
            if missing:
                raise KeyError(f"{label} names unknown models: {missing}")
            if label == "only":
                members = [item for item in members if item in names]
            else:
                members = [item for item in members if item not in names]
        if not members:
            raise ValueError("no framework models selected for the merge")
        selected = set(members)
        couplings = (
            [item for item in self._couplings.values() if item.members <= selected]
            if include_couplings
            else []
        )
        merged, paths = self._build_merge(name, members, couplings)
        # Meshes live in the members' directories; keep them reachable by the viewer.
        merged.geometry_directories = tuple(
            dict.fromkeys(
                [
                    *(
                        Path(directory)
                        for member in members
                        for directory in getattr(
                            dict.__getitem__(self, member), "geometry_directories", ()
                        )
                    ),
                    MESHES_DIR,
                ]
            )
        )
        if return_mapping:
            return merged, paths
        return merged

    def _build_merge(
        self,
        name: str | None,
        members: list[str],
        couplings: list[Coupling],
    ) -> tuple[opensim.Model, dict[tuple[str, str], str]]:
        merged = opensim.Model()
        merged.setName(name or "framework")
        existing_names: dict[str, set[str]] = {
            set_name: set() for set_name, _ in _MODEL_SETS
        }
        component_paths: dict[tuple[str, str], str] = {}
        gravity: tuple[float, float, float] | None = None

        for model_name in members:
            model = dict.__getitem__(self, model_name)
            model_gravity = (
                float(model.getGravity().get(0)),
                float(model.getGravity().get(1)),
                float(model.getGravity().get(2)),
            )
            if gravity is None:
                gravity = model_gravity
            elif any(
                not isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
                for left, right in zip(gravity, model_gravity)
            ):
                raise ValueError(
                    "All framework models must use the same gravity to compose"
                )
            coordinate_values = _current_coordinate_values(model)
            source = opensim.Model(model)
            _apply_coordinate_defaults(source, coordinate_values)
            source.initSystem()

            ground_parent_joints: set[str] = set()
            ground_child_joints: set[str] = set()
            for joint in _set_items(source.getJointSet()):
                if opensim.Ground.safeDownCast(joint.getParentFrame()) is not None:
                    ground_parent_joints.add(joint.getName())
                if opensim.Ground.safeDownCast(joint.getChildFrame()) is not None:
                    ground_child_joints.add(joint.getName())

            clones: dict[str, list[Any]] = {
                set_name: [] for set_name, _ in _MODEL_SETS
            }
            renamed_paths: dict[str, str] = {}
            joints_by_original_name: dict[str, Any] = {}
            for set_name, getter_name in _MODEL_SETS:
                source_set = getattr(source, getter_name.replace("upd", "get"))()
                for component in _set_items(source_set):
                    clone = component.clone()
                    original_name = clone.getName()
                    if original_name in existing_names[set_name]:
                        final_name = _unique_name(
                            model_name, original_name, existing_names[set_name]
                        )
                        renamed_paths[f"/{set_name}/{original_name}"] = (
                            f"/{set_name}/{final_name}"
                        )
                        clone.setName(final_name)
                    else:
                        final_name = original_name
                    existing_names[set_name].add(final_name)
                    if final_name != original_name:
                        renamed_paths[f"/{set_name}/{original_name}"] = (
                            f"/{set_name}/{final_name}"
                        )
                    clones[set_name].append(clone)
                    if set_name == "jointset":
                        joints_by_original_name[original_name] = clone

            for set_name, getter_name in _MODEL_SETS:
                target_set = getattr(merged, getter_name)()
                for clone in clones[set_name]:
                    target_set.adoptAndAppend(clone)
                    clone.thisown = False

            merged.finalizeFromProperties()
            for components in clones.values():
                for clone in components:
                    _patch_socket_paths(clone, renamed_paths)

            for component in source.getComponentsList():
                source_path = _component_path(component)
                component_paths[(model_name, source_path)] = (
                    _remap_component_path(source_path, renamed_paths)
                )

            if ground_parent_joints or ground_child_joints:
                anchor_name = _unique_name(
                    model_name, "ground_anchor", existing_names.setdefault("frames", set())
                )
                existing_names["frames"].add(anchor_name)
                anchor = opensim.PhysicalOffsetFrame(
                    anchor_name, merged.getGround(), opensim.Transform()
                )
                merged.addComponent(anchor)
                anchor.thisown = False
                anchor_path = f"/{anchor_name}"
                for original_name in ground_parent_joints:
                    joints_by_original_name[original_name].updSocket(
                        "parent_frame"
                    ).setConnecteePath(anchor_path)
                for original_name in ground_child_joints:
                    joints_by_original_name[original_name].updSocket(
                        "child_frame"
                    ).setConnecteePath(anchor_path)

        if gravity is not None:
            merged.setGravity(opensim.Vec3(*gravity))
        if couplings:
            self._apply_couplings(merged, component_paths, couplings, existing_names)
        merged.finalizeConnections()
        merged.initSystem()
        return merged, component_paths

    @staticmethod
    def _merged_path(
        component_paths: Mapping[tuple[str, str], str],
        coupling: Coupling,
        reference: ComponentRef,
    ) -> str:
        try:
            return component_paths[reference]
        except KeyError:
            raise ValueError(
                f"coupling {coupling.name!r} no longer resolves "
                f"{reference[1]!r} in model {reference[0]!r}; remove it with "
                "uncouple()"
            ) from None

    def _apply_couplings(
        self,
        merged: opensim.Model,
        component_paths: Mapping[tuple[str, str], str],
        couplings: list[Coupling],
        existing_names: dict[str, set[str]],
    ) -> None:
        def final_name(coupling: Coupling, set_name: str = "constraintset") -> str:
            used = existing_names[set_name]
            result = coupling.name
            if result in used:
                result = _unique_name("coupling", coupling.name, used)
            used.add(result)
            return result

        for coupling in couplings:
            paths = [
                self._merged_path(component_paths, coupling, reference)
                for reference in coupling.references
            ]
            if coupling.kind == "weld":
                constraint = opensim.WeldConstraint()
                constraint.setName(final_name(coupling))
                constraint.updSocket("frame1").setConnecteePath(paths[0])
                constraint.updSocket("frame2").setConnecteePath(paths[1])
            elif coupling.kind == "point":
                constraint = opensim.PointConstraint()
                constraint.setName(final_name(coupling))
                constraint.updSocket("body_1").setConnecteePath(paths[0])
                constraint.updSocket("body_2").setConnecteePath(paths[1])
                constraint.set_location_body_1(
                    opensim.Vec3(*coupling.option("first_location"))
                )
                constraint.set_location_body_2(
                    opensim.Vec3(*coupling.option("second_location"))
                )
            elif coupling.kind == "coordinate":
                names = [path.rsplit("/", 1)[-1] for path in paths]
                coordinates = merged.getCoordinateSet()
                all_names = [
                    coordinates.get(index).getName()
                    for index in range(coordinates.getSize())
                ]
                ambiguous = sorted({item for item in names if all_names.count(item) > 1})
                if ambiguous:
                    raise ValueError(
                        f"coupling {coupling.name!r} references coordinate names "
                        f"that are not unique in the composite: {ambiguous}"
                    )
                constraint = opensim.CoordinateCouplerConstraint()
                constraint.setName(final_name(coupling))
                independent = opensim.ArrayStr()
                for item in names[:-1]:
                    independent.append(item)
                constraint.setIndependentCoordinateNames(independent)
                constraint.setDependentCoordinateName(names[-1])
                function = coupling.option("function")
                constraint.setFunction(
                    function.clone() if function is not None else opensim.LinearFunction(1.0, 0.0)
                )
            elif coupling.kind == "joint":
                joint = opensim.Joint.safeDownCast(merged.getComponent(paths[1]))
                if joint is None:
                    raise ValueError(f"coupling {coupling.name!r} no longer resolves a joint")
                # Connections are not finalized yet, so resolve the parent
                # frame from its socket path.
                frame_path = joint.getSocket("parent_frame").getConnecteePath()
                if not frame_path.startswith("/"):
                    frame_path = f"{paths[1]}/{frame_path}"
                offset = opensim.PhysicalOffsetFrame.safeDownCast(
                    merged.getComponent(frame_path)
                )
                if offset is not None:
                    offset.updSocket("parent").setConnecteePath(paths[0])
                else:
                    joint.updSocket("parent_frame").setConnecteePath(paths[0])
                continue
            else:
                template = coupling.option("component")
                component = template.clone()
                set_name, add_method = _COMPONENT_SETS[coupling.kind]
                component.setName(final_name(coupling, set_name))
                for socket_name, indices in coupling.option("sockets"):
                    socket = component.updSocket(socket_name)
                    if socket.isListSocket():
                        socket.setConnecteePath("")
                        for index in indices:
                            socket.appendConnecteePath(paths[index])
                    else:
                        socket.setConnecteePath(paths[indices[0]])
                getattr(merged, add_method)(component)
                component.thisown = False
                continue
            merged.addConstraint(constraint)
            constraint.thisown = False

    def _add_coupling(self, coupling: Coupling) -> Coupling:
        self._couplings[coupling.name] = coupling
        self._invalidate_snapshot()
        return coupling

    def _coupling_name(self, kind: str, name: str | None) -> str:
        if name is None:
            index = 1
            while f"{kind}_{index}" in self._couplings:
                index += 1
            return f"{kind}_{index}"
        if not isinstance(name, str) or not name:
            raise ValueError("coupling name must be a non-empty string")
        if name in self._couplings:
            raise ValueError(f"coupling {name!r} already exists")
        return name

    def _reference(self, reference: Any, expected: Any, label: str) -> ComponentRef:
        member_name, path = normalize_reference(reference)
        if member_name not in self:
            raise KeyError(f"unknown framework model {member_name!r}")
        member = dict.__getitem__(self, member_name)
        try:
            component = member.getComponent(path)
        except RuntimeError as error:
            raise ValueError(
                f"{path!r} does not exist in model {member_name!r}"
            ) from error
        if expected.safeDownCast(component) is None:
            raise TypeError(
                f"{label} {member_name!r}:{path!r} must be a {expected.__name__}"
            )
        return (member_name, path)

    @staticmethod
    def _location(value: Any, label: str) -> tuple[float, float, float]:
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must contain three finite values") from error
        if len(result) != 3 or any(not isfinite(item) for item in result):
            raise ValueError(f"{label} must contain three finite values")
        return (result[0], result[1], result[2])

    @staticmethod
    def _require_distinct(first: ComponentRef, second: ComponentRef) -> None:
        if first[0] == second[0]:
            raise ValueError("a coupling must link components of distinct models")

    @property
    def couplings(self) -> Mapping[str, Coupling]:
        """Return the declared couplings by name, in creation order."""
        return MappingProxyType(dict(self._couplings))

    def couple_weld(
        self,
        first: ComponentRef,
        second: ComponentRef,
        *,
        name: str | None = None,
    ) -> Coupling:
        """Weld two physical frames of distinct models in the composite.

        Parameters
        ----------
        first, second : tuple[str, str]
            ``(model_name, frame_path)`` references, for example
            ``("user", "/bodyset/hand_r")``. Use offset frames to weld at a
            specific pose.
        name : str or None, default=None
            Coupling name; generated when omitted.

        Returns
        -------
        Coupling
            The registered coupling. Member models are not modified.

        Raises
        ------
        KeyError
            If a model is unknown.
        ValueError
            If a path does not exist, the models are the same, or the name
            is already used.
        TypeError
            If a reference is malformed or not a physical frame.
        """
        left = self._reference(first, opensim.PhysicalFrame, "weld frame")
        right = self._reference(second, opensim.PhysicalFrame, "weld frame")
        self._require_distinct(left, right)
        return self._add_coupling(
            Coupling(self._coupling_name("weld", name), "weld", (left, right))
        )

    def couple_point(
        self,
        first: ComponentRef,
        second: ComponentRef,
        *,
        first_location: Any = (0.0, 0.0, 0.0),
        second_location: Any = (0.0, 0.0, 0.0),
        name: str | None = None,
    ) -> Coupling:
        """Make a point on one frame coincide with a point on another model's frame.

        Parameters
        ----------
        first, second : tuple[str, str]
            ``(model_name, frame_path)`` references.
        first_location, second_location : iterable of float, default=(0, 0, 0)
            Points in metres, expressed in each frame.
        name : str or None, default=None
            Coupling name; generated when omitted.

        Returns
        -------
        Coupling
            The registered coupling.

        Raises
        ------
        KeyError, ValueError, TypeError
            As for :meth:`couple_weld`; also if a location is not three
            finite numbers.
        """
        left = self._reference(first, opensim.PhysicalFrame, "point frame")
        right = self._reference(second, opensim.PhysicalFrame, "point frame")
        self._require_distinct(left, right)
        options = (
            ("first_location", self._location(first_location, "first_location")),
            ("second_location", self._location(second_location, "second_location")),
        )
        return self._add_coupling(
            Coupling(self._coupling_name("point", name), "point", (left, right), options)
        )

    def couple_coordinate(
        self,
        independent: ComponentRef | Iterable[ComponentRef],
        dependent: ComponentRef,
        *,
        function: Any = None,
        name: str | None = None,
    ) -> Coupling:
        """Drive a coordinate of one model from coordinates of other models.

        Parameters
        ----------
        independent : tuple[str, str] or iterable of tuple[str, str]
            ``(model_name, coordinate_path)`` reference(s) of the driving
            coordinate(s), for example ``("a", "/jointset/j/j_coord_0")``.
        dependent : tuple[str, str]
            Coordinate whose value follows the function.
        function : opensim.Function or None, default=None
            Function of the independent values; identity
            (``LinearFunction(1, 0)``) when omitted. It is cloned for every
            rebuild.
        name : str or None, default=None
            Coupling name; generated when omitted.

        Returns
        -------
        Coupling
            The registered coupling.

        Raises
        ------
        KeyError, ValueError, TypeError
            As for :meth:`couple_weld`; also if the dependent model drives
            itself or ``function`` is not an ``opensim.Function``.

        Notes
        -----
        OpenSim resolves coordinate couplers by coordinate name, so the names
        used must be unique in the composite; otherwise rebuilding raises
        ``ValueError``.
        """
        if (
            isinstance(independent, Sequence)
            and len(independent) == 2
            and all(isinstance(item, str) for item in independent)
        ):
            sources = [independent]
        else:
            sources = list(independent)
        if not sources:
            raise ValueError("at least one independent coordinate is required")
        references = [
            self._reference(item, opensim.Coordinate, "independent coordinate")
            for item in sources
        ]
        target = self._reference(dependent, opensim.Coordinate, "dependent coordinate")
        for reference in references:
            self._require_distinct(reference, target)
        if function is not None and not isinstance(function, opensim.Function):
            raise TypeError("function must be an opensim.Function or None")
        return self._add_coupling(
            Coupling(
                self._coupling_name("coordinate", name),
                "coordinate",
                (*references, target),
                (("function", function.clone() if function is not None else None),),
            )
        )

    def couple_joint(
        self,
        parent_frame: ComponentRef,
        child_joint: ComponentRef,
        *,
        name: str | None = None,
    ) -> Coupling:
        """Attach a model's ground-rooted joint to a frame of another model.

        The joint's parent side, which is attached to ground in its own
        model, is attached to ``parent_frame`` in the composite, so the child
        model moves relative to the parent model through that joint.

        Parameters
        ----------
        parent_frame : tuple[str, str]
            ``(model_name, frame_path)`` of the frame that becomes the parent.
        child_joint : tuple[str, str]
            ``(model_name, joint_path)`` of a joint attached to ground, for
            example ``("tool", "/jointset/tool_root")``.
        name : str or None, default=None
            Coupling name; generated when omitted.

        Returns
        -------
        Coupling
            The registered coupling.

        Raises
        ------
        KeyError, ValueError, TypeError
            As for :meth:`couple_weld`; also if the joint is not attached to
            ground or is already the child of another joint coupling.
        """
        parent = self._reference(parent_frame, opensim.PhysicalFrame, "parent frame")
        child = self._reference(child_joint, opensim.Joint, "child joint")
        self._require_distinct(parent, child)
        joint = opensim.Joint.safeDownCast(
            dict.__getitem__(self, child[0]).getComponent(child[1])
        )
        frame = joint.getParentFrame() if joint is not None else None
        attached = opensim.Ground.safeDownCast(frame)
        offset = opensim.PhysicalOffsetFrame.safeDownCast(frame)
        if attached is None and offset is not None:
            attached = opensim.Ground.safeDownCast(offset.getParentFrame())
        if attached is None:
            raise ValueError("child_joint must be attached to ground in its own model")
        if any(
            item.kind == "joint" and item.references[1] == child
            for item in self._couplings.values()
        ):
            raise ValueError("child_joint is already coupled to a parent frame")
        return self._add_coupling(
            Coupling(self._coupling_name("joint", name), "joint", (parent, child))
        )

    def _add_component(
        self,
        kind: str,
        expected: Any,
        component: Any,
        sockets: Mapping[str, Any],
        name: str | None,
    ) -> Coupling:
        if not isinstance(component, expected):
            raise TypeError(f"component must be an opensim.{expected.__name__}")
        if not isinstance(sockets, Mapping) or not sockets:
            raise ValueError("sockets must map socket names to member components")
        template = component.clone()
        references: list[ComponentRef] = []
        assignments: list[tuple[str, tuple[int, ...]]] = []
        for socket_name, value in sockets.items():
            if socket_name not in list(template.getSocketNames()):
                raise ValueError(
                    f"{type(component).__name__} has no socket named {socket_name!r}"
                )
            if (
                isinstance(value, Sequence)
                and len(value) == 2
                and all(isinstance(item, str) for item in value)
            ):
                items = [value]
            else:
                items = list(value)
            if not items:
                raise ValueError(f"socket {socket_name!r} needs at least one component")
            if len(items) > 1 and not template.updSocket(socket_name).isListSocket():
                raise ValueError(f"socket {socket_name!r} accepts a single component")
            indices = []
            for item in items:
                member_name, path = normalize_reference(item)
                if member_name not in self:
                    raise KeyError(f"unknown framework model {member_name!r}")
                try:
                    dict.__getitem__(self, member_name).getComponent(path)
                except RuntimeError as error:
                    raise ValueError(
                        f"{path!r} does not exist in model {member_name!r}"
                    ) from error
                indices.append(len(references))
                references.append((member_name, path))
            assignments.append((socket_name, tuple(indices)))
        if name is None:
            candidate = str(template.getName())
            name = candidate if candidate and candidate not in self._couplings else None
        return self._add_coupling(
            Coupling(
                self._coupling_name(kind, name),
                kind,  # type: ignore[arg-type]
                tuple(references),
                (("component", template), ("sockets", tuple(assignments))),
            )
        )

    def add_force(
        self,
        force: Any,
        sockets: Mapping[str, Any],
        *,
        name: str | None = None,
    ) -> Coupling:
        """Add a force to the composite, wiring its sockets to member components.

        Parameters
        ----------
        force : opensim.Force
            Template force; it is cloned, so later changes to it do not
            affect the framework.
        sockets : Mapping[str, tuple[str, str] or iterable of tuple[str, str]]
            Socket name to ``(model_name, component_path)`` reference (or a
            list of references for list sockets), for example
            ``{"frame1": ("a", "/bodyset/b"), "frame2": ("b", "/bodyset/c")}``.
        name : str or None, default=None
            Coupling name, also used for the force; defaults to the template
            name or a generated one.

        Returns
        -------
        Coupling
            The registered coupling.

        Raises
        ------
        TypeError
            If ``force`` is not an ``opensim.Force`` or a reference is malformed.
        KeyError
            If a model is unknown.
        ValueError
            If a socket or path does not exist, or the name is already used.

        Notes
        -----
        References and socket names are validated now; socket type
        compatibility is checked by OpenSim when the composite is built.
        """
        return self._add_component("force", opensim.Force, force, sockets, name)

    def add_constraint(
        self,
        constraint: Any,
        sockets: Mapping[str, Any],
        *,
        name: str | None = None,
    ) -> Coupling:
        """Add a constraint to the composite; see :meth:`add_force`.

        Parameters
        ----------
        constraint : opensim.Constraint
            Template constraint, cloned.
        sockets : Mapping[str, tuple[str, str] or iterable of tuple[str, str]]
            Socket name to member component reference(s).
        name : str or None, default=None
            Coupling name; defaults to the template name or a generated one.

        Returns
        -------
        Coupling
            The registered coupling.
        """
        return self._add_component(
            "constraint", opensim.Constraint, constraint, sockets, name
        )

    def add_controller(
        self,
        controller: Any,
        sockets: Mapping[str, Any],
        *,
        name: str | None = None,
    ) -> Coupling:
        """Add a controller to the composite; see :meth:`add_force`.

        Parameters
        ----------
        controller : opensim.Controller
            Template controller, cloned.
        sockets : Mapping[str, tuple[str, str] or iterable of tuple[str, str]]
            Socket name to member component reference(s), for example
            ``{"actuators": [("a", "/forceset/m1"), ("b", "/forceset/m2")]}``.
        name : str or None, default=None
            Coupling name; defaults to the template name or a generated one.

        Returns
        -------
        Coupling
            The registered coupling.
        """
        return self._add_component(
            "controller", opensim.Controller, controller, sockets, name
        )

    def uncouple(self, name: str | None = None) -> None:
        """Remove a coupling, or every coupling when ``name`` is ``None``.

        Parameters
        ----------
        name : str or None, default=None
            Coupling name. Member models are never modified.

        Raises
        ------
        KeyError
            If ``name`` is not a declared coupling.
        """
        if name is None:
            self._couplings.clear()
        else:
            del self._couplings[name]
        self._invalidate_snapshot()

    def show(
        self,
        motion: Any = None,
        *,
        loop: bool = False,
        fps: float = 30.0,
    ) -> Any:
        """Open the viewer for a merged snapshot of this framework.

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
        from ..viewer import VTKViewer

        return VTKViewer().show(self, motion, loop=loop, fps=fps)
