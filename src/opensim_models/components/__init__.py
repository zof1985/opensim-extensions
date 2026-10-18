"""Python-friendly wrappers around individual OpenSim components.

``OpenSimModel`` is a *container* (one ``opensim.Model``); the parts inside
it -- a ``Body``, a ``Joint``, a ``Marker``, ... -- used to come back from
its accessors (``.body(name)``, ``.bodies``, ...) as the raw SWIG-bound
``opensim.*`` object, with the full (and large) C++-mirroring API attached
and nothing about it documented in a way an IDE could show. Every class
here wraps one such raw object instead, exposing only a small,
consistently-named set of properties/``set_x()`` method pairs (never an
``@x.setter`` -- matching every other model class in this package, e.g.
:class:`~opensim_models.components.Box`), with real docstrings.

This wraps the categories :class:`~opensim_models.model.OpenSimModel`
already directly touches -- the 8 ``_MERGE_SETS`` categories (body, joint,
force, marker, constraint, controller, contact geometry, probe) plus
coordinate and (standalone) frame -- not OpenSim's entire ~800-class
surface. ``Force``/``Constraint``/``Controller``/``ContactGeometry``/
``Probe``/``Frame`` are thin, intentionally minimal wrappers for now
(OpenSim has many concrete subtypes of each, with very different APIs);
the pattern here -- a :class:`_ComponentWrapper` subclass plus
property/``set_x()`` pairs -- is meant to be extended the same way later
for any subtype that needs its own dedicated properties.

Every wrapper's :attr:`~_ComponentWrapper.raw` is the escape hatch back to
the underlying ``opensim`` object, for anything not (yet) wrapped here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal import geometry as _geometry

if TYPE_CHECKING:
    from ..model import OpenSimModel

__all__ = [
    "Body",
    "Coordinate",
    "Marker",
    "Muscle",
    "Joint",
    "Frame",
    "OffsetFrame",
    "Force",
    "ExponentialContactForce",
    "Constraint",
    "WeldConstraint",
    "PointConstraint",
    "ConstantDistanceConstraint",
    "Controller",
    "ContactGeometry",
    "ContactSphere",
    "ContactHalfSpace",
    "ContactMesh",
    "Probe",
]


def _positive(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"value must be strictly positive, got {value!r}")
    return float(value)


def _position_and_rotation_in_ground(
    owner: "OpenSimModel", frame: Any
) -> tuple[Any, Any]:
    """Return ``frame``'s current ``(position, rotation)`` in the ground frame.

    ``position`` is a shape-``(3,)`` numpy array, in metres; ``rotation`` is
    the matching shape-``(3, 3)`` rotation matrix (ground axes expressed in
    terms of ``frame``'s own axes, i.e. ``ground_vector = rotation @
    frame_vector``). Realizes ``owner.state`` through ``Stage::Position``
    first (``owner.model.realizePosition``), so this is always safe to call
    regardless of what has or hasn't been realized since the last posture
    change.

    This is the one piece of raw-SWIG plumbing -- ``getPositionInGround()``/
    ``getRotationInGround().asMat33()``/rebuilding a plain 3x3 numpy array
    from it element by element -- every ground-frame position/orientation
    property in this module is now built on top of
    (:attr:`Body.position_global`, :attr:`Body.com`, :attr:`Body.inclination`,
    :attr:`Body.corners`, :class:`Marker`/:class:`ContactGeometry`/
    :class:`Joint`'s own ``position_global``, :class:`Box`/:class:`Screen`'s
    ``origin``/``angle_deg``/``corners``), instead of re-deriving it inline
    in each one -- the same hand-duplicated dance that previously showed up
    independently in at least four different places across this package.

    Parameters
    ----------
    owner : OpenSimModel
        The model ``frame`` belongs to -- needed to realize/read its state.
    frame : opensim.Frame (or anything with matching getPositionInGround/getRotationInGround methods)
        The frame to query (e.g. an ``opensim.Body``, a
        ``PhysicalOffsetFrame``, or ``opensim.Ground`` itself).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(position, rotation)`` -- ``position`` has shape ``(3,)``,
        ``rotation`` has shape ``(3, 3)``.
    """
    owner.model.realizePosition(owner.state)
    position = np.asarray(frame.getPositionInGround(owner.state).to_numpy())
    rotation_matrix = frame.getRotationInGround(owner.state).asMat33()
    rotation = np.array(
        [[rotation_matrix.get(i, j) for j in range(3)] for i in range(3)]
    )
    return position, rotation


def _point_in_ground(
    owner: "OpenSimModel", frame: Any, local_point: Any
) -> tuple[float, float, float]:
    """Transform ``local_point`` (metres, in ``frame``'s own axes) into the ground frame.

    Built on :func:`_position_and_rotation_in_ground`; shared by every
    property that must express an offset *within* some frame (a body's own
    ``mass_center``, a marker's or contact geometry's ``location``) as a
    ground-frame point.

    Parameters
    ----------
    owner : OpenSimModel
        The model ``frame`` belongs to.
    frame : opensim.Frame
        The frame ``local_point`` is expressed in.
    local_point : sequence of 3 floats
        Point, in metres, in ``frame``'s own local axes.

    Returns
    -------
    tuple[float, float, float]
        The equivalent point in the ground frame, in metres.
    """
    position, rotation = _position_and_rotation_in_ground(owner, frame)
    local = np.asarray(local_point, dtype=float)
    return tuple(float(value) for value in (position + rotation @ local))


class _ComponentWrapper:
    """Base for every Python-friendly wrapper around a raw opensim component.

    Parameters
    ----------
    owner : OpenSimModel
        The model this component belongs to -- needed for anything that
        reads/writes ``owner.state`` or calls back into OpenSim through
        ``owner.model``/``owner.opensim``.
    raw : Any
        The underlying raw ``opensim`` object (e.g. an ``opensim.Body``).
    """

    def __init__(self, owner: "OpenSimModel", raw: Any) -> None:
        self._owner = owner
        self._raw = raw

    @property
    def name(self) -> str:
        """OpenSim name of this component."""
        return self._raw.getName()

    def set_name(self, name: str) -> None:
        """Rename this component.

        Also re-finalizes ``owner.model``'s connections, so every other
        component's socket that refers to this one (e.g. a joint's
        offset frame naming its parent/child body) gets its stored
        connectee path refreshed to the new name. Skipping this would
        leave those paths pointing at the old name -- harmless for the
        live, already-resolved connection (it keeps working off the
        direct object reference), but fatal the next time the model is
        cloned or merged elsewhere (:meth:`~opensim_models.model.OpenSimModel.copy`/
        :meth:`~opensim_models.model.OpenSimModel.add_model`/``+``), since
        that re-resolves every connection from its stored path string: a
        stale one raises a native ``RuntimeError`` from
        ``finalizeConnections()`` (``Component ... could not find
        '/bodyset/<old name>'``) instead of finding the renamed component.
        This is purely a bookkeeping refresh (no structural change), so
        ``owner.state`` stays valid -- no :meth:`~opensim_models.model.OpenSimModel.reinitialize`
        needed.

        Parameters
        ----------
        name : str
            New OpenSim name.
        """
        self._raw.setName(name)
        self._owner.model.finalizeConnections()

    @property
    def raw(self) -> Any:
        """The underlying raw ``opensim`` object, for anything this wrapper doesn't cover."""
        return self._raw

    def __eq__(self, other: object) -> bool:
        # SWIG hands out a fresh Python proxy on every access (same reason
        # model.py keeps a pointer-identity _owners map, see its
        # _register_owner/_find_owner) -- plain `is`/default `==` would
        # wrongly treat two wrappers around the *same* underlying component
        # as different objects.
        if isinstance(other, _ComponentWrapper):
            return int(self._raw.this) == int(other._raw.this)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(int(self._raw.this))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class Body(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Body``."""

    @property
    def mass(self) -> float:
        """Mass, in kilograms."""
        return float(self._raw.getMass())

    def set_mass(self, kilograms: float) -> None:
        """Set this body's mass and rebuild the system so the change takes effect.

        A body's mass feeds into the multibody system's mass matrix, which
        OpenSim builds once in ``initSystem()``: changing the property
        alone would have no effect on dynamics until the system is
        rebuilt, so this calls :meth:`~opensim_models.model.OpenSimModel.reinitialize`
        afterward, preserving the current posture and velocities.

        Parameters
        ----------
        kilograms : float
            New mass, in kilograms. Must be strictly positive.

        Raises
        ------
        ValueError
            If ``kilograms`` is not finite or not strictly positive.
        """
        self._raw.setMass(_positive(kilograms))
        self._owner.reinitialize()

    def _position_and_rotation(self) -> tuple[Any, Any]:
        return _position_and_rotation_in_ground(self._owner, self._raw)

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This body's own frame origin, in the ground frame, in metres.

        Computed fresh from the model's current state on every access (via
        :func:`_position_and_rotation_in_ground`), so it always reflects
        this body's current placement -- never a stale cached value. Not
        the same thing as :attr:`com` (which additionally accounts for
        this body's own ``mass_center`` offset, so the two only coincide
        when ``mass_center`` is ``(0, 0, 0)``, as for e.g. :class:`Box`/
        :class:`Screen`).
        """
        position, _ = self._position_and_rotation()
        return tuple(float(value) for value in position)

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This body's own origin, expressed in its own local frame: always ``(0.0, 0.0, 0.0)``.

        A body *is* its own local frame's origin, so this is trivially
        zero -- not informative on its own, but present (alongside
        :attr:`position_global`) so code written generically against "any
        spatial element" (e.g. :class:`Marker`, :class:`ContactGeometry`,
        :class:`Joint`, all of which have a non-trivial ``position_local``)
        can read both properties uniformly, without needing to know ahead
        of time whether a given element is a whole body or an offset
        within some parent frame.
        """
        return (0.0, 0.0, 0.0)

    @property
    def com(self) -> tuple[float, float, float]:
        """Centre of mass in the ground frame, in metres.

        This body's own ``mass_center`` (in its local frame) transformed
        through its current ground-frame placement -- not the whole
        model's centre of mass (see ``Model.calcMassCenterPosition``,
        across every body, for that).
        """
        local_com = self._raw.get_mass_center()
        local = (local_com.get(0), local_com.get(1), local_com.get(2))
        return _point_in_ground(self._owner, self._raw, local)

    @property
    def inclination(self) -> tuple[float, float, float]:
        """Orientation relative to ground, as X-Y-Z body-fixed Euler degrees.

        Same convention as :attr:`opensim_models.components.Box.angle_deg`.
        """
        self._owner.model.realizePosition(self._owner.state)
        rotation = self._raw.getRotationInGround(self._owner.state)
        euler = rotation.convertRotationToBodyFixedXYZ()
        return tuple(float(np.degrees(euler.get(i))) for i in range(3))

    @property
    def corners(self) -> tuple[tuple[float, float, float], ...]:
        """Ground-frame corners of this body's attached geometry, in metres.

        The 8 corners of the axis-aligned bounding box of every geometry
        item attached to this body (local-frame bounds unioned across all
        of them, via the same ``Mesh``/``Brick``/``Cylinder``/``Sphere``
        dispatch :class:`~opensim_models._gui.visualizer.VTKVisualizer`
        uses to render them -- see :mod:`opensim_models._internal.geometry`),
        transformed through this body's current ground-frame placement.
        Empty if this body has no attached geometry, or none of it could
        be resolved (e.g. a mesh file that can't be found).
        """
        import vtk

        position, rotation = self._position_and_rotation()
        opensim = self._owner.opensim
        geometry_property = self._raw.getPropertyByName("attached_geometry")
        bounds_list = []
        for index in range(geometry_property.size()):
            geometry = self._raw.get_attached_geometry(index)
            bounds = _geometry.local_bounds(
                vtk, opensim, geometry, self._owner._resolve_geometry_file
            )
            if bounds is not None:
                bounds_list.append(bounds)
        if not bounds_list:
            return ()

        xmin = min(b[0] for b in bounds_list)
        xmax = max(b[1] for b in bounds_list)
        ymin = min(b[2] for b in bounds_list)
        ymax = max(b[3] for b in bounds_list)
        zmin = min(b[4] for b in bounds_list)
        zmax = max(b[5] for b in bounds_list)

        corners = []
        for x in (xmin, xmax):
            for y in (ymin, ymax):
                for z in (zmin, zmax):
                    local = np.array([x, y, z])
                    corners.append(tuple(position + rotation @ local))
        return tuple(corners)

    @property
    def parents(self) -> tuple[Any, ...]:
        """Every other component in the model that references this body, found by scanning it.

        Every other wrapper's ``parents`` in this module is a *forward*
        relation, read straight off a native property/socket the
        component already carries (e.g. :attr:`Marker.parents`, straight
        from its own parent frame). This one is the one genuinely
        *backward* relation -- a plain ``opensim.Body`` carries no native
        back-reference to whatever joints/muscles/constraints/markers
        attach to it at all -- so it is discovered instead by scanning the
        owning model's own :attr:`~opensim_models.model.OpenSimModel.joints`/
        :attr:`~opensim_models.model.OpenSimModel.muscles`/
        :attr:`~opensim_models.model.OpenSimModel.constraints`/
        :attr:`~opensim_models.model.OpenSimModel.markers` dict properties,
        keeping whichever ones list this body among their own (forward)
        ``parents``. This is exactly as expensive as that sounds -- every
        component in those four categories is visited, and each one's own
        ``parents`` resolved fresh, every single time this property is
        read, with no caching (same "always live, never stale" philosophy
        as :attr:`position_global`/:attr:`com`, just with a much higher
        constant factor here) -- avoid calling this in a tight loop over
        many bodies; read it once per body of interest instead.

        Deliberately **not** scanned: any non-muscle ``Force`` (most
        concrete subtypes, e.g.
        :class:`~opensim_models.components.ExponentialContactForce`,
        expose no generically discoverable body/frame reference at all --
        confirmed directly, see that class's own docstring for why), and
        any ``Constraint`` without a dedicated wrapper (e.g. a
        ``CoordinateCouplerConstraint``, which references coordinates, not
        bodies, so has no ``parents`` of its own for this to check against).

        Returns
        -------
        tuple
            Every referencing component found, already wrapped in its own
            type (:class:`Joint`, :class:`Muscle`, one of
            :class:`WeldConstraint`/:class:`PointConstraint`/
            :class:`ConstantDistanceConstraint`, :class:`Marker`), in that
            order. Empty if nothing in the model references this body.
        """
        result: list[Any] = []
        for joint in self._owner.joints.values():
            if any(_same_component(parent, self) for parent in joint.parents):
                result.append(joint)
        for muscle in self._owner.muscles.values():
            if any(_same_component(parent, self) for parent in muscle.parents):
                result.append(muscle)
        for constraint in self._owner.constraints.values():
            constraint_parents = getattr(constraint, "parents", ())
            if any(_same_component(parent, self) for parent in constraint_parents):
                result.append(constraint)
        for marker in self._owner.markers.values():
            if any(_same_component(parent, self) for parent in marker.parents):
                result.append(marker)
        return tuple(result)


class Coordinate(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Coordinate``."""

    @property
    def value(self) -> float:
        """Current value, in the coordinate's native unit (radians or metres)."""
        return float(self._raw.getValue(self._owner.state))

    def set_value(self, value: float, *, enforce_constraints: bool = False) -> None:
        """Set the value, in the coordinate's native unit (radians or metres).

        This only writes the raw value into ``owner.state``; it does not
        propagate it to anything derived (body positions, muscle lengths,
        forces, ...). Call :meth:`~opensim_models.model.OpenSimModel.update_state`
        before reading any derived quantity, or after a batch of several
        ``set_*``/direct ``state`` edits, to bring the whole state up to
        date in one pass.

        Parameters
        ----------
        value : float
            New value, in radians (rotational coordinate) or metres
            (translational coordinate).
        enforce_constraints : bool, optional
            Passed through to OpenSim's own ``Coordinate.setValue``.
            Defaults to ``False`` (matching this package's previous
            ``set_coordinate_degrees``, which always used ``False``):
            setting ``True`` makes OpenSim immediately re-solve any
            constraint coupled to this coordinate (e.g. a
            ``CoordinateCouplerConstraint``), which is more expensive and
            not always needed mid-edit.

        Raises
        ------
        ValueError
            If ``value`` is not finite, or this coordinate is locked.
        """
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        if self._raw.get_locked():
            raise ValueError(f"OpenSim coordinate {self.name!r} is locked")
        self._raw.setValue(self._owner.state, float(value), enforce_constraints)

    @property
    def value_degrees(self) -> float:
        """Current value, converted from radians to degrees.

        A convenience for a rotational coordinate; for a translational one
        this still converts as if it were an angle, same as this
        package's previous ``coordinate_degrees`` always did.
        """
        return float(np.rad2deg(self.value))

    def set_value_degrees(
        self, degrees: float, *, enforce_constraints: bool = False
    ) -> None:
        """Set the value, converting from degrees to radians first.

        See :meth:`set_value` for the ``enforce_constraints``/state-update
        caveats; this just converts units before calling it.

        Parameters
        ----------
        degrees : float
            New value in degrees.
        enforce_constraints : bool, optional
            See :meth:`set_value`.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or this coordinate is locked.
        """
        if not np.isfinite(degrees):
            raise ValueError("degrees must be finite")
        self.set_value(
            float(np.deg2rad(degrees)), enforce_constraints=enforce_constraints
        )

    @property
    def speed(self) -> float:
        """Current speed, in the coordinate's native unit per second."""
        return float(self._raw.getSpeedValue(self._owner.state))

    def set_speed(self, value: float) -> None:
        """Set the speed, in the coordinate's native unit per second.

        Same state-update caveat as :meth:`set_value`.

        Parameters
        ----------
        value : float
            New speed, in radians/s (rotational) or metres/s (translational).

        Raises
        ------
        ValueError
            If ``value`` is not finite, or this coordinate is locked.
        """
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        if self._raw.get_locked():
            raise ValueError(f"OpenSim coordinate {self.name!r} is locked")
        self._raw.setSpeedValue(self._owner.state, float(value))

    @property
    def speed_degrees(self) -> float:
        """Current speed, converted from radians/s to degrees/s."""
        return float(np.rad2deg(self.speed))

    def set_speed_degrees(self, degrees_per_second: float) -> None:
        """Set the speed, converting from degrees/s to radians/s first.

        Parameters
        ----------
        degrees_per_second : float
            New speed in degrees per second.

        Raises
        ------
        ValueError
            If ``degrees_per_second`` is not finite, or this coordinate is locked.
        """
        if not np.isfinite(degrees_per_second):
            raise ValueError("degrees_per_second must be finite")
        self.set_speed(float(np.deg2rad(degrees_per_second)))

    @property
    def locked(self) -> bool:
        """Whether this coordinate currently rejects new values."""
        return bool(self._raw.get_locked())

    def set_locked(self, locked: bool) -> None:
        """Lock or unlock this coordinate.

        Parameters
        ----------
        locked : bool
            Whether the coordinate should reject new values.
        """
        self._raw.set_locked(bool(locked))

    @property
    def range(self) -> tuple[float, float]:
        """Allowed range of motion, in the coordinate's native unit."""
        return (float(self._raw.getRangeMin()), float(self._raw.getRangeMax()))

    def set_range(self, range_: tuple[float, float]) -> None:
        """Set the allowed range of motion, in the coordinate's native unit.

        Parameters
        ----------
        range_ : tuple[float, float]
            ``(min, max)`` bounds, in radians (rotational) or metres
            (translational).

        Raises
        ------
        ValueError
            If a bound is not finite, or ``min >= max``.
        """
        min_value, max_value = range_
        if not (np.isfinite(min_value) and np.isfinite(max_value)):
            raise ValueError("range bounds must be finite")
        if min_value >= max_value:
            raise ValueError("min must be less than max")
        self._raw.setRangeMin(float(min_value))
        self._raw.setRangeMax(float(max_value))


class Marker(_ComponentWrapper):
    """Create a body-attached marker or wrap an existing ``opensim.Marker``.

    To create a marker directly, pass its name, owning body, coordinates,
    and optionally whether those coordinates are in the ground frame::

        marker = Marker(
            marker_name="tool_tip",
            body=model.body("tool"),
            coordinates=(0.0, 0.2, 0.0),
            coordinates_are_global=False,
        )
        model.add_marker(marker, reinitialize=True)

    Coordinates are metres. By default they are local to ``body``. If
    ``coordinates_are_global`` is ``True``, the ground-frame point is
    converted into the body's local frame using the body's current model
    state. The marker uses the body's OpenSim frame as its parent. This
    constructor creates the marker but does not add it to the model.

    The internal ``Marker(owner, raw_marker)`` form remains supported for
    wrapping markers already in a model.
    """

    def __init__(
        self,
        marker_name: "OpenSimModel | str",
        body: Any,
        coordinates: tuple[float, float, float] | None = None,
        *,
        coordinates_are_global: bool = False,
    ) -> None:
        """Create a marker attached to ``body``, or wrap an existing marker.

        Public construction uses ``marker_name``, a :class:`Body` wrapper,
        and a finite 3D ``coordinates`` point in metres. The body must
        belong to the model where the marker will be added.
        ``coordinates_are_global=False`` interprets the point in the
        body's local frame; when ``True``, the point is interpreted in the
        ground frame and converted to local coordinates using the body's
        current pose.

        Construction only creates the OpenSim marker. Add it with
        ``model.add_marker(marker, reinitialize=True)`` before querying
        model-derived values such as :attr:`position_global`.

        The internal ``(owner, raw_marker)`` form is used to wrap a marker
        already in a model; the coordinate arguments are not used in that
        form.

        Raises
        ------
        TypeError
            If ``body`` is not a :class:`Body` wrapper, ``coordinates``
            are missing, or invalid coordinate arguments are supplied
            while wrapping an existing marker.
        ValueError
            If ``coordinates`` do not contain exactly three finite values.
        """
        if isinstance(marker_name, str):
            if not isinstance(body, Body):
                raise TypeError(
                    "body must be a components.Body from the model to which "
                    "the marker will be added"
                )
            if coordinates is None:
                raise TypeError("coordinates are required when creating a marker")

            point = np.asarray(coordinates, dtype=float)
            if point.shape != (3,):
                raise ValueError(
                    "coordinates must contain exactly three values (x, y, z)"
                )
            if not np.all(np.isfinite(point)):
                raise ValueError("coordinates must be finite")

            if coordinates_are_global:
                from ..operators.geometry import from_global_to_local

                point = np.asarray(from_global_to_local(point, body), dtype=float)

            model = body._owner
            raw_marker = model.opensim.Marker(
                marker_name,
                body.raw,
                model.opensim.Vec3(*(float(value) for value in point)),
            )
            super().__init__(model, raw_marker)
            return

        if coordinates is not None or coordinates_are_global:
            raise TypeError(
                "coordinates and coordinates_are_global are only accepted "
                "when creating a marker with Marker(marker_name, body, coordinates)"
            )
        super().__init__(marker_name, body)

    @property
    def location(self) -> tuple[float, float, float]:
        """Offset within this marker's parent frame, in metres."""
        location = self._raw.get_location()
        return (location.get(0), location.get(1), location.get(2))

    def set_location(self, location: tuple[float, float, float]) -> None:
        """Set this marker's offset within its parent frame, in metres.

        Parameters
        ----------
        location : tuple[float, float, float]
            ``(x, y, z)`` offset coordinates in metres.

        Raises
        ------
        ValueError
            If a coordinate is not finite.
        """
        x, y, z = location
        if not all(np.isfinite(value) for value in (x, y, z)):
            raise ValueError("location must be finite")
        self._raw.set_location(self._owner.opensim.Vec3(float(x), float(y), float(z)))

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This marker's location, in the ground frame, in metres.

        :attr:`location` transformed through its parent frame's current
        ground-frame placement -- computed directly via OpenSim's own
        ``Marker.getLocationInGround`` (equivalent to, but cheaper than,
        going through :func:`_point_in_ground` with this marker's own
        parent frame), after realizing the model's state through
        ``Stage::Position``. Computed fresh on every access.
        """
        self._owner.model.realizePosition(self._owner.state)
        location = self._raw.getLocationInGround(self._owner.state)
        return (float(location.get(0)), float(location.get(1)), float(location.get(2)))

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This marker's offset within its parent frame, in metres -- the same value as :attr:`location`.

        Present alongside :attr:`position_global` purely for naming
        consistency with every other wrapper's ``position_global``/
        ``position_local`` pair (e.g. :attr:`Body.position_local`,
        :attr:`ContactGeometry.position_local`, :attr:`Joint.position_local`)
        -- prefer :attr:`location`/:meth:`set_location` when working with a
        ``Marker`` specifically, since those also support writing.
        """
        return self.location

    @property
    def parents(self) -> tuple[Any, ...]:
        """The body (or ground) this marker is attached to, as a 1-tuple.

        Resolved from this marker's own parent frame, walked down to its
        ultimate base frame (see :func:`_wrap_base_frame`) -- wrapped as a
        :class:`Body` when it is one, or the raw ``opensim.Ground`` object
        otherwise. Always exactly one element: a marker has a single
        parent frame, never more than one.
        """
        base = self._raw.getParentFrame().findBaseFrame()
        return (_wrap_base_frame(self._owner, base),)


class Muscle(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Muscle``."""

    @property
    def max_isometric_force(self) -> float:
        """Maximum isometric force, in newtons."""
        return float(self._raw.getMaxIsometricForce())

    def set_max_isometric_force(self, newtons: float) -> None:
        """Set the maximum isometric force, in newtons.

        Parameters
        ----------
        newtons : float
            New maximum isometric force, in newtons. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``newtons`` is not finite or not strictly positive.
        """
        self._raw.setMaxIsometricForce(_positive(newtons))

    @property
    def optimal_fiber_length(self) -> float:
        """Optimal fiber length, in metres."""
        return float(self._raw.getOptimalFiberLength())

    def set_optimal_fiber_length(self, meters: float) -> None:
        """Set the optimal fiber length, in metres.

        Parameters
        ----------
        meters : float
            New optimal fiber length, in metres. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``meters`` is not finite or not strictly positive.
        """
        self._raw.setOptimalFiberLength(_positive(meters))

    @property
    def tendon_slack_length(self) -> float:
        """Tendon slack length, in metres."""
        return float(self._raw.getTendonSlackLength())

    def set_tendon_slack_length(self, meters: float) -> None:
        """Set the tendon slack length, in metres.

        Parameters
        ----------
        meters : float
            New tendon slack length, in metres. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``meters`` is not finite or not strictly positive.
        """
        self._raw.setTendonSlackLength(_positive(meters))

    @property
    def pennation_angle(self) -> float:
        """Pennation angle at optimal fiber length, in degrees."""
        return float(np.rad2deg(self._raw.getPennationAngleAtOptimalFiberLength()))

    def set_pennation_angle(self, degrees: float) -> None:
        """Set the pennation angle at optimal fiber length, in degrees.

        Parameters
        ----------
        degrees : float
            New pennation angle, in degrees. Must be finite and within
            ``[0, 90)`` -- ``90`` itself is excluded since a pennation
            angle that large would leave no fiber length component along
            the line of action.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite or not in ``[0, 90)``.
        """
        if not np.isfinite(degrees) or not (0.0 <= degrees < 90.0):
            raise ValueError(
                "pennation angle must be a finite value in [0, 90) degrees"
            )
        self._raw.setPennationAngleAtOptimalFiberLength(float(np.deg2rad(degrees)))

    @property
    def parents(self) -> tuple[Any, ...]:
        """Every distinct body this muscle's path crosses, in path-point order.

        Via this muscle's own ``getGeometryPath().getPathPointSet()``,
        each path point's ``getBody()`` (confirmed directly: resolves
        straight to the body a point lives on -- origin, insertion, and
        any via point alike, no ``findBaseFrame()`` walk needed here).
        Wrapped as a :class:`Body` when possible, or the raw
        ``opensim.Ground`` object. Duplicates are removed, keeping each
        body's first occurrence along the path -- several path points
        commonly share the same origin/insertion body, so a simple
        2-point muscle (no via points) has exactly 2 entries here, not
        more.
        """
        points = self._raw.getGeometryPath().getPathPointSet()
        seen: set[int] = set()
        result = []
        for i in range(points.getSize()):
            body = points.get(i).getBody()
            key = int(body.this)
            if key in seen:
                continue
            seen.add(key)
            result.append(_wrap_base_frame(self._owner, body))
        return tuple(result)


class Joint(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Joint``."""

    @property
    def coordinates(self) -> dict[str, Coordinate]:
        """Coordinates owned by this joint, keyed by name.

        Via ``numCoordinates()``/``get_coordinates(i)`` -- the generic,
        multi-DOF-safe OpenSim API (``getCoordinate()``, singular, is
        documented single-DOF-only, so it is never used here). There is no
        equivalent on :class:`~opensim_models.model.OpenSimModel` itself:
        ``.coordinate(name)``/``.coordinates`` there go through the
        model's global ``CoordinateSet``, independent of which joint owns
        each one.
        """
        return {
            self._raw.get_coordinates(i).getName(): Coordinate(
                self._owner, self._raw.get_coordinates(i)
            )
            for i in range(self._raw.numCoordinates())
        }

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This joint's child frame position, in the ground frame, in metres.

        A ``Joint`` connects a parent frame and a child frame, each
        independently placeable -- resolving it to a *single* ground-frame
        point means picking one of the two, and this follows the same
        convention already used wherever else this package resolves a
        ``Joint`` to one ground-frame point (e.g.
        :func:`opensim_models.operators.rotate_object`'s ``origin``
        argument, and every joint-centre property on
        :class:`~opensim_models.models.User`, e.g. ``.left_knee``): the
        **child** frame, not the parent's. The two coincide only while
        this joint's own coordinate(s) sit at the value that makes them so
        (typically zero, for a joint built by this package's
        ``add_*_joint`` helpers, before it's actually been posed/driven).
        Computed fresh from the model's current state on every access; use
        :attr:`parent_frame` if the parent side's position is what you
        actually need.
        """
        child = self._owner.opensim.Frame.safeDownCast(self._raw.getChildFrame())
        position, _ = _position_and_rotation_in_ground(self._owner, child)
        return tuple(float(value) for value in position)

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This joint's child frame offset from its own immediate parent, in metres.

        Matches :attr:`position_global`'s child-frame convention. Every
        joint built by this package's ``add_*_joint`` helpers (and every
        joint loaded from a real ``.osim`` model, confirmed directly
        against the bundled Rajagopal-based base model) gives its child
        frame as an ``opensim.PhysicalOffsetFrame`` -- even when built with
        a zero offset -- in which case this is that frame's own
        ``translation`` property: the offset from the body it's welded to,
        in that body's own local axes. Falls back to ``(0.0, 0.0, 0.0)``
        in the (currently unobserved, but not provably impossible) case
        where the child frame is *not* an offset frame at all -- consistent
        with :attr:`Body.position_local`, since a frame is trivially at
        the origin of itself.
        """
        child = self._raw.getChildFrame()
        offset = self._owner.opensim.PhysicalOffsetFrame.safeDownCast(child)
        if offset is None:
            return (0.0, 0.0, 0.0)
        translation = offset.get_translation()
        return (
            float(translation.get(0)),
            float(translation.get(1)),
            float(translation.get(2)),
        )

    @property
    def parent_frame(self) -> "OffsetFrame | Any":
        """This joint's parent frame, wrapped.

        Returns an :class:`OffsetFrame` when the parent frame is an
        ``opensim.PhysicalOffsetFrame`` -- true for every joint built by
        this package's ``add_*_joint`` helpers, even with a zero offset,
        and for every joint loaded from the bundled Rajagopal-based model
        (see :attr:`position_local`). Falls back to the raw, unwrapped
        ``opensim.PhysicalFrame`` in the (currently unobserved) case where
        it is not an offset frame at all -- there would be no
        translation/orientation of its own for :class:`OffsetFrame` to
        expose.
        """
        frame = self._raw.getParentFrame()
        offset = self._owner.opensim.PhysicalOffsetFrame.safeDownCast(frame)
        return OffsetFrame(self._owner, offset) if offset is not None else frame

    @property
    def child_frame(self) -> "OffsetFrame | Any":
        """This joint's child frame, wrapped. See :attr:`parent_frame` for the exact conditions."""
        frame = self._raw.getChildFrame()
        offset = self._owner.opensim.PhysicalOffsetFrame.safeDownCast(frame)
        return OffsetFrame(self._owner, offset) if offset is not None else frame

    @property
    def parents(self) -> tuple[Any, Any]:
        """The two bodies (or ground) this joint connects: ``(parent_body, child_body)``.

        Unlike :attr:`parent_frame`/:attr:`child_frame` (which may be an
        intermediate :class:`OffsetFrame`), each side here is walked down
        to its ultimate base frame first (``findBaseFrame()``, see
        :func:`_wrap_base_frame`) -- the actual body (or ``opensim.Ground``)
        this joint attaches to, skipping past the offset frame itself.
        """
        parent_base = self._raw.getParentFrame().findBaseFrame()
        child_base = self._raw.getChildFrame().findBaseFrame()
        return (
            _wrap_base_frame(self._owner, parent_base),
            _wrap_base_frame(self._owner, child_base),
        )


class Frame(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Frame``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has several concrete ``Frame`` subtypes,
    each with its own API). Provides only the base
    :class:`_ComponentWrapper` surface:
    :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Frame`` object for
    anything not wrapped here.

    ``opensim.Body`` and ``opensim.Ground`` are themselves ``Frame``
    subtypes too, but are never wrapped as a plain :class:`Frame` (or
    reached through it) -- see
    :attr:`~opensim_models.model.OpenSimModel.frames` for the full
    reasoning; in short, those two (and a joint's or ``WeldConstraint``'s
    own attachment frames) are already reachable through their own
    dedicated accessors, so that property excludes them rather than
    re-exposing them under a second name. See :class:`OffsetFrame` for
    the one concrete subtype with a dedicated wrapper so far (see
    :func:`_wrap_frame`) -- in practice, confirmed empirically, every
    frame :attr:`~opensim_models.model.OpenSimModel.frames` actually
    returns comes back as an :class:`OffsetFrame`, never this generic
    fallback; it exists purely for a currently-unobserved standalone
    ``Frame`` subtype that is neither a ``Body``/``Ground`` nor a
    ``PhysicalOffsetFrame``.
    """


class OffsetFrame(Frame):
    """Python-friendly wrapper around an ``opensim.PhysicalOffsetFrame``.

    Also reachable, filtered to just the standalone cases, via
    :attr:`~opensim_models.model.OpenSimModel.frames` (see
    :func:`_wrap_frame`) -- in addition to the uses already documented
    below (a joint's own parent/child frame, a ``WeldConstraint``'s own
    attachment frames).

    A ``PhysicalOffsetFrame`` is the small, usually-unnamed frame every
    ``add_*_joint`` (:func:`~opensim_models.operators.add_weld_joint`, etc.)
    builds on each side of a joint to hold its ``position``/
    ``orientation_deg`` (parent side) or ``child_position``/
    ``child_orientation_deg`` (child side) -- see
    :attr:`Joint.parent_frame`/:attr:`Joint.child_frame` for the normal way
    to reach one already wrapped. Also the attachment frame
    :class:`WeldConstraint` uses on each side (:attr:`WeldConstraint.frame1`/
    :attr:`WeldConstraint.frame2`). Before this wrapper existed, every one
    of these was manipulated through raw SWIG calls (``get_translation()``/
    ``set_translation()``, ``get_orientation()``/``set_orientation()``,
    manually rebuilt rotation matrices for ground-frame queries) --
    confirmed hand-duplicated this way in
    :mod:`opensim_models.operators`'s own ``rotate_object``/
    ``translate_object`` implementation, now refactored to use this wrapper
    for the parts that read/write the frame's own translation/orientation
    (see ``operators/rotation.py``/``operators/translation.py``).

    **A confirmed, counter-intuitive gotcha when this frame is a joint's
    own parent/child frame** (as opposed to, e.g., a
    :class:`WeldConstraint`'s attachment frame): :meth:`set_translation`/
    :meth:`set_orientation_deg` change where this frame sits *relative to
    the body it's attached to*, not where it ends up *in ground*. A
    joint's whole job is to force its parent and child frames to coincide
    (modulo the joint's own dof) -- so changing one side's own local
    offset, with the joint's coordinate(s) and the other side unchanged,
    moves **the body this frame belongs to** to re-satisfy that
    constraint, leaving this frame's own :attr:`position_global` exactly
    where it was before (confirmed directly: setting a hip joint's child
    frame translation, then calling
    :meth:`~opensim_models.model.OpenSimModel.reinitialize`, leaves
    :attr:`position_global` unchanged while the femur's own
    :attr:`Body.position_global` shifts instead). This is precisely why
    :func:`~opensim_models.operators.rotate_object`/
    :func:`~opensim_models.operators.translate_object` never simply nudge
    an existing local translation by a delta: they always solve for the
    local value that places this frame at a specific *ground-frame*
    target first, then write that down -- see
    ``operators/rotation.py``/``operators/translation.py``.
    """

    @property
    def translation(self) -> tuple[float, float, float]:
        """This frame's offset from its own parent frame, in metres.

        OpenSim's own native ``get_translation()`` property, in the parent
        frame's own local axes -- the "position" half of what an
        ``add_*_joint`` call's ``position=``/``child_position=`` argument
        sets. Same value as :attr:`position_local`.
        """
        translation = self._raw.get_translation()
        return (translation.get(0), translation.get(1), translation.get(2))

    def set_translation(self, translation: tuple[float, float, float]) -> None:
        """Set this frame's offset from its own parent frame, in metres.

        Parameters
        ----------
        translation : tuple[float, float, float]
            ``(x, y, z)`` offset, in metres, in the parent frame's own
            local axes.

        Raises
        ------
        ValueError
            If a coordinate is not finite.
        """
        x, y, z = translation
        if not all(np.isfinite(value) for value in (x, y, z)):
            raise ValueError("translation must be finite")
        self._raw.set_translation(
            self._owner.opensim.Vec3(float(x), float(y), float(z))
        )

    @property
    def orientation_deg(self) -> tuple[float, float, float]:
        """This frame's orientation relative to its own parent frame, as X-Y-Z body-fixed Euler degrees.

        OpenSim's own native ``get_orientation()`` property (stored in
        radians internally), converted to degrees to match every other
        angle in this package's public API (e.g.
        :attr:`~opensim_models.components.Box.angle_deg`) -- the
        "orientation" half of what an ``add_*_joint`` call's
        ``orientation_deg=``/``child_orientation_deg=`` argument sets.
        """
        orientation = self._raw.get_orientation()
        return tuple(float(np.degrees(orientation.get(i))) for i in range(3))

    def set_orientation_deg(self, orientation_deg: tuple[float, float, float]) -> None:
        """Set this frame's orientation relative to its own parent frame, as X-Y-Z body-fixed Euler degrees.

        Parameters
        ----------
        orientation_deg : tuple[float, float, float]
            ``(x, y, z)`` X-Y-Z body-fixed Euler angles, in degrees, about
            the parent frame's own axes.

        Raises
        ------
        ValueError
            If a value is not finite.
        """
        x, y, z = orientation_deg
        if not all(np.isfinite(value) for value in (x, y, z)):
            raise ValueError("orientation_deg must be finite")
        self._raw.set_orientation(
            self._owner.opensim.Vec3(*np.radians((float(x), float(y), float(z))))
        )

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This frame's origin, in the ground frame, in metres.

        Computed fresh from the model's current state on every access (via
        :func:`_position_and_rotation_in_ground`).
        """
        position, _ = _position_and_rotation_in_ground(self._owner, self._raw)
        return tuple(float(value) for value in position)

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This frame's offset from its own parent frame, in metres -- the same value as :attr:`translation`.

        Present alongside :attr:`position_global` purely for naming
        consistency with every other wrapper's ``position_global``/
        ``position_local`` pair; prefer :attr:`translation`/
        :meth:`set_translation` when working with an ``OffsetFrame``
        specifically, since those names match OpenSim's own terminology
        for this property.
        """
        return self.translation

    @property
    def parents(self) -> tuple[Any, ...]:
        """The body (or ground) this frame is ultimately attached to, as a 1-tuple.

        Resolved via ``findBaseFrame()`` from this frame's own immediate
        parent (see :func:`_wrap_base_frame`) -- so an intermediate
        offset frame in a longer chain is skipped in favour of the real
        base, same as every other ``parents`` property in this module.
        """
        base = self._raw.getParentFrame().findBaseFrame()
        return (_wrap_base_frame(self._owner, base),)


def _wrap_constraint(owner: "OpenSimModel", raw: Any) -> "Constraint":
    """Wrap ``raw`` (an ``opensim.Constraint``) in the most specific wrapper available.

    Tries each concrete subtype this module has a dedicated wrapper for
    (:class:`WeldConstraint`, :class:`PointConstraint`,
    :class:`ConstantDistanceConstraint`), via ``safeDownCast`` -- these are
    three independent, unrelated concrete OpenSim classes (none a subclass
    of another), so check order does not matter between them. Falls back
    to the thin, generic :class:`Constraint` for anything else (e.g. a
    ``CoordinateCouplerConstraint``, which has no single spatial point to
    add a dedicated wrapper for).

    Wraps the *result* of ``safeDownCast``, not ``raw`` itself: confirmed
    directly that a constraint fetched back from the model's own
    ``ConstraintSet`` (e.g. via ``model.constraints``, unlike one just
    returned fresh from ``add_weld_constraint``) comes back as a
    generically-typed ``opensim.Constraint`` proxy even though the
    underlying C++ object is a concrete subtype -- storing that generic
    proxy as ``self._raw`` on e.g. a :class:`WeldConstraint` would make
    its own ``getFrame1()`` raise ``AttributeError`` (silently swallowed
    into an empty result anywhere this is read through ``getattr(...,
    "parents", ())``, which is exactly how it was first caught).
    """
    opensim = owner.opensim
    weld = opensim.WeldConstraint.safeDownCast(raw)
    if weld is not None:
        return WeldConstraint(owner, weld)
    point = opensim.PointConstraint.safeDownCast(raw)
    if point is not None:
        return PointConstraint(owner, point)
    constant_distance = opensim.ConstantDistanceConstraint.safeDownCast(raw)
    if constant_distance is not None:
        return ConstantDistanceConstraint(owner, constant_distance)
    return Constraint(owner, raw)


def _wrap_force(owner: "OpenSimModel", raw: Any) -> "Force":
    """Wrap ``raw`` (an ``opensim.Force``) in the most specific wrapper available.

    Tries :class:`ExponentialContactForce` via ``safeDownCast``, falling
    back to the thin, generic :class:`Force` for anything else. Does
    *not* special-case a muscle here (even though ``opensim.Muscle`` is
    itself a ``Force`` subtype): :class:`Muscle` is reached through its
    own dedicated accessors (``OpenSimModel.muscles``/``.muscle(name)``,
    via OpenSim's own ``getMuscles()``), kept deliberately separate from
    the generic ``forces``/``force(...)`` accessors, same as before this
    function existed.

    Wraps the *result* of ``safeDownCast``, not ``raw`` itself -- see
    :func:`_wrap_constraint`'s docstring for why (same confirmed gotcha,
    same fix).
    """
    opensim = owner.opensim
    contact_force = opensim.ExponentialContactForce.safeDownCast(raw)
    if contact_force is not None:
        return ExponentialContactForce(owner, contact_force)
    return Force(owner, raw)


def _wrap_frame(owner: "OpenSimModel", raw: Any) -> "Frame":
    """Wrap ``raw`` (an ``opensim.Frame``) in the most specific wrapper available.

    Tries :class:`OffsetFrame` via ``safeDownCast``, falling back to the
    thin, generic :class:`Frame` for anything else -- same dispatch shape
    as :func:`_wrap_constraint`/:func:`_wrap_force`. Used by
    :attr:`~opensim_models.model.OpenSimModel.frames`, after that
    property has already filtered ``raw`` down to a standalone frame (not
    a ``Body``/``Ground``, not a joint's or constraint's own attachment
    frame) -- :class:`OffsetFrame` is, confirmed empirically, the concrete
    type of every single result that filtering lets through in practice
    (there is no other common standalone ``Frame`` subtype that this
    package, or plain model-building code, constructs), so the
    :class:`Frame` branch here is an (currently unobserved) safety net,
    not a normal code path.

    Unlike :func:`_wrap_constraint`/:func:`_wrap_force` (where
    ``safeDownCast`` on the result fetched back from a native ``Set`` is
    required -- see that function's docstring), ``raw`` here is already
    confirmed to come back correctly, concretely typed directly from
    ``opensim.Model.getFrameList()`` (``isinstance``/``safeDownCast``
    against it already succeed with no prior cast needed); ``safeDownCast``
    is kept anyway, for the same uniform, defensive shape every other
    ``_wrap_*`` helper in this module follows.
    """
    opensim = owner.opensim
    offset = opensim.PhysicalOffsetFrame.safeDownCast(raw)
    if offset is not None:
        return OffsetFrame(owner, offset)
    return Frame(owner, raw)


def _wrap_base_frame(owner: "OpenSimModel", frame: Any) -> Any:
    """Wrap ``frame`` -- already resolved to its ultimate base (e.g. via ``Frame.findBaseFrame()``) -- as a :class:`Body` if possible.

    Every ``parents`` property in this module resolves "which body does
    this ultimately attach to" the same way: walk any chain of
    ``PhysicalOffsetFrame``\\ s down to the real, underlying frame via
    OpenSim's own ``findBaseFrame()`` (confirmed directly to skip straight
    through any number of offset frames to the actual ``Body``/``Ground``),
    then wrap it here. Returns the raw ``frame`` unchanged when it is not
    a ``Body`` -- in practice this means ``opensim.Ground`` itself, which
    has no dedicated wrapper in this module.
    """
    body = owner.opensim.Body.safeDownCast(frame)
    return Body(owner, body) if body is not None else frame


def _same_component(a: Any, b: Any) -> bool:
    """True if ``a`` and ``b`` refer to the same underlying ``opensim`` object.

    Each may be a raw ``opensim`` object or a :class:`_ComponentWrapper`
    (its ``.raw`` is compared instead) -- used by :attr:`Body.parents` to
    check a forward ``parents`` entry (e.g. one of :attr:`Joint.parents`'s
    two bodies) against ``self`` without caring which side is wrapped.
    """
    raw_a = a.raw if isinstance(a, _ComponentWrapper) else a
    raw_b = b.raw if isinstance(b, _ComponentWrapper) else b
    return int(raw_a.this) == int(raw_b.this)


class Force(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Force``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``Force`` subtypes, each
    with its own API). Provides only the base :class:`_ComponentWrapper`
    surface: :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Force`` object for
    anything not wrapped here (e.g. force-specific getters/setters). A
    muscle is a ``Force`` subtype in OpenSim; see :class:`Muscle` for
    muscle-specific properties on top of this base set. See
    :class:`ExponentialContactForce` for the one other concrete subtype
    with a dedicated wrapper so far (see :func:`_wrap_force`).
    """


class ExponentialContactForce(Force):
    """Python-friendly wrapper around an ``opensim.ExponentialContactForce``.

    The force :func:`~opensim_models.operators.add_sliding_point_contact`
    builds: a contact point (OpenSim's own "station") on one body, repelled
    from and sliding along a plane fixed on another. Exposes each as a
    ground-frame point, :attr:`point_global` (the station) and
    :attr:`plane_point_global` (a point on the plane) -- but **no**
    ``..._local`` counterpart for either, breaking the
    ``position_global``/``position_local`` pair every other wrapper in
    this module follows. This is a confirmed, genuine OpenSim binding
    limitation, not a shortcut: ``getSocketNames()`` returns empty for this
    force (its body/frame references are plain internal C++ pointers, not
    exposed as OpenSim ``Socket``s the way almost every other component's
    attachments are), and its own ``getStation()`` returns an untyped,
    method-less raw SWIG pointer in this installation's Python bindings --
    there is no way to read back, from this object alone, which body/frame
    the original ``point``/``plane_point`` arguments to
    ``add_sliding_point_contact`` were expressed in. A third candidate
    point, ``getAnchorPointPosition()``, was deliberately left unwrapped
    too: confirmed directly that it requires ``Stage::Dynamics`` to be
    realized first (unlike every other property here, realized through
    ``Stage::Position`` only) and represents a penetration-dependent
    contact-mechanics result, not a fixed geometric point of this force's
    definition.
    """

    @property
    def point_global(self) -> tuple[float, float, float]:
        """The contact point (OpenSim's own "station"), in the ground frame, in metres.

        Via OpenSim's own ``getStationPosition(state)``, after realizing
        the model's state through ``Stage::Position``. See the class
        docstring for why no ``point_local`` is exposed.
        """
        self._owner.model.realizePosition(self._owner.state)
        position = self._raw.getStationPosition(self._owner.state)
        return (float(position.get(0)), float(position.get(1)), float(position.get(2)))

    @property
    def plane_point_global(self) -> tuple[float, float, float]:
        """A point on the contact plane, in the ground frame, in metres.

        Via OpenSim's own ``getContactPlaneTransform()`` (confirmed
        directly to already be expressed in the ground frame, not local to
        the plane's own body), after realizing the model's state through
        ``Stage::Position``. See the class docstring for why no
        ``plane_point_local`` is exposed.
        """
        self._owner.model.realizePosition(self._owner.state)
        position = self._raw.getContactPlaneTransform().p()
        return (float(position.get(0)), float(position.get(1)), float(position.get(2)))


class Constraint(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Constraint``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``Constraint`` subtypes,
    each with its own API). Provides only the base
    :class:`_ComponentWrapper` surface:
    :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Constraint`` object for
    anything not wrapped here (e.g. a ``CoordinateCouplerConstraint``'s own
    function/coordinates, which has no single spatial point of its own).
    See :class:`WeldConstraint`, :class:`PointConstraint` and
    :class:`ConstantDistanceConstraint` for the concrete subtypes with a
    dedicated wrapper so far (see :func:`_wrap_constraint`).
    """


class WeldConstraint(Constraint):
    """Python-friendly wrapper around an ``opensim.WeldConstraint``.

    Rigidly welds two frames together (6 dof removed) -- see
    :func:`~opensim_models.operators.add_weld_constraint`. Confirmed
    directly: both attachment frames (``getFrame1()``/``getFrame2()``) are
    genuine ``opensim.PhysicalOffsetFrame``\\ s, built internally by OpenSim
    from ``add_weld_constraint``'s ``position1``/``orientation1_deg``/
    ``position2``/``orientation2_deg`` arguments even when all-zero -- so
    they are wrapped in :class:`OffsetFrame`, the very same wrapper a
    joint's own parent/child frame uses (:attr:`Joint.parent_frame`/
    :attr:`Joint.child_frame`).
    """

    @property
    def frame1(self) -> OffsetFrame:
        """The attachment frame on ``body1``, wrapped.

        ``getFrame1()`` hands back a plain ``opensim.PhysicalFrame``-typed
        SWIG proxy even though the underlying object is always a
        ``PhysicalOffsetFrame`` (confirmed directly via
        ``getConcreteClassName()``) -- ``safeDownCast`` first, same as
        every other spot in this package that needs the offset-frame-only
        API (``get_translation()``/``get_orientation()``) off of a
        generically-typed frame reference.
        """
        frame = self._owner.opensim.PhysicalOffsetFrame.safeDownCast(
            self._raw.getFrame1()
        )
        return OffsetFrame(self._owner, frame)

    @property
    def frame2(self) -> OffsetFrame:
        """The attachment frame on ``body2``, wrapped. See :attr:`frame1` for the ``safeDownCast`` note."""
        frame = self._owner.opensim.PhysicalOffsetFrame.safeDownCast(
            self._raw.getFrame2()
        )
        return OffsetFrame(self._owner, frame)

    @property
    def point1_global(self) -> tuple[float, float, float]:
        """:attr:`frame1`'s origin, in the ground frame, in metres -- same as ``self.frame1.position_global``."""
        return self.frame1.position_global

    @property
    def point1_local(self) -> tuple[float, float, float]:
        """:attr:`frame1`'s offset from ``body1``, in metres -- same as ``self.frame1.position_local``."""
        return self.frame1.position_local

    @property
    def point2_global(self) -> tuple[float, float, float]:
        """:attr:`frame2`'s origin, in the ground frame, in metres -- same as ``self.frame2.position_global``."""
        return self.frame2.position_global

    @property
    def point2_local(self) -> tuple[float, float, float]:
        """:attr:`frame2`'s offset from ``body2``, in metres -- same as ``self.frame2.position_local``."""
        return self.frame2.position_local

    @property
    def parents(self) -> tuple[Any, Any]:
        """The two bodies this weld constraint ties together: ``(body1, body2)``.

        Each resolved via ``findBaseFrame()`` from :attr:`frame1`/
        :attr:`frame2` (see :func:`_wrap_base_frame`) -- the actual body,
        not the attachment frame itself.
        """
        return (
            _wrap_base_frame(self._owner, self.frame1.raw.findBaseFrame()),
            _wrap_base_frame(self._owner, self.frame2.raw.findBaseFrame()),
        )


class PointConstraint(Constraint):
    """Python-friendly wrapper around an ``opensim.PointConstraint``.

    Constrains a point fixed on each of two bodies to coincide (3 dof
    removed, orientation left free) -- see
    :func:`~opensim_models.operators.add_point_constraint`. Unlike
    :class:`WeldConstraint`, OpenSim defines each point here as a plain
    ``Vec3`` offset directly on the constraint (``get_location_body_1()``/
    ``get_location_body_2()``), not through its own
    ``PhysicalOffsetFrame`` -- so there is no :class:`OffsetFrame` to
    delegate to; ``body1``/``body2`` themselves are read through this
    constraint's own ``body_1``/``body_2`` sockets (confirmed directly:
    ``getSocketNames()`` does list them, unlike
    :class:`ExponentialContactForce`), and each ground-frame point is
    resolved with the same shared :func:`_point_in_ground` helper every
    other wrapper's ``position_global`` is built on.
    """

    def _body(self, socket_name: str) -> Any:
        connectee = self._raw.getSocket(socket_name).getConnecteeAsObject()
        return self._owner.opensim.PhysicalFrame.safeDownCast(connectee)

    @property
    def point1_local(self) -> tuple[float, float, float]:
        """Constrained point on ``body1``, in ``body1``'s own local frame, in metres."""
        location = self._raw.get_location_body_1()
        return (location.get(0), location.get(1), location.get(2))

    @property
    def point1_global(self) -> tuple[float, float, float]:
        """Constrained point on ``body1``, in the ground frame, in metres."""
        return _point_in_ground(self._owner, self._body("body_1"), self.point1_local)

    @property
    def point2_local(self) -> tuple[float, float, float]:
        """Constrained point on ``body2``, in ``body2``'s own local frame, in metres."""
        location = self._raw.get_location_body_2()
        return (location.get(0), location.get(1), location.get(2))

    @property
    def point2_global(self) -> tuple[float, float, float]:
        """Constrained point on ``body2``, in the ground frame, in metres."""
        return _point_in_ground(self._owner, self._body("body_2"), self.point2_local)

    @property
    def parents(self) -> tuple[Any, Any]:
        """The two bodies this point constraint ties together: ``(body1, body2)``.

        Each resolved via ``findBaseFrame()`` from this constraint's own
        ``body_1``/``body_2`` sockets (see :func:`_wrap_base_frame`).
        """
        return (
            _wrap_base_frame(self._owner, self._body("body_1").findBaseFrame()),
            _wrap_base_frame(self._owner, self._body("body_2").findBaseFrame()),
        )


class ConstantDistanceConstraint(Constraint):
    """Python-friendly wrapper around an ``opensim.ConstantDistanceConstraint``.

    Holds two points -- one fixed on each of two bodies -- at a constant
    distance apart (1 dof removed). Built by
    :func:`~opensim_models.operators.add_point_on_plane_constraint` as its
    long-rod plane approximation: there, ``point2`` below is that
    function's *computed anchor point* (``anchor_point``, placed
    ``anchor_distance`` behind ``plane_point``), not the ``plane_point``
    argument itself -- see that function's own docstring for the full
    geometry. Same plain-``Vec3``-offset shape as :class:`PointConstraint`
    (``get_location_body_1()``/``get_location_body_2()``), but with direct
    ``getBody1()``/``getBody2()`` accessors instead of a socket lookup.
    """

    @property
    def point1_local(self) -> tuple[float, float, float]:
        """Constrained point on ``body1``, in ``body1``'s own local frame, in metres."""
        location = self._raw.get_location_body_1()
        return (location.get(0), location.get(1), location.get(2))

    @property
    def point1_global(self) -> tuple[float, float, float]:
        """Constrained point on ``body1``, in the ground frame, in metres."""
        return _point_in_ground(self._owner, self._raw.getBody1(), self.point1_local)

    @property
    def point2_local(self) -> tuple[float, float, float]:
        """Anchor point on ``body2`` (``plane_body``), in ``body2``'s own local frame, in metres.

        See the class docstring: for a constraint built by
        ``add_point_on_plane_constraint``, this is that function's computed
        ``anchor_point``, not the ``plane_point`` argument it was given.
        """
        location = self._raw.get_location_body_2()
        return (location.get(0), location.get(1), location.get(2))

    @property
    def point2_global(self) -> tuple[float, float, float]:
        """Anchor point on ``body2``, in the ground frame, in metres. See :attr:`point2_local`."""
        return _point_in_ground(self._owner, self._raw.getBody2(), self.point2_local)

    @property
    def distance(self) -> float:
        """The fixed distance enforced between :attr:`point1_global` and :attr:`point2_global`, in metres."""
        return float(self._raw.get_constant_distance())

    @property
    def parents(self) -> tuple[Any, Any]:
        """The two bodies this constraint ties together: ``(body1, body2)``.

        Each resolved via ``findBaseFrame()`` from this constraint's own
        ``getBody1()``/``getBody2()`` (see :func:`_wrap_base_frame`). For a
        constraint built by ``add_point_on_plane_constraint``, ``body2`` is
        ``plane_body`` -- see :attr:`point2_local`.
        """
        return (
            _wrap_base_frame(self._owner, self._raw.getBody1().findBaseFrame()),
            _wrap_base_frame(self._owner, self._raw.getBody2().findBaseFrame()),
        )


class Controller(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Controller``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``Controller`` subtypes,
    each with its own API). Provides only the base
    :class:`_ComponentWrapper` surface:
    :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Controller`` object for
    anything not wrapped here (e.g. which actuators it controls).
    """


class ContactGeometry(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.ContactGeometry``.

    Every concrete subtype (``ContactSphere``, ``ContactHalfSpace``,
    ``ContactMesh``) shares this base's ``location``/``orientation_deg``
    (an offset/orientation within whichever ``opensim.PhysicalFrame`` it is
    attached to, via its ``frame`` socket) -- this class provides those,
    plus :attr:`position_global`/:attr:`position_local`/:attr:`parents`,
    for all of them at once. See :class:`ContactSphere` (``radius``),
    :class:`ContactHalfSpace` (no extra property -- see its own docstring
    for the "which side is solid" convention) and :class:`ContactMesh`
    (``filename``) for what each subtype adds on top; use
    :func:`~opensim_models.operators.add_contact_sphere`/
    :func:`~opensim_models.operators.add_contact_half_space`/
    :func:`~opensim_models.operators.add_contact_mesh` to build and attach
    one in a single call, or :attr:`~_ComponentWrapper.raw` as the escape
    hatch to anything not wrapped here, for any subtype not yet given its
    own dedicated wrapper.
    """

    @property
    def location(self) -> tuple[float, float, float]:
        """This contact geometry's offset within its attached frame, in metres.

        OpenSim's own native ``get_location()`` property, shared by every
        concrete ``ContactGeometry`` subtype -- same value as
        :attr:`position_local`, provided under this name too since it
        matches OpenSim's own terminology (same relationship as
        :attr:`Marker.location`/:attr:`Marker.position_local`).
        """
        local = self._raw.get_location()
        return (float(local.get(0)), float(local.get(1)), float(local.get(2)))

    def set_location(self, location: tuple[float, float, float]) -> None:
        """Set this contact geometry's offset within its attached frame, in metres.

        Parameters
        ----------
        location : tuple[float, float, float]
            ``(x, y, z)`` offset, in metres, in the attached frame's own
            local axes.

        Raises
        ------
        ValueError
            If a coordinate is not finite.
        """
        x, y, z = location
        if not all(np.isfinite(value) for value in (x, y, z)):
            raise ValueError("location must be finite")
        self._raw.set_location(self._owner.opensim.Vec3(float(x), float(y), float(z)))

    @property
    def orientation_deg(self) -> tuple[float, float, float]:
        """This contact geometry's orientation within its attached frame, as X-Y-Z body-fixed Euler degrees.

        OpenSim's own native ``get_orientation()`` property (stored in
        radians internally), converted to degrees to match every other
        angle in this package's public API (e.g.
        :attr:`~opensim_models.components.OffsetFrame.orientation_deg`).
        """
        orientation = self._raw.get_orientation()
        return tuple(float(np.degrees(orientation.get(i))) for i in range(3))

    def set_orientation_deg(self, orientation_deg: tuple[float, float, float]) -> None:
        """Set this contact geometry's orientation within its attached frame, as X-Y-Z body-fixed Euler degrees.

        Parameters
        ----------
        orientation_deg : tuple[float, float, float]
            ``(x, y, z)`` X-Y-Z body-fixed Euler angles, in degrees, about
            the attached frame's own axes.

        Raises
        ------
        ValueError
            If a value is not finite.
        """
        x, y, z = orientation_deg
        if not all(np.isfinite(value) for value in (x, y, z)):
            raise ValueError("orientation_deg must be finite")
        self._raw.set_orientation(
            self._owner.opensim.Vec3(*np.radians((float(x), float(y), float(z))))
        )

    @property
    def position_global(self) -> tuple[float, float, float]:
        """This contact geometry's location, in the ground frame, in metres.

        Every concrete ``opensim.ContactGeometry`` subtype (``ContactSphere``,
        ``ContactHalfSpace``, ``ContactMesh``, ...) shares the same base
        ``location`` property (see :attr:`position_local`): an offset
        within whatever ``opensim.PhysicalFrame`` it is attached to (its
        ``frame`` socket, read via ``getFrame()``). This transforms that
        offset through the attached frame's current ground-frame placement
        (:func:`_point_in_ground`), after realizing the model's state
        through ``Stage::Position``. Verified directly against a
        ``ContactSphere`` welded to an offset ``WeldJoint`` child frame:
        matches the sphere's actual ground-frame position (``frame``'s
        position plus its rotation applied to ``location``).
        """
        frame = self._raw.getFrame()
        local = self._raw.get_location()
        local_point = (local.get(0), local.get(1), local.get(2))
        return _point_in_ground(self._owner, frame, local_point)

    @property
    def position_local(self) -> tuple[float, float, float]:
        """This contact geometry's offset within its attached frame, in metres -- the same value as :attr:`location`.

        Present alongside :attr:`position_global` purely for naming
        consistency with every other wrapper's ``position_global``/
        ``position_local`` pair; prefer :attr:`location`/
        :meth:`set_location` when working with a ``ContactGeometry``
        specifically, since those match OpenSim's own terminology.
        """
        return self.location

    @property
    def parents(self) -> tuple[Any, ...]:
        """The body (or ground) this contact geometry is attached to, as a 1-tuple.

        Resolved from ``getFrame()``, walked down to its ultimate base
        frame (see :func:`_wrap_base_frame`).
        """
        base = self._raw.getFrame().findBaseFrame()
        return (_wrap_base_frame(self._owner, base),)


class ContactSphere(ContactGeometry):
    """Python-friendly wrapper around an ``opensim.ContactSphere``.

    A sphere of :attr:`radius`, centred at :attr:`location` (inherited from
    :class:`ContactGeometry`) within its attached frame -- built in one
    call by :func:`~opensim_models.operators.add_contact_sphere`.
    """

    @property
    def radius(self) -> float:
        """Sphere radius, in metres."""
        return float(self._raw.getRadius())

    def set_radius(self, radius: float) -> None:
        """Set the sphere radius, in metres.

        Parameters
        ----------
        radius : float
            New radius, in metres. Must be finite and strictly positive.

        Raises
        ------
        ValueError
            If ``radius`` is not finite or not strictly positive.
        """
        self._raw.setRadius(_positive(radius))


class ContactHalfSpace(ContactGeometry):
    """Python-friendly wrapper around an ``opensim.ContactHalfSpace``.

    An infinite half-space, split by the plane through :attr:`location`
    perpendicular to its attached frame's local X axis (after
    :attr:`orientation_deg`, both inherited from :class:`ContactGeometry`)
    -- OpenSim's own documented convention: every point with a *positive*
    local X coordinate (relative to this geometry's own, possibly tilted,
    frame) is considered solid/inside, so the open, contactable half-space
    is the local **negative** X side. No extra property beyond
    :class:`ContactGeometry`'s own -- built in one call by
    :func:`~opensim_models.operators.add_contact_half_space`.
    """


class ContactMesh(ContactGeometry):
    """Python-friendly wrapper around an ``opensim.ContactMesh``.

    An arbitrary triangulated surface loaded from :attr:`filename`,
    positioned at :attr:`location`/:attr:`orientation_deg` (inherited from
    :class:`ContactGeometry`) within its attached frame -- built in one
    call by :func:`~opensim_models.operators.add_contact_mesh`.
    """

    @property
    def filename(self) -> str:
        """Path to the mesh file this geometry was loaded from.

        Exactly as stored (OpenSim resolves it relative to the model file's
        own directory, or its registered geometry search paths, the same
        way an attached body ``Mesh`` does -- not necessarily an absolute
        path or one valid relative to the current working directory).
        """
        return self._raw.get_filename()

    def set_filename(self, filename: str) -> None:
        """Set the mesh file this geometry loads its surface from.

        Parameters
        ----------
        filename : str
            Path to the new mesh file (``.obj``/``.vtp``/``.stl``), same
            resolution rules as :attr:`filename`.
        """
        self._raw.set_filename(str(filename))


def _wrap_contact_geometry(owner: "OpenSimModel", raw: Any) -> "ContactGeometry":
    """Wrap ``raw`` (an ``opensim.ContactGeometry``) in the most specific wrapper available.

    Tries each concrete subtype this module has a dedicated wrapper for
    (:class:`ContactSphere`, :class:`ContactHalfSpace`, :class:`ContactMesh`),
    via ``safeDownCast`` -- three independent, unrelated concrete OpenSim
    classes (none a subclass of another), so check order does not matter
    between them. Falls back to the thin, generic :class:`ContactGeometry`
    for anything else (e.g. an ``opensim.ContactCylinder``/``ContactTorus``,
    which this module has no dedicated wrapper for yet).

    Wraps the *result* of ``safeDownCast``, not ``raw`` itself -- see
    :func:`_wrap_constraint`'s docstring for why (same confirmed gotcha,
    same fix: a generically-typed ``opensim.ContactGeometry`` proxy fetched
    back from the model's own ``ContactGeometrySet`` would otherwise make
    e.g. a :class:`ContactSphere`'s own ``getRadius()`` raise
    ``AttributeError``).
    """
    opensim = owner.opensim
    sphere = opensim.ContactSphere.safeDownCast(raw)
    if sphere is not None:
        return ContactSphere(owner, sphere)
    half_space = opensim.ContactHalfSpace.safeDownCast(raw)
    if half_space is not None:
        return ContactHalfSpace(owner, half_space)
    mesh = opensim.ContactMesh.safeDownCast(raw)
    if mesh is not None:
        return ContactMesh(owner, mesh)
    return ContactGeometry(owner, raw)


class Probe(_ComponentWrapper):
    """Python-friendly wrapper around an ``opensim.Probe``.

    A thin, intentionally minimal wrapper for now -- see the module
    docstring for why (OpenSim has many concrete ``Probe`` subtypes, each
    with its own API). Provides only the base :class:`_ComponentWrapper`
    surface: :attr:`~_ComponentWrapper.name`/:meth:`~_ComponentWrapper.set_name`,
    equality/hashing by underlying identity, and :attr:`~_ComponentWrapper.raw`
    as the escape hatch to the underlying ``opensim.Probe`` object for
    anything not wrapped here (e.g. a probe's operation/report settings).
    """


# Concrete standalone components -- a Body subclass bundling its own
# private OpenSimModel, rather than a wrapper around a piece of someone
# else's model (see each module's own docstring for why).
from .box import Box
from .screen import Screen
from .cylinder import Cylinder

__all__ += ["Box", "Screen", "Cylinder"]
