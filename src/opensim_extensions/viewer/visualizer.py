"""An in-process, interactive VTK visualizer for an ``opensim.Model``.

Replaces the native Simbody visualizer (``opensim.ModelVisualizer``) that
:meth:`~opensim_extensions.viewer.show` used to open: that
visualizer runs as a *separate process* (a bundled ``simbody-visualizer``
executable) communicating over a pipe, and its own
``Visualizer.InputListener`` interface exposes exactly three callbacks --
``keyPressed``, ``menuSelected``, ``sliderMoved`` -- with no mouse-move/
hover event, no camera-transform getter (only a setter, and the user can
freely orbit/pan/zoom with the mouse, which is never reported back), and
no picking/ray-intersection API at all. There is therefore no way to know
where the mouse is over that window, or what is under it.

VTK's ``vtkRenderWindowInteractor``, by contrast, runs *in this same
process*: its ``MouseMoveEvent`` fires here, its camera transform is
always readable, and ``vtkCellPicker`` resolves a screen pixel to both a
3D world point and the actor under it -- exactly what a live coordinate/
component readout needs. The tradeoff is reimplementing geometry loading
(meshes and the native ``Brick``/``Cylinder``/``Sphere`` primitives) and
visuals that the Simbody visualizer provided for free; this module covers
only what :meth:`opensim.Model.show` and the playback window
(:mod:`opensim_extensions.viewer.player`) actually need.

The interactor's own blocking ``Start()`` event loop is never used: it runs
a native Win32 message loop in C++ that does not release the GIL, so even
on its own background thread it stalls every other Python thread in the
process (confirmed directly -- a plain
``threading.Thread(target=interactor.Start).start()`` hangs the calling
thread's very next line). Instead, :meth:`VTKVisualizer.process_events`
wraps ``vtkRenderWindowInteractor.ProcessEvents()``, a single non-blocking
pump of pending window messages; :mod:`opensim_extensions.viewer.player` calls it
once per tick of its own Tk ``after()`` loop, on the same thread as Tk's
event processing, so one thread drives both GUIs without contention.

For the same reason, a ``VTKVisualizer`` must be *constructed* on that
same thread too, not handed over from another one afterward: its native
window is thread-affine (a plain Win32 rule -- a window's messages are
only ever delivered to the thread that created it), and
:mod:`opensim_extensions.viewer.player` additionally reparents that window into its
own Tk frame, a call that blocks waiting for the *creating* thread to
process it -- which deadlocks if that thread is the caller's and has
already moved on past ``show()``, as confirmed directly. ``_player.py``'s
``start_player`` is therefore what constructs every ``VTKVisualizer``,
never :meth:`~opensim_extensions.viewer.show` itself.

A pure-white actor on a near-white background looks exactly like a blank/
broken render at a glance -- confirmed the hard way, chasing a GL-context
red herring across several reparenting-timing variants before noticing
the *background* was rendering correctly all along and only the
(uncoloured, default-white) actor itself was invisible against it.
:meth:`_add_body_actors` sets an explicit material colour for exactly
this reason -- it is a visibility fix, not a cosmetic one.
"""

from __future__ import annotations

import threading
from typing import Any

from . import geometry as _geometry

__all__ = ["VTKVisualizer", "DEFAULT_SIZE", "VIEW_NAMES"]

#: Initial render window size in pixels, as (width, height). Exposed so
#: :mod:`opensim_extensions.viewer.player` can size its viewer frame to match before
#: a ``VTKVisualizer`` even exists (it must embed that frame's handle
#: *during* construction -- see ``VTKVisualizer.__init__``'s ``embed``
#: parameter -- so construction can't happen first).
DEFAULT_SIZE = (900, 700)

# Ground reference plane: a flat grid in OpenSim's XZ plane (Y=0), matching
# every ``opensim.Model``'s default gravity direction ((0, -9.80665, 0),
# confirmed directly) -- i.e. "down" is -Y, so the ground belongs at Y=0,
# not Z=0. Size/resolution are just enough to read as a floor/scale
# reference without dominating the view for a human-scale (~1-2 m) model.
_GROUND_HALF_SIZE = 5.0  # metres
_GROUND_RESOLUTION = 10  # grid divisions per side

# Ground-frame origin triad (X/Y/Z axes, at the same (0, 0, 0) point the
# ground plane is centred on). Half the ground plane's half-size would
# still read as "the whole floor's scale reference"; a short arm length
# instead reads as "which way is which axis" without competing with the
# model itself for visual weight on a human-scale (~1-2 m) model -- same
# sizing rationale as the ground plane above, just one order of magnitude
# smaller since this marks a point, not an extent.
_AXES_LENGTH = 0.5  # metres, per arm

# Muscle path lines and marker spheres, in the same (0.75, 0.76, 0.8)-grey
# world as body geometry -- saturated colours so both stay readable against
# bone-coloured actors and each other.
_MUSCLE_COLOR = (0.75, 0.1, 0.1)
_MUSCLE_RADIUS = 0.004  # metres; cosmetic tube thickness, not a real diameter
_MARKER_COLOR = (1.0, 0.85, 0.0)
_MARKER_RADIUS = 0.008  # metres

# Axis-aligned preset camera views: (direction to place the camera from the
# model's centre, in ground frame) -> view-up. Generic ground-frame axes,
# not anatomical ones -- they happen to line up with a `User`'s own
# forward(+X)/up(+Y)/right(+Z) convention since that is what the ground
# frame is built against, but apply the same way to any model.
_VIEW_DIRECTIONS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "back": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "right": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "left": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "top": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "bottom": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
}

#: Names accepted by :meth:`VTKVisualizer.set_view`, in a stable display order.
VIEW_NAMES = ("front", "back", "left", "right", "top", "bottom")


def _build_interactor_style(vtk: Any) -> Any:
    """Build the camera interactor style: left-drag rotates, Ctrl+left-drag pans, wheel zooms.

    A thin remap of ``vtkInteractorStyleTrackballCamera``'s own modifier ->
    action dispatch (which already reserves Shift+left for Pan and plain
    Ctrl+left for Spin): a Ctrl-held left-drag is presented to the base
    implementation as if Shift were held instead (and Ctrl released),
    since that is the branch it already treats as Pan -- reusing its own
    ``FindPokedRenderer``/``GrabFocus``/``StartPan`` machinery rather than
    reimplementing it. The real modifier state is restored immediately
    after, since the base class's ``OnMouseMove``/``OnLeftButtonUp`` only
    ever dispatch off ``State`` (set by ``StartPan``/``StartRotate``
    during ``OnLeftButtonDown``), never by re-reading modifier keys
    themselves mid-drag. Plain rotate and mouse-wheel zoom are left as the
    base class's own default behaviour unchanged -- which already does
    nothing on a plain click (rotation amount comes purely from mouse-move
    delta while the button is down, so zero movement is zero rotation).

    The returned style must be kept alive for as long as it's installed on
    an interactor (e.g. as an instance attribute): VTK's Python-override
    dispatch for a virtual method like ``OnLeftButtonDown`` requires the
    Python wrapper object itself to still be alive, independently of the
    underlying C++ object's own reference count.
    """

    class _CameraInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
        def OnLeftButtonDown(self) -> None:
            interactor = self.GetInteractor()
            ctrl = bool(interactor.GetControlKey())
            shift = bool(interactor.GetShiftKey())
            if ctrl and not shift:
                interactor.SetControlKey(0)
                interactor.SetShiftKey(1)
                try:
                    super().OnLeftButtonDown()
                finally:
                    interactor.SetControlKey(1)
                    interactor.SetShiftKey(0)
            else:
                super().OnLeftButtonDown()

    return _CameraInteractorStyle()


class VTKVisualizer:
    """Opens an interactive VTK window rendering ``model``'s current geometry.

    One actor per piece of geometry attached to one of ``model``'s bodies
    (a mesh file, or a native ``Brick``/``Cylinder``/``Sphere``), built
    once at construction; call :meth:`show` after any posture/structural
    change to re-sync every actor's transform and redraw (this mirrors
    ``opensim.ModelVisualizer.show(state)``'s own signature, so
    :mod:`opensim_extensions.viewer.player` drives either one the same way).

    Construction returns as soon as the first frame is rendered; the
    window stays interactive (camera orbit *and* hover tracking) only for
    as long as something calls :meth:`process_events` regularly (see that
    method, and the module docstring, for why that is a deliberate
    non-blocking pump rather than the interactor's own ``Start()``) --
    :mod:`opensim_extensions.viewer.player` is what actually does this, once per
    tick, for every ``VTKVisualizer`` :meth:`~opensim_extensions.viewer.show`
    creates.

    Parameters
    ----------
    model : opensim.Model
        Model to render. Read once, at construction, for which bodies and
        geometry exist; call :meth:`show` to reflect a structural change
        (an added/removed body) with a new ``VTKVisualizer`` instead --
        this one's actors are fixed at construction time.
    embed : callable or None, optional
        If given, called once, with this visualizer's native window
        handle (:meth:`get_window_id`'s value), after the window exists
        but before its first paint -- :mod:`opensim_extensions.viewer.player`
        passes a callback that reparents it into its own Tk frame right
        then.
    offscreen : bool, optional
        Render without creating an interactive render-window interactor.
        Used by :func:`opensim_extensions.save_animation` to export video
        without opening a window. Defaults to ``False``.
    size : tuple[int, int], optional
        Render-window dimensions in pixels as ``(width, height)``.
        Defaults to :data:`DEFAULT_SIZE`.

    Raises
    ------
    RuntimeError
        If the ``vtk`` package is not installed.
    """

    def __init__(
        self,
        model: Any,
        *,
        embed: Any = None,
        offscreen: bool = False,
        size: tuple[int, int] = DEFAULT_SIZE,
    ) -> None:
        try:
            import vtk
        except ImportError as error:
            raise RuntimeError(
                "The 'vtk' package is required to show an opensim.Model. "
                "Install it with 'pip install vtk'."
            ) from error
        self._vtk = vtk
        self._model = model

        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.92, 0.93, 0.95)
        self._render_window = vtk.vtkRenderWindow()
        if offscreen:
            self._render_window.SetOffScreenRendering(1)
        self._render_window.AddRenderer(self._renderer)
        self._render_window.SetSize(*size)
        self._render_window.SetWindowName(self._window_title())
        self._interactor = None
        self._interactor_style = None
        if not offscreen:
            self._interactor = vtk.vtkRenderWindowInteractor()
            self._interactor.SetRenderWindow(self._render_window)
            # kept as an instance attribute (not a local), since VTK's Python
            # dispatch into an overridden virtual method requires the Python
            # wrapper object itself to stay alive for as long as it is
            # installed on the interactor -- see _build_interactor_style.
            self._interactor_style = _build_interactor_style(vtk)
            self._interactor.SetInteractorStyle(self._interactor_style)
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)

        # (opensim.Body, vtkActor) pairs, to resync transforms in show();
        # actor -> (model name, component/body name), for hover_info().
        self._actor_bodies: list[tuple[Any, Any]] = []
        self._actor_components: dict[Any, tuple[str, str]] = {}
        self._hover_lock = threading.Lock()
        self._hover_info: tuple[str, float, float, float] | None = None

        # (opensim.Muscle, vtkPolyDataMapper, vtkActor) triples -- unlike a
        # body's fixed-shape mesh (just re-transformed in show()), a
        # muscle's path can change point *count* with posture (wrapping
        # points appearing/disappearing), so its mapper's input polydata is
        # rebuilt from scratch every show() instead.
        self._muscle_items: list[tuple[Any, Any, Any]] = []
        # (opensim.Marker, vtkActor) pairs -- a sphere glyph repositioned
        # (not reshaped) every show(), same idea as a body actor but
        # position-only since a marker carries no orientation.
        self._marker_items: list[tuple[Any, Any]] = []
        self._muscles_visible = True
        self._markers_visible = True
        self._reachability_visible = True
        self._vision_cone_visible = True

        self._build_actors()
        self._build_muscle_actors()
        self._build_marker_actors()
        if self._interactor is not None:
            self._interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
            self._interactor.AddObserver("LeaveEvent", self._on_mouse_leave)
            self._interactor.Initialize()
        if embed is not None:
            embed(self.get_window_id())
        self.show(model.state)
        # framed on the model's own actors *before* the ground plane is
        # added below -- added after, its 10 m-wide extent would otherwise
        # dominate `ResetCamera`'s fit for any human-scale (~1-2 m) model.
        self._renderer.ResetCamera()
        self._ground_actor = self._build_ground_actor()
        self._renderer.AddActor(self._ground_actor)
        # same ordering reasoning as the ground plane just above: added
        # after ResetCamera so its triad never factors into the initial
        # fit (immaterial at this actor's size, but kept consistent with
        # the ground plane's own placement rather than relying on that).
        self._axes_actor = self._build_axes_actor()
        self._renderer.AddActor(self._axes_actor)
        self._render_window.Render()

    @property
    def has_reachability_regions(self) -> bool:
        """Report whether at least one reachability surface is present.

        Returns
        -------
        bool
            ``True`` when at least one reachability surface has renderable
            geometry in the model.
        """
        return any(
            body.getName().rsplit("/", 1)[-1].startswith("reach_")
            for body, _ in self._actor_bodies
        )

    @property
    def has_vision_cone_and_axis(self) -> bool:
        """Report whether both the visual cone and its central axis exist.

        Returns
        -------
        bool
            ``True`` when the model contains renderable geometry for both the
            visual cone and its central-axis line.
        """
        names = {
            body.getName().rsplit("/", 1)[-1] for body, _ in self._actor_bodies
        }
        return any(name.startswith("vision_cone_shape") for name in names) and any(
            name.startswith("vision_cone_axis_shape") for name in names
        )

    def _window_title(self) -> str:
        name = self._model.getName()
        return f"opensim-extensions: {name}" if name else "opensim-extensions"

    def _build_ground_actor(self) -> Any:
        """Build the ground reference plane actor (see the module-level constants)."""
        plane = self._vtk.vtkPlaneSource()
        half = _GROUND_HALF_SIZE
        plane.SetOrigin(-half, 0.0, -half)
        plane.SetPoint1(half, 0.0, -half)
        plane.SetPoint2(-half, 0.0, half)
        plane.SetXResolution(_GROUND_RESOLUTION)
        plane.SetYResolution(_GROUND_RESOLUTION)
        mapper = self._vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = self._vtk.vtkActor()
        actor.SetMapper(mapper)
        properties = actor.GetProperty()
        properties.SetColor(0.85, 0.85, 0.82)
        properties.SetAmbient(0.6)
        properties.EdgeVisibilityOn()
        properties.SetEdgeColor(0.6, 0.6, 0.6)
        return actor

    def set_ground_visible(self, visible: bool) -> None:
        """Show or hide the ground reference plane, and redraw.

        Backed by ``vtkActor.SetVisibility``: when hidden, the ground is
        skipped entirely by the renderer -- not drawn, not in the depth
        buffer -- so this controls the same thing whether what is looking
        at the render is a human watching the live window or something
        reading pixels back from it (e.g. a future screenshot/video
        export).

        Parameters
        ----------
        visible : bool
            Whether the ground plane is rendered.
        """
        self._ground_actor.SetVisibility(bool(visible))
        self._render_window.Render()

    def _build_axes_actor(self) -> Any:
        """Build the ground-frame origin axes triad actor (see ``_AXES_LENGTH``).

        Uses VTK's own ``vtkAxesActor`` rather than three hand-built
        ``vtkArrowSource`` actors: it is the idiomatic VTK way to draw
        exactly this (a labelled XYZ triad at a frame's origin), and its
        defaults already match the convention this triad is meant to
        communicate -- X/Y/Z shafts coloured red/green/blue, with
        matching "X"/"Y"/"Z" text captions at each tip (confirmed
        directly; unlike the plain, unlabelled ground plane, labelling
        here is the point -- a bare set of lines without colour/text
        cues would not actually tell which axis is which). ``vtkAxesActor``
        is a ``vtkProp3D`` rather than a ``vtkActor``, but exposes the
        same ``SetVisibility``/``GetVisibility`` pair (confirmed
        directly) and is accepted by ``vtkRenderer.AddActor`` the same
        way, so it slots into this module's existing per-layer pattern
        without any special-casing.
        """
        axes = self._vtk.vtkAxesActor()
        axes.SetTotalLength(_AXES_LENGTH, _AXES_LENGTH, _AXES_LENGTH)
        return axes

    def set_axes_visible(self, visible: bool) -> None:
        """Show or hide the ground-frame origin axes triad, and redraw.

        Backed by ``vtkProp3D.SetVisibility`` (the same mechanism
        :meth:`set_ground_visible` uses on its ``vtkActor``): when
        hidden, the triad is skipped entirely by the renderer -- not
        drawn, not in the depth buffer -- so this controls the same
        thing whether what is looking at the render is a human watching
        the live window or something reading pixels back from it (e.g. a
        future screenshot/video export).

        Parameters
        ----------
        visible : bool
            Whether the ground-frame axes are rendered.
        """
        self._axes_actor.SetVisibility(bool(visible))
        self._render_window.Render()

    def set_reachability_visible(self, visible: bool) -> None:
        """Show or hide both palm reachability surfaces, and redraw.

        Parameters
        ----------
        visible : bool
            Whether the right and left reachability regions are rendered.
        """
        self._reachability_visible = bool(visible)
        for body, actor in self._actor_bodies:
            if body.getName().rsplit("/", 1)[-1].startswith("reach_"):
                actor.SetVisibility(self._reachability_visible)
        self._render_window.Render()

    def set_vision_cone_visible(self, visible: bool) -> None:
        """Show or hide the visual cone and its central axis, and redraw.

        Parameters
        ----------
        visible : bool
            Whether the visual cone and central-axis line are rendered.
        """
        self._vision_cone_visible = bool(visible)
        for body, actor in self._actor_bodies:
            if body.getName().rsplit("/", 1)[-1].startswith("vision_cone"):
                actor.SetVisibility(self._vision_cone_visible)
        self._render_window.Render()

    def _model_display_name(self) -> str:
        return self._model.getName() or type(self._model).__name__

    def _build_actors(self) -> None:
        opensim = self._model.opensim
        model_name = self._model_display_name()
        # Geometry may be attached to a body or to a PhysicalOffsetFrame
        # nested in it (as bodies.attach_shapes does), so scan every frame.
        for component in self._model.getComponentsList():
            frame = opensim.PhysicalFrame.safeDownCast(component)
            if frame is not None:
                self._add_body_actors(frame, model_name, opensim)

    def _add_body_actors(self, body: Any, model_name: str, opensim: Any) -> None:
        geometry_property = body.getPropertyByName("attached_geometry")
        for index in range(geometry_property.size()):
            geometry = body.get_attached_geometry(index)
            source = self._build_geometry_source(geometry, opensim)
            if source is None:
                continue
            mapper = self._vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = self._vtk.vtkActor()
            actor.SetMapper(mapper)
            body_name = body.getName().rsplit("/", 1)[-1]
            # an actor's default (unset) material is plain white, which
            # under VTK's default headlight looks indistinguishable from
            # the light-gray background it was mistaken for exactly that,
            # at a glance, more than once while debugging this module.
            actor.GetProperty().SetColor(0.75, 0.76, 0.8)
            appearance = geometry.get_Appearance()
            opacity = appearance.get_opacity()
            if opacity < 1.0 or body_name.startswith("vision_cone_axis"):
                color = geometry.getColor()
                actor.GetProperty().SetColor(color.get(0), color.get(1), color.get(2))
            if opacity < 1.0:
                actor.GetProperty().SetOpacity(opacity)
            if body_name.startswith("reach_"):
                actor.SetVisibility(self._reachability_visible)
            elif body_name.startswith("vision_cone"):
                actor.SetVisibility(self._vision_cone_visible)
            self._renderer.AddActor(actor)
            self._actor_bodies.append((body, actor))
            self._actor_components[actor] = (model_name, body.getName())

    def _build_muscle_actors(self) -> None:
        model_name = self._model_display_name()
        muscles = self._model.getMuscles()
        for index in range(muscles.getSize()):
            muscle = muscles.get(index)
            mapper = self._vtk.vtkPolyDataMapper()
            actor = self._vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*_MUSCLE_COLOR)
            actor.SetVisibility(self._muscles_visible)
            self._renderer.AddActor(actor)
            self._muscle_items.append((muscle, mapper, actor))
            self._actor_components[actor] = (model_name, muscle.getName())

    def _build_marker_actors(self) -> None:
        model_name = self._model_display_name()
        marker_set = self._model.getMarkerSet()
        sphere = self._vtk.vtkSphereSource()
        sphere.SetRadius(_MARKER_RADIUS)
        sphere.SetThetaResolution(12)
        sphere.SetPhiResolution(12)
        for index in range(marker_set.getSize()):
            marker = marker_set.get(index)
            mapper = self._vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = self._vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*_MARKER_COLOR)
            actor.SetVisibility(self._markers_visible)
            self._renderer.AddActor(actor)
            self._marker_items.append((marker, actor))
            self._actor_components[actor] = (model_name, marker.getName())

    def _update_muscle_actor(self, muscle: Any, mapper: Any, state: Any) -> None:
        path = muscle.getGeometryPath().getCurrentPath(state)
        count = path.getSize()
        if count < 2:
            mapper.SetInputConnection(None)
            return
        points = self._vtk.vtkPoints()
        polyline = self._vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(count)
        for index in range(count):
            location = path.get(index).getLocationInGround(state)
            points.InsertNextPoint(location.get(0), location.get(1), location.get(2))
            polyline.GetPointIds().SetId(index, index)
        lines = self._vtk.vtkCellArray()
        lines.InsertNextCell(polyline)
        polydata = self._vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        tube = self._vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(_MUSCLE_RADIUS)
        tube.SetNumberOfSides(8)
        tube.CappingOn()
        mapper.SetInputConnection(tube.GetOutputPort())

    def set_muscles_visible(self, visible: bool) -> None:
        """Show or hide all rendered muscle paths.

        Parameters
        ----------
        visible : bool
            Whether muscle paths are rendered.
        """
        self._muscles_visible = bool(visible)
        for _, _, actor in self._muscle_items:
            actor.SetVisibility(self._muscles_visible)
        self._render_window.Render()

    def set_markers_visible(self, visible: bool) -> None:
        """Show or hide all rendered marker spheres.

        Parameters
        ----------
        visible : bool
            Whether marker spheres are rendered.
        """
        self._markers_visible = bool(visible)
        for _, actor in self._marker_items:
            actor.SetVisibility(self._markers_visible)
        self._render_window.Render()

    def set_camera(
        self,
        *,
        position: tuple[float, float, float] | None = None,
        focal_point: tuple[float, float, float] | None = None,
        view_up: tuple[float, float, float] | None = None,
    ) -> None:
        """Set camera position, target, and up direction selectively.

        Omitted vectors retain their current values. The requested vectors
        must define a non-degenerate viewing direction.

        Parameters
        ----------
        position : tuple of float or None, optional
            Camera ``(x, y, z)`` position in world coordinates.
        focal_point : tuple of float or None, optional
            World-frame ``(x, y, z)`` point at which the camera looks.
        view_up : tuple of float or None, optional
            World-frame direction defining the camera's vertical orientation.

        Raises
        ------
        ValueError
            If the resulting position, focal point, and up direction are
            degenerate or collinear.
        """
        camera = self._renderer.GetActiveCamera()
        current_position = camera.GetPosition()
        current_focal_point = camera.GetFocalPoint()
        current_view_up = camera.GetViewUp()
        position = position or current_position
        focal_point = focal_point or current_focal_point
        view_up = view_up or current_view_up
        direction = tuple(position[index] - focal_point[index] for index in range(3))
        up_length = sum(value * value for value in view_up) ** 0.5
        direction_length = sum(value * value for value in direction) ** 0.5
        cross = (
            direction[1] * view_up[2] - direction[2] * view_up[1],
            direction[2] * view_up[0] - direction[0] * view_up[2],
            direction[0] * view_up[1] - direction[1] * view_up[0],
        )
        if direction_length == 0 or up_length == 0 or sum(
            value * value for value in cross
        ) <= (direction_length * up_length * 1e-12) ** 2:
            raise ValueError(
                "camera position, focal point, and view-up direction must define "
                "a non-degenerate view"
            )
        if position is not None:
            camera.SetPosition(*position)
        if focal_point is not None:
            camera.SetFocalPoint(*focal_point)
        if view_up is not None:
            camera.SetViewUp(*view_up)
        self._renderer.ResetCameraClippingRange()
        self._render_window.Render()

    def set_view(self, name: str) -> None:
        """Point the camera at one of :data:`VIEW_NAMES`'s preset directions.

        Reframes on the model's current visible bounds (so this works the
        same regardless of what posture/zoom the camera was at before), but
        does not change what is visible/hidden -- toggle
        :meth:`set_ground_visible`/:meth:`set_muscles_visible`/
        :meth:`set_markers_visible`/:meth:`set_axes_visible` first if those
        should be excluded from the framing.

        Parameters
        ----------
        name : str
            Preset view name from :data:`VIEW_NAMES`.

        Returns
        -------
        None
            The active camera is updated and the render window is redrawn.

        Raises
        ------
        ValueError
            If ``name`` is not one of :data:`VIEW_NAMES`.
        """
        try:
            direction, view_up = _VIEW_DIRECTIONS[name]
        except KeyError:
            raise ValueError(
                f"Unknown view {name!r}; expected one of {VIEW_NAMES}"
            ) from None
        bounds = self._renderer.ComputeVisiblePropBounds()
        center = (
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )
        diagonal = (
            (bounds[1] - bounds[0]) ** 2
            + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2
        ) ** 0.5
        distance = max(diagonal, 1.0) * 1.5
        camera = self._renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(
            center[0] + direction[0] * distance,
            center[1] + direction[1] * distance,
            center[2] + direction[2] * distance,
        )
        camera.SetViewUp(*view_up)
        self._renderer.ResetCameraClippingRange()
        self._render_window.Render()

    def _build_geometry_source(self, geometry: Any, opensim: Any) -> Any | None:
        """Return a VTK source/reader for one attached ``DecorativeGeometry``.

        ``None`` for a mesh whose file can't be resolved, or a geometry
        type not handled -- see :func:`opensim_extensions._geometry.build_source`,
        shared with :class:`~opensim_extensions.components.Body`'s ``.corners``.
        """
        return _geometry.build_source(
            self._vtk, opensim, geometry, self._model._resolve_geometry_file
        )

    def _on_mouse_move(self, _obj: Any, _event: str) -> None:
        x, y = self._interactor.GetEventPosition()
        self._picker.Pick(x, y, 0, self._renderer)
        actor = self._picker.GetActor()
        if actor in self._actor_components:
            model_name, component_name = self._actor_components[actor]
            position = self._picker.GetPickPosition()
            info = (f"{model_name}-{component_name}", position[0], position[1], position[2])
        else:
            info = None
        with self._hover_lock:
            self._hover_info = info

    def _on_mouse_leave(self, _obj: Any, _event: str) -> None:
        with self._hover_lock:
            self._hover_info = None

    def hover_info(self) -> tuple[str, float, float, float] | None:
        """Return ``(label, x, y, z)`` for whatever the mouse is currently over.

        ``None`` when the pointer is outside the window or not over a
        rendered piece of geometry; ``label`` is ``"<model>-<component>"``,
        ``(x, y, z)`` its ground-frame pick position, in metres. The lock
        guarding this is a leftover safety net from the previous threaded
        design, now harmless rather than load-bearing: the mouse-move
        observer only ever fires inside :meth:`process_events`, called
        from the same thread that reads this back (see
        :mod:`opensim_extensions.viewer.player`'s tick loop, which renders it as
        a floating tooltip next to the cursor via
        :class:`~opensim_extensions.viewer.tooltip.HoverTooltip`).

        Returns
        -------
        tuple[str, float, float, float] or None
            Picked component label and its ground-frame ``(x, y, z)``
            position in metres, or ``None`` if no rendered geometry is under
            the pointer.
        """
        with self._hover_lock:
            return self._hover_info

    def capture_frame(self) -> Any:
        """Return the current 3D view as an ``(height, width, 3)`` uint8 RGB array.

        Used by :mod:`opensim_extensions.viewer.export` to save the view as a
        PNG, or a sequence of these (one per sampled motion time) as an
        MP4. Captures whatever is in the render window *right now* -- call
        :meth:`show` first if ``state`` just changed. Row 0 is the top of
        the image (standard image-array convention): VTK's own pixel
        buffer is bottom-up, flipped here so callers don't have to know
        that.

        Returns
        -------
        numpy.ndarray, shape (height, width, 3), dtype uint8
            RGB image of the current render window, with row zero at the top.
        """
        import numpy as np
        from vtkmodules.util.numpy_support import vtk_to_numpy

        window_to_image = self._vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self._render_window)
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()
        image_data = window_to_image.GetOutput()
        width, height, _ = image_data.GetDimensions()
        array = vtk_to_numpy(image_data.GetPointData().GetScalars()).reshape(height, width, 3)
        return np.flipud(array)

    def save_svg(self, path: str | Path) -> Path:
        """Write the current view as a vector SVG (see :mod:`.svg`).

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            The destination path.
        """
        from .svg import write_svg

        return write_svg(self._renderer, self.get_size(), path)

    def show(self, state: Any) -> None:
        """Sync every actor's transform from ``state`` and redraw.

        Same signature as ``opensim.ModelVisualizer.show(state)``, so
        anything already written against that (e.g.
        :mod:`opensim_extensions.viewer.player`) drives this the same way.

        Parameters
        ----------
        state : opensim.State
            State whose body transforms, muscle paths, and marker positions
            are copied into the rendered actors.
        """
        self._model.realizePosition(state)
        for body, actor in self._actor_bodies:
            position = body.getPositionInGround(state)
            rotation_matrix = body.getRotationInGround(state).asMat33()
            matrix = self._vtk.vtkMatrix4x4()
            for row in range(3):
                for column in range(3):
                    matrix.SetElement(row, column, rotation_matrix.get(row, column))
                matrix.SetElement(row, 3, position.get(row))
            actor.SetUserMatrix(matrix)
        for muscle, mapper, _ in self._muscle_items:
            self._update_muscle_actor(muscle, mapper, state)
        for marker, actor in self._marker_items:
            location = marker.getLocationInGround(state)
            actor.SetPosition(location.get(0), location.get(1), location.get(2))
        self._render_window.Render()

    def process_events(self) -> None:
        """Pump pending window messages once (mouse move, drag/orbit, resize, close).

        Non-blocking -- unlike ``vtkRenderWindowInteractor.Start()``, which
        loops internally until the window closes (see the module
        docstring for why that cannot run on a background thread here).
        Must be called repeatedly, from whatever thread owns this
        process's single native event-pump duty, for the window to stay
        responsive at all; :mod:`opensim_extensions.viewer.player` calls it once per
        tick of its own Tk loop.
        """
        if self._interactor is not None:
            self._interactor.ProcessEvents()

    def get_size(self) -> tuple[int, int]:
        """Return the current render-window dimensions.

        Returns
        -------
        tuple[int, int]
            Window size in pixels, ordered as ``(width, height)``.
        """
        width, height = self._render_window.GetSize()
        return (width, height)

    def resize(self, width: int, height: int) -> None:
        """Resize the render window to ``(width, height)`` pixels and redraw.

        Called by :mod:`opensim_extensions.viewer.player` whenever the Tk frame this
        window is embedded into is resized, to keep the two in sync.

        Parameters
        ----------
        width : int
            New render-window width in pixels.
        height : int
            New render-window height in pixels.
        """
        self._render_window.SetSize(width, height)
        self._render_window.Render()

    def get_window_id(self) -> Any:
        """Return the native window handle (``HWND`` on Windows).

        For :mod:`opensim_extensions.viewer.player` to embed this window directly
        into its own Tk frame -- no window search needed (unlike the old
        Simbody visualizer, a separate process), since this window was
        created in this same process.

        Returns
        -------
        object
            Native window identifier. On Windows this is a SWIG-encoded
            pointer that :func:`hwnd_from_vtk_window_id` can decode.
        """
        return self._render_window.GetGenericWindowId()

    def close(self) -> None:
        """Release the render window.

        Idempotent. There is no interactor loop to stop (see the module
        docstring): :meth:`process_events` simply stops being called once
        the owning :mod:`opensim_extensions.viewer.player` window closes.
        """
        self._render_window.Finalize()
