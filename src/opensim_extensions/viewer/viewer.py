"""Public viewer entry point for native OpenSim models and frameworks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..animation import save_animation as _render_animation
from ..bodies.shapes import MESHES_DIR
from ..framework import Framework
from ..opensim import import_opensim, resolve_geometry_file
from .export import save_image as _save_image
from .player import MotionData, PlayerWindow, _apply_time, start_player
from .visualizer import DEFAULT_SIZE, VTKVisualizer

__all__ = ["VTKViewer"]


class _ViewerModel:
    """Provide viewer state and geometry lookup around a native model."""

    def __init__(self, model: Any, state: Any = None) -> None:
        self.model = model
        self.opensim = import_opensim()
        if state is None:
            state = getattr(model, "state", None)
        self.state = state if state is not None else model.initSystem()
        self._geometry_dirs = (
            *getattr(model, "geometry_directories", ()),
            MESHES_DIR,
        )
        self._visualizer = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    @property
    def visualizer(self) -> VTKVisualizer | None:
        return self._visualizer

    @visualizer.setter
    def visualizer(self, value: VTKVisualizer | None) -> None:
        self._visualizer = value
        if hasattr(self.model, "_visualizer"):
            self.model._visualizer = value

    def _resolve_geometry_file(self, filename: str) -> Path | None:
        return resolve_geometry_file(
            self.opensim, self.model, filename, self._geometry_dirs
        )


def _camera_vector(
    name: str, value: tuple[float, float, float] | None
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three finite numbers") from error
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain three finite numbers")
    return vector


class VTKViewer:
    """Display and export native OpenSim models and frameworks with VTK.

    Examples
    --------
    >>> viewer = VTKViewer()
    >>> viewer.show(framework)  # doctest: +SKIP
    >>> viewer.save_snapshot(framework, path="pose.svg")  # doctest: +SKIP
    >>> viewer.save_animation(model, "motion.mp4", motion="motion.sto")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._opensim = import_opensim()

    def _native_model(self, obj: Any) -> Any:
        if isinstance(obj, Framework):
            return obj.merge_to_model()
        if isinstance(obj, self._opensim.Model):
            return obj
        raise TypeError("object must be an opensim.Model or Framework")

    def _table(self, motion: Any) -> Any:
        if isinstance(motion, (str, Path)):
            return self._opensim.TimeSeriesTable(str(motion))
        if isinstance(motion, self._opensim.TimeSeriesTable):
            return motion
        raise TypeError("motion must be a path or opensim.TimeSeriesTable")

    def show(
        self,
        object: Framework | Any,
        motion: Any = None,
        *,
        loop: bool = False,
        fps: float = 30.0,
    ) -> PlayerWindow:
        """Open the interactive viewer.

        Parameters
        ----------
        object : opensim.Model or Framework
            Model to display. A framework is displayed as a merged snapshot;
            its member models remain unchanged.
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

        Raises
        ------
        TypeError
            If ``object`` is neither an OpenSim model nor a ``Framework``.
        """
        model = self._native_model(object)
        return start_player(_ViewerModel(model), motion, loop=loop, fps=fps)

    def save_snapshot(
        self,
        object: Framework | Any,
        frame: int | None = None,
        time_s: float | None = 0.0,
        path: str | Path = "snapshot.svg",
        *,
        motion: Any = None,
        size: tuple[int, int] = DEFAULT_SIZE,
        dpi: int = 300,
        camera_position: tuple[float, float, float] | None = None,
        camera_focal_point: tuple[float, float, float] | None = None,
        camera_view_up: tuple[float, float, float] | None = None,
        show_ground: bool = True,
        show_muscles: bool = True,
        show_markers: bool = True,
        show_axes: bool = True,
        show_reachability: bool = True,
        show_vision_cone: bool = True,
    ) -> Path:
        """Save the posed model as an SVG or PNG image, without opening a window.

        Parameters
        ----------
        object : opensim.Model or Framework
            Model to render. It is not modified.
        frame : int or None, default=None
            Row of ``motion`` to render. Takes precedence over ``time_s``.
        time_s : float or None, default=0.0
            Time in seconds on ``motion`` to render (linearly interpolated).
        path : str or pathlib.Path
            Destination; the extension must be ``.svg`` or ``.png``.
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, default=None
            Motion that ``frame`` and ``time_s`` refer to. Without a motion
            the model is rendered in its current posture, and ``frame`` and
            ``time_s`` must be ``None``, ``0`` or ``0.0``.
        size : tuple[int, int], default=DEFAULT_SIZE
            Render size in pixels; sets PNG dimensions and SVG layout.
        dpi : int, default=300
            Resolution metadata for PNG output. Ignored for SVG output.
        camera_position : tuple of float or None, default=None
            Camera position ``(x, y, z)`` in ground-frame metres. ``None``
            uses the automatically framed position.
        camera_focal_point : tuple of float or None, default=None
            Ground-frame point in metres for the camera to look at. ``None``
            uses the automatically framed target.
        camera_view_up : tuple of float or None, default=None
            Ground-frame camera up direction. ``None`` keeps the automatic
            direction unless another camera vector is supplied.
        show_ground : bool, default=True
            Show the ground reference grid.
        show_muscles : bool, default=True
            Show muscle paths.
        show_markers : bool, default=True
            Show model markers.
        show_axes : bool, default=True
            Show the ground-frame XYZ coordinate axes.
        show_reachability : bool, default=True
            Show palm reachability regions when present.
        show_vision_cone : bool, default=True
            Show the visual cone and its centre line when present.

        Returns
        -------
        pathlib.Path
            The destination path.

        Raises
        ------
        ValueError
            If the extension is not ``.svg`` or ``.png``, PNG ``dpi`` is not
            a positive integer, ``frame`` is out of range, or ``frame``/
            ``time_s`` is requested without a motion.
        TypeError
            If ``object`` or ``motion`` has an unsupported type.
        ValueError
            If a camera vector is malformed or the requested camera vectors
            do not define a valid view.
        """
        destination = Path(path)
        extension = destination.suffix.lower()
        if extension not in {".svg", ".png"}:
            raise ValueError("path must have a .svg or .png extension")
        if extension == ".png" and (
            not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0
        ):
            raise ValueError("dpi must be a positive integer for PNG output")
        visibility_options = {
            "show_ground": show_ground,
            "show_muscles": show_muscles,
            "show_markers": show_markers,
            "show_axes": show_axes,
            "show_reachability": show_reachability,
            "show_vision_cone": show_vision_cone,
        }
        for name, value in visibility_options.items():
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean")
        camera_vectors = {
            "camera_position": _camera_vector("camera_position", camera_position),
            "camera_focal_point": _camera_vector(
                "camera_focal_point", camera_focal_point
            ),
            "camera_view_up": _camera_vector("camera_view_up", camera_view_up),
        }
        model = self._native_model(object)
        meta = _ViewerModel(model)
        try:
            state = self._opensim.State(model.getWorkingState())
        except RuntimeError:
            state = self._opensim.State(meta.state)
        meta.state = state

        if motion is None:
            if frame not in (None, 0) or time_s not in (None, 0, 0.0):
                raise ValueError("frame and time_s require a motion")
        else:
            data = MotionData(meta, self._table(motion))
            if frame is not None:
                if not -len(data.times) <= frame < len(data.times):
                    raise ValueError("frame is out of range")
                time = float(data.times[frame])
            else:
                time = data.start_time if time_s is None else float(time_s)
            _apply_time(meta, data, time)

        destination.parent.mkdir(parents=True, exist_ok=True)
        visualizer = VTKVisualizer(meta, offscreen=True, size=size)
        try:
            visualizer.set_ground_visible(show_ground)
            visualizer.set_muscles_visible(show_muscles)
            visualizer.set_markers_visible(show_markers)
            visualizer.set_axes_visible(show_axes)
            visualizer.set_reachability_visible(show_reachability)
            visualizer.set_vision_cone_visible(show_vision_cone)
            if any(value is not None for value in camera_vectors.values()):
                visualizer.set_camera(
                    position=camera_vectors["camera_position"],
                    focal_point=camera_vectors["camera_focal_point"],
                    view_up=camera_vectors["camera_view_up"],
                )
            if extension == ".svg":
                visualizer.save_svg(destination)
            else:
                _save_image(visualizer, destination, dpi=dpi)
        finally:
            visualizer.close()
        return destination

    def save_animation(
        self,
        object: Framework | Any,
        file: str | Path,
        *,
        motion: Any,
        size: tuple[int, int] = DEFAULT_SIZE,
        camera_position: tuple[float, float, float] | None = None,
        camera_focal_point: tuple[float, float, float] | None = None,
        camera_view_up: tuple[float, float, float] | None = None,
        show_ground: bool = True,
        show_muscles: bool = True,
        show_markers: bool = True,
        show_axes: bool = True,
        show_reachability: bool = True,
        show_vision_cone: bool = True,
    ) -> Path:
        """Render a motion of the model to an MP4 without opening a window.

        Parameters
        ----------
        object : opensim.Model or Framework
            Model driven by ``motion``. It is not modified.
        file : str or pathlib.Path
            Destination ``.mp4`` path.
        motion : str, pathlib.Path or opensim.TimeSeriesTable
            Motion whose coordinate columns drive the model.
        size : tuple[int, int], default=DEFAULT_SIZE
            Output frame dimensions in pixels.
        camera_position : tuple of float or None, default=None
            Camera position ``(x, y, z)`` in ground-frame metres. ``None``
            uses the automatically framed position.
        camera_focal_point : tuple of float or None, default=None
            Ground-frame point in metres for the camera to look at. ``None``
            uses the automatically framed target.
        camera_view_up : tuple of float or None, default=None
            Ground-frame camera up direction.
        show_ground : bool, default=True
            Show the ground reference grid.
        show_muscles : bool, default=True
            Show muscle paths.
        show_markers : bool, default=True
            Show model markers.
        show_axes : bool, default=True
            Show the ground-frame XYZ coordinate axes.
        show_reachability : bool, default=True
            Show palm reachability regions when present.
        show_vision_cone : bool, default=True
            Show the visual cone and its centre line when present.

        Returns
        -------
        pathlib.Path
            The destination path.
        """
        model = self._native_model(object)
        return _render_animation(
            model,
            self._table(motion),
            file,
            size=size,
            camera_position=camera_position,
            camera_focal_point=camera_focal_point,
            camera_view_up=camera_view_up,
            show_ground=show_ground,
            show_muscles=show_muscles,
            show_markers=show_markers,
            show_axes=show_axes,
            show_reachability=show_reachability,
            show_vision_cone=show_vision_cone,
        )
