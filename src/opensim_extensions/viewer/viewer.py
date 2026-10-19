"""Public viewer entry point for native OpenSim models and frameworks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..animation import save_animation as _render_animation
from ..framework import Framework
from ..opensim import import_opensim
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
        self._geometry_dirs = tuple(getattr(model, "geometry_directories", ()))
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
        for directory in self._geometry_dirs:
            candidate = Path(directory) / filename
            if candidate.is_file():
                return candidate
        try:
            resolved = self.opensim.ModelVisualizer.findGeometryFile(filename)
        except RuntimeError:
            return None
        return Path(str(resolved)) if resolved else None


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
    ) -> Path:
        """Save the posed model as an SVG image, without opening a window.

        Parameters
        ----------
        object : opensim.Model or Framework
            Model to render. It is not modified.
        frame : int or None, default=None
            Row of ``motion`` to render. Takes precedence over ``time_s``.
        time_s : float or None, default=0.0
            Time in seconds on ``motion`` to render (linearly interpolated).
        path : str or pathlib.Path
            Destination; the extension must be ``.svg``.
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, default=None
            Motion that ``frame`` and ``time_s`` refer to. Without a motion
            the model is rendered in its current posture, and ``frame`` and
            ``time_s`` must be ``None``, ``0`` or ``0.0``.
        size : tuple[int, int], default=DEFAULT_SIZE
            Render size in pixels used to lay out the vector drawing.

        Returns
        -------
        pathlib.Path
            The destination path.

        Raises
        ------
        ValueError
            If the extension is not ``.svg``, ``frame`` is out of range, or
            ``frame``/``time_s`` is requested without a motion.
        TypeError
            If ``object`` or ``motion`` has an unsupported type.
        """
        destination = Path(path)
        if destination.suffix.lower() != ".svg":
            raise ValueError("path must have an .svg extension")
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
            visualizer.save_svg(destination)
        finally:
            visualizer.close()
        return destination

    def save_animation(
        self,
        object: Framework | Any,
        file: str | Path,
        *,
        motion: Any,
        **options: Any,
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
        **options
            Rendering options accepted by
            :func:`opensim_extensions.animation.save_animation` (``size``,
            camera vectors and ``show_*`` flags).

        Returns
        -------
        pathlib.Path
            The destination path.
        """
        model = self._native_model(object)
        return _render_animation(model, self._table(motion), file, **options)
