"""The unified viewer+playback window for an ``opensim.Model``.

Used by :meth:`~opensim_extensions.viewer.show`: opens a single Tk
window embedding the (VTK-based, see :mod:`opensim_extensions.viewer.visualizer`)
3D view in its upper area, with three groups of controls below it:

- Playback (play/pause, stop, fast-forward, fast-backward, a loop/cycle
  toggle, and a draggable progress slider) driving a motion loaded via
  ``show(motion=...)`` -- disabled whenever no motion is loaded, rather
  than the controls disappearing, so there is always somewhere to look
  for them.
- View (ground/muscles/markers/axes visibility toggles, and a preset-camera
  dropdown) -- unlike the playback controls, never disabled, since these
  act on the 3D view itself rather than a loaded motion.
- Export (save the current view as a PNG, or a loaded motion as an MP4 --
  see :mod:`opensim_extensions.viewer.export`); the animation button is disabled
  under the same condition as the playback controls, since there is
  nothing to animate without a loaded motion.

A floating tooltip (:class:`~opensim_extensions.viewer.tooltip.HoverTooltip`) is
shown next to the mouse cursor whenever it is over a piece of geometry,
with the hovered component's name and ground-frame position.

This used to be two separate windows (the native 3D view plus a Tk window
kept docked under it), from when this project drove the *native* Simbody
visualizer (a separate process) and the only way to add playback/status
chrome to it at all was a second, independent window -- Simbody's own
widget API (``Visualizer.addSlider``/``addMenu``/``setWindowTitle`` --
anything taking a ``SimTK::String``) is not callable from Python on at
least some OpenSim builds. The visualizer is no longer Simbody's, and
runs in this same process now, so its native window is instead reparented
(Windows only -- see :mod:`opensim_extensions.viewer.win32_embed`) directly into
a frame inside this one Tk window: one window, no inter-window docking to
keep in sync.

Embedding was briefly suspected of corrupting VTK's OpenGL context (a
blank window, logging ``wglMakeCurrent failed ... resource is in use``)
and reverted to the two-window design once -- that diagnosis was wrong.
The window was never blank: its actors had no material colour set, so
VTK's default (plain white) rendered indistinguishably from the
near-white background underneath real geometry that was there all along,
proven by sampling actual pixel values on and off the actor rather than
eyeballing a screenshot. :meth:`~opensim_extensions.viewer.visualizer.VTKVisualizer._add_body_actors`
now sets an explicit colour for exactly this reason. The
``wglMakeCurrent`` log lines turned out to be unrelated, triggered by
*this project's own test scripts* calling VTK interactor methods directly
from a thread other than the one that created the window while chasing
the (nonexistent) embedding bug -- not something a real caller, driving
input through the OS's normal per-thread message dispatch, would ever
hit.

This module's ``start_player`` still constructs ``model``'s
:class:`~opensim_extensions.viewer.visualizer.VTKVisualizer` itself (setting
:attr:`~opensim_extensions.viewer.VTKVisualizer`), on the same
background thread that runs this window and calls
:meth:`~opensim_extensions.viewer.visualizer.VTKVisualizer.process_events`
every tick: a Win32 window's messages are only ever delivered to the
thread that created it, so both the embedding reparent call and every
later ``process_events()`` pump need to happen on that same thread.
"""

from __future__ import annotations

import threading
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

from . import export as _export
from .tooltip import HoverTooltip
from .win32_embed import (
    IS_WINDOWS,
    embed_visualizer_window,
    hwnd_from_vtk_window_id,
    resize_embedded_window,
    set_process_dpi_aware,
)

__all__ = ["MotionData", "MotionPlayer", "PlayerWindow", "start_player"]

_SPEED_STEPS = (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0)
_NEUTRAL_SPEED_INDEX = _SPEED_STEPS.index(1.0)

# OpenSim's own Coordinate::MotionType C++ enum (Rotational=1): not exposed
# as a named constant through these Python bindings, so used as a literal.
_ROTATIONAL_MOTION_TYPE = 1

_NO_MOTION_TEXT = "(nessuna animazione)"


class MotionData:
    """A time-indexed table of coordinate values, ready to drive a model.

    Parameters
    ----------
    model : opensim.Model
        Model the motion will be applied to -- used only to tell a
        rotational coordinate (its column may need a degrees-to-radians
        conversion) from a translational one (never converted, regardless
        of the table's own ``inDegrees`` metadata), matching the convention
        every OpenSim motion/``.mot`` file already uses.
    table : opensim.TimeSeriesTable
        Already-loaded motion data: one column per OpenSim coordinate name
        (e.g. ``"hip_flexion_r"``; columns that don't match a coordinate of
        ``model`` are kept but simply never applied), one row per time
        sample.

    Raises
    ------
    ValueError
        If ``table`` has fewer than 2 rows.
    """

    def __init__(self, model: Any, table: Any) -> None:
        # opensim.TimeSeriesTable itself already rejects a non-increasing
        # time column (both built via appendRow and loaded from file), so
        # that isn't re-checked here.
        times = np.asarray(table.getIndependentColumn(), dtype=float)
        if times.size < 2:
            raise ValueError("motion must have at least 2 rows")

        in_degrees = (
            table.hasTableMetaDataKey("inDegrees")
            and str(table.getTableMetaDataAsString("inDegrees")).strip().lower() == "yes"
        )

        native_model = getattr(model, "model", model)
        coordinate_set = native_model.getCoordinateSet()
        coordinate_names = {
            coordinate_set.get(index).getName()
            for index in range(coordinate_set.getSize())
        }
        columns: dict[str, np.ndarray] = {}
        for name in table.getColumnLabels():
            values = np.asarray(table.getDependentColumn(name).to_numpy(), dtype=float)
            if (
                in_degrees
                and name in coordinate_names
                and coordinate_set.get(name).getMotionType()
                == _ROTATIONAL_MOTION_TYPE
            ):
                values = np.radians(values)
            columns[name] = values

        self.times = times
        self.columns = columns

    @property
    def start_time(self) -> float:
        """Return the motion's first sample time, in seconds."""
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        """Return the motion's last sample time, in seconds."""
        return float(self.times[-1])

    def values_at(self, time: float) -> dict[str, float]:
        """Return every column's value at ``time``, in metres/radians.

        Linearly interpolated between the two bracketing samples; ``time``
        outside ``[start_time, end_time]`` is clamped to that range first.

        Parameters
        ----------
        time : float
            Query time, in seconds.
        """
        clamped = min(max(time, self.start_time), self.end_time)
        return {
            name: float(np.interp(clamped, self.times, values))
            for name, values in self.columns.items()
        }


class MotionPlayer:
    """Pure playback state machine for a motion -- no GUI, no OpenSim call.

    Kept independent of both so its transitions (play/pause, stop,
    fast-forward/fast-backward, cycle, seek, advancing by elapsed real
    time) are exercised directly by unit tests, without a live visualizer.

    Parameters
    ----------
    start_time, end_time : float
        The motion's own time range, in seconds (e.g.
        :attr:`MotionData.start_time`/:attr:`MotionData.end_time`).
    loop : bool, optional
        Whether playback wraps back around at either end (``True``)
        instead of stopping there (``False``, default).

    Raises
    ------
    ValueError
        If ``end_time`` is not strictly greater than ``start_time``.
    """

    def __init__(self, start_time: float, end_time: float, *, loop: bool = False) -> None:
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        self.start_time = start_time
        self.end_time = end_time
        self.loop = loop
        self.time = start_time
        self.playing = False
        self._speed_index = _NEUTRAL_SPEED_INDEX

    @property
    def speed(self) -> float:
        """Return the current signed playback speed (multiples of real time)."""
        return _SPEED_STEPS[self._speed_index]

    @property
    def fraction(self) -> float:
        """Return how far through the motion :attr:`time` is, in ``[0, 1]``."""
        span = self.end_time - self.start_time
        return (self.time - self.start_time) / span if span else 0.0

    def play_pause(self) -> None:
        """Toggle between playing and paused, keeping the current position/speed."""
        self.playing = not self.playing

    def stop(self) -> None:
        """Stop playback and reset to the start, at normal forward speed."""
        self.playing = False
        self._speed_index = _NEUTRAL_SPEED_INDEX
        self.time = self.start_time

    def toggle_loop(self) -> None:
        """Toggle whether playback wraps around at either end instead of stopping."""
        self.loop = not self.loop

    def fast_forward(self) -> None:
        """Step the playback speed one notch faster forward, and start playing.

        Repeated calls step through ``1x, 2x, 4x, 8x`` (capped); from a
        negative (reverse) speed, steps back up toward ``1x`` first.
        """
        self._speed_index = min(self._speed_index + 1, len(_SPEED_STEPS) - 1)
        self.playing = True

    def fast_backward(self) -> None:
        """Step the playback speed one notch faster backward, and start playing.

        Repeated calls step through ``-1x, -2x, -4x, -8x`` (capped); from a
        positive (forward) speed, steps back down toward ``-1x`` first.
        """
        self._speed_index = max(self._speed_index - 1, 0)
        self.playing = True

    def seek_fraction(self, fraction: float) -> None:
        """Jump directly to ``fraction`` (in ``[0, 1]``) through the motion."""
        fraction = min(max(fraction, 0.0), 1.0)
        self.time = self.start_time + fraction * (self.end_time - self.start_time)

    def advance(self, dt: float) -> None:
        """Move playback forward by ``dt`` real seconds, at the current speed.

        A no-op while paused. Past either end: wraps around if
        :attr:`loop`, otherwise clamps there and pauses.

        Parameters
        ----------
        dt : float
            Elapsed real time, in seconds. Non-positive values are ignored.
        """
        if not self.playing or dt <= 0:
            return
        self.time += dt * self.speed
        span = self.end_time - self.start_time
        if self.time > self.end_time:
            if self.loop and span > 0:
                self.time = self.start_time + (self.time - self.end_time) % span
            else:
                self.time = self.end_time
                self.playing = False
        elif self.time < self.start_time:
            if self.loop and span > 0:
                self.time = self.end_time - (self.start_time - self.time) % span
            else:
                self.time = self.start_time
                self.playing = False



def _apply_time(model: Any, data: MotionData, time: float) -> None:
    native_model = getattr(model, "model", model)
    state = getattr(model, "state", None)
    if state is None:
        state = native_model.getWorkingState()
    coordinate_set = native_model.getCoordinateSet()
    for name, value in data.values_at(time).items():
        if not coordinate_set.contains(name):
            continue
        coordinate = coordinate_set.get(name)
        if coordinate.get_locked():
            continue
        coordinate.setValue(state, value, False)
    native_model.realizePosition(state)

class PlayerWindow:
    """Handle to a running playback/status window (see :func:`start_player`).

    Attributes
    ----------
    motion_player : MotionPlayer or None
        The playback state machine, if a motion was loaded; ``None`` if
        the window was opened without one (controls shown disabled).
    """

    def __init__(
        self,
        motion_player: MotionPlayer | None,
        stop_event: threading.Event,
        closed_event: threading.Event,
    ) -> None:
        self.motion_player = motion_player
        self._stop_event = stop_event
        self._closed_event = closed_event

    def close(self) -> None:
        """Close the window and stop its background thread.

        Safe to call from any thread; idempotent. The window closes on
        its own next tick (at most one frame interval later), since
        Tk widgets may only safely be touched from the thread running
        their own ``mainloop``. Does not wait for that to actually happen
        -- see :meth:`wait_closed` for callers that need to.
        """
        self._stop_event.set()

    def wait_closed(self, timeout: float | None = None) -> bool:
        """Block until the window has actually finished closing.

        Needed before reusing/mutating whatever ``opensim.Model`` this
        window's visualizer was showing: :meth:`close` only *signals* the
        background thread to stop, so code right after it that goes on to
        mutate ``model``/``model.state`` can otherwise race that
        thread's last tick, still reading them, before it notices the
        signal (confirmed directly: rebuilding a model's components
        immediately after ``close()`` -- instead of after this -- left the
        process unable to exit afterward).

        Parameters
        ----------
        timeout : float or None, optional
            Maximum time to wait, in seconds. ``None`` (default) waits
            indefinitely.

        Returns
        -------
        bool
            ``True`` if the window closed before ``timeout`` elapsed.
        """
        return self._closed_event.wait(timeout)


def start_player(
    model: Any, motion: Any = None, *, loop: bool = False, fps: float = 30.0
) -> PlayerWindow:
    """Open the unified viewer+playback window for ``model``.

    Also constructs ``model``'s :class:`~opensim_extensions.viewer.visualizer.VTKVisualizer`
    (setting :attr:`~opensim_extensions.viewer.VTKVisualizer`) -- on
    the same background thread this starts for the window itself, rather
    than on the caller's thread, so that one thread both creates the VTK
    window and is the one later embedding/resizing/pumping it. A Win32
    window may only safely be manipulated by the thread that created it;
    constructing it here instead of in :meth:`~opensim_extensions.viewer.show`
    avoids a cross-thread deadlock in the embedding step confirmed by
    direct testing (``SetParent``/``ShowWindow`` block forever waiting for
    the *creating* thread to pump messages it was never going to pump,
    since that thread -- the caller's -- had already moved on).

    Runs its own ``Tk`` event loop on that same background daemon thread,
    so this call returns once that window is up (or construction failed);
    call :meth:`PlayerWindow.close` to shut it down (e.g. before opening a
    new one for the same model). Only that background thread should touch
    ``model`` (its ``state``, coordinates, visualizer) while the window is
    open -- OpenSim's ``State``/``Model`` are not thread-safe, so driving
    the same model concurrently from the caller's own thread is not safe.

    Parameters
    ----------
    model : opensim.Model
        Model to show/animate; any previous :attr:`~opensim_extensions.viewer.VTKVisualizer`
        should already be closed (see :meth:`~opensim_extensions.viewer.show`).
    motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, optional
        A motion file path (``.mot``/``.sto``), or an already-loaded/built
        table (e.g. one written by a prior analysis) -- see
        :class:`MotionData`. When ``None`` (default), the window still
        opens (with its hover tooltip and Export image button live), but
        the playback controls and the Export animation button are shown
        disabled, since there is nothing to play/animate.
    loop : bool, optional
        Initial state of the cycle/loop toggle. Ignored if ``motion`` is
        ``None``. Defaults to ``False``.
    fps : float, optional
        Target refresh rate for advancing playback, refreshing the
        tooltip, and redrawing the visualizer, in frames per second; also
        the frame rate of an exported animation (see :mod:`opensim_extensions.viewer.export`).
        Defaults to ``30.0``.

    Returns
    -------
    PlayerWindow
        Handle exposing the playback state machine (``.motion_player``,
        ``None`` if ``motion`` was ``None``) and a way to close the window
        (``.close()``).

    Raises
    ------
    RuntimeError
        If the 3D visualizer could not be constructed (e.g. ``vtk`` is not
        installed, or no graphical backend is available).
    ValueError
        If ``motion`` resolves to fewer than 2 rows.
    """
    if IS_WINDOWS:
        set_process_dpi_aware()

    data: MotionData | None = None
    player: MotionPlayer | None = None
    if motion is not None:
        opensim = model.opensim
        table = opensim.TimeSeriesTable(str(motion)) if isinstance(motion, (str, Path)) else motion
        data = MotionData(model, table)
        player = MotionPlayer(data.start_time, data.end_time, loop=loop)

    stop_event = threading.Event()
    closed_event = threading.Event()
    ready = threading.Event()
    failure: dict[str, Exception] = {}

    def run() -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from tkinter import font as tkfont
        from tkinter import ttk

        import sv_ttk

        from .visualizer import DEFAULT_SIZE, VIEW_NAMES, VTKVisualizer

        root = tk.Tk()
        # A flat, modern ttk theme (Windows 11-style) instead of the dated
        # default OS one -- applied before building any widget below, so
        # every ttk.* constructor already picks it up (no retroactive
        # re-styling needed). "light" chosen to sit close to the
        # visualizer's own light-grey background (see VTKVisualizer's
        # renderer) rather than clashing with a dark control panel.
        sv_ttk.set_theme("light")
        root.title(model.getName() or "opensim-extensions")

        style = ttk.Style()
        # Bigger glyphs for the Unicode icon buttons below -- sv_ttk's own
        # button font is sized for text labels, which renders e.g. "⏪" or
        # "📷" uncomfortably small/off-centre otherwise.
        style.configure("Icon.TButton", font=("Segoe UI", 12))

        default_width, default_height = DEFAULT_SIZE
        viewer_frame = tk.Frame(root, bg="black", width=default_width, height=default_height)
        viewer_frame.pack(side="top", fill="both", expand=True)
        # has no Tk-managed children (the embedded VTK window is a raw
        # Win32 child, invisible to Tk's own geometry management) -- lock
        # in the explicit size above instead of collapsing to one that fits
        # (empty) content, until <Configure> (bound once VTKVisualizer
        # embeds into this frame, below) takes over sizing it from then on.
        viewer_frame.pack_propagate(False)

        controls_frame = ttk.Frame(root, padding=(8, 8, 8, 0))
        controls_frame.pack(side="bottom", fill="x")
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(2, weight=1)

        slider_var = tk.DoubleVar(value=0.0)
        time_var = tk.StringVar(value=_NO_MOTION_TEXT)
        # guards the slider's own `command` callback (fired on *any* value
        # change, including the programmatic ones `tick` below makes every
        # frame) so only an actual user drag seeks playback
        user_is_dragging = {"value": False}

        def on_slider_drag(value: str) -> None:
            if user_is_dragging["value"] and player is not None:
                player.seek_fraction(float(value) / 1000.0)
                _apply_time(model, data, player.time)

        slider = ttk.Scale(
            controls_frame, from_=0, to=1000, orient="horizontal",
            variable=slider_var, command=on_slider_drag, length=360,
        )
        slider.bind("<Button-1>", lambda event: user_is_dragging.update(value=True))
        slider.bind("<ButtonRelease-1>", lambda event: user_is_dragging.update(value=False))
        slider.grid(row=0, column=0, columnspan=3, padx=4, pady=(0, 2), sticky="ew")

        ttk.Label(controls_frame, textvariable=time_var).grid(
            row=1, column=0, columnspan=3, pady=(0, 8)
        )

        def do_play_pause() -> None:
            player.play_pause()
            play_button.config(text="⏸" if player.playing else "▶")

        def do_stop() -> None:
            player.stop()
            play_button.config(text="▶")
            _apply_time(model, data, player.time)

        def do_fast_forward() -> None:
            player.fast_forward()
            play_button.config(text="⏸")

        def do_fast_backward() -> None:
            player.fast_backward()
            play_button.config(text="⏸")

        cycle_var = tk.BooleanVar(value=player.loop if player is not None else loop)

        def do_toggle_loop() -> None:
            player.toggle_loop()
            cycle_var.set(player.loop)

        # Three side-by-side groups -- playback transport, scene
        # visibility/camera, and export -- each in its own bordered,
        # titled LabelFrame instead of one undifferentiated row of
        # buttons, so the three unrelated concerns (driving a loaded
        # motion, looking at the model, saving something to disk) read as
        # visually distinct at a glance.
        playback_group = ttk.LabelFrame(controls_frame, text="Playback", padding=(8, 4))
        playback_group.grid(row=2, column=0, padx=(0, 4), pady=(0, 8), sticky="nsew")
        view_group = ttk.LabelFrame(controls_frame, text="View", padding=(8, 4))
        view_group.grid(row=2, column=1, padx=4, pady=(0, 8), sticky="nsew")
        export_group = ttk.LabelFrame(controls_frame, text="Export", padding=(8, 4))
        export_group.grid(row=2, column=2, padx=(4, 0), pady=(0, 8), sticky="nsew")

        rew_button = ttk.Button(
            playback_group, text="⏪", width=3, style="Icon.TButton", command=do_fast_backward,
        )
        rew_button.grid(row=0, column=0, padx=2)
        play_button = ttk.Button(
            playback_group, text="▶", width=3, style="Icon.TButton", command=do_play_pause,
        )
        play_button.grid(row=0, column=1, padx=2)
        stop_button = ttk.Button(
            playback_group, text="⏹", width=3, style="Icon.TButton", command=do_stop,
        )
        stop_button.grid(row=0, column=2, padx=2)
        ff_button = ttk.Button(
            playback_group, text="⏩", width=3, style="Icon.TButton", command=do_fast_forward,
        )
        ff_button.grid(row=0, column=3, padx=2)
        cycle_check = ttk.Checkbutton(
            playback_group, text="Cycle", variable=cycle_var, command=do_toggle_loop,
        )
        cycle_check.grid(row=0, column=4, padx=(10, 2))

        # Matches vtkActor's own default visibility (True), so this needs
        # no sync against the visualizer at startup -- ground_check's
        # `command` only reads `model.visualizer` later, once a click can
        # actually happen (construction has long since finished by then).
        ground_var = tk.BooleanVar(value=True)

        def do_toggle_ground() -> None:
            model.visualizer.set_ground_visible(ground_var.get())

        ground_check = ttk.Checkbutton(
            view_group, text="Ground", variable=ground_var, command=do_toggle_ground,
        )
        ground_check.grid(row=0, column=0, padx=2, sticky="w")

        # Same deferred-read trick as ground_var above: these commands only
        # touch model.visualizer once a click can actually happen.
        muscles_var = tk.BooleanVar(value=True)

        def do_toggle_muscles() -> None:
            model.visualizer.set_muscles_visible(muscles_var.get())

        muscles_check = ttk.Checkbutton(
            view_group, text="Muscles", variable=muscles_var, command=do_toggle_muscles,
        )
        muscles_check.grid(row=0, column=1, padx=2, sticky="w")

        markers_var = tk.BooleanVar(value=True)

        def do_toggle_markers() -> None:
            model.visualizer.set_markers_visible(markers_var.get())

        markers_check = ttk.Checkbutton(
            view_group, text="Markers", variable=markers_var, command=do_toggle_markers,
        )
        markers_check.grid(row=0, column=2, padx=2, sticky="w")

        axes_var = tk.BooleanVar(value=True)

        def do_toggle_axes() -> None:
            model.visualizer.set_axes_visible(axes_var.get())

        axes_check = ttk.Checkbutton(
            view_group, text="Axes", variable=axes_var, command=do_toggle_axes,
        )
        axes_check.grid(row=0, column=3, padx=2, sticky="w")

        def do_set_view(event: Any = None) -> None:
            model.visualizer.set_view(view_var.get().lower())

        ttk.Label(view_group, text="Camera:").grid(row=1, column=0, pady=(6, 0), sticky="w")
        view_var = tk.StringVar(value=VIEW_NAMES[0].capitalize())
        view_combo = ttk.Combobox(
            view_group, textvariable=view_var, state="readonly", width=9,
            values=[name.capitalize() for name in VIEW_NAMES],
        )
        view_combo.bind("<<ComboboxSelected>>", do_set_view)
        view_combo.grid(row=1, column=1, columnspan=2, pady=(6, 0), sticky="w")

        def do_save_image() -> None:
            path = filedialog.asksaveasfilename(
                parent=root, title="Save image", defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
            )
            if not path:
                return
            try:
                _export.save_image(visualizer, path)
            except Exception as error:  # noqa: BLE001 -- reported via a dialog, not swallowed
                messagebox.showerror("Export failed", str(error), parent=root)

        image_button = ttk.Button(
            export_group, text="📷", width=3, style="Icon.TButton", command=do_save_image,
        )
        image_button.grid(row=0, column=0, padx=2)

        def do_save_animation() -> None:
            path = filedialog.asksaveasfilename(
                parent=root, title="Save animation", defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")],
            )
            if not path:
                return
            was_playing = player.playing
            if was_playing:
                player.play_pause()
                play_button.config(text="▶")
            original_time = player.time
            root.config(cursor="wait")
            root.update()
            try:
                _export.save_animation(model, data, visualizer, _apply_time, path, fps=fps)
            except Exception as error:  # noqa: BLE001 -- reported via a dialog, not swallowed
                messagebox.showerror("Export failed", str(error), parent=root)
            finally:
                root.config(cursor="")
                _apply_time(model, data, original_time)
                if was_playing:
                    player.play_pause()
                    play_button.config(text="⏸")

        animation_button = ttk.Button(
            export_group, text="🎬", width=3, style="Icon.TButton", command=do_save_animation,
        )
        animation_button.grid(row=0, column=1, padx=2)

        playback_widgets = (slider, rew_button, play_button, stop_button, ff_button, cycle_check)
        if player is None:
            for widget in playback_widgets:
                widget.config(state="disabled")
            animation_button.config(state="disabled")

        # Floating hover tooltip (see module docstring): a borderless
        # window following the mouse, shown only while it is over a piece
        # of geometry -- see opensim_extensions.viewer.tooltip.
        tooltip = HoverTooltip(root)

        # winfo_id() alone realizes viewer_frame's native window without
        # pumping Tk's event queue, unlike update()/update_idletasks() --
        # calling either of those *before* constructing VTKVisualizer
        # (next) was tried and rejected, confirmed directly to hang
        # forever inside vtkRenderWindowInteractor.Initialize(), seemingly
        # from some conflict with Tk's own event-loop setup already having
        # touched this thread's message queue first.
        hwnd_box: dict[str, int] = {}
        embed = None
        if IS_WINDOWS:
            parent_hwnd = viewer_frame.winfo_id()

            def embed(window_id: Any) -> None:
                # called by VTKVisualizer.__init__ itself, between its
                # window's creation and its first paint. Only reparents/
                # restyles here, deliberately not also sizing it (below,
                # once construction is done) in the same call: an
                # immediate `MoveWindow` forces a synchronous repaint
                # before this window's *first* real `Render()` -- still a
                # couple lines away at this point, inside
                # `VTKVisualizer.__init__`.
                hwnd = hwnd_from_vtk_window_id(window_id)
                hwnd_box["hwnd"] = hwnd
                embed_visualizer_window(hwnd, parent_hwnd)

        try:
            model.visualizer = VTKVisualizer(model, embed=embed)
        except Exception as error:  # noqa: BLE001 -- reported via `failure`, not swallowed
            failure["error"] = error
            ready.set()
            root.destroy()
            return
        # captured here, by this specific run()/thread, rather than read
        # back from `model.visualizer` inside tick() below: a later show()
        # call reassigns `model._visualizer` to a *different* instance as
        # soon as it starts (on the caller's thread, concurrently with
        # this thread's own teardown), so reading it late could close the
        # wrong (new) visualizer instead of this thread's own (old) one.
        visualizer = model._visualizer

        root.update_idletasks()
        root.geometry(f"{default_width}x{default_height + controls_frame.winfo_reqheight()}")
        root.update()

        if "hwnd" in hwnd_box:
            hwnd = hwnd_box["hwnd"]
            resize_embedded_window(hwnd, viewer_frame.winfo_width(), viewer_frame.winfo_height())
            model.visualizer.resize(viewer_frame.winfo_width(), viewer_frame.winfo_height())

            def on_viewer_configure(event: Any) -> None:
                if event.width > 1 and event.height > 1:
                    resize_embedded_window(hwnd, event.width, event.height)
                    model.visualizer.resize(event.width, event.height)

            viewer_frame.bind("<Configure>", on_viewer_configure)

        last_tick = _time.monotonic()

        def tick() -> None:
            if stop_event.is_set():
                # closing the *visualizer* here too (not just destroying
                # `root`), on this thread rather than the caller's: closing
                # it from a model.show() call running on a different
                # thread -- even just this one call later, after this
                # thread's already-scheduled tick() fires -- was tried and
                # rejected, confirmed directly to hang forever inside
                # vtkRenderWindow.Finalize(), the same class of cross-
                # thread Win32 hazard documented throughout this module.
                visualizer.close()
                root.destroy()
                closed_event.set()
                return
            nonlocal last_tick
            now = _time.monotonic()
            dt = now - last_tick
            last_tick = now
            visualizer.process_events()
            if player is not None:
                if player.playing:
                    player.advance(dt)
                    _apply_time(model, data, player.time)
                    if not player.playing:
                        play_button.config(text="▶")
                slider_var.set(player.fraction * 1000.0)
                time_var.set(
                    f"{player.time - player.start_time:6.2f} / "
                    f"{player.end_time - player.start_time:.2f} s  ({player.speed:+.0f}x)"
                )
            tooltip.update(model.visualizer.hover_info())
            root.after(max(1, round(1000.0 / fps)), tick)

        if player is not None:
            _apply_time(model, data, player.time)
        tick()
        ready.set()
        root.mainloop()

    gui_thread = threading.Thread(target=run, daemon=True)
    # Tell pydevd/debugpy (VS Code's debugger) not to suspend this thread
    # when a breakpoint is hit elsewhere: without this, hitting a
    # breakpoint on the caller's thread freezes this window's Tk/VTK
    # message pump too (pydevd suspends every thread by default), which
    # looks exactly like a hang/crash with no error. Harmless outside a
    # debugger -- plain threading.Thread instances ignore unknown
    # attributes.
    gui_thread.pydev_do_not_trace = True
    gui_thread.start()
    # generous: construction (VTK window + actors + Tk widgets, all on the
    # background thread this call is waiting on) can take far longer than
    # it looks like it should under a line-tracing debugger (e.g. VS
    # Code's debugpy with "justMyCode": false, which traces into this
    # library's own code too, not just the caller's) -- a short timeout
    # here previously meant silently returning a PlayerWindow around a
    # visualizer that was still mid-construction on timeout, with no error
    # at all, confirmed directly to look like a silent hang/freeze from
    # the caller's side.
    if not ready.wait(timeout=60.0):
        raise RuntimeError(
            "Timed out waiting for the 3D visualizer/playback window to "
            "initialize (60s). If this is running under a debugger that "
            "traces into library code (e.g. VS Code's debugpy with "
            "\"justMyCode\": false), try disabling that for this run -- "
            "line-by-line tracing of VTK/Tk's own setup code can slow it "
            "down enough to hit this."
        )
    if "error" in failure:
        raise RuntimeError(
            "Unable to open the 3D visualizer. Check that 'vtk' is "
            "installed (pip install vtk) and that a graphical backend "
            "is available."
        ) from failure["error"]
    return PlayerWindow(player, stop_event, closed_event)
