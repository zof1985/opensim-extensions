"""Save the player's current 3D view as an image, or a loaded motion as a video.

Used by the "Export" controls in :mod:`opensim_extensions.viewer.player` -- kept
separate from that module since these are pure render/encode operations
(no Tk widgets), independently testable/reusable without a live window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

__all__ = ["save_image", "save_animation"]


def save_image(visualizer: Any, path: str | Path, *, dpi: int = 300) -> None:
    """Save ``visualizer``'s current frame as a PNG.

    Parameters
    ----------
    visualizer : opensim_extensions.viewer.visualizer.VTKVisualizer
        The visualizer to capture (see :meth:`~opensim_extensions.viewer.visualizer.VTKVisualizer.capture_frame`).
    path : str or pathlib.Path
        Destination file. Any existing file at ``path`` is overwritten.
    dpi : int, optional
        Resolution metadata embedded in the PNG (does not change the pixel
        dimensions, which always match the visualizer's current window
        size -- this only affects the physical size the image prints/
        displays at in tools that honour it). Defaults to ``300``.
    """
    from PIL import Image

    Image.fromarray(visualizer.capture_frame()).save(str(path), dpi=(dpi, dpi))


def save_animation(
    model: Any,
    data: "MotionData",
    visualizer: Any,
    apply_time: Callable[[Any, "MotionData", float], None],
    path: str | Path,
    *,
    fps: float = 30.0,
) -> None:
    """Render ``data`` from start to end at ``fps`` and encode it as an MP4.

    Drives ``model`` through the motion exactly like live playback does
    (via ``apply_time``), capturing one frame per sample -- the caller is
    responsible for restoring whatever posture/playback state ``model``
    had before this call once it returns (this function always leaves
    ``model`` sitting at the motion's last sampled frame).

    Parameters
    ----------
    model : opensim.Model
        Model the motion drives.
    data : opensim_extensions.viewer.player.MotionData
        The motion to render, start to end.
    visualizer : opensim_extensions.viewer.visualizer.VTKVisualizer
        The visualizer to capture each frame from.
    apply_time : callable
        ``(model, data, time) -> None``, applying ``data``'s coordinate
        values at ``time`` to ``model`` and redrawing ``visualizer`` --
        pass :func:`opensim_extensions.viewer.player._apply_time`.
    path : str or pathlib.Path
        Destination ``.mp4`` file. Any existing file at ``path`` is
        overwritten.
    fps : float, optional
        Frames per second, both for sampling ``data`` and for the encoded
        video's own playback rate. Defaults to ``30.0``.
    """
    import imageio

    duration = data.end_time - data.start_time
    frame_count = max(1, round(duration * fps))
    # macro_block_size=1 disables imageio's default macroblock-alignment
    # padding (which would otherwise silently resize an odd frame size up
    # to the next multiple of 16): libx264 handles arbitrary even
    # dimensions directly, and the visualizer's own render window size is
    # already controllable (DEFAULT_SIZE/resize()), so the saved video
    # should match it exactly rather than being padded.
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
    )
    try:
        for index in range(frame_count + 1):
            time = data.start_time + duration * index / frame_count
            apply_time(model, data, time)
            writer.append_data(visualizer.capture_frame())
    finally:
        writer.close()
