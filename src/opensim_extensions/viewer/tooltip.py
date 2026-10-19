"""A floating, rounded-corner tooltip following the mouse cursor.

Used by :mod:`opensim_extensions.viewer.player` to show the hovered component's
name and ground-frame position next to the cursor, instead of a fixed
bottom status bar. Rounded corners are faked with a Windows-only trick
(``-transparentcolor`` keys out every pixel of that exact colour, making
the area outside the rounded shape drawn on the canvas invisible *and*
click-through) -- on other platforms this falls back to a plain
rectangular popup, the same per-platform split as the visualizer embedding
in :mod:`opensim_extensions.viewer.win32_embed`.
"""

from __future__ import annotations

from typing import Any

from .win32_embed import IS_WINDOWS

__all__ = ["HoverTooltip"]

_BG = "#2b2b2b"
_FG = "#ffffff"
_TITLE_FG = "#4fc3f7"
_TRANSPARENT_KEY = "#123456"  # arbitrary colour, unlikely to appear in real content
_RADIUS = 10
_PAD_X = 12
_PAD_Y = 10
_OFFSET = 18  # pixels from the cursor, so the tooltip never sits under it
_LINE_SPACING = 4


def _rounded_rect_points(x1: float, y1: float, x2: float, y2: float, radius: float) -> list[float]:
    """Return the point list for a rounded-rectangle ``create_polygon(..., smooth=True)``."""
    radius = max(0.0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    return [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]


class HoverTooltip:
    """A borderless, always-on-top popup showing ``<label>``/``X:``/``Y:``/``Z:``.

    Parameters
    ----------
    root : tkinter.Tk or tkinter.Toplevel
        The window this tooltip belongs to -- used to read the current
        mouse position (:meth:`update` positions the tooltip relative to
        it) and as the ``Toplevel`` parent, so it is destroyed
        automatically along with ``root``.
    """

    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self._root = root
        self._window = tk.Toplevel(root)
        self._window.withdraw()
        self._window.overrideredirect(True)
        self._window.attributes("-topmost", True)
        if IS_WINDOWS:
            self._window.configure(bg=_TRANSPARENT_KEY)
            self._window.attributes("-transparentcolor", _TRANSPARENT_KEY)
            canvas_bg = _TRANSPARENT_KEY
        else:
            canvas_bg = _BG
        self._canvas = tk.Canvas(self._window, bg=canvas_bg, highlightthickness=0, bd=0)
        self._canvas.pack()
        self._title_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self._body_font = tkfont.Font(family="Segoe UI", size=9)

    def update(self, info: tuple[str, float, float, float] | None) -> None:
        """Show ``info`` next to the cursor, or hide the tooltip if ``None``.

        Parameters
        ----------
        info : tuple[str, float, float, float] or None
            ``(label, x, y, z)`` -- see :meth:`~opensim_extensions.viewer.visualizer.VTKVisualizer.hover_info`
            -- or ``None`` when nothing is hovered, which hides the tooltip.
        """
        if info is None:
            self._window.withdraw()
            return
        label, x, y, z = info
        lines = [(label, self._title_font, _TITLE_FG)]
        lines.extend(
            (f"{axis}: {value:.3f}", self._body_font, _FG)
            for axis, value in (("X", x), ("Y", y), ("Z", z))
        )

        line_heights = [font.metrics("linespace") for _, font, _ in lines]
        text_width = max(font.measure(text) for text, font, _ in lines)
        width = text_width + 2 * _PAD_X
        height = sum(line_heights) + _LINE_SPACING * (len(lines) - 1) + 2 * _PAD_Y

        self._canvas.delete("all")
        self._canvas.config(width=width, height=height)
        self._canvas.create_polygon(
            _rounded_rect_points(1, 1, width - 1, height - 1, _RADIUS),
            smooth=True, fill=_BG, outline="",
        )
        y_cursor = _PAD_Y
        for (text, font, color), line_height in zip(lines, line_heights):
            self._canvas.create_text(
                _PAD_X, y_cursor, anchor="nw", text=text, font=font, fill=color,
            )
            y_cursor += line_height + _LINE_SPACING

        pointer_x = self._root.winfo_pointerx() + _OFFSET
        pointer_y = self._root.winfo_pointery() + _OFFSET
        self._window.geometry(f"{width}x{height}+{pointer_x}+{pointer_y}")
        self._window.deiconify()
