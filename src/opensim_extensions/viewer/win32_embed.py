"""Embedding the visualizer's native window into a Tk frame (Windows only).

Windows-only: reparenting an arbitrary native window by handle is a
Win32-specific operation; there is no equivalent here for other platforms,
so there the visualizer keeps its own separate top-level window instead
(still fully functional, just not embedded).

VTK's own ``SetParentId`` (create the render window as a child to begin
with) was tried first and rejected: it fails outright on this VTK build
(``vtkWin32OpenGLRenderWindow``, error 1400/ERROR_INVALID_WINDOW_HANDLE,
confirmed by direct testing, both via the SWIG-encoded pointer string and a
``ctypes.c_void_p``). Reparenting the already-created top-level window
after the fact -- strip its WS_POPUP/title-bar styles, ``SetParent`` it
under the Tk frame, ``MoveWindow`` it to fill that frame -- works reliably
and is the standard Win32 technique for embedding a foreign window.
"""

from __future__ import annotations

import re
import sys
from typing import Any

__all__ = [
    "IS_WINDOWS",
    "hwnd_from_vtk_window_id",
    "embed_visualizer_window",
    "resize_embedded_window",
]

IS_WINDOWS = sys.platform.startswith("win")

_VTK_WINDOW_ID_PATTERN = re.compile(r"_([0-9a-fA-F]+)_p_void")


def hwnd_from_vtk_window_id(window_id: Any) -> int:
    """Convert ``VTKVisualizer.get_window_id()``'s SWIG pointer encoding to a plain int HWND."""
    match = _VTK_WINDOW_ID_PATTERN.match(str(window_id))
    if match is None:
        raise ValueError(f"Unrecognized VTK window id format: {window_id!r}")
    return int(match.group(1), 16)


if IS_WINDOWS:
    import ctypes

    _user32 = ctypes.windll.user32

    _GWL_STYLE = -16
    _WS_CHILD = 0x40000000
    _WS_VISIBLE = 0x10000000
    _WS_CLIPSIBLINGS = 0x04000000
    _WS_CLIPCHILDREN = 0x02000000
    _WS_POPUP = 0x80000000
    _WS_CAPTION = 0x00C00000
    _WS_THICKFRAME = 0x00040000
    _WS_SYSMENU = 0x00080000
    _SWP_NOMOVE = 0x0002
    _SWP_NOSIZE = 0x0001
    _SWP_NOZORDER = 0x0004
    _SWP_FRAMECHANGED = 0x0020
    _SW_HIDE = 0

    def embed_visualizer_window(hwnd: int, parent_hwnd: int) -> None:
        """Reparent the visualizer's top-level ``hwnd`` into ``parent_hwnd``, filling it.

        Hides ``hwnd`` first as cheap insurance against it painting a
        frame as a top-level window before this call lands (it is called
        right after that window is created, before its first paint, so in
        practice there is nothing to hide yet). Sets
        ``WS_CLIPCHILDREN``/``WS_CLIPSIBLINGS`` alongside ``WS_CHILD``,
        per Microsoft's own documented rules for any window used for
        OpenGL rendering: the *top-level* window this used to be had no
        siblings/children for clipping to matter for, so it never needed
        them until now.
        """
        _user32.ShowWindow(hwnd, _SW_HIDE)
        style = _user32.GetWindowLongPtrW(hwnd, _GWL_STYLE)
        style &= ~(_WS_POPUP | _WS_CAPTION | _WS_THICKFRAME | _WS_SYSMENU)
        style |= _WS_CHILD | _WS_VISIBLE | _WS_CLIPSIBLINGS | _WS_CLIPCHILDREN
        _user32.SetWindowLongPtrW(hwnd, _GWL_STYLE, style)
        _user32.SetParent(hwnd, parent_hwnd)
        _user32.SetWindowPos(
            hwnd, None, 0, 0, 0, 0, _SWP_FRAMECHANGED | _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER
        )

    def resize_embedded_window(hwnd: int, width: int, height: int) -> None:
        _user32.MoveWindow(hwnd, 0, 0, width, height, True)

    def set_process_dpi_aware() -> None:
        # keeps the embedded window's MoveWindow calls and Tk's own
        # winfo_width/height agreeing on the same (unscaled) pixel
        # coordinates; without this, the embedded 3D view would drift out
        # of sync with its frame on a display with Windows display scaling.
        _user32.SetProcessDPIAware()
