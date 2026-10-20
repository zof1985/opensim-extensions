"""Utilities for loading and configuring the native OpenSim bindings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["import_opensim", "register_geometry_search_path"]


def _ensure_visualizer_dll_path() -> None:
    if not sys.platform.startswith("win"):
        return
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if not library_bin.is_dir():
        return
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(library_bin) not in path_entries:
        os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")


def register_geometry_search_path(opensim: Any, directory: str | Path) -> None:
    """Add a directory to OpenSim's native mesh-geometry search paths."""
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(
        str(Path(directory).expanduser().resolve())
    )


def import_opensim() -> Any:
    """Import and return the OpenSim Python bindings.

    On Windows, makes the active environment's native libraries available
    before importing OpenSim. Routine OpenSim informational logs are
    suppressed while warnings and errors remain visible.

    Returns
    -------
    module
        The imported ``opensim`` module.

    Raises
    ------
    RuntimeError
        If the OpenSim Python bindings are unavailable.
    """
    _ensure_visualizer_dll_path()
    try:
        import opensim
    except ImportError as error:
        raise RuntimeError(
            "OpenSim Python bindings are required. "
            "Install a compatible OpenSim/Conda environment first."
        ) from error
    opensim.Logger.removeFileSink()
    opensim.Logger.setLevelString("Warn")
    return opensim
