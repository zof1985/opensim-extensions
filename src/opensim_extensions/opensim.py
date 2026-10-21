"""Utilities for loading and configuring the native OpenSim bindings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["import_opensim", "register_geometry_search_path", "resolve_geometry_file"]


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
    """Register a directory with OpenSim's native mesh search paths.

    Parameters
    ----------
    opensim : module
        Imported OpenSim Python bindings exposing ``ModelVisualizer``.
    directory : str or pathlib.Path
        Directory containing geometry files. User-home markers are expanded
        and the path is resolved before registration.
    """
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(
        str(Path(directory).expanduser().resolve())
    )


def resolve_geometry_file(
    opensim: Any, model: Any, filename: str, directories: Any = ()
) -> Path | None:
    """Resolve a mesh file against local and OpenSim geometry search paths.

    Parameters
    ----------
    opensim : module
        Imported OpenSim Python bindings.
    model : opensim.Model
        Model used by OpenSim's native geometry resolver.
    filename : str
        Mesh filename or path stored in a model geometry component.
    directories : iterable of str or pathlib.Path, optional
        Directories searched in order before consulting OpenSim's registered
        geometry paths.

    Returns
    -------
    pathlib.Path or None
        First matching local path, or the native resolved path. Returns
        ``None`` if the file cannot be resolved.
    """
    for directory in directories:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    try:
        resolved = opensim.ModelVisualizer.findGeometryFile(
            model, filename, False, opensim.ArrayStr()
        )
    except (RuntimeError, TypeError):
        return None
    return Path(str(resolved)) if resolved else None


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
