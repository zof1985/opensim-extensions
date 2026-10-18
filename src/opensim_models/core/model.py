"""Public core-model import path.

The implementation remains in :mod:`opensim_models.model` while it is split
into smaller core modules.
"""

from ..model import OpenSimModel, import_opensim

__all__ = ["OpenSimModel", "import_opensim"]
