"""Declarative couplings between components of different framework members."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["Coupling", "ComponentRef"]

ComponentRef = tuple[str, str]
"""``(member_name, component_path)`` reference into a framework member."""

CouplingKind = Literal[
    "weld", "point", "coordinate", "joint", "force", "constraint", "controller"
]


@dataclass(frozen=True)
class Coupling:
    """Describe a link applied only to a framework's composite model.

    Instances are created by the ``Framework.couple_*`` methods and are
    immutable. ``references`` holds the ``(member, path)`` pairs the coupling
    needs, in the order described by each kind.

    Attributes
    ----------
    name : str
        Unique coupling name within the framework.
    kind : {"weld", "point", "coordinate", "joint", "force", "constraint", "controller"}
        Coupling type.
    references : tuple[tuple[str, str], ...]
        Referenced member components. ``weld`` and ``point``: the two
        physical frames. ``coordinate``: the independent coordinates followed
        by the dependent one. ``joint``: the new parent frame followed by the
        child joint. ``force``, ``constraint`` and ``controller``: every
        component referenced by the added component's sockets.
    options : tuple[tuple[str, Any], ...]
        Extra kind-specific settings (point locations, coupling function, or
        the component template and its socket assignments).
    """

    name: str
    kind: CouplingKind
    references: tuple[ComponentRef, ...]
    options: tuple[tuple[str, Any], ...] = ()

    @property
    def members(self) -> frozenset[str]:
        """Names of the framework members this coupling involves."""
        return frozenset(member for member, _ in self.references)

    def option(self, key: str, default: Any = None) -> Any:
        """Return the option stored under ``key``, or ``default``."""
        return dict(self.options).get(key, default)


def normalize_reference(reference: Sequence[str]) -> ComponentRef:
    """Return ``reference`` as a validated ``(member, path)`` tuple."""
    if (
        isinstance(reference, (str, bytes))
        or len(reference) != 2
        or not all(isinstance(part, str) and part for part in reference)
    ):
        raise TypeError("a component reference must be a (member, path) pair of strings")
    member, path = reference
    if not path.startswith("/"):
        raise ValueError(f"component path must be absolute, got {path!r}")
    return (member, path)
