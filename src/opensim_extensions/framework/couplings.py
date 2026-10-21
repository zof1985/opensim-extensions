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
        """Return the framework members referenced by this coupling.

        Returns
        -------
        frozenset of str
            Distinct member names referenced by the coupling's component
            paths.
        """
        return frozenset(member for member, _ in self.references)

    def option(self, key: str, default: Any = None) -> Any:
        """Look up a kind-specific coupling option.

        Parameters
        ----------
        key : str
            Option name to retrieve.
        default : object, optional
            Value returned when the option is not present.

        Returns
        -------
        object
            Stored option value, or ``default`` when the key is absent.
        """
        return dict(self.options).get(key, default)


def normalize_reference(reference: Sequence[str]) -> ComponentRef:
    """Validate and normalize a member/component reference.

    Parameters
    ----------
    reference : sequence of str
        Pair containing a framework member name and an absolute component
        path within that member.

    Returns
    -------
    tuple[str, str]
        The validated ``(member_name, component_path)`` pair.

    Raises
    ------
    TypeError
        If the reference is not a pair of non-empty strings.
    ValueError
        If the component path is not absolute.
    """
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
