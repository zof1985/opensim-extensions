"""Generic OpenSim equipment models and screen geometry."""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from ...opensim import import_opensim

opensim = import_opensim()

__all__ = ["Equipment", "Screen"]

_MM = 1e-3


@dataclass(frozen=True)
class Screen:
    """Screen dimensions, mounting offsets, and world-frame pose.

    The screen is modelled in the package world convention: ``x`` is forward,
    ``y`` is up, and ``z`` is lateral. The lower-centre mounting point is
    calculated from ``position_anchor_m`` and the two offsets.

    Parameters
    ----------
    width_mm : float
        Screen width across the lateral axis, in millimetres.
    height_mm : float
        Screen height along the panel, in millimetres.
    inclination_deg : float
        Directed angle of the panel's upward direction from ``+x`` toward
        ``+y``, in degrees.
    position_anchor_m : sequence of float
        World-frame reference point for the screen offsets, in metres.
    horizontal_offset_mm : float
        Forward offset of the lower-centre mounting point from the reference,
        in millimetres.
    vertical_offset_mm : float
        Upward offset of the lower-centre mounting point from the reference,
        in millimetres.
    horizontal_offset_min_mm, vertical_offset_min_mm : float or None
        Inclusive lower limits on the corresponding offsets. ``None`` means
        the screen implementation does not impose a separate lower limit.
    """

    width_mm: float
    height_mm: float
    inclination_deg: float
    position_anchor_m: tuple[float, float, float]
    horizontal_offset_mm: float
    vertical_offset_mm: float
    horizontal_offset_min_mm: float | None = None
    vertical_offset_min_mm: float | None = None

    def __post_init__(self) -> None:
        for name in ("width_mm", "height_mm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value")
            object.__setattr__(self, name, value)
        for name in (
            "inclination_deg",
            "horizontal_offset_mm",
            "vertical_offset_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

        anchor = np.asarray(self.position_anchor_m, dtype=float)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            raise ValueError("position_anchor_m must contain three finite values")
        object.__setattr__(self, "position_anchor_m", tuple(map(float, anchor)))
        for name in ("horizontal_offset_min_mm", "vertical_offset_min_mm"):
            minimum = getattr(self, name)
            if minimum is None:
                continue
            minimum = float(minimum)
            if not math.isfinite(minimum):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, minimum)

    @property
    def base_position_m(self) -> np.ndarray:
        """Return the lower-centre mounting point in world coordinates.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Lower-centre screen position in metres.
        """
        return np.asarray(self.position_anchor_m) + _MM * np.array(
            [self.horizontal_offset_mm, self.vertical_offset_mm, 0.0]
        )

    @property
    def center_position_m(self) -> np.ndarray:
        """Return the geometric centre of the panel in world coordinates.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Screen-centre position in metres.
        """
        return self.base_position_m + (
            self.height_mm * _MM / 2
        ) * self._up_direction(self.inclination_deg)

    @staticmethod
    def _up_direction(inclination_deg: float) -> np.ndarray:
        angle = math.radians(inclination_deg)
        return np.array([math.cos(angle), math.sin(angle), 0.0])

    def corners(
        self, position: tuple[float, float, float] | None = None
    ) -> np.ndarray:
        """Return the four panel corners for the current or candidate pose.

        Parameters
        ----------
        position : tuple of float or None, optional
            Candidate ``(horizontal_offset_mm, vertical_offset_mm,
            inclination_deg)``. ``None`` uses the configured pose.

        Returns
        -------
        numpy.ndarray, shape (4, 3)
            Panel corners in metres in the world frame.
        """
        horizontal, vertical, inclination = (
            (
                self.horizontal_offset_mm,
                self.vertical_offset_mm,
                self.inclination_deg,
            )
            if position is None
            else position
        )
        base = np.asarray(self.position_anchor_m) + _MM * np.array(
            [horizontal, vertical, 0.0]
        )
        up = self._up_direction(inclination)
        half_width = self.width_mm * _MM / 2
        height = self.height_mm * _MM
        return np.asarray(
            [
                base + side * np.array([0.0, 0.0, 1.0]) + rise * up
                for rise in (0.0, height)
                for side in (-half_width, half_width)
            ]
        )

    def samples(
        self,
        grid_size: int,
        position: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Sample a uniform rectangular grid over the current or candidate pose.

        Parameters
        ----------
        grid_size : int
            Number of evenly spaced cell centres along each panel dimension.
        position : tuple of float or None, optional
            Candidate ``(horizontal_offset_mm, vertical_offset_mm,
            inclination_deg)``. ``None`` uses the configured pose.

        Returns
        -------
        numpy.ndarray, shape (grid_size**2, 3)
            Sample points in metres in the world frame.
        """
        if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 1:
            raise ValueError("grid_size must be a positive integer")
        horizontal, vertical, inclination = (
            (
                self.horizontal_offset_mm,
                self.vertical_offset_mm,
                self.inclination_deg,
            )
            if position is None
            else position
        )
        base = np.asarray(self.position_anchor_m) + _MM * np.array(
            [horizontal, vertical, 0.0]
        )
        up = self._up_direction(inclination)
        values = (np.arange(grid_size) + 0.5) / grid_size
        return np.asarray(
            [
                base
                + self.height_mm * _MM * rise * up
                + self.width_mm
                * _MM
                * (2 * side - 1)
                / 2
                * np.array([0.0, 0.0, 1.0])
                for rise in values
                for side in values
            ]
        )


class Equipment(opensim.Model, ABC):
    """Abstract base class for user-facing OpenSim equipment.

    Concrete equipment provides its own user-fitting procedure. If it has a
    screen, :attr:`screen` returns the complete screen dimensions and pose;
    otherwise it returns ``None``.
    """

    @property
    def screen(self) -> Screen | None:
        """Return this equipment's screen description, if present.

        Returns
        -------
        Screen or None
            Current screen dimensions and world-frame pose, or ``None`` when
            this equipment has no screen.
        """
        return None

    def has_screen(self) -> bool:
        """Report whether this equipment provides a screen.

        Returns
        -------
        bool
            ``True`` when :attr:`screen` contains a screen description.
        """
        return self.screen is not None

    def set_screen_position(
        self,
        horizontal_offset_mm: float,
        vertical_offset_mm: float,
        inclination_deg: float,
    ) -> None:
        """Apply a candidate pose to the equipment screen.

        Screen-bearing equipment must override this method so optimizers can
        apply the selected screen pose to its native model.

        Parameters
        ----------
        horizontal_offset_mm, vertical_offset_mm : float
            Screen mounting offsets from its equipment-specific reference, in
            millimetres.
        inclination_deg : float
            Directed panel inclination in degrees.

        Raises
        ------
        NotImplementedError
            If the equipment has a screen but does not support repositioning.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement screen repositioning"
        )

    def expand_screen_offset_limits(
        self,
        horizontal_min_mm: float | None = None,
        vertical_min_mm: float | None = None,
    ) -> None:
        """Expand equipment-specific lower bounds for screen offsets.

        Screen-bearing equipment with configurable offset limits should
        override this method. The default is suitable for equipment whose
        screen offsets do not have separate lower limits.
        """
        if self.screen is not None and (
            (horizontal_min_mm is not None and self.screen.horizontal_offset_min_mm is not None)
            or (vertical_min_mm is not None and self.screen.vertical_offset_min_mm is not None)
        ):
            raise NotImplementedError(
                f"{type(self).__name__} does not support expanding screen offset limits"
            )

    def screen_position_constraints(
        self, position: tuple[float, float, float]
    ) -> np.ndarray:
        """Return feasibility margins for a candidate screen pose.

        Positive values indicate satisfied constraints. Equipment without
        special placement constraints inherits an always-feasible margin.

        Parameters
        ----------
        position : tuple of float
            Candidate ``(horizontal_offset_mm, vertical_offset_mm,
            inclination_deg)``.

        Returns
        -------
        numpy.ndarray
            Non-negative feasibility margins, or negative values for violated
            constraints.
        """
        return np.ones(1)

    def clone(self) -> Self:
        """Create an independent native and Python-level copy of this equipment.

        Concrete subclasses with non-copyable Python state may override this
        method. The copied OpenSim model is reinitialized so its working state
        belongs to the clone.

        Returns
        -------
        Equipment
            Independent copy of this equipment instance.
        """
        result = type(self).__new__(type(self))
        opensim.Model.__init__(result, self)
        for name, value in vars(self).items():
            if name == "opensim":
                setattr(result, name, value)
            elif name == "_visualizer":
                setattr(result, name, None)
            elif name == "state":
                continue
            else:
                setattr(result, name, copy.deepcopy(value))
        result.state = result.initSystem()
        result.realizePosition(result.state)
        return result

    @abstractmethod
    def fit_user(self, *args: Any, **kwargs: Any) -> Any:
        """Fit a user to this equipment using a subclass-specific procedure."""
        raise NotImplementedError
