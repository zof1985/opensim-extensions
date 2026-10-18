"""Generated wrapper for :class:`opensim.Body`."""

from __future__ import annotations

import opensim


class Body(opensim.Body):
    """OpenSim rigid body.\n\n    An OpenSim::Body is a PhysicalFrame (reference frame) with associated

    Parameters
    ----------
    See the native :class:`opensim.Body` constructor.

    Notes
    -----
    This generated subclass adds statically discoverable properties.
    """

    @property
    def mass(self) -> float:
        """The mass of the body (kg)

        Returns
        -------
        float
            Value of the ``mass`` property.
        """
        return self.get_mass()

    def set_mass(self, value: float) -> None:
        """Set the ``mass`` property.

        Parameters
        ----------
        value : float
            New value for ``mass``.
        """
        opensim.Body.set_mass(self, value)

    @property
    def mass_center(self) -> opensim.Vec3:
        """The location (Vec3) of the mass center in the body frame.

        Returns
        -------
        opensim.Vec3
            Value of the ``mass_center`` property.
        """
        return self.get_mass_center()

    def set_mass_center(self, value: opensim.Vec3) -> None:
        """Set the ``mass_center`` property.

        Parameters
        ----------
        value : opensim.Vec3
            New value for ``mass_center``.
        """
        opensim.Body.set_mass_center(self, value)

    @property
    def inertia(self) -> opensim.Vec6:
        """The elements of the inertia tensor (Vec6) as [Ixx Iyy Izz Ixy Ixz Iyz] measured about the mass_center and not the body origin.

        Returns
        -------
        opensim.Vec6
            Value of the ``inertia`` property.
        """
        return self.get_inertia()

    def set_inertia(self, value: opensim.Vec6) -> None:
        """Set the ``inertia`` property.

        Parameters
        ----------
        value : opensim.Vec6
            New value for ``inertia``.
        """
        opensim.Body.set_inertia(self, value)

__all__ = ["Body"]
