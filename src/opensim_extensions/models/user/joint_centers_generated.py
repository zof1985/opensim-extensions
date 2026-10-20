from __future__ import annotations

__all__ = []


class _JointCenterMixin:
    """Every joint-center property."""

    @property
    def pelvis(self) -> tuple[float, float, float]:
        """Return the pelvis joint centre, in the ground frame.

        This is the child frame origin of the ``ground_pelvis`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('ground_pelvis')

    @property
    def left_hip(self) -> tuple[float, float, float]:
        """Return the left hip joint centre, in the ground frame.

        This is the child frame origin of the ``hip_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('hip_l')

    @property
    def right_hip(self) -> tuple[float, float, float]:
        """Return the right hip joint centre, in the ground frame.

        This is the child frame origin of the ``hip_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('hip_r')

    @property
    def left_knee(self) -> tuple[float, float, float]:
        """Return the left knee joint centre, in the ground frame.

        This is the child frame origin of the ``walker_knee_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('walker_knee_l')

    @property
    def right_knee(self) -> tuple[float, float, float]:
        """Return the right knee joint centre, in the ground frame.

        This is the child frame origin of the ``walker_knee_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('walker_knee_r')

    @property
    def left_patella(self) -> tuple[float, float, float]:
        """Return the left patellofemoral joint centre, in the ground frame.

        This is the child frame origin of the ``patellofemoral_l`` joint.
        The patella's own coordinate (``knee_angle_l_beta``) is kinematically
        coupled to ``knee_angle_l`` via a ``CoordinateCouplerConstraint``;
        this position reflects the model's current posture, but that
        coupling is only resolved by ``realizePosition``/``assemble``-level
        updates, not by a plain :meth:`update_state` after a mid-edit knee
        angle change (see :meth:`set_left_knee_flexionextension` and the
        README's "Limiti noti di update_state()").

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('patellofemoral_l')

    @property
    def right_patella(self) -> tuple[float, float, float]:
        """Return the right patellofemoral joint centre, in the ground frame.

        This is the child frame origin of the ``patellofemoral_r`` joint.
        The patella's own coordinate (``knee_angle_r_beta``) is kinematically
        coupled to ``knee_angle_r`` via a ``CoordinateCouplerConstraint``;
        this position reflects the model's current posture, but that
        coupling is only resolved by ``realizePosition``/``assemble``-level
        updates, not by a plain :meth:`update_state` after a mid-edit knee
        angle change (see :meth:`set_right_knee_flexionextension` and the
        README's "Limiti noti di update_state()").

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('patellofemoral_r')

    @property
    def left_ankle(self) -> tuple[float, float, float]:
        """Return the left ankle joint centre, in the ground frame.

        This is the child frame origin of the ``ankle_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first). Also used
        as the approximation for :attr:`left_foot_height`.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('ankle_l')

    @property
    def right_ankle(self) -> tuple[float, float, float]:
        """Return the right ankle joint centre, in the ground frame.

        This is the child frame origin of the ``ankle_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first). Also used
        as the approximation for :attr:`right_foot_height`.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('ankle_r')

    @property
    def left_subtalar(self) -> tuple[float, float, float]:
        """Return the left subtalar joint centre, in the ground frame.

        This is the child frame origin of the ``subtalar_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('subtalar_l')

    @property
    def right_subtalar(self) -> tuple[float, float, float]:
        """Return the right subtalar joint centre, in the ground frame.

        This is the child frame origin of the ``subtalar_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('subtalar_r')

    @property
    def left_mtp(self) -> tuple[float, float, float]:
        """Return the left metatarsophalangeal (toe) joint centre, in the ground frame.

        This is the child frame origin of the ``mtp_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('mtp_l')

    @property
    def right_mtp(self) -> tuple[float, float, float]:
        """Return the right metatarsophalangeal (toe) joint centre, in the ground frame.

        This is the child frame origin of the ``mtp_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('mtp_r')

    @property
    def torso(self) -> tuple[float, float, float]:
        """Return the torso (back) joint centre, in the ground frame.

        This is the child frame origin of the ``back`` joint (the lumbar
        joint connecting pelvis and torso), reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('back')

    @property
    def left_shoulder(self) -> tuple[float, float, float]:
        """Return the left shoulder joint centre, in the ground frame.

        This is the child frame origin of the ``acromial_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first). See
        :attr:`shoulder_width` and :attr:`biacromial_breadth` for two
        different measures derived from this and :attr:`right_shoulder`.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('acromial_l')

    @property
    def right_shoulder(self) -> tuple[float, float, float]:
        """Return the right shoulder joint centre, in the ground frame.

        This is the child frame origin of the ``acromial_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first). See
        :attr:`shoulder_width` and :attr:`biacromial_breadth` for two
        different measures derived from this and :attr:`left_shoulder`.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('acromial_r')

    @property
    def left_elbow(self) -> tuple[float, float, float]:
        """Return the left elbow joint centre, in the ground frame.

        This is the child frame origin of the ``elbow_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('elbow_l')

    @property
    def right_elbow(self) -> tuple[float, float, float]:
        """Return the right elbow joint centre, in the ground frame.

        This is the child frame origin of the ``elbow_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('elbow_r')

    @property
    def left_radioulnar(self) -> tuple[float, float, float]:
        """Return the left radioulnar joint centre, in the ground frame.

        This is the child frame origin of the ``radioulnar_l`` joint (the
        joint whose coordinate is driven by :meth:`set_left_forearm_pronation`),
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('radioulnar_l')

    @property
    def right_radioulnar(self) -> tuple[float, float, float]:
        """Return the right radioulnar joint centre, in the ground frame.

        This is the child frame origin of the ``radioulnar_r`` joint (the
        joint whose coordinate is driven by :meth:`set_right_forearm_pronation`),
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('radioulnar_r')

    @property
    def left_wrist(self) -> tuple[float, float, float]:
        """Return the left wrist joint centre, in the ground frame.

        This is the child frame origin of the ``radius_hand_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('radius_hand_l')

    @property
    def right_wrist(self) -> tuple[float, float, float]:
        """Return the right wrist joint centre, in the ground frame.

        This is the child frame origin of the ``radius_hand_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._joint_center('radius_hand_r')
