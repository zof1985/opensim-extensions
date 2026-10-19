"""Table of every joint-center property User exposes (names and coordinates)."""

from __future__ import annotations

__all__ = ["JOINT_CENTER_ENTRIES", "JOINT_CENTER_NAMES"]

from dataclasses import dataclass


@dataclass(frozen=True)
class JointCenterEntry:
    """One joint-center property: its friendly name, OpenSim joint name, and
    its exact docstring text (verbatim -- several carry an extra
    cross-reference sentence beyond the standard template, e.g. a
    CoordinateCouplerConstraint caveat for the patella, so the whole
    paragraph is stored rather than decomposed).
    """

    attr_name: str
    joint_name: str
    docstring: str


JOINT_CENTER_ENTRIES: tuple[JointCenterEntry, ...] = (
    JointCenterEntry(
        attr_name='pelvis',
        joint_name='ground_pelvis',
        docstring=(
            """Return the pelvis joint centre, in the ground frame.

        This is the child frame origin of the ``ground_pelvis`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='left_hip',
        joint_name='hip_l',
        docstring=(
            """Return the left hip joint centre, in the ground frame.

        This is the child frame origin of the ``hip_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='right_hip',
        joint_name='hip_r',
        docstring=(
            """Return the right hip joint centre, in the ground frame.

        This is the child frame origin of the ``hip_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='left_knee',
        joint_name='walker_knee_l',
        docstring=(
            """Return the left knee joint centre, in the ground frame.

        This is the child frame origin of the ``walker_knee_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='right_knee',
        joint_name='walker_knee_r',
        docstring=(
            """Return the right knee joint centre, in the ground frame.

        This is the child frame origin of the ``walker_knee_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='left_patella',
        joint_name='patellofemoral_l',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='right_patella',
        joint_name='patellofemoral_r',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='left_ankle',
        joint_name='ankle_l',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='right_ankle',
        joint_name='ankle_r',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='left_subtalar',
        joint_name='subtalar_l',
        docstring=(
            """Return the left subtalar joint centre, in the ground frame.

        This is the child frame origin of the ``subtalar_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='right_subtalar',
        joint_name='subtalar_r',
        docstring=(
            """Return the right subtalar joint centre, in the ground frame.

        This is the child frame origin of the ``subtalar_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='left_mtp',
        joint_name='mtp_l',
        docstring=(
            """Return the left metatarsophalangeal (toe) joint centre, in the ground frame.

        This is the child frame origin of the ``mtp_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='right_mtp',
        joint_name='mtp_r',
        docstring=(
            """Return the right metatarsophalangeal (toe) joint centre, in the ground frame.

        This is the child frame origin of the ``mtp_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='torso',
        joint_name='back',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='left_shoulder',
        joint_name='acromial_l',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='right_shoulder',
        joint_name='acromial_r',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='left_elbow',
        joint_name='elbow_l',
        docstring=(
            """Return the left elbow joint centre, in the ground frame.

        This is the child frame origin of the ``elbow_l`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='right_elbow',
        joint_name='elbow_r',
        docstring=(
            """Return the right elbow joint centre, in the ground frame.

        This is the child frame origin of the ``elbow_r`` joint, reflecting
        the model's current posture (``realizePosition`` is called
        automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='left_radioulnar',
        joint_name='radioulnar_l',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='right_radioulnar',
        joint_name='radioulnar_r',
        docstring=(
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
        ),
    ),
    JointCenterEntry(
        attr_name='left_wrist',
        joint_name='radius_hand_l',
        docstring=(
            """Return the left wrist joint centre, in the ground frame.

        This is the child frame origin of the ``radius_hand_l`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
    JointCenterEntry(
        attr_name='right_wrist',
        joint_name='radius_hand_r',
        docstring=(
            """Return the right wrist joint centre, in the ground frame.

        This is the child frame origin of the ``radius_hand_r`` joint,
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need for :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        ),
    ),
)

# Derived view matching the pre-refactor _JOINT_CENTER_NAMES mapping,
# kept so existing imports/tests of that exact name keep working unchanged.
JOINT_CENTER_NAMES: dict[str, str] = {e.attr_name: e.joint_name for e in JOINT_CENTER_ENTRIES}
