from __future__ import annotations

__all__ = []


class _MeasurementMixin:
    """Explicit anthropometric measurement setters."""

    def set_height(self, value: float) -> None:
        """Set height and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in cm.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('height', 'stature_m', value, source_multiplier=0.01)

    def set_biacromial_breadth(self, value: float) -> None:
        """Set biacromial breadth and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('biacromial_breadth', 'biacromialbreadth', value, source_multiplier=1000.0)

    def set_neck_circumference(self, value: float) -> None:
        """Set neck circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('neck_circumference', 'neckcircumference', value, source_multiplier=1000.0)

    def set_chest_circumference(self, value: float) -> None:
        """Set chest circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('chest_circumference', 'chestcircumference', value, source_multiplier=1000.0)

    def set_chest_depth(self, value: float) -> None:
        """Set chest depth and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('chest_depth', 'chestdepth', value, source_multiplier=1000.0)

    def set_chest_width(self, value: float) -> None:
        """Set chest width and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('chest_width', 'chestbreadth', value, source_multiplier=1000.0)

    def set_waist_circumference(self, value: float) -> None:
        """Set waist circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('waist_circumference', 'waistcircumference', value, source_multiplier=1000.0)

    def set_waist_depth(self, value: float) -> None:
        """Set waist depth and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('waist_depth', 'waistdepth', value, source_multiplier=1000.0)

    def set_waist_width(self, value: float) -> None:
        """Set waist width and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('waist_width', 'waistbreadth', value, source_multiplier=1000.0)

    def set_hip_circumference(self, value: float) -> None:
        """Set hip circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('hip_circumference', 'buttockcircumference', value, source_multiplier=1000.0)

    def set_hip_depth(self, value: float) -> None:
        """Set hip depth and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('hip_depth', 'buttockdepth', value, source_multiplier=1000.0)

    def set_hip_width(self, value: float) -> None:
        """Set hip width and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_reference_measurement('hip_width', 'hipbreadth', value, source_multiplier=1000.0)

    def set_left_foot_length(self, value: float) -> None:
        """Set left foot length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_foot_length', value * 1.0, body_axes=(('calcn_l', ('y',)), ('toes_l', ('y',))), store_override=True)

    def set_right_foot_length(self, value: float) -> None:
        """Set right foot length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_foot_length', value * 1.0, body_axes=(('calcn_r', ('y',)), ('toes_r', ('y',))), store_override=True)

    def set_left_arm_length(self, value: float) -> None:
        """Set left arm length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_arm_length', value * 1.0, body_axes=(('humerus_l', ('y',)),), store_override=False)

    def set_right_arm_length(self, value: float) -> None:
        """Set right arm length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_arm_length', value * 1.0, body_axes=(('humerus_r', ('y',)),), store_override=False)

    def set_left_forearm_length(self, value: float) -> None:
        """Set left forearm length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_forearm_length', value * 1.0, body_axes=(('ulna_l', ('y',)), ('radius_l', ('y',))), store_override=False)

    def set_right_forearm_length(self, value: float) -> None:
        """Set right forearm length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_forearm_length', value * 1.0, body_axes=(('ulna_r', ('y',)), ('radius_r', ('y',))), store_override=False)

    def set_left_thigh_length(self, value: float) -> None:
        """Set left thigh length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_thigh_length', value * 1.0, body_axes=(('femur_l', ('y',)),), store_override=False)

    def set_right_thigh_length(self, value: float) -> None:
        """Set right thigh length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_thigh_length', value * 1.0, body_axes=(('femur_r', ('y',)),), store_override=False)

    def set_left_shank_length(self, value: float) -> None:
        """Set left shank length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_shank_length', value * 1.0, body_axes=(('tibia_l', ('y',)),), store_override=False)

    def set_right_shank_length(self, value: float) -> None:
        """Set right shank length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_shank_length', value * 1.0, body_axes=(('tibia_r', ('y',)),), store_override=False)

    def set_left_palm_length(self, value: float) -> None:
        """Set left palm length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_palm_length', value * 1.0, body_axes=(('hand_l', ('y',)),), store_override=True)

    def set_right_palm_length(self, value: float) -> None:
        """Set right palm length and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_palm_length', value * 1.0, body_axes=(('hand_r', ('y',)),), store_override=True)

    def set_left_arm_circumference(self, value: float) -> None:
        """Set left arm circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_arm_circumference', value * 1.0, body_axes=(('humerus_l', ('x', 'z')),), store_override=True)

    def set_right_arm_circumference(self, value: float) -> None:
        """Set right arm circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_arm_circumference', value * 1.0, body_axes=(('humerus_r', ('x', 'z')),), store_override=True)

    def set_left_forearm_circumference(self, value: float) -> None:
        """Set left forearm circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_forearm_circumference', value * 1.0, body_axes=(('ulna_l', ('x', 'z')), ('radius_l', ('x', 'z'))), store_override=True)

    def set_right_forearm_circumference(self, value: float) -> None:
        """Set right forearm circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_forearm_circumference', value * 1.0, body_axes=(('ulna_r', ('x', 'z')), ('radius_r', ('x', 'z'))), store_override=True)

    def set_left_thigh_circumference(self, value: float) -> None:
        """Set left thigh circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_thigh_circumference', value * 1.0, body_axes=(('femur_l', ('x', 'z')),), store_override=True)

    def set_right_thigh_circumference(self, value: float) -> None:
        """Set right thigh circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_thigh_circumference', value * 1.0, body_axes=(('femur_r', ('x', 'z')),), store_override=True)

    def set_left_calf_circumference(self, value: float) -> None:
        """Set left calf circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_calf_circumference', value * 1.0, body_axes=(('tibia_l', ('x', 'z')),), store_override=True)

    def set_right_calf_circumference(self, value: float) -> None:
        """Set right calf circumference and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_calf_circumference', value * 1.0, body_axes=(('tibia_r', ('x', 'z')),), store_override=True)

    def set_left_thigh_depth(self, value: float) -> None:
        """Set left thigh depth and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_thigh_circumference', value * 3.141592653589793, body_axes=(('femur_l', ('x', 'z')),), store_override=True)

    def set_right_thigh_depth(self, value: float) -> None:
        """Set right thigh depth and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_thigh_circumference', value * 3.141592653589793, body_axes=(('femur_r', ('x', 'z')),), store_override=True)

    def set_left_thigh_width(self, value: float) -> None:
        """Set left thigh width and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_thigh_circumference', value * 3.141592653589793, body_axes=(('femur_l', ('x', 'z')),), store_override=True)

    def set_right_thigh_width(self, value: float) -> None:
        """Set right thigh width and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_thigh_circumference', value * 3.141592653589793, body_axes=(('femur_r', ('x', 'z')),), store_override=True)

    def set_left_calf_depth(self, value: float) -> None:
        """Set left calf depth and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_calf_circumference', value * 3.141592653589793, body_axes=(('tibia_l', ('x', 'z')),), store_override=True)

    def set_right_calf_depth(self, value: float) -> None:
        """Set right calf depth and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_calf_circumference', value * 3.141592653589793, body_axes=(('tibia_r', ('x', 'z')),), store_override=True)

    def set_left_calf_width(self, value: float) -> None:
        """Set left calf width and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('left_calf_circumference', value * 3.141592653589793, body_axes=(('tibia_l', ('x', 'z')),), store_override=True)

    def set_right_calf_width(self, value: float) -> None:
        """Set right calf width and rescale the model immediately.

        Parameters
        ----------
        value : float
            Target value in m.

        Raises
        ------
        ValueError
            If ``value`` is not finite and strictly positive.
        """
        self._set_side_measurement('right_calf_circumference', value * 3.141592653589793, body_axes=(('tibia_r', ('x', 'z')),), store_override=True)
