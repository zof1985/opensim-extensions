from __future__ import annotations

__all__ = ["segment_scale_factors"]

from collections.abc import Mapping

from .data import AnthropometricReference

DEFAULT_BODY_NAMES = (
    "pelvis",
    "femur_r",
    "tibia_r",
    "patella_r",
    "talus_r",
    "calcn_r",
    "toes_r",
    "femur_l",
    "tibia_l",
    "patella_l",
    "talus_l",
    "calcn_l",
    "toes_l",
    "torso",
    "humerus_r",
    "ulna_r",
    "radius_r",
    "hand_r",
    "humerus_l",
    "ulna_l",
    "radius_l",
    "hand_l",
)

# Entries are (ANSUR measure, OpenSim body, axis). Measurements are applied
# to both sides. The remaining axes and bodies use the stature ratio as a
# conservative, deterministic fallback rather than remaining unscaled.
SEGMENT_MEASURES: Mapping[str, tuple[tuple[str, str, str], ...]] = {
    "pelvis": (("hipbreadth", "pelvis", "x"), ("buttockdepth", "pelvis", "z")),
    "torso": (
        ("chestbreadth", "torso", "x"),
        ("sittingheight", "torso", "y"),
        ("chestdepth", "torso", "z"),
    ),
    "femur": (("buttockkneelength", "femur", "y"),),
    "tibia": (("tibialheight", "tibia", "y"),),
    "calcn": (("footlength", "calcn", "y"), ("footbreadthhorizontal", "calcn", "x")),
    "toes": (("footlength", "toes", "y"), ("footbreadthhorizontal", "toes", "x")),
    "humerus": (("shoulderelbowlength", "humerus", "y"),),
    "ulna": (("acromionradialelength", "ulna", "y"),),
    "radius": (("radialestylionlength", "radius", "y"),),
    "hand": (("handlength", "hand", "y"), ("handbreadth", "hand", "x")),
}


def segment_scale_factors(
    target: AnthropometricReference,
    baseline: AnthropometricReference,
) -> dict[str, tuple[float, float, float]]:
    """Build complete body scale factors from anthropometric references.

    Directly mapped ANSUR measurements override the corresponding axis. All
    remaining axes use the stature ratio, which keeps every body in the model
    scaled while making the fallback explicit and deterministic.

    Parameters
    ----------
    target : AnthropometricReference
        Anthropometric measurements requested for the user.
    baseline : AnthropometricReference
        Measurements describing the unscaled reference model.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        OpenSim body names mapped to positive ``(x, y, z)`` scale factors.

    Raises
    ------
    KeyError
        If either reference does not contain ``stature_m``.
    """
    stature_ratio = target.values["stature_m"] / baseline.values["stature_m"]
    factors: dict[str, list[float]] = {
        body: [stature_ratio, stature_ratio, stature_ratio]
        for body in DEFAULT_BODY_NAMES
    }
    for mappings in SEGMENT_MEASURES.values():
        for measure, body, axis in mappings:
            target_value = target.values.get(measure)
            base_value = baseline.values.get(measure)
            if target_value is None or base_value is None or base_value <= 0:
                continue
            axes = factors.setdefault(body, [stature_ratio] * 3)
            axes["xyz".index(axis)] = target_value / base_value
    return {body: tuple(axes) for body, axes in factors.items()}
