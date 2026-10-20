import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import OpenSimModel, User

# User only exposes DEFAULT_DATASET/DEFAULT_MESHES_DIR/load_ansur/resolve_reference
# internally; tests reach into the implementation modules directly to exercise them.
from opensim_models.models.user._data import DEFAULT_DATASET, load_ansur, resolve_reference
from opensim_models.models.user.user import (
    DEFAULT_MESHES_DIR,
    _FOOT_MARKER_NAMES,
    _JOINT_CENTER_NAMES,
    _POSTURE_COORDINATE_NAMES,
)

try:
    import opensim
except ImportError:
    opensim = None

requires_opensim = pytest.mark.skipif(
    opensim is None, reason="OpenSim bindings are not installed"
)

DATASET = DEFAULT_DATASET

POSTURE_SETTERS = [
    ("set_left_hip_flexionextension", "hip_flexion_l"),
    ("set_right_hip_flexionextension", "hip_flexion_r"),
    ("set_left_hip_adduction", "hip_adduction_l"),
    ("set_right_hip_adduction", "hip_adduction_r"),
    ("set_left_hip_rotation", "hip_rotation_l"),
    ("set_right_hip_rotation", "hip_rotation_r"),
    ("set_left_knee_flexionextension", "knee_angle_l"),
    ("set_right_knee_flexionextension", "knee_angle_r"),
    ("set_left_ankle_flexiondorsiflexion", "ankle_angle_l"),
    ("set_right_ankle_flexiondorsiflexion", "ankle_angle_r"),
    ("set_left_subtalar_inversion", "subtalar_angle_l"),
    ("set_right_subtalar_inversion", "subtalar_angle_r"),
    ("set_left_mtp_flexion", "mtp_angle_l"),
    ("set_right_mtp_flexion", "mtp_angle_r"),
    ("set_left_shoulder_flexion", "arm_flex_l"),
    ("set_right_shoulder_flexion", "arm_flex_r"),
    ("set_left_shoulder_adduction", "arm_add_l"),
    ("set_right_shoulder_adduction", "arm_add_r"),
    ("set_left_shoulder_rotation", "arm_rot_l"),
    ("set_right_shoulder_rotation", "arm_rot_r"),
    ("set_left_elbow_flexion", "elbow_flex_l"),
    ("set_right_elbow_flexion", "elbow_flex_r"),
    ("set_left_wrist_flexion", "wrist_flex_l"),
    ("set_right_wrist_flexion", "wrist_flex_r"),
    ("set_left_wrist_deviation", "wrist_dev_l"),
    ("set_right_wrist_deviation", "wrist_dev_r"),
    ("set_left_forearm_pronation", "pro_sup_l"),
    ("set_right_forearm_pronation", "pro_sup_r"),
    ("set_lumbar_extension", "lumbar_extension"),
    ("set_lumbar_bending", "lumbar_bending"),
    ("set_lumbar_rotation", "lumbar_rotation"),
]

# (property name, ANSUR column) -- every measurement read straight from
# anthropometry.values, with a simple unit conversion (mm -> m).
ANSUR_DIRECT_PROPERTIES = [
    ("left_foot_length", "footlength"),
    ("right_foot_length", "footlength"),
    ("left_palm_length", "palmlength"),
    ("right_palm_length", "palmlength"),
    ("biacromial_breadth", "biacromialbreadth"),
    ("left_arm_circumference", "bicepscircumferenceflexed"),
    ("right_arm_circumference", "bicepscircumferenceflexed"),
    ("left_forearm_circumference", "forearmcircumferenceflexed"),
    ("right_forearm_circumference", "forearmcircumferenceflexed"),
    ("neck_circumference", "neckcircumference"),
    ("chest_circumference", "chestcircumference"),
    ("chest_depth", "chestdepth"),
    ("chest_width", "chestbreadth"),
    ("waist_circumference", "waistcircumference"),
    ("waist_depth", "waistdepth"),
    ("waist_width", "waistbreadth"),
    ("hip_circumference", "buttockcircumference"),
    ("hip_depth", "buttockdepth"),
    ("hip_width", "hipbreadth"),
    ("left_thigh_circumference", "thighcircumference"),
    ("right_thigh_circumference", "thighcircumference"),
    ("left_calf_circumference", "calfcircumference"),
    ("right_calf_circumference", "calfcircumference"),
]

# (property name, ANSUR circumference column) -- thigh/calf depth and width
# have no ANSUR breadth to fit an ellipse against, so both are derived as a
# circular-section diameter (circumference / pi) from the same column.
CIRCULAR_SECTION_PROPERTIES = [
    ("left_thigh_depth", "thighcircumference"),
    ("right_thigh_depth", "thighcircumference"),
    ("left_thigh_width", "thighcircumference"),
    ("right_thigh_width", "thighcircumference"),
    ("left_calf_depth", "calfcircumference"),
    ("right_calf_depth", "calfcircumference"),
    ("left_calf_width", "calfcircumference"),
    ("right_calf_width", "calfcircumference"),
]


def make_user(**kwargs):
    return User(dataset=DATASET, **kwargs)


# ---------------------------------------------------------------------------
# ANSUR data layer (no OpenSim binding required)
# ---------------------------------------------------------------------------


def test_loader_repairs_implicit_subject_id_and_gender():
    frame = load_ansur(DATASET)

    assert frame.columns[0] == "subject_id"
    assert len(frame) == 6068
    assert set(frame["Gender"]) == {"M", "F"}
    assert frame["stature_m"].dtype.kind == "f"


def test_default_percentile_is_used_for_gender_only():
    reference = resolve_reference("M", dataset=DATASET)

    assert reference.gender == "M"
    assert reference.percentile == 50.0
    assert reference.height_cm == pytest.approx(175.5, abs=0.1)


def test_explicit_percentile_applies_to_height_and_measurements():
    reference = resolve_reference("F", percentile=75.0, dataset=DATASET)
    female = load_ansur(DATASET).query("Gender == 'F'")

    assert reference.percentile == 75.0
    assert reference.height_m == pytest.approx(
        np.percentile(female["stature_m"], 75.0, method="linear")
    )
    assert reference.values["footlength"] == pytest.approx(
        np.percentile(female["footlength"], 75.0, method="linear")
    )


def test_height_resolves_a_common_target_percentile():
    reference = resolve_reference("M", height=175.0, dataset=DATASET)
    male = load_ansur(DATASET).query("Gender == 'M'")
    expected_percentile = (
        np.searchsorted(np.sort(male["stature_m"].to_numpy()), 1.75, side="right")
        * 100
        / len(male)
    )

    assert reference.percentile == pytest.approx(
        min(99.9, max(0.1, expected_percentile))
    )
    assert reference.height_cm == pytest.approx(175.0)


@pytest.mark.parametrize("gender", ["X", "male", ""])
def test_gender_is_validated(gender):
    with pytest.raises(ValueError, match="gender"):
        resolve_reference(gender, dataset=DATASET)


def test_height_outside_dataset_range_warns_and_extrapolates():
    male = load_ansur(DATASET).query("Gender == 'M'")
    assert 205.0 > male["stature_m"].max() * 100

    with pytest.warns(UserWarning, match="outside ANSUR range"):
        reference = resolve_reference("M", height=205.0, dataset=DATASET)

    baseline = resolve_reference("M", percentile=99.0, dataset=DATASET)

    assert reference.height_cm == pytest.approx(205.0)
    assert reference.percentile > 99.0
    # Measurements correlated with stature should extrapolate above the
    # tallest in-range percentile, not collapse to a percentile lookup.
    assert reference.values["acromialheight"] > baseline.values["acromialheight"]


# ---------------------------------------------------------------------------
# Constructing a User (requires the OpenSim bindings)
# ---------------------------------------------------------------------------


@requires_opensim
def test_default_percentile_is_used_when_only_gender_is_given():
    male = make_user(gender="M")
    female = make_user(gender="F")

    assert male.gender == "M"
    assert male.percentile == 50.0
    assert female.gender == "F"
    assert female.percentile == 50.0


@requires_opensim
def test_explicit_percentile_is_honored():
    user = make_user(gender="M", percentile=75.0)

    assert user.percentile == 75.0


@requires_opensim
def test_explicit_height_resolves_its_empirical_percentile():
    user = make_user(gender="M", height_cm=175.0)
    reference = resolve_reference("M", height=175.0, dataset=DATASET)

    assert user.percentile == pytest.approx(reference.percentile)
    assert user.height == pytest.approx(reference.height_cm)


@requires_opensim
def test_anthropometry_property_matches_the_resolved_reference():
    user = make_user(gender="F", percentile=75.0)

    assert user.anthropometry.gender == "F"
    assert user.anthropometry.percentile == 75.0
    assert user.anthropometry.height_cm == pytest.approx(user.height)


@requires_opensim
def test_gender_only_accepts_m_or_f():
    with pytest.raises(ValueError, match="gender"):
        make_user(gender="X")


@requires_opensim
def test_instances_are_backed_by_independent_opensim_models():
    user_50 = make_user(gender="M", percentile=50)
    user_75 = make_user(gender="M", percentile=75)

    assert user_50.model is not user_75.model


@requires_opensim
def test_copy_preserves_type_and_anthropometry_without_a_user_override():
    user = make_user(gender="F", percentile=75.0)
    user.set_left_knee_flexionextension(90.0)

    duplicate = user.copy()

    assert type(duplicate) is User
    assert duplicate.gender == "F"
    assert duplicate.percentile == 75.0
    assert duplicate.coordinate("knee_angle_l").value_degrees == pytest.approx(90.0)
    assert duplicate.model is not user.model


# ---------------------------------------------------------------------------
# Total body mass (mass_kg)
# ---------------------------------------------------------------------------


@requires_opensim
def test_default_mass_matches_the_resolved_ansur_weight_kg():
    user = make_user(gender="M", height_cm=180.0)
    reference = resolve_reference("M", height=180.0, dataset=DATASET)

    assert user.mass_kg == pytest.approx(reference.values["weight_kg"])
    assert user.model.getTotalMass(user.state) == pytest.approx(user.mass_kg)


@requires_opensim
def test_explicit_mass_kg_overrides_the_ansur_default():
    default_user = make_user(gender="M", height_cm=180.0)
    heavy_user = make_user(gender="M", height_cm=180.0, mass_kg=120.0)

    # The requested mass must actually differ from what height alone would
    # have produced, otherwise this test would not exercise the override.
    assert default_user.mass_kg != pytest.approx(120.0)
    assert heavy_user.mass_kg == pytest.approx(120.0)
    assert heavy_user.model.getTotalMass(heavy_user.state) == pytest.approx(120.0)


@requires_opensim
def test_mass_kg_does_not_affect_geometric_scaling():
    # mass_kg must only rescale mass/inertia (opensim.Body.scaleMass), never
    # the geometric, height-driven per-body scale factors.
    default_user = make_user(gender="M", height_cm=180.0)
    heavy_user = make_user(gender="M", height_cm=180.0, mass_kg=200.0)
    light_user = make_user(gender="M", height_cm=180.0, mass_kg=50.0)

    assert heavy_user.right_thigh_length == pytest.approx(default_user.right_thigh_length)
    assert light_user.right_thigh_length == pytest.approx(default_user.right_thigh_length)
    assert heavy_user.left_hip == pytest.approx(default_user.left_hip)
    assert light_user.left_hip == pytest.approx(default_user.left_hip)


@requires_opensim
def test_mass_kg_rescales_every_body_mass_by_the_same_factor():
    default_user = make_user(gender="M", height_cm=180.0)
    heavy_user = make_user(gender="M", height_cm=180.0, mass_kg=200.0)
    expected_factor = 200.0 / default_user.mass_kg

    for name in default_user.bodies:
        ratio = heavy_user.body(name).mass / default_user.body(name).mass
        assert ratio == pytest.approx(expected_factor)


@pytest.mark.parametrize("mass_kg", [0.0, -10.0, float("inf"), float("nan")])
@requires_opensim
def test_mass_kg_rejects_non_positive_or_non_finite_values(mass_kg):
    with pytest.raises(ValueError, match="mass_kg"):
        make_user(gender="M", mass_kg=mass_kg)


@requires_opensim
def test_copy_preserves_mass_kg():
    user = make_user(gender="M", height_cm=180.0, mass_kg=95.0)

    duplicate = user.copy()

    assert duplicate.mass_kg == pytest.approx(95.0)
    assert duplicate.model.getTotalMass(duplicate.state) == pytest.approx(95.0)


@requires_opensim
def test_set_mass_kg_recalibrates_an_already_built_user():
    user = make_user(gender="M", height_cm=180.0)

    user.set_mass_kg(130.0)

    assert user.mass_kg == pytest.approx(130.0)
    assert user.model.getTotalMass(user.state) == pytest.approx(130.0)


@requires_opensim
def test_set_mass_kg_is_idempotent_and_works_from_either_starting_point():
    default_user = make_user(gender="M", height_cm=180.0)
    explicit_user = make_user(gender="M", height_cm=180.0, mass_kg=200.0)

    default_user.set_mass_kg(75.0)
    explicit_user.set_mass_kg(75.0)

    assert default_user.mass_kg == pytest.approx(75.0)
    assert explicit_user.mass_kg == pytest.approx(75.0)
    assert default_user.model.getTotalMass(default_user.state) == pytest.approx(75.0)
    assert explicit_user.model.getTotalMass(explicit_user.state) == pytest.approx(75.0)


@requires_opensim
def test_set_mass_kg_preserves_posture():
    user = make_user(gender="M", height_cm=180.0)
    user.set_left_knee_flexionextension(33.0)
    user.update_state()

    user.set_mass_kg(80.0)

    assert user.left_knee_flexionextension == pytest.approx(33.0)
    assert user.mass_kg == pytest.approx(80.0)


@pytest.mark.parametrize("mass_kg", [0.0, -10.0, float("inf"), float("nan")])
@requires_opensim
def test_set_mass_kg_rejects_non_positive_or_non_finite_values(mass_kg):
    user = make_user(gender="M", height_cm=180.0)

    with pytest.raises(ValueError, match="mass_kg"):
        user.set_mass_kg(mass_kg)


# ---------------------------------------------------------------------------
# Anthropometric scaling of the bundled OpenSim model
# ---------------------------------------------------------------------------


@requires_opensim
def test_model_components_match_the_bundled_rajagopal_model():
    user = make_user(gender="M", percentile=75.0)

    assert len(user.bodies) == 22
    assert len(user.joints) == 22
    assert len(user.muscles) == 80
    assert len(user.markers) == 66
    assert len(user.coordinates) == 39


@requires_opensim
def test_scaling_moves_markers_away_from_the_unscaled_baseline():
    baseline_user = make_user(gender="M", percentile=50.0)
    scaled_user = make_user(gender="M", percentile=95.0)

    baseline = baseline_user.marker("RTOE").location
    scaled = scaled_user.marker("RTOE").location

    assert scaled != pytest.approx(baseline)


@requires_opensim
def test_originally_locked_coordinates_are_unlocked_for_full_configurability():
    user = make_user(gender="F")

    assert user.coordinate("subtalar_angle_l").locked is False
    assert user.coordinate("mtp_angle_r").locked is False
    assert user.coordinate("wrist_flex_r").locked is False


# ---------------------------------------------------------------------------
# Posture setters (exhaustive: every left/right/lumbar coordinate alias)
# ---------------------------------------------------------------------------


@requires_opensim
@pytest.mark.parametrize("setter_name, coordinate_name", POSTURE_SETTERS)
def test_every_posture_setter_writes_its_target_coordinate(setter_name, coordinate_name):
    user = make_user(gender="F")

    getattr(user, setter_name)(12.0)

    assert user.coordinate(coordinate_name).value_degrees == pytest.approx(12.0)


@requires_opensim
def test_posture_setters_do_not_cross_talk_between_sides():
    user = make_user(gender="M")

    user.set_left_knee_flexionextension(10.0)
    user.set_right_hip_flexionextension(25.0)

    assert user.coordinate("knee_angle_l").value_degrees == pytest.approx(10.0)
    assert user.coordinate("knee_angle_r").value_degrees == pytest.approx(0.0)
    assert user.coordinate("hip_flexion_r").value_degrees == pytest.approx(25.0)
    assert user.coordinate("hip_flexion_l").value_degrees == pytest.approx(0.0)


def test_posture_properties_mirror_every_posture_setter():
    expected = {
        setter_name[len("set_") :]: coordinate_name
        for setter_name, coordinate_name in POSTURE_SETTERS
    }

    assert _POSTURE_COORDINATE_NAMES == expected


@requires_opensim
@pytest.mark.parametrize("property_name, coordinate_name", list(_POSTURE_COORDINATE_NAMES.items()))
def test_posture_angle_property_reads_back_the_coordinate(property_name, coordinate_name):
    user = make_user(gender="F")

    user.coordinate(coordinate_name).set_value_degrees(12.0)

    assert getattr(user, property_name) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Joint centres and derived measurements
# ---------------------------------------------------------------------------


@requires_opensim
@pytest.mark.parametrize("property_name, joint_name", list(_JOINT_CENTER_NAMES.items()))
def test_joint_center_property_matches_the_opensim_joint_position(property_name, joint_name):
    user = make_user(gender="M")

    frame = user.joint(joint_name).raw.getChildFrame()
    expected = frame.getPositionInGround(user.state)

    assert getattr(user, property_name) == pytest.approx(
        (expected.get(0), expected.get(1), expected.get(2))
    )


@requires_opensim
def test_joint_centers_dict_matches_every_named_property():
    user = make_user(gender="M")

    centers = user.joint_centers

    assert set(centers) == set(_JOINT_CENTER_NAMES)
    for name in _JOINT_CENTER_NAMES:
        assert centers[name] == pytest.approx(getattr(user, name))


@requires_opensim
def test_joint_center_reflects_posture_without_calling_update_state():
    user = make_user(gender="M")
    before = user.right_knee

    user.set_right_knee_flexionextension(45.0)

    assert user.right_knee != pytest.approx(before)


@requires_opensim
@pytest.mark.parametrize("property_name, marker_name", list(_FOOT_MARKER_NAMES.items()))
def test_foot_marker_property_matches_the_opensim_marker_location(property_name, marker_name):
    user = make_user(gender="M")

    expected = user.marker(marker_name).raw.getLocationInGround(user.state)

    assert getattr(user, property_name) == pytest.approx(
        (expected.get(0), expected.get(1), expected.get(2))
    )


@requires_opensim
def test_foot_markers_dict_matches_every_named_property():
    user = make_user(gender="M")

    markers = user.foot_markers

    assert set(markers) == set(_FOOT_MARKER_NAMES)
    for name in _FOOT_MARKER_NAMES:
        assert markers[name] == pytest.approx(getattr(user, name))


@requires_opensim
def test_thigh_and_shank_length_match_joint_center_distances():
    user = make_user(gender="M")

    assert user.right_thigh_length == pytest.approx(math.dist(user.right_hip, user.right_knee))
    assert user.right_shank_length == pytest.approx(math.dist(user.right_knee, user.right_ankle))


@requires_opensim
def test_arm_and_forearm_length_match_joint_center_distances():
    user = make_user(gender="M")

    assert user.right_arm_length == pytest.approx(
        math.dist(user.right_shoulder, user.right_elbow)
    )
    assert user.right_forearm_length == pytest.approx(
        math.dist(user.right_elbow, user.right_wrist)
    )


@requires_opensim
def test_torso_height_and_shoulder_width_match_joint_center_geometry():
    user = make_user(gender="M")
    hip_center = tuple((a + b) / 2 for a, b in zip(user.left_hip, user.right_hip))
    shoulder_center = tuple((a + b) / 2 for a, b in zip(user.left_shoulder, user.right_shoulder))

    assert user.torso_height == pytest.approx(math.dist(hip_center, shoulder_center))
    assert user.shoulder_width == pytest.approx(math.dist(user.left_shoulder, user.right_shoulder))


@requires_opensim
def test_shoulder_width_and_biacromial_breadth_are_independent_measures():
    user = make_user(gender="M")

    assert user.shoulder_width != pytest.approx(user.biacromial_breadth)
    assert user.shoulder_width > 0
    assert user.biacromial_breadth > 0


@requires_opensim
def test_left_foot_height_is_the_ankle_joint_height():
    user = make_user(gender="M")

    assert user.left_foot_height == pytest.approx(user.left_ankle[1])
    assert user.right_foot_height == pytest.approx(user.right_ankle[1])


@requires_opensim
@pytest.mark.parametrize("property_name, ansur_column", ANSUR_DIRECT_PROPERTIES)
def test_ansur_direct_property_matches_resolved_measurement(property_name, ansur_column):
    user = make_user(gender="M")

    assert getattr(user, property_name) == pytest.approx(
        user.anthropometry.values[ansur_column] / 1000.0
    )


@requires_opensim
@pytest.mark.parametrize("property_name, circumference_column", CIRCULAR_SECTION_PROPERTIES)
def test_circular_section_property_derives_diameter_from_circumference(
    property_name, circumference_column
):
    user = make_user(gender="M")

    expected = user.anthropometry.values[circumference_column] / 1000.0 / math.pi

    assert getattr(user, property_name) == pytest.approx(expected)


@requires_opensim
def test_thigh_and_calf_width_equal_depth_under_the_circular_assumption():
    user = make_user(gender="M")

    assert user.left_thigh_width == pytest.approx(user.left_thigh_depth)
    assert user.right_thigh_width == pytest.approx(user.right_thigh_depth)
    assert user.left_calf_width == pytest.approx(user.left_calf_depth)
    assert user.right_calf_width == pytest.approx(user.right_calf_depth)


@requires_opensim
def test_com_is_finite_and_above_the_ground():
    user = make_user(gender="M")

    x, y, z = user.com

    assert all(math.isfinite(value) for value in (x, y, z))
    assert y > 0


@requires_opensim
def test_cop_is_the_vertical_projection_of_com_onto_the_ground():
    user = make_user(gender="M")

    com = user.com

    assert user.cop == pytest.approx((com[0], 0.0, com[2]))


@requires_opensim
def test_set_position_moves_the_reference_point_to_the_target():
    user = make_user(gender="M")

    user.set_position(user.cop, 1.0, 0.0, 0.5)

    assert user.cop == pytest.approx((1.0, 0.0, 0.5))


@requires_opensim
def test_set_position_is_a_rigid_translation_that_preserves_posture():
    user = make_user(gender="M")
    user.set_right_knee_flexionextension(30.0)
    angle_before = user.right_knee_flexionextension
    thigh_length_before = user.right_thigh_length

    user.set_position(user.com, 3.0, 0.0, -1.0)

    assert user.right_knee_flexionextension == pytest.approx(angle_before)
    assert user.right_thigh_length == pytest.approx(thigh_length_before)


@requires_opensim
def test_set_position_rejects_non_finite_target():
    user = make_user(gender="M")

    with pytest.raises(ValueError, match="finite"):
        user.set_position(user.com, float("nan"), 0.0, 0.0)


@requires_opensim
def test_out_of_range_height_user_builds_with_positive_derived_measurements():
    with pytest.warns(UserWarning, match="outside ANSUR range"):
        user = make_user(gender="F", height_cm=210.0)

    for name in (
        "left_thigh_circumference",
        "chest_depth",
        "chest_width",
        "left_arm_length",
        "right_shank_length",
        "torso_height",
    ):
        assert getattr(user, name) > 0


# ---------------------------------------------------------------------------
# Inherited OpenSimModel facade, exercised through User
# ---------------------------------------------------------------------------


@requires_opensim
def test_inherited_coordinate_locking_is_enforced():
    user = make_user(gender="M")

    user.coordinate("knee_angle_r").set_locked(True)
    with pytest.raises(ValueError, match="locked"):
        user.coordinate("knee_angle_r").set_value_degrees(10.0)


@requires_opensim
def test_inherited_marker_and_muscle_accessors_work():
    user = make_user(gender="M")

    user.marker("RTOE").set_location((0.1, 0.02, 0.03))
    user.muscle("addbrev_r").set_max_isometric_force(1500.0)

    assert user.marker("RTOE").location == pytest.approx((0.1, 0.02, 0.03))
    assert user.muscle("addbrev_r").max_isometric_force == pytest.approx(1500.0)


@requires_opensim
def test_export_writes_a_reloadable_osim_file_with_current_posture():
    user = make_user(gender="M", percentile=75.0)
    user.set_right_hip_flexionextension(25.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        destination = user.export(Path(tmp_dir) / "exported.osim")

        assert destination.is_file()
        reloaded = opensim.Model(str(destination))
        reloaded.initSystem()
        hip_flexion_r = reloaded.getCoordinateSet().get("hip_flexion_r")
        assert hip_flexion_r.get_default_value() == pytest.approx(np.deg2rad(25.0))


# ---------------------------------------------------------------------------
# Rendering assets and composition with other OpenSimModel instances
# ---------------------------------------------------------------------------


@requires_opensim
def test_user_registers_its_own_mesh_directory_for_show():
    user = make_user(gender="M")

    assert DEFAULT_MESHES_DIR in user.geometry_directories
    assert user.visualizer is None


@requires_opensim
def test_adding_two_users_degrades_to_a_generic_opensimmodel():
    user_a = make_user(gender="M")
    user_b = make_user(gender="F")

    combined = user_a + user_b

    assert type(combined) is OpenSimModel
    assert len(combined.bodies) == len(user_a.bodies) + len(user_b.bodies) == 44
