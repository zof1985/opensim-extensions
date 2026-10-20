import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import OpenSimModel, components, operators

# DEFAULT_MODEL_PATH is User's internal default, reused here as a real .osim
# fixture to test OpenSimModel's generic facade rather than User specifically.
from opensim_models.models.user.user import DEFAULT_MODEL_PATH

opensim = pytest.importorskip("opensim")

REAL_MODEL = DEFAULT_MODEL_PATH


def make_model(**kwargs):
    return OpenSimModel(model_path=REAL_MODEL, **kwargs)


def build_single_body_model(directory, body_name, joint_name):
    """Build a tiny synthetic one-body OpenSim model file for merge tests
    that must not depend on any specific bundled model."""
    raw = opensim.Model()
    body = opensim.Body(body_name, 1.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1))
    raw.addBody(body)
    joint = opensim.FreeJoint(joint_name, raw.getGround(), body)
    raw.addJoint(joint)
    raw.finalizeConnections()
    raw.initSystem()
    path = Path(directory) / f"{body_name}.osim"
    raw.printToXML(str(path))
    return OpenSimModel(model_path=path)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_blank_model_has_no_components():
    model = OpenSimModel(model_path=None)

    assert model.model_path is None
    assert len(model.bodies) == 0
    assert len(model.joints) == 0
    assert len(model.muscles) == 0
    assert len(model.markers) == 0
    assert len(model.coordinates) == 0
    assert len(model.frames) == 0
    assert model.state is not None


def test_model_loaded_from_file_exposes_expected_components():
    model = make_model()

    assert len(model.bodies) == 22
    assert len(model.joints) == 22
    assert len(model.muscles) == 80
    assert len(model.markers) == 66
    # The bundled base model has no standalone offset frame of its own --
    # every PhysicalOffsetFrame it contains is a joint's own parent/child
    # frame, already excluded by .frames (see test_operators.py's
    # test_model_frames_excludes_bodies_ground_and_a_joints_own_offset_frames
    # for why), even though getFrameList() itself is far from empty here.
    assert len(model.frames) == 0


def test_missing_model_file_raises():
    with pytest.raises(FileNotFoundError):
        OpenSimModel(model_path="does/not/exist.osim")


def test_locked_coordinates_are_unlocked_on_load():
    model = make_model()

    assert model.coordinate("subtalar_angle_l").locked is False
    assert model.coordinate("mtp_angle_r").locked is False
    assert model.coordinate("wrist_flex_r").locked is False


def test_instances_loaded_from_the_same_file_are_independent():
    first = make_model()
    second = make_model()

    assert first.model is not second.model
    first.coordinate("hip_flexion_r").set_value_degrees(25.0)
    assert second.coordinate("hip_flexion_r").value_degrees == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# copy()
# ---------------------------------------------------------------------------


def test_copy_returns_an_instance_of_the_same_type():
    model = make_model()

    assert type(model.copy()) is type(model)


def test_copy_is_backed_by_an_independent_opensim_model():
    model = make_model()

    duplicate = model.copy()

    assert duplicate.model is not model.model
    duplicate.coordinate("hip_flexion_r").set_value_degrees(25.0)
    assert model.coordinate("hip_flexion_r").value_degrees == pytest.approx(0.0)


def test_copy_preserves_posture_and_speed():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)
    model.coordinate("hip_flexion_r").set_speed_degrees(45.0)

    duplicate = model.copy()

    assert duplicate.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)
    assert duplicate.coordinate("hip_flexion_r").speed_degrees == pytest.approx(45.0)


def test_copy_preserves_a_coupled_dependent_coordinate():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)

    duplicate = model.copy()

    assert duplicate.coordinate("knee_angle_r_beta").value_degrees == pytest.approx(90.0)


def test_copy_carries_over_geometry_directories_independently():
    model = make_model()
    model.add_geometry_directory("some/dir")

    duplicate = model.copy()
    duplicate.add_geometry_directory("other/dir")

    assert Path("other/dir") not in model.geometry_directories


# ---------------------------------------------------------------------------
# update_state()
# ---------------------------------------------------------------------------


def test_coordinate_setters_do_not_auto_realize():
    model = make_model()
    tibia = model.body("tibia_r")
    origin = opensim.Vec3(0, 0, 0)
    tibia.raw.findStationLocationInGround(model.state, origin)  # baseline: works

    model.coordinate("knee_angle_r").set_value_degrees(90.0)

    # The raw value is set immediately (no realize needed to read it back)...
    assert model.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)
    # ...but nothing derived is recomputed: OpenSim's own cache-versioning
    # catches this and raises rather than silently returning a stale value.
    with pytest.raises(RuntimeError):
        tibia.raw.findStationLocationInGround(model.state, origin)

    model.update_state()
    tibia.raw.findStationLocationInGround(model.state, origin)  # works again


def test_update_state_realizes_position_and_velocity():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)
    model.coordinate("hip_flexion_r").set_speed_degrees(45.0)

    model.update_state()

    assert model.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)
    assert model.coordinate("hip_flexion_r").speed_degrees == pytest.approx(45.0)


def test_update_state_equilibrates_muscles_without_crashing():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)

    model.update_state()  # must not raise or crash the process

    assert len(model.muscles) > 0


# ---------------------------------------------------------------------------
# Named-component accessors
# ---------------------------------------------------------------------------


def test_named_component_accessors_return_matching_objects():
    model = make_model()

    assert model.body("pelvis").name == "pelvis"
    assert model.joint("hip_r").name == "hip_r"
    assert model.muscle("glmax1_r").name == "glmax1_r"
    assert model.marker("RASI").name == "RASI"
    assert model.coordinate("hip_flexion_r").name == "hip_flexion_r"


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def test_coordinate_degrees_round_trip_through_radians():
    model = make_model()

    model.coordinate("hip_flexion_r").set_value_degrees(25.0)

    assert model.coordinate("hip_flexion_r").value_degrees == pytest.approx(25.0)


def test_set_coordinate_degrees_rejects_non_finite_values():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.coordinate("hip_flexion_r").set_value_degrees(float("nan"))


def test_coordinate_locking_can_be_toggled_and_is_enforced():
    model = make_model()
    coordinate = model.coordinate("knee_angle_r")

    coordinate.set_locked(True)
    assert coordinate.locked is True
    with pytest.raises(ValueError, match="locked"):
        coordinate.set_value_degrees(10.0)

    coordinate.set_locked(False)
    coordinate.set_value_degrees(10.0)
    assert coordinate.value_degrees == pytest.approx(10.0)


def test_coordinate_range_can_be_read_and_updated():
    model = make_model()
    coordinate = model.coordinate("knee_angle_r")

    # Coordinate.set_range/.range are in the coordinate's native unit
    # (radians, for this rotational coordinate), not degrees -- convert at
    # the test boundary so the assertions can still be expressed in degrees.
    coordinate.set_range((np.radians(-100.0), np.radians(5.0)))

    assert tuple(np.degrees(coordinate.range)) == pytest.approx((-100.0, 5.0))
    with pytest.raises(ValueError, match="min must be less than max"):
        coordinate.set_range((np.radians(5.0), np.radians(-100.0)))


def test_coordinate_range_rejects_non_finite_bounds():
    model = make_model()
    coordinate = model.coordinate("knee_angle_r")

    with pytest.raises(ValueError, match="finite"):
        coordinate.set_range((float("nan"), np.radians(5.0)))


def test_coordinate_speed_degrees_round_trip_through_radians():
    model = make_model()

    model.coordinate("hip_flexion_r").set_speed_degrees(45.0)

    assert model.coordinate("hip_flexion_r").speed_degrees == pytest.approx(45.0)


def test_set_coordinate_speed_degrees_rejects_non_finite_values():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.coordinate("hip_flexion_r").set_speed_degrees(float("nan"))


def test_set_coordinate_speed_degrees_is_rejected_when_locked():
    model = make_model()
    coordinate = model.coordinate("knee_angle_r")
    coordinate.set_locked(True)

    with pytest.raises(ValueError, match="locked"):
        coordinate.set_speed_degrees(10.0)


# ---------------------------------------------------------------------------
# Body properties
# ---------------------------------------------------------------------------


def test_body_mass_can_be_read_and_updated():
    model = make_model()

    model.body("tibia_r").set_mass(5.0)

    assert model.body("tibia_r").mass == pytest.approx(5.0)


def test_set_body_mass_rejects_non_positive_values():
    model = make_model()
    tibia = model.body("tibia_r")

    with pytest.raises(ValueError, match="positive"):
        tibia.set_mass(0.0)
    with pytest.raises(ValueError, match="positive"):
        tibia.set_mass(-1.0)


def test_set_body_mass_preserves_posture():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)

    model.body("tibia_r").set_mass(5.0)

    assert model.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Reinitializing the system (structural property changes)
# ---------------------------------------------------------------------------


def test_reinitialize_preserves_posture_and_speed():
    model = make_model()
    model.coordinate("knee_angle_r").set_value_degrees(90.0)
    model.coordinate("hip_flexion_r").set_speed_degrees(45.0)

    model.reinitialize()

    assert model.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)
    assert model.coordinate("hip_flexion_r").speed_degrees == pytest.approx(45.0)


def test_reinitialize_updates_a_coupled_dependent_coordinate():
    model = make_model()
    # knee_angle_r_beta (the patella) is coupled to knee_angle_r via a
    # CoordinateCouplerConstraint in the bundled Rajagopal model.
    assert "knee_angle_r_beta" in model.coordinates

    model.coordinate("knee_angle_r").set_value_degrees(90.0)
    model.reinitialize()

    assert model.coordinate("knee_angle_r_beta").value_degrees == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def test_marker_location_can_be_read_and_updated():
    model = make_model()

    model.marker("RTOE").set_location((0.1, 0.02, 0.03))

    assert model.marker("RTOE").location == pytest.approx((0.1, 0.02, 0.03))


def test_marker_can_be_created_for_a_body_and_added_to_its_model():
    model = make_model()
    body = model.body("pelvis")

    assert "custom_pelvis_marker" not in model.markers
    added = model.add_marker(
        marker_name="custom_pelvis_marker",
        body=body,
        coordinates=(0.1, 0.2, 0.3),
        reinitialize=True,
    )

    assert added.name == "custom_pelvis_marker"
    stored = model.marker("custom_pelvis_marker")
    assert added == stored
    assert int(added.raw.this) == int(stored.raw.this)
    assert added.location == pytest.approx((0.1, 0.2, 0.3))
    assert stored.parents[0] == body
    assert model.body("pelvis").parents[-1].name == "custom_pelvis_marker"

    model.model.realizePosition(model.state)
    raw_position = (
        model.model.getMarkerSet()
        .get("custom_pelvis_marker")
        .getLocationInGround(model.state)
    )
    assert added.position_global == pytest.approx(
        tuple(raw_position.get(i) for i in range(3))
    )


def test_add_marker_requires_body_and_coordinates():
    model = make_model()
    body = model.body("pelvis")

    with pytest.raises(TypeError, match="coordinates are required"):
        model.add_marker("missing_coordinates", body)
    with pytest.raises(TypeError, match="body is required"):
        model.add_marker("missing_body", coordinates=(0.0, 0.0, 0.0))
    with pytest.raises(TypeError, match="body must be a body name"):
        model.add_marker("raw_body", body.raw, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        model.add_marker(
            "non_finite", body, (float("inf"), 0.0, 0.0)
        )


def test_add_marker_accepts_global_coordinates_and_converts_to_body_local():
    model = make_model()
    body = model.body("pelvis")
    global_point = (0.2, 1.1, -0.3)

    marker = model.add_marker(
        "global_pelvis_marker",
        body,
        global_point,
        coordinates_are_global=True,
        reinitialize=True,
    )

    assert marker.position_global == pytest.approx(global_point)
    assert marker.location != pytest.approx(global_point)


def test_marker_constructor_accepts_global_coordinates():
    model = make_model()
    body = model.body("pelvis")
    global_point = (0.2, 1.1, -0.3)

    marker = components.Marker(
        marker_name="constructed_global_marker",
        body=body,
        coordinates=global_point,
        coordinates_are_global=True,
    )

    expected_local = operators.from_global_to_local(global_point, body)
    assert marker.location == pytest.approx(expected_local)


def test_add_marker_accepts_body_name():
    model = make_model()

    marker = model.add_marker(
        "named_body_marker",
        "pelvis",
        (0.0, 0.0, 0.0),
        reinitialize=True,
    )

    assert marker.parents[0].name == "pelvis"


def test_body_attached_marker_cannot_be_added_to_a_different_model():
    model = make_model()
    marker = components.Marker(
        marker_name="custom_pelvis_marker",
        body=model.body("pelvis"),
        coordinates=(0.0, 0.0, 0.0),
    )
    other_model = OpenSimModel(model_path=None)

    with pytest.raises(ValueError, match="model that owns its body"):
        other_model.add_marker(marker)


def test_set_marker_location_rejects_non_finite_values():
    model = make_model()

    with pytest.raises(ValueError, match="finite"):
        model.marker("RTOE").set_location((float("inf"), 0.0, 0.0))


# ---------------------------------------------------------------------------
# Ground-frame / local positions (position_global / position_local)
# ---------------------------------------------------------------------------


def test_body_position_global_matches_raw_opensim_position_in_ground():
    model = make_model()
    tibia = model.body("tibia_r")

    model.model.realizePosition(model.state)
    raw_position = model.model.getBodySet().get("tibia_r").getPositionInGround(model.state)
    expected = (raw_position.get(0), raw_position.get(1), raw_position.get(2))

    assert tibia.position_global == pytest.approx(expected)


def test_body_position_local_is_always_the_origin():
    model = make_model()

    assert model.body("tibia_r").position_local == (0.0, 0.0, 0.0)


def test_body_position_global_reflects_a_posture_change():
    model = make_model()
    before = model.body("tibia_r").position_global

    model.coordinate("knee_angle_r").set_value_degrees(45.0)
    model.update_state()
    after = model.body("tibia_r").position_global

    assert after != pytest.approx(before)


def test_marker_position_global_matches_get_location_in_ground():
    model = make_model()
    marker = model.marker("RTOE")

    model.model.realizePosition(model.state)
    raw_location = model.model.getMarkerSet().get("RTOE").getLocationInGround(model.state)
    expected = (raw_location.get(0), raw_location.get(1), raw_location.get(2))

    assert marker.position_global == pytest.approx(expected)


def test_marker_position_local_matches_location():
    model = make_model()
    marker = model.marker("RTOE")
    marker.set_location((0.1, 0.02, 0.03))

    assert marker.position_local == marker.location == pytest.approx((0.1, 0.02, 0.03))


def test_joint_position_global_matches_child_frame_position_in_ground():
    model = make_model()
    joint = model.joint("hip_r")

    model.model.realizePosition(model.state)
    raw_child = model.model.getJointSet().get("hip_r").getChildFrame()
    raw_position = raw_child.getPositionInGround(model.state)
    expected = (raw_position.get(0), raw_position.get(1), raw_position.get(2))

    assert joint.position_global == pytest.approx(expected)


def test_joint_position_local_matches_child_offset_frame_translation():
    model = make_model()
    joint = model.joint("hip_r")

    raw_child = model.model.getJointSet().get("hip_r").getChildFrame()
    offset = opensim.PhysicalOffsetFrame.safeDownCast(raw_child)
    assert offset is not None  # confirmed for every joint in the bundled model
    translation = offset.get_translation()
    expected = (translation.get(0), translation.get(1), translation.get(2))

    assert joint.position_local == pytest.approx(expected)


def test_joint_child_frame_is_an_offset_frame_matching_position_local():
    from opensim_models import components

    model = make_model()
    joint = model.joint("hip_r")

    child_frame = joint.child_frame

    assert isinstance(child_frame, components.OffsetFrame)
    assert child_frame.translation == pytest.approx(joint.position_local)
    assert child_frame.position_local == pytest.approx(joint.position_local)
    assert child_frame.position_global == pytest.approx(joint.position_global)


def test_joint_parent_frame_is_an_offset_frame():
    from opensim_models import components

    model = make_model()
    joint = model.joint("hip_r")

    parent_frame = joint.parent_frame

    assert isinstance(parent_frame, components.OffsetFrame)
    # hip_r's parent frame is on the pelvis, not at the ground origin.
    assert parent_frame.position_global != pytest.approx((0.0, 0.0, 0.0))


def test_offset_frame_translation_can_be_read_and_updated():
    from opensim_models import components

    model = make_model()
    offset_frame = model.joint("hip_r").child_frame

    offset_frame.set_translation((0.01, 0.02, 0.03))

    assert offset_frame.translation == pytest.approx((0.01, 0.02, 0.03))
    assert offset_frame.position_local == pytest.approx((0.01, 0.02, 0.03))


def test_offset_frame_set_translation_rejects_non_finite_values():
    model = make_model()
    offset_frame = model.joint("hip_r").child_frame

    with pytest.raises(ValueError, match="finite"):
        offset_frame.set_translation((float("inf"), 0.0, 0.0))


def test_offset_frame_orientation_deg_can_be_read_and_updated():
    model = make_model()
    offset_frame = model.joint("hip_r").child_frame

    offset_frame.set_orientation_deg((10.0, 20.0, 30.0))

    assert offset_frame.orientation_deg == pytest.approx((10.0, 20.0, 30.0))


def test_offset_frame_set_orientation_deg_rejects_non_finite_values():
    model = make_model()
    offset_frame = model.joint("hip_r").child_frame

    with pytest.raises(ValueError, match="finite"):
        offset_frame.set_orientation_deg((float("nan"), 0.0, 0.0))


def test_offset_frame_parents_resolves_to_the_base_body():
    from opensim_models import components

    model = make_model()
    offset_frame = model.joint("hip_r").child_frame  # on the femur

    parents = offset_frame.parents

    assert len(parents) == 1
    assert isinstance(parents[0], components.Body)
    assert parents[0].name == "femur_r"


# ---------------------------------------------------------------------------
# parents: forward relations (Marker, Joint) and the backward one (Body)
# ---------------------------------------------------------------------------


def test_marker_parents_is_the_body_it_is_attached_to():
    from opensim_models import components

    model = make_model()
    marker = model.marker("RASI")  # a pelvis landmark

    parents = marker.parents

    assert len(parents) == 1
    assert isinstance(parents[0], components.Body)
    assert parents[0].name == "pelvis"


def test_joint_parents_is_the_parent_and_child_body():
    from opensim_models import components

    model = make_model()
    joint = model.joint("hip_r")

    parents = joint.parents

    assert len(parents) == 2
    assert all(isinstance(p, components.Body) for p in parents)
    assert [p.name for p in parents] == ["pelvis", "femur_r"]


def test_body_parents_finds_every_referencing_joint_muscle_and_marker():
    from opensim_models import components

    model = make_model()
    femur = model.body("femur_r")

    parents = femur.parents

    names_by_type = {}
    for parent in parents:
        names_by_type.setdefault(type(parent), set()).add(parent.name)

    # femur_r is the child body of these three joints in the bundled model.
    assert {"hip_r", "walker_knee_r", "patellofemoral_r"} <= names_by_type[components.Joint]
    # A handful of muscles known to cross the femur.
    assert {"glmax1_r", "iliacus_r", "psoas_r", "vasint_r"} <= names_by_type[components.Muscle]
    # A handful of femur-attached markers.
    assert "RHJC" in names_by_type[components.Marker]
    # No non-muscle Force or coordinate-only Constraint in the bundled
    # model references any body directly -- confirms nothing beyond the
    # three scanned categories leaked in.
    assert components.Constraint not in names_by_type


def test_body_parents_is_empty_for_a_body_with_no_joint_muscle_or_marker():
    # add_body alone (no add_joint call) still gets an automatic FreeJoint
    # from OpenSim itself at initSystem() -- so this exercises "truly
    # unreferenced" by adding a second, deliberately disconnected body
    # alongside one connected normally, and checking only the first.
    from opensim_models import components, operators

    model = OpenSimModel(model_path=None)
    with model.structural_change():
        operators.add_body(model, "connected", mass=1.0)
        operators.add_body(model, "disconnected", mass=1.0)
        operators.add_joint(
            model, opensim.FreeJoint("connected_to_ground", model.model.getGround(), model.body("connected").raw)
        )

    # Both bodies get OpenSim's own automatic FreeJoint fallback at
    # initSystem() when they have none explicitly -- so "disconnected"
    # still ends up with one joint (its own auto-added FreeJoint), but
    # zero muscles/markers/constraints reference it.
    parents = model.body("disconnected").parents
    assert all(not isinstance(p, components.Muscle) for p in parents)
    assert all(not isinstance(p, components.Marker) for p in parents)


# ---------------------------------------------------------------------------
# Muscles
# ---------------------------------------------------------------------------


def test_muscle_mechanical_parameters_can_be_read_and_updated():
    model = make_model()
    muscle = model.muscle("addbrev_r")

    muscle.set_max_isometric_force(1500.0)
    muscle.set_optimal_fiber_length(0.15)
    muscle.set_tendon_slack_length(0.1)
    muscle.set_pennation_angle(10.0)

    assert muscle.max_isometric_force == pytest.approx(1500.0)
    assert muscle.optimal_fiber_length == pytest.approx(0.15)
    assert muscle.tendon_slack_length == pytest.approx(0.1)
    assert muscle.pennation_angle == pytest.approx(10.0)


@pytest.mark.parametrize(
    "setter, value",
    [
        ("set_max_isometric_force", 0.0),
        ("set_optimal_fiber_length", -0.1),
        ("set_tendon_slack_length", float("nan")),
    ],
)
def test_muscle_setters_reject_non_positive_or_non_finite_values(setter, value):
    model = make_model()
    muscle = model.muscle("addbrev_r")

    with pytest.raises(ValueError):
        getattr(muscle, setter)(value)


def test_muscle_pennation_angle_must_be_within_valid_range():
    model = make_model()

    with pytest.raises(ValueError, match="pennation angle"):
        model.muscle("addbrev_r").set_pennation_angle(90.0)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_scale_bodies_applies_positive_factors():
    model = make_model()
    baseline = model.marker("RTOE").location

    model.scale_bodies({"calcn_r": (1.2, 1.2, 1.2), "toes_r": (1.2, 1.2, 1.2)})

    assert model.marker("RTOE").location != pytest.approx(baseline)


def test_scale_bodies_ignores_unknown_body_names():
    model = make_model()

    model.scale_bodies({"not_a_real_body": (1.5, 1.5, 1.5)})

    assert len(model.bodies) == 22


def test_scale_bodies_rejects_non_positive_factors():
    model = make_model()

    with pytest.raises(ValueError):
        model.scale_bodies({"pelvis": (1.0, -1.0, 1.0)})


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_writes_a_reloadable_osim_file_with_current_posture():
    model = make_model()
    model.coordinate("hip_flexion_r").set_value_degrees(25.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        destination = model.export(Path(tmp_dir) / "exported.osim")

        assert destination.is_file()
        reloaded = opensim.Model(str(destination))
        reloaded.initSystem()
        hip_flexion_r = reloaded.getCoordinateSet().get("hip_flexion_r")
        assert hip_flexion_r.get_default_value() == pytest.approx(np.deg2rad(25.0))


def test_export_copies_referenced_meshes_into_a_geometry_folder():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        geometry_dir = tmp_path / "assets"
        geometry_dir.mkdir()
        mesh_file = geometry_dir / "part.stl"
        mesh_file.write_bytes(b"solid fake\nendsolid fake\n")

        model = build_single_body_model(tmp_path, "part_body", "part_joint")
        model.body("part_body").raw.attachGeometry(opensim.Mesh("part.stl"))
        model.add_geometry_directory(geometry_dir)

        export_dir = tmp_path / "exported"
        export_dir.mkdir()
        destination = model.export(export_dir / "model.osim")

        exported_mesh = destination.parent / "Geometry" / "part.stl"
        assert exported_mesh.is_file()
        assert exported_mesh.read_bytes() == mesh_file.read_bytes()


def test_export_without_referenced_meshes_creates_no_geometry_folder():
    model = make_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        destination = model.export(Path(tmp_dir) / "exported.osim")

        assert not (destination.parent / "Geometry").exists()


# ---------------------------------------------------------------------------
# Geometry / visualizer
# ---------------------------------------------------------------------------


def test_geometry_directories_start_empty_and_accept_registrations():
    model = OpenSimModel(model_path=None)

    assert model.geometry_directories == ()
    assert model.visualizer is None

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.add_geometry_directory(tmp_dir)

        assert model.geometry_directories == (Path(tmp_dir),)


# ---------------------------------------------------------------------------
# Composing models: add_model / remove_model / __add__ / __radd__
# ---------------------------------------------------------------------------


def test_add_model_renames_colliding_components_with_a_prefix():
    base = make_model()
    extra = make_model()  # loaded from the same file: every name collides

    base.add_model(extra, name="extra")

    assert len(base.bodies) == 44
    assert len(base.joints) == 44
    assert len(base.muscles) == 80 * 2
    assert len(base.markers) == 66 * 2
    assert "extra_pelvis" in base.bodies
    # the original operand is left untouched
    assert len(extra.bodies) == 22
    assert "extra_pelvis" not in extra.bodies


def test_add_model_default_prefix_is_the_operand_class_name():
    base = make_model()
    extra = make_model()

    base.add_model(extra)

    assert "opensimmodel_pelvis" in base.bodies


def test_add_model_does_not_rename_non_colliding_components():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        extra = build_single_body_model(tmp_dir, "gadget", "gadget_to_ground")

        base.add_model(extra, name="extra")

        assert len(base.bodies) == 2
        assert "widget" in base.bodies
        assert "gadget" in base.bodies  # kept as-is, no collision


def test_add_model_merged_model_initializes_and_exports():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        extra = build_single_body_model(tmp_dir, "widget", "widget_to_ground")

        base.add_model(extra, name="extra")
        destination = Path(tmp_dir) / "combined.osim"
        base.export(destination)

        reloaded = opensim.Model(str(destination))
        reloaded.initSystem()
        assert reloaded.getBodySet().getSize() == 2


def test_add_model_preserves_the_merged_operands_posture():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = OpenSimModel(model_path=None)
        extra = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        # Just need *some* coordinate name; .coordinates is a dict (no
        # positional index), so go straight to the raw CoordinateSet for
        # this one arbitrary, order-dependent lookup.
        coordinate_name = extra.model.getCoordinateSet().get(0).getName()
        extra.coordinate(coordinate_name).set_value_degrees(30.0)

        base.add_model(extra)

        assert base.coordinate(coordinate_name).value_degrees == pytest.approx(30.0)


def test_add_model_carries_over_geometry_directories():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = OpenSimModel(model_path=None)
        extra = build_single_body_model(tmp_dir, "widget", "widget_to_ground")
        extra.add_geometry_directory(tmp_dir)

        base.add_model(extra)

        assert Path(tmp_dir) in base.geometry_directories


def test_add_model_then_remove_model_restores_original_state():
    base = OpenSimModel(model_path=None)
    extra = make_model()

    base.add_model(extra)
    base.remove_model(extra)

    assert len(base.bodies) == 0
    assert len(base.joints) == 0
    assert len(base.muscles) == 0
    assert len(base.markers) == 0


def test_remove_model_preserves_the_remaining_posture():
    base = make_model()
    extra = OpenSimModel(model_path=None)
    base.coordinate("knee_angle_r").set_value_degrees(90.0)

    base.add_model(extra)
    base.remove_model(extra)

    assert base.coordinate("knee_angle_r").value_degrees == pytest.approx(90.0)


def test_remove_model_rejects_a_model_that_was_never_added():
    base = OpenSimModel(model_path=None)
    stranger = make_model()

    with pytest.raises(ValueError):
        base.remove_model(stranger)


@pytest.mark.parametrize(
    "operation",
    [
        lambda model: model.add_model("not a model"),
        lambda model: model.remove_model("not a model"),
        lambda model: model + "not a model",
        lambda model: "not a model" + model,
    ],
)
def test_merge_operations_reject_non_model_operands(operation):
    model = OpenSimModel(model_path=None)

    with pytest.raises(TypeError):
        operation(model)


# `+` always merges both operands into one new, generic OpenSimModel (via
# add_model(), applied twice to a fresh OpenSimModel) -- neither operand is
# mutated, and the result is never a subclass of either operand.


def test_add_operator_returns_a_new_merged_model_without_mutating_operands():
    first = make_model()
    second = make_model()

    combined = first + second

    assert type(combined) is OpenSimModel
    assert len(combined.bodies) == len(first.bodies) + len(second.bodies)
    assert len(first.bodies) == 22
    assert len(second.bodies) == 22


def test_radd_delegates_to_the_left_operand():
    first = make_model()
    second = make_model()

    # first.__radd__(second) supports `second + first` when second doesn't
    # implement __add__; the result merges second's components first,
    # matching `second + first`'s left-to-right order.
    combined = first.__radd__(second)

    assert type(combined) is OpenSimModel
    assert len(combined.bodies) == len(first.bodies) + len(second.bodies)


def test_chained_add_merges_three_models():
    first = make_model()
    second = make_model()
    third = make_model()

    combined = first + second + third

    assert type(combined) is OpenSimModel
    assert len(combined.bodies) == 22 * 3


# ---------------------------------------------------------------------------
# rotate / translate
# ---------------------------------------------------------------------------


def build_one_body_model(position=(1.0, 0.0, 0.0)):
    model = OpenSimModel(model_path=None)
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=position)
    return model, body


def test_rotate_delegates_to_rotate_object():
    model, body = build_one_body_model(position=(1.0, 0.0, 0.0))

    new_position = model.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )


def test_translate_delegates_to_translate_object():
    model, body = build_one_body_model(position=(1.0, 0.0, 0.0))

    new_position = model.translate((0.0, 2.0, 0.0))

    assert new_position == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 2.0, 0.0), abs=1e-9
    )


def test_rotate_not_inplace_returns_a_rotated_copy_and_leaves_self_untouched():
    model, body = build_one_body_model(position=(1.0, 0.0, 0.0))

    rotated_copy = model.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False)

    assert isinstance(rotated_copy, OpenSimModel)
    assert rotated_copy is not model
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    copy_body_position = rotated_copy.body("b1").raw.getPositionInGround(rotated_copy.state)
    assert tuple(copy_body_position.to_numpy()) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_translate_not_inplace_returns_a_translated_copy_and_leaves_self_untouched():
    model, body = build_one_body_model(position=(1.0, 0.0, 0.0))

    translated_copy = model.translate((0.0, 2.0, 0.0), inplace=False)

    assert isinstance(translated_copy, OpenSimModel)
    assert translated_copy is not model
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    copy_body_position = translated_copy.body("b1").raw.getPositionInGround(translated_copy.state)
    assert tuple(copy_body_position.to_numpy()) == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Contact geometry convenience methods (delegation to operators)
# ---------------------------------------------------------------------------


def test_add_contact_sphere_delegates_to_operators():
    from opensim_models import components

    model, body = build_one_body_model(position=(0.0, 0.0, 0.0))

    sphere = model.add_contact_sphere("cs1", body, 0.05, location=(0.1, 0.2, 0.3), reinitialize=True)

    assert isinstance(sphere, components.ContactSphere)
    assert sphere.radius == pytest.approx(0.05)
    assert sphere.location == pytest.approx((0.1, 0.2, 0.3))
    assert type(model.contact_geometries["cs1"]) is components.ContactSphere


def test_add_contact_half_space_delegates_to_operators():
    from opensim_models import components

    model, body = build_one_body_model(position=(0.0, 0.0, 0.0))

    half_space = model.add_contact_half_space(
        "chs1", body, orientation_deg=(0.0, 0.0, 90.0), reinitialize=True
    )

    assert isinstance(half_space, components.ContactHalfSpace)
    assert half_space.orientation_deg == pytest.approx((0.0, 0.0, 90.0))
    assert type(model.contact_geometries["chs1"]) is components.ContactHalfSpace


def test_add_offset_frame_delegates_to_operators():
    from opensim_models import components

    model, body = build_one_body_model(position=(1.0, 2.0, 3.0))

    frame = model.add_offset_frame("f1", body, translation=(0.1, 0.2, 0.3), reinitialize=True)

    assert isinstance(frame, components.OffsetFrame)
    assert frame.position_local == pytest.approx((0.1, 0.2, 0.3))
    assert frame.position_global == pytest.approx((1.1, 2.2, 3.3))
    assert frame.parents == (model.body("b1"),)
