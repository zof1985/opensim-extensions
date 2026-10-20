import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import Box, OpenSimModel, components, operators

opensim = pytest.importorskip("opensim")


def make_model():
    return OpenSimModel(model_path=None)


def add_free_body(model, name):
    """Add a body with a FreeJoint to ground, as a single structural change."""
    with model.structural_change():
        body = operators.add_body(model, name, mass=2.0, inertia=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
        operators.add_joint(
            model,
            opensim.FreeJoint(f"{name}_to_ground", model.model.getGround(), body.raw),
        )
    return body


# ---------------------------------------------------------------------------
# Generic add_component/remove_component
# ---------------------------------------------------------------------------


def test_add_component_rejects_unknown_kind():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(0, 0, 0))

    with pytest.raises(ValueError, match="Unknown component kind"):
        operators.add_component(model, "not_a_kind", marker)


def test_remove_component_rejects_unknown_name():
    model = make_model()

    with pytest.raises(ValueError, match="No marker named"):
        operators.remove_component(model, "marker", "does_not_exist")


# ---------------------------------------------------------------------------
# Bodies and joints
# ---------------------------------------------------------------------------


def test_add_body_and_joint_as_a_batch():
    model = make_model()

    body = add_free_body(model, "b1")

    assert len(model.bodies) == 1
    assert len(model.joints) == 1
    assert model.body("b1").mass == pytest.approx(2.0)
    assert body.name == "b1"


def test_remove_body_and_joint_as_a_batch():
    model = make_model()
    add_free_body(model, "b1")

    with model.structural_change():
        operators.remove_joint(model, "b1_to_ground")
        operators.remove_body(model, "b1")

    assert len(model.bodies) == 0
    assert len(model.joints) == 0


def test_structural_change_preserves_posture_of_surviving_coordinates():
    model = make_model()
    add_free_body(model, "b1")
    add_free_body(model, "b2")
    coordinate_name = next(
        name for name in model.coordinates if name.startswith("b1_to_ground")
    )
    model.coordinate(coordinate_name).set_value_degrees(30.0)

    with model.structural_change():
        operators.remove_joint(model, "b2_to_ground")
        operators.remove_body(model, "b2")

    assert model.coordinate(coordinate_name).value_degrees == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# attach_component
# ---------------------------------------------------------------------------


def test_attach_component_reattaches_to_a_new_parent_at_com():
    model = make_model()
    parent = add_free_body(model, "parent")
    child = add_free_body(model, "child")

    joint = operators.attach_component(
        model, child.raw, to=parent.raw, reinitialize=True,
    )

    assert joint.name == "child_to_parent"
    assert len(model.joints) == 2
    assert "child_to_ground" not in model.joints
    assert "parent" in joint.raw.getParentFrame().getName()


def test_attach_component_tilts_a_slider_axis_with_parent_orientation_deg():
    model = make_model()
    parent = add_free_body(model, "parent")
    child = add_free_body(model, "child")

    joint = operators.attach_component(
        model, child.raw, to=parent.raw,
        child_point=(0.0, 0.0, 0.0), parent_point=(0.0, 0.0, 0.0),
        parent_orientation_deg=(0.0, 0.0, 45.0),
        joint_type="slider", reinitialize=True,
    )

    coordinate = next(iter(joint.coordinates.values()))
    coordinate.set_value(1.0, enforce_constraints=False)
    model.update_state()
    model.model.realizePosition(model.state)

    position = child.raw.getPositionInGround(model.state)
    assert (position.get(0), position.get(1)) == pytest.approx(
        (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)))
    )


def test_attach_component_resolves_explicit_points():
    model = make_model()
    parent = add_free_body(model, "parent")
    child = add_free_body(model, "child")

    operators.attach_component(
        model, child.raw, to=parent.raw,
        child_point=(0.0, 0.0, 0.0), parent_point=(0.0, 0.5, 0.0),
        reinitialize=True,
    )

    model.model.realizePosition(model.state)
    position = child.raw.getPositionInGround(model.state)
    assert (position.get(0), position.get(1), position.get(2)) == pytest.approx((0.0, 0.5, 0.0))


def test_attach_component_supports_a_standalone_component_once_merged():
    model = make_model()
    parent = add_free_body(model, "parent")
    box = Box(0.1, 0.1, 0.1, origin=(9.0, 9.0, 9.0))

    merged = model + box
    merged_parent = merged.body("parent")
    merged_box = merged.body("box")

    merged.attach_component(merged_box, to=merged_parent, reinitialize=True)

    merged.model.realizePosition(merged.state)
    box_position = merged_box.raw.getPositionInGround(merged.state)
    parent_position = merged_parent.raw.getPositionInGround(merged.state)
    assert (box_position.get(0), box_position.get(1), box_position.get(2)) == pytest.approx(
        (parent_position.get(0), parent_position.get(1), parent_position.get(2))
    )


def test_attach_component_rejects_a_component_not_yet_in_the_model():
    model = make_model()
    parent = add_free_body(model, "parent")
    lone_box = Box(0.1, 0.1, 0.1)

    with pytest.raises(ValueError, match="not connected by any joint"):
        operators.attach_component(model, lone_box.raw, to=parent.raw)


def test_attach_component_rejects_an_unknown_joint_type():
    model = make_model()
    parent = add_free_body(model, "parent")
    child = add_free_body(model, "child")

    with pytest.raises(ValueError, match="Unknown joint_type"):
        operators.attach_component(model, child.raw, to=parent.raw, joint_type="bogus")


def test_attach_component_accepts_an_offset_frame_as_to_with_default_com():
    # Regression for the _resolve_attachment_point fix: "com" used to
    # assume `to` was an opensim.Body (get_mass_center()); an OffsetFrame
    # has no mass at all, so "com" must fall back to its own origin
    # instead of raising AttributeError.
    model = make_model()
    body = add_welded_body(model, "b1", (1.0, 2.0, 3.0))
    frame = operators.add_offset_frame(model, "f1", body, translation=(0.1, 0.2, 0.3), reinitialize=True)
    child = add_free_body(model, "child")

    joint = operators.attach_component(model, child.raw, to=frame.raw, joint_type="weld", reinitialize=True)

    assert isinstance(joint, components.Joint)
    assert model.body("child").position_global == pytest.approx(frame.position_global)


# ---------------------------------------------------------------------------
# Offset frames
# ---------------------------------------------------------------------------


def test_add_offset_frame_builds_and_attaches_an_offset_frame():
    model = make_model()
    body = add_welded_body(model, "b1", (1.0, 2.0, 3.0))

    frame = operators.add_offset_frame(
        model, "f1", body, translation=(0.1, 0.2, 0.3), reinitialize=True
    )

    assert isinstance(frame, components.OffsetFrame)
    assert frame.position_local == pytest.approx((0.1, 0.2, 0.3))
    assert frame.position_global == pytest.approx((1.1, 2.2, 3.3))
    assert frame.parents == (model.body("b1"),)
    assert frame.raw.getAbsolutePathString() == "/bodyset/b1/f1"


def test_add_offset_frame_applies_orientation_deg():
    model = make_model()
    body = add_welded_body(model, "b1", (0.0, 0.0, 0.0))

    frame = operators.add_offset_frame(
        model, "f1", body, orientation_deg=(10.0, 20.0, 30.0), reinitialize=True
    )

    assert frame.orientation_deg == pytest.approx((10.0, 20.0, 30.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"translation": (float("nan"), 0.0, 0.0)},
        {"orientation_deg": (float("inf"), 0.0, 0.0)},
    ],
)
def test_add_offset_frame_rejects_non_finite_values(kwargs):
    model = make_model()
    body = add_welded_body(model, "b1", (0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="finite"):
        operators.add_offset_frame(model, "bad", body, **kwargs)


def test_add_offset_frame_is_usable_as_a_body_downstream():
    # Confirmed in the task this was added for: a PhysicalOffsetFrame is a
    # genuine opensim.PhysicalFrame, so it works as `body=` to add_marker/
    # add_contact_sphere, not just as a read-only landmark.
    model = make_model()
    body = add_welded_body(model, "b1", (1.0, 2.0, 3.0))
    frame = operators.add_offset_frame(model, "f1", body, translation=(0.1, 0.2, 0.3), reinitialize=True)

    marker = operators.add_marker(
        model, opensim.Marker("m1", frame.raw, opensim.Vec3(0, 0, 0)), reinitialize=True
    )
    sphere = operators.add_contact_sphere(model, "cs1", frame, 0.05, reinitialize=True)

    assert marker.position_global == pytest.approx(frame.position_global)
    assert sphere.position_global == pytest.approx(frame.position_global)
    assert sphere.parents == (model.body("b1"),)


def test_add_offset_frame_survives_add_model_merge():
    # The key reason this is attached via body.addComponent(frame) rather
    # than at the model's own root (unlike the internal, unnamed "ground
    # anchor" add_model builds for its own bookkeeping): a root-level
    # component falls outside every _MERGE_SETS category and is silently
    # left behind when merging into another model, but a body's own
    # subcomponent is cloned right along with it.
    model = make_model()
    body = add_welded_body(model, "b1", (1.0, 2.0, 3.0))
    frame = operators.add_offset_frame(model, "f1", body, translation=(0.1, 0.2, 0.3), reinitialize=True)

    other = make_model()
    combined = model + other

    found = combined.model.getComponent("/bodyset/b1/f1")
    combined_frame = components.OffsetFrame(combined, found)
    assert combined_frame.position_global == pytest.approx(frame.position_global)


def test_model_frames_dispatches_a_standalone_offset_frame():
    model = make_model()
    body = add_welded_body(model, "b1", (1.0, 2.0, 3.0))
    frame = operators.add_offset_frame(
        model, "f1", body, translation=(0.1, 0.2, 0.3), reinitialize=True
    )

    assert type(model.frames["f1"]) is components.OffsetFrame
    assert model.frames["f1"].position_global == pytest.approx(frame.position_global)
    assert len(model.frames) == 1


def test_model_frames_excludes_bodies_ground_and_a_joints_own_offset_frames():
    # getFrameList() returns every Frame-typed component in the model,
    # which includes Ground, every Body, and the two PhysicalOffsetFrames
    # every joint owns for its own parent/child attachment (already
    # reachable via Joint.parent_frame/child_frame) -- none of that is a
    # *standalone* frame, so .frames must stay empty here even though the
    # raw getFrameList() for this model is not.
    model = make_model()
    add_free_body(model, "b1")

    assert len(list(model.model.getFrameList())) > 0
    assert model.frames == {}


def test_model_frames_excludes_a_weld_constraints_own_attachment_frames():
    # A WeldConstraint builds two more PhysicalOffsetFrames of its own
    # (constraint.frame1/frame2) -- also not a standalone frame.
    model = make_model()
    body1 = add_welded_body(model, "b1", (0, 0, 0))
    body2 = add_welded_body(model, "b2", (0, 0, 0))

    operators.add_weld_constraint(model, "weld1", body1, body2, reinitialize=True)

    assert model.frames == {}


def test_model_frames_is_rebuilt_fresh_and_not_cached():
    model = make_model()
    body = add_welded_body(model, "b1", (0.0, 0.0, 0.0))

    assert model.frames == {}

    operators.add_offset_frame(model, "f1", body, reinitialize=True)

    assert list(model.frames) == ["f1"]


# ---------------------------------------------------------------------------
# Forces and muscles
# ---------------------------------------------------------------------------


def test_add_muscle_and_remove_muscle():
    model = make_model()
    body = add_free_body(model, "b1")

    operators.add_muscle(
        model, "mus1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        max_isometric_force=500.0, reinitialize=True,
    )
    assert len(model.muscles) == 1

    operators.remove_muscle(model, "mus1", reinitialize=True)
    assert len(model.muscles) == 0


def test_add_muscle_is_visible_through_the_force_set():
    model = make_model()
    body = add_free_body(model, "b1")

    operators.add_muscle(
        model, "mus1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        max_isometric_force=500.0, reinitialize=True,
    )

    assert model.model.getForceSet().getIndex("mus1") >= 0


def test_add_muscle_with_via_points_creates_a_longer_path():
    model = make_model()
    body = add_free_body(model, "b1")

    muscle = operators.add_muscle(
        model, "mus1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        max_isometric_force=500.0, via_points=[(model.model.getGround(), (0.05, 0, 0))],
        reinitialize=True,
    )

    assert muscle.raw.getGeometryPath().getCurrentPath(model.state).getSize() == 3


# ---------------------------------------------------------------------------
# Markers, constraints
# ---------------------------------------------------------------------------


def test_add_marker_and_remove_marker():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(0, 0, 0))

    operators.add_marker(model, marker, reinitialize=True)
    assert len(model.markers) == 1

    operators.remove_marker(model, "mk1", reinitialize=True)
    assert len(model.markers) == 0


def test_add_marker_constructs_and_attaches_marker_from_coordinates():
    model = OpenSimModel(model_path=None)
    body = operators.add_body(
        model, "marker_parent", mass=1.0, inertia=(1, 1, 1, 0, 0, 0)
    )
    operators.add_joint(
        model,
        opensim.FreeJoint(
            "marker_parent_to_ground", model.model.getGround(), body.raw
        ),
    )
    model.reinitialize()

    marker = operators.add_marker(
        model,
        marker_name="created_by_operator",
        body=body,
        coordinates=(1.0, 2.0, 3.0),
        coordinates_are_global=True,
        reinitialize=True,
    )

    assert marker.name == "created_by_operator"
    assert marker.position_global == pytest.approx((1.0, 2.0, 3.0))
    assert marker.parents[0].name == "marker_parent"


def test_add_coordinate_coupler_constraint_and_remove_constraint():
    model = make_model()
    add_free_body(model, "b1")
    add_free_body(model, "b2")
    independent = next(
        name for name in model.coordinates if name.startswith("b1_to_ground")
    )
    dependent = next(
        name for name in model.coordinates if name.startswith("b2_to_ground")
    )

    operators.add_coordinate_coupler_constraint(
        model, "coupler1", independent, dependent, opensim.LinearFunction(1.0, 0.0),
        reinitialize=True,
    )
    assert model.model.getConstraintSet().getSize() == 1

    operators.remove_constraint(model, "coupler1", reinitialize=True)
    assert model.model.getConstraintSet().getSize() == 0


def test_add_weld_constraint_removes_relative_motion():
    model = make_model()
    body1 = add_free_body(model, "b1")
    body2 = add_free_body(model, "b2")

    operators.add_weld_constraint(model, "weld1", body1.raw, body2.raw, reinitialize=True)

    assert model.model.getConstraintSet().getSize() == 1
    assert model.model.getConstraintSet().get("weld1") is not None


def test_add_point_constraint_connects_a_body_to_ground():
    # Confirmed directly (OpenSim 4.6): a PointConstraint between two
    # non-ground bodies crashes the process natively in initSystem() --
    # see add_point_constraint's docstring. Ground-to-body is the
    # confirmed-safe case, so that's what this exercises.
    model = make_model()
    body = add_free_body(model, "b1")

    constraint = operators.add_point_constraint(
        model, "point1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        reinitialize=True,
    )

    assert constraint.name == "point1"
    assert model.model.getConstraintSet().getSize() == 1


# ---------------------------------------------------------------------------
# Constraint/force subtype dispatch: WeldConstraint, PointConstraint,
# ConstantDistanceConstraint, ExponentialContactForce
# ---------------------------------------------------------------------------


def add_welded_body(model, name, position):
    """Add a body rigidly welded to ground at ``position`` (no dof)."""
    body = opensim.Body(name, 1.0, opensim.Vec3(0, 0, 0), opensim.Inertia(1, 1, 1, 0, 0, 0))
    model.model.addBody(body)
    joint = opensim.WeldJoint(
        f"{name}_to_ground",
        model.model.getGround(),
        opensim.Vec3(*position),
        opensim.Vec3(0, 0, 0),
        body,
        opensim.Vec3(0, 0, 0),
        opensim.Vec3(0, 0, 0),
    )
    model.model.addJoint(joint)
    model.model.finalizeConnections()
    model.state = model.model.initSystem()
    return body


def test_add_weld_constraint_returns_a_weld_constraint_with_both_points():
    model = make_model()
    body1 = add_welded_body(model, "b1", (0, 0, 0))
    body2 = add_welded_body(model, "b2", (0, 0, 0))

    constraint = operators.add_weld_constraint(
        model, "weld1", body1, body2,
        position1=(0.1, 0.2, 0.3), position2=(0.1, 0.2, 0.3),
        reinitialize=True,
    )

    assert isinstance(constraint, components.WeldConstraint)
    assert isinstance(constraint.frame1, components.OffsetFrame)
    assert isinstance(constraint.frame2, components.OffsetFrame)
    assert constraint.point1_local == pytest.approx((0.1, 0.2, 0.3))
    assert constraint.point1_global == pytest.approx((0.1, 0.2, 0.3))
    assert constraint.point2_local == pytest.approx((0.1, 0.2, 0.3))
    assert constraint.point2_global == pytest.approx((0.1, 0.2, 0.3))
    # model.constraints must dispatch to the same specific wrapper type.
    assert type(model.constraints["weld1"]) is components.WeldConstraint

    parents = constraint.parents
    assert [p.name for p in parents] == ["b1", "b2"]
    assert all(isinstance(p, components.Body) for p in parents)
    # The backward relation must find this constraint from either body.
    assert constraint in model.body("b1").parents
    assert constraint in model.body("b2").parents


def test_add_point_constraint_returns_a_point_constraint_with_both_points():
    model = make_model()
    body = add_welded_body(model, "b1", (2.0, 3.0, 4.0))

    constraint = operators.add_point_constraint(
        model, "point1", model.model.getGround(), (2.1, 3.2, 4.3), body, (0.1, 0.2, 0.3),
        reinitialize=True,
    )

    assert isinstance(constraint, components.PointConstraint)
    assert constraint.point1_local == pytest.approx((2.1, 3.2, 4.3))
    assert constraint.point1_global == pytest.approx((2.1, 3.2, 4.3))  # ground: local == global
    assert constraint.point2_local == pytest.approx((0.1, 0.2, 0.3))
    assert constraint.point2_global == pytest.approx((2.1, 3.2, 4.3))  # constraint satisfied
    assert type(model.constraints["point1"]) is components.PointConstraint

    parents = constraint.parents
    assert opensim.Ground.safeDownCast(parents[0]) is not None
    assert isinstance(parents[1], components.Body) and parents[1].name == "b1"
    assert constraint in model.body("b1").parents


def test_add_coordinate_coupler_constraint_still_returns_the_generic_wrapper():
    # CoordinateCouplerConstraint has no single spatial point -- no
    # dedicated subtype exists, so dispatch must fall through to the plain
    # Constraint wrapper (not raise, not silently return something wrong).
    model = make_model()
    add_free_body(model, "b1")
    add_free_body(model, "b2")
    independent = next(
        name for name in model.coordinates if name.startswith("b1_to_ground")
    )
    dependent = next(
        name for name in model.coordinates if name.startswith("b2_to_ground")
    )

    constraint = operators.add_coordinate_coupler_constraint(
        model, "coupler1", independent, dependent, opensim.LinearFunction(1.0, 0.0),
        reinitialize=True,
    )

    assert type(constraint) is components.Constraint
    assert type(model.constraints["coupler1"]) is components.Constraint


def test_add_point_on_plane_constraint_returns_a_constant_distance_constraint():
    model = make_model()
    body = add_welded_body(model, "b1", (0, 0, 0))
    plane_body = add_welded_body(model, "pb", (0, 0, 0))

    constraint = operators.add_point_on_plane_constraint(
        model, "cdc1", body, (0, 0, 0), plane_body, (0, 0, 0), (0, 1, 0),
        reinitialize=True,
    )

    assert isinstance(constraint, components.ConstantDistanceConstraint)
    assert constraint.point1_global == pytest.approx((0.0, 0.0, 0.0))
    assert constraint.point2_global == pytest.approx((0.0, -50.0, 0.0))  # default anchor_distance
    assert constraint.distance == pytest.approx(50.0)
    assert type(model.constraints["cdc1"]) is components.ConstantDistanceConstraint

    parents = constraint.parents
    assert [p.name for p in parents] == ["b1", "pb"]
    assert constraint in model.body("b1").parents
    assert constraint in model.body("pb").parents


def test_add_sliding_point_contact_returns_an_exponential_contact_force():
    model = make_model()
    body = add_welded_body(model, "b1", (0, 1, 0))
    plane_body = add_welded_body(model, "pb", (0, 0, 0))

    force = operators.add_sliding_point_contact(
        model, "contact1", body, (0.1, 0.2, 0.3), plane_body, (1, 2, 3), (0, 1, 0),
        reinitialize=True,
    )

    assert isinstance(force, components.ExponentialContactForce)
    assert force.point_global == pytest.approx((0.1, 1.2, 0.3))
    assert force.plane_point_global == pytest.approx((1.0, 2.0, 3.0))
    assert type(model.forces["contact1"]) is components.ExponentialContactForce


def test_add_muscle_still_returns_a_muscle_not_the_generic_force_or_wrapper():
    # components.Muscle stays reachable only via add_muscle/.muscles/.muscle()
    # -- _wrap_force deliberately does not special-case it (see its
    # docstring); this guards that .forces still returns the generic Force
    # for a muscle, not Muscle, even after the dispatch helper was added.
    model = make_model()
    body = add_free_body(model, "b1")
    muscle = operators.add_muscle(
        model, "m1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        reinitialize=True,
    )

    assert isinstance(muscle, components.Muscle)
    assert type(model.forces["m1"]) is components.Force


# ---------------------------------------------------------------------------
# Contact geometry: build-and-attach (add_contact_sphere/half_space/mesh)
# ---------------------------------------------------------------------------


def test_add_contact_sphere_builds_and_attaches_a_contact_sphere():
    model = make_model()
    body = add_welded_body(model, "b1", (1.0, 2.0, 3.0))

    sphere = operators.add_contact_sphere(
        model, "cs1", body, 0.05, location=(0.1, 0.2, 0.3), reinitialize=True
    )

    assert isinstance(sphere, components.ContactSphere)
    assert sphere.radius == pytest.approx(0.05)
    assert sphere.location == pytest.approx((0.1, 0.2, 0.3))
    assert sphere.position_local == pytest.approx((0.1, 0.2, 0.3))
    assert sphere.position_global == pytest.approx((1.1, 2.2, 3.3))
    assert sphere.parents == (model.body("b1"),)
    # model.contact_geometries must dispatch to the same specific wrapper.
    assert type(model.contact_geometries["cs1"]) is components.ContactSphere


def test_add_contact_sphere_rejects_non_positive_radius():
    model = make_model()
    body = add_welded_body(model, "b1", (0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="strictly positive"):
        operators.add_contact_sphere(model, "bad", body, -1.0)


def test_add_contact_half_space_builds_and_attaches_a_contact_half_space():
    model = make_model()
    body = add_welded_body(model, "b1", (0.0, 0.0, 0.0))

    half_space = operators.add_contact_half_space(
        model, "chs1", body, orientation_deg=(0.0, 0.0, 90.0), reinitialize=True
    )

    assert isinstance(half_space, components.ContactHalfSpace)
    assert half_space.orientation_deg == pytest.approx((0.0, 0.0, 90.0))
    assert type(model.contact_geometries["chs1"]) is components.ContactHalfSpace


def test_add_contact_mesh_builds_and_attaches_a_contact_mesh_without_crashing():
    # Confirmed directly (see add_contact_mesh's own docstring): OpenSim's
    # own ContactMesh(filename, location, orientation, frame, name)
    # all-at-once constructor segfaults the process in this installation,
    # with both a freshly-written .stl and a bundled .vtp file -- this is
    # the regression test for the piecemeal workaround add_contact_mesh
    # uses instead. A crash here would kill the whole test process, not
    # raise a catchable failure, so passing at all (not just the
    # assertions below) is itself part of what this test guards.
    from opensim_models._primitives import write_box_mesh

    model = make_model()
    body = add_welded_body(model, "b1", (1.0, 2.0, 3.0))
    with tempfile.TemporaryDirectory() as tmp_dir:
        mesh_path = Path(tmp_dir) / "box.stl"
        write_box_mesh(mesh_path, 0.1, 0.1, 0.1)

        mesh = operators.add_contact_mesh(
            model, "cm1", body, str(mesh_path), location=(0.1, 0.2, 0.3), reinitialize=True
        )

        assert isinstance(mesh, components.ContactMesh)
        assert mesh.filename == str(mesh_path)
        assert mesh.position_local == pytest.approx((0.1, 0.2, 0.3))
        assert mesh.position_global == pytest.approx((1.1, 2.2, 3.3))
        assert mesh.parents == (model.body("b1"),)
        assert type(model.contact_geometries["cm1"]) is components.ContactMesh


def test_add_contact_mesh_does_not_validate_the_file_eagerly():
    # Matches the lazy-resolution behaviour of an attached body Mesh,
    # documented explicitly in add_contact_mesh's own docstring -- confirmed
    # directly that even reinitialize=True (which runs initSystem()) does
    # not raise for a mesh file that doesn't exist.
    model = make_model()
    body = add_welded_body(model, "b1", (0.0, 0.0, 0.0))

    mesh = operators.add_contact_mesh(model, "cm1", body, "does_not_exist.obj", reinitialize=True)

    assert mesh.filename == "does_not_exist.obj"


# ---------------------------------------------------------------------------
# Contact geometry: position_global / position_local
# ---------------------------------------------------------------------------


def test_contact_geometry_parents_is_the_body_it_is_attached_to():
    model = make_model()
    body = add_free_body(model, "b1")

    sphere = opensim.ContactSphere()
    sphere.setName("cs1")
    sphere.setRadius(0.05)
    sphere.connectSocket_frame(body.raw)
    operators.add_contact_geometry(model, sphere, reinitialize=True)

    parents = model.contact_geometries["cs1"].parents

    assert len(parents) == 1
    assert isinstance(parents[0], components.Body)
    assert parents[0].name == "b1"


def test_muscle_parents_lists_distinct_bodies_crossed_by_the_path():
    model = make_model()
    body = add_free_body(model, "b1")

    muscle = operators.add_muscle(
        model, "mus1", model.model.getGround(), (0, 0, 0), body.raw, (0, 0, 0),
        via_points=[(model.model.getGround(), (0.05, 0, 0))],
        reinitialize=True,
    )

    parents = muscle.parents

    # 3 path points (origin on ground, via point on ground, insertion on
    # body), but ground's duplicate is dropped -- 2 distinct bodies.
    assert len(parents) == 2
    assert opensim.Ground.safeDownCast(parents[0]) is not None
    assert isinstance(parents[1], components.Body) and parents[1].name == "b1"
    assert muscle in model.body("b1").parents


def test_contact_geometry_position_local_matches_its_own_location():
    model = make_model()
    body = add_free_body(model, "b1")

    sphere = opensim.ContactSphere()
    sphere.setName("cs1")
    sphere.setRadius(0.05)
    sphere.set_location(opensim.Vec3(0.1, 0.2, 0.3))
    sphere.connectSocket_frame(body.raw)
    operators.add_contact_geometry(model, sphere, reinitialize=True)

    contact_geometry = model.contact_geometries["cs1"]
    assert contact_geometry.position_local == pytest.approx((0.1, 0.2, 0.3))


def test_contact_geometry_position_global_matches_its_frame_transformed_by_hand():
    # A body welded to ground with a non-trivial offset/rotation, so this
    # actually exercises the position+rotation transform (not just a
    # zero-rotation pass-through) -- confirmed by hand against the same
    # frame.getPositionInGround()/getRotationInGround() calls
    # ContactGeometry.position_global is built on top of.
    model = make_model()
    body = opensim.Body("b1", 1.0, opensim.Vec3(0, 0, 0), opensim.Inertia(1, 1, 1, 0, 0, 0))
    model.model.addBody(body)
    joint = opensim.WeldJoint(
        "j1",
        model.model.getGround(),
        opensim.Vec3(1.0, 2.0, 3.0),
        opensim.Vec3(0.0, 0.0, 0.5),
        body,
        opensim.Vec3(0.0, 0.0, 0.0),
        opensim.Vec3(0.0, 0.0, 0.0),
    )
    model.model.addJoint(joint)

    sphere = opensim.ContactSphere()
    sphere.setName("cs1")
    sphere.setRadius(0.05)
    sphere.set_location(opensim.Vec3(0.1, 0.2, 0.3))
    sphere.connectSocket_frame(body)
    operators.add_contact_geometry(model, sphere, reinitialize=True)

    contact_geometry = model.contact_geometries["cs1"]

    model.model.realizePosition(model.state)
    raw_frame = model.model.getContactGeometrySet().get("cs1").getFrame()
    raw_position = np.array(raw_frame.getPositionInGround(model.state).to_numpy())
    raw_rotation_matrix = raw_frame.getRotationInGround(model.state).asMat33()
    raw_rotation = np.array(
        [[raw_rotation_matrix.get(i, j) for j in range(3)] for i in range(3)]
    )
    expected = raw_position + raw_rotation @ np.array([0.1, 0.2, 0.3])

    assert contact_geometry.position_global == pytest.approx(tuple(expected))


# ---------------------------------------------------------------------------
# add_body mesh attachment
# ---------------------------------------------------------------------------


def test_add_body_attaches_an_existing_mesh_file():
    model = make_model()
    with tempfile.TemporaryDirectory() as tmp_dir:
        mesh_path = Path(tmp_dir) / "part.stl"
        from opensim_models._primitives import write_box_mesh

        write_box_mesh(mesh_path, 0.1, 0.1, 0.1)

        with model.structural_change():
            body = operators.add_body(model, "b1", mass=1.0, mesh_files=mesh_path)
            operators.add_weld_joint(model, "b1_joint", body)

        assert body.raw.getPropertyByName("attached_geometry").size() == 1
        assert Path(tmp_dir) in model.geometry_directories


# ---------------------------------------------------------------------------
# Named joint constructors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adder, expected_dof",
    [
        (operators.add_free_joint, 6),
        (operators.add_pin_joint, 1),
        (operators.add_ball_joint, 3),
        (operators.add_slider_joint, 1),
        (operators.add_weld_joint, 0),
    ],
)
def test_named_joint_constructors_create_the_expected_number_of_coordinates(
    adder, expected_dof
):
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        joint = adder(model, "b1_joint", body)

    assert len(joint.coordinates) == expected_dof


def test_named_joint_constructor_places_the_joint_at_position_and_orientation():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_pin_joint(
            model,
            "b1_joint",
            body,
            position=(1.0, 2.0, 3.0),
            orientation_deg=(90.0, 0.0, 0.0),
        )

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx((1.0, 2.0, 3.0))
    assert parent.get_orientation()[0] == pytest.approx(math.radians(90.0))


def test_add_joint_rejects_unknown_kind_when_called_generically():
    model = make_model()
    body = operators.add_body(model, "b1", mass=1.0)

    with pytest.raises(ValueError, match="Unknown component kind"):
        operators.add_component(model, "not_a_joint_kind", body)


# ---------------------------------------------------------------------------
# Primitive-shaped bodies
# ---------------------------------------------------------------------------


def test_add_box_body_computes_mass_and_inertia_analytically():
    model = make_model()

    body, joint = operators.add_box_body(
        model, "box1", (0.2, 0.3, 0.4), density=1200.0, reinitialize=True
    )

    expected_mass = 0.2 * 0.3 * 0.4 * 1200.0
    assert body.mass == pytest.approx(expected_mass)
    moments = body.raw.getInertia().getMoments()
    assert moments.get(0) == pytest.approx(expected_mass / 12.0 * (0.3**2 + 0.4**2))
    assert moments.get(1) == pytest.approx(expected_mass / 12.0 * (0.2**2 + 0.4**2))
    assert moments.get(2) == pytest.approx(expected_mass / 12.0 * (0.2**2 + 0.3**2))
    assert opensim.WeldJoint.safeDownCast(joint.raw) is not None


def test_add_cylinder_body_computes_mass_and_inertia_analytically():
    model = make_model()

    body, _ = operators.add_cylinder_body(model, "cyl1", 0.05, 0.3, reinitialize=True)

    expected_mass = math.pi * 0.05**2 * 0.3 * 1000.0
    assert body.mass == pytest.approx(expected_mass)
    moments = body.raw.getInertia().getMoments()
    assert moments.get(1) == pytest.approx(expected_mass * 0.05**2 / 2.0)


def test_add_sphere_body_computes_mass_and_inertia_analytically():
    model = make_model()

    body, _ = operators.add_sphere_body(model, "sph1", 0.1, reinitialize=True)

    expected_mass = 4.0 / 3.0 * math.pi * 0.1**3 * 1000.0
    assert body.mass == pytest.approx(expected_mass)
    moments = body.raw.getInertia().getMoments()
    expected_moment = 2.0 / 5.0 * expected_mass * 0.1**2
    assert moments.get(0) == pytest.approx(expected_moment)
    assert moments.get(1) == pytest.approx(expected_moment)
    assert moments.get(2) == pytest.approx(expected_moment)


def test_primitive_body_is_placed_at_the_given_position_and_orientation():
    model = make_model()

    _, joint = operators.add_box_body(
        model,
        "box1",
        (0.1, 0.1, 0.1),
        position=(1.0, 2.0, 3.0),
        orientation_deg=(0.0, 45.0, 0.0),
        reinitialize=True,
    )

    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx((1.0, 2.0, 3.0))
    assert parent.get_orientation()[1] == pytest.approx(math.radians(45.0))


def test_primitive_body_default_joint_type_is_weld():
    model = make_model()

    operators.add_sphere_body(model, "sph1", 0.05, reinitialize=True)

    assert len(model.coordinates) == 0


def test_primitive_body_joint_type_can_be_changed():
    model = make_model()

    operators.add_sphere_body(model, "sph1", 0.05, joint_type="free", reinitialize=True)

    assert len(model.coordinates) == 6


def test_primitive_body_rejects_unknown_joint_type():
    model = make_model()

    with pytest.raises(ValueError, match="Unknown joint_type"):
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), joint_type="bogus")

    assert len(model.bodies) == 0


def test_primitive_body_mesh_requires_mesh_dir():
    model = make_model()

    with pytest.raises(ValueError, match="mesh_dir"):
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), mesh=True)

    assert len(model.bodies) == 0


def test_primitive_body_can_generate_an_actual_mesh_file():
    model = make_model()
    with tempfile.TemporaryDirectory() as tmp_dir:
        body, _ = operators.add_box_body(
            model, "box1", (0.1, 0.2, 0.3), mesh=True, mesh_dir=tmp_dir, reinitialize=True
        )

        assert (Path(tmp_dir) / "box1.stl").is_file()
        assert body.raw.getPropertyByName("attached_geometry").size() == 1


def test_primitive_bodies_can_be_batched_together():
    model = make_model()

    with model.structural_change():
        operators.add_box_body(model, "box1", (0.1, 0.1, 0.1), position=(0.0, 0.0, 0.0))
        operators.add_sphere_body(model, "sph1", 0.05, position=(1.0, 0.0, 0.0))

    assert len(model.bodies) == 2
    assert len(model.joints) == 2


# ---------------------------------------------------------------------------
# rotate_object
# ---------------------------------------------------------------------------


def test_rotate_object_rotates_a_marker_about_the_origin():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(1.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    new_position = operators.rotate_object(marker, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert model.marker("mk1").location == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotate_object_rotates_a_physical_offset_frame_translation_and_orientation():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    new_position = operators.rotate_object(parent, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )
    assert parent.get_orientation()[2] == pytest.approx(math.radians(90.0))


def test_rotate_object_pivots_about_an_external_point():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    new_position = operators.rotate_object(parent, (2.0, 0.0, 0.0), (0.0, 0.0, 1.0), 180.0)

    assert new_position == pytest.approx((3.0, 0.0, 0.0), abs=1e-9)


def test_rotate_object_origin_can_be_another_component():
    model = make_model()
    pivot_marker = opensim.Marker(
        "pivot", model.model.getGround(), opensim.Vec3(2.0, 0.0, 0.0)
    )
    operators.add_marker(model, pivot_marker, reinitialize=True)
    moving_marker = opensim.Marker(
        "moving", model.model.getGround(), opensim.Vec3(3.0, 0.0, 0.0)
    )
    operators.add_marker(model, moving_marker, reinitialize=True)

    new_position = operators.rotate_object(
        moving_marker, pivot_marker, (0.0, 0.0, 1.0), 180.0
    )

    assert new_position == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_rotate_object_on_a_body_is_read_only():
    model = make_model()
    body = add_free_body(model, "b1")
    original_position = tuple(body.raw.getPositionInGround(model.state).to_numpy())

    new_position = operators.rotate_object(body, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position != pytest.approx(original_position)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        original_position
    )


def test_rotate_object_rejects_unresolvable_object():
    with pytest.raises(TypeError, match="Cannot resolve a ground-frame position"):
        operators.rotate_object(
            "not an opensim component", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0
        )


def test_rotate_object_rejects_zero_direction():
    with pytest.raises(ValueError, match="direction must be a non-zero vector"):
        operators.rotate_object((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 90.0)


def test_rotate_object_works_standalone_on_plain_coordinates():
    new_position = operators.rotate_object((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotate_object_rotates_the_whole_model_about_its_root_joint():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    new_position = operators.rotate_object(model, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)

    assert new_position == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )


def test_rotate_object_rejects_a_model_with_no_ground_joint():
    model = make_model()

    with pytest.raises(ValueError, match="no joint attached to ground"):
        operators.rotate_object(model, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)


# ---------------------------------------------------------------------------
# rotate_object(..., inplace=False)
# ---------------------------------------------------------------------------


def test_rotate_object_not_inplace_leaves_the_whole_model_untouched():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    rotated_copy = operators.rotate_object(
        model, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
    )

    assert rotated_copy is not model
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    copy_body_position = rotated_copy.body("b1").raw.getPositionInGround(rotated_copy.state)
    assert tuple(copy_body_position.to_numpy()) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_rotate_object_not_inplace_leaves_a_marker_untouched():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(1.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    rotated_copy = operators.rotate_object(
        marker, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
    )

    assert rotated_copy is not marker
    assert tuple(marker.get_location().to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert tuple(rotated_copy.get_location().to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )


def test_rotate_object_not_inplace_leaves_a_physical_offset_frame_untouched():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    rotated_copy = operators.rotate_object(
        parent, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
    )

    assert rotated_copy is not parent
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert tuple(rotated_copy.get_translation().to_numpy()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9
    )


def test_rotate_object_not_inplace_on_a_body_still_returns_a_position():
    model = make_model()
    body = add_free_body(model, "b1")
    original_position = tuple(body.raw.getPositionInGround(model.state).to_numpy())

    new_position = operators.rotate_object(
        body, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0, inplace=False
    )

    assert new_position != pytest.approx(original_position)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        original_position
    )


# ---------------------------------------------------------------------------
# translate_object
# ---------------------------------------------------------------------------


def test_translate_object_translates_a_marker():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(1.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    new_position = operators.translate_object(marker, (0.0, 2.0, 3.0))

    assert new_position == pytest.approx((1.0, 2.0, 3.0), abs=1e-9)
    assert model.marker("mk1").location == pytest.approx((1.0, 2.0, 3.0), abs=1e-9)


def test_translate_object_translates_a_physical_offset_frame():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    new_position = operators.translate_object(parent, (0.0, 2.0, 0.0))

    assert new_position == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx(
        (1.0, 2.0, 0.0), abs=1e-9
    )
    # a translation does not rotate anything
    orientation = parent.get_orientation()
    assert (orientation[0], orientation[1], orientation[2]) == pytest.approx(
        (0.0, 0.0, 0.0)
    )


def test_translate_object_on_a_body_is_read_only():
    model = make_model()
    body = add_free_body(model, "b1")
    original_position = tuple(body.raw.getPositionInGround(model.state).to_numpy())

    new_position = operators.translate_object(body, (1.0, 0.0, 0.0))

    assert new_position != pytest.approx(original_position)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        original_position
    )


def test_translate_object_rejects_unresolvable_object():
    with pytest.raises(TypeError, match="Cannot resolve a ground-frame position"):
        operators.translate_object("not an opensim component", (0.0, 0.0, 0.0))


def test_translate_object_rejects_malformed_direction():
    with pytest.raises(ValueError, match="direction must have exactly 3 values"):
        operators.translate_object((1.0, 0.0, 0.0), (0.0, 0.0))


def test_translate_object_works_standalone_on_plain_coordinates():
    new_position = operators.translate_object((1.0, 0.0, 0.0), (0.0, 2.0, 3.0))

    assert new_position == pytest.approx((1.0, 2.0, 3.0), abs=1e-9)


def test_translate_object_translates_the_whole_model_about_its_root_joint():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    new_position = operators.translate_object(model, (0.0, 2.0, 0.0))

    assert new_position == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 2.0, 0.0), abs=1e-9
    )


def test_translate_object_rejects_a_model_with_no_ground_joint():
    model = make_model()

    with pytest.raises(ValueError, match="no joint attached to ground"):
        operators.translate_object(model, (0.0, 2.0, 0.0))


# ---------------------------------------------------------------------------
# translate_object(..., inplace=False)
# ---------------------------------------------------------------------------


def test_translate_object_not_inplace_leaves_the_whole_model_untouched():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    translated_copy = operators.translate_object(model, (0.0, 2.0, 0.0), inplace=False)

    assert translated_copy is not model
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    copy_body_position = translated_copy.body("b1").raw.getPositionInGround(translated_copy.state)
    assert tuple(copy_body_position.to_numpy()) == pytest.approx((1.0, 2.0, 0.0), abs=1e-9)


def test_translate_object_not_inplace_leaves_a_marker_untouched():
    model = make_model()
    marker = opensim.Marker("mk1", model.model.getGround(), opensim.Vec3(1.0, 0.0, 0.0))
    operators.add_marker(model, marker, reinitialize=True)

    translated_copy = operators.translate_object(marker, (0.0, 2.0, 3.0), inplace=False)

    assert translated_copy is not marker
    assert tuple(marker.get_location().to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert tuple(translated_copy.get_location().to_numpy()) == pytest.approx(
        (1.0, 2.0, 3.0), abs=1e-9
    )


def test_translate_object_not_inplace_leaves_a_physical_offset_frame_untouched():
    model = make_model()
    with model.structural_change():
        body = operators.add_body(model, "b1", mass=1.0, inertia=(1, 1, 1, 0, 0, 0))
        operators.add_weld_joint(model, "b1_joint", body, position=(1.0, 0.0, 0.0))

    joint = model.joints.get("b1_joint")
    parent = opensim.PhysicalOffsetFrame.safeDownCast(joint.raw.getParentFrame())

    translated_copy = operators.translate_object(parent, (0.0, 2.0, 0.0), inplace=False)

    assert translated_copy is not parent
    assert tuple(parent.get_translation().to_numpy()) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert tuple(translated_copy.get_translation().to_numpy()) == pytest.approx(
        (1.0, 2.0, 0.0), abs=1e-9
    )


def test_translate_object_not_inplace_on_a_body_still_returns_a_position():
    model = make_model()
    body = add_free_body(model, "b1")
    original_position = tuple(body.raw.getPositionInGround(model.state).to_numpy())

    new_position = operators.translate_object(body, (1.0, 0.0, 0.0), inplace=False)

    assert new_position != pytest.approx(original_position)
    assert tuple(body.raw.getPositionInGround(model.state).to_numpy()) == pytest.approx(
        original_position
    )
