from __future__ import annotations

import copy
import math

import pytest

opensim = pytest.importorskip("opensim")

from opensim_extensions import Framework


def make_pendulum(prefix: str, parent_offset=(0.0, 0.0, 0.0)) -> opensim.Model:
    model = opensim.Model()
    model.setName(prefix)
    body = opensim.Body(f"{prefix}_body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    model.addBody(body)
    joint = opensim.PinJoint(
        f"{prefix}_joint",
        model.getGround(),
        opensim.Vec3(*parent_offset),
        opensim.Vec3(0),
        body,
        opensim.Vec3(0),
        opensim.Vec3(0),
    )
    joint.upd_coordinates(0).setName(f"{prefix}_q")
    model.addJoint(joint)
    model.state = model.initSystem()
    return model


def make_pair() -> Framework:
    return Framework({"a": make_pendulum("a"), "b": make_pendulum("b")})


def constraint_names(model: opensim.Model) -> list[str]:
    constraints = model.getConstraintSet()
    return [constraints.get(i).getName() for i in range(constraints.getSize())]


def test_declared_couplings_are_listed_validated_and_removable():
    framework = make_pair()
    frames = (("a", "/bodyset/a_body"), ("b", "/bodyset/b_body"))

    weld = framework.couple_weld(*frames)
    point = framework.couple_point(
        *frames, second_location=(0.0, 0.5, 0.0), name="hinge"
    )
    coordinate = framework.couple_coordinate(
        ("a", "/jointset/a_joint/a_q"), ("b", "/jointset/b_joint/b_q")
    )

    assert list(framework.couplings) == ["weld_1", "hinge", "coordinate_1"]
    assert framework.couplings["weld_1"] is weld
    assert point.kind == "point" and coordinate.members == {"a", "b"}
    with pytest.raises(ValueError, match="already exists"):
        framework.couple_weld(*frames, name="hinge")
    with pytest.raises(ValueError, match="distinct models"):
        framework.couple_weld(frames[0], frames[0])
    with pytest.raises(KeyError):
        framework.couple_weld(frames[0], ("missing", "/bodyset/x"))
    with pytest.raises(ValueError, match="does not exist"):
        framework.couple_weld(frames[0], ("b", "/bodyset/nope"))
    with pytest.raises(TypeError, match="PhysicalFrame"):
        framework.couple_weld(frames[0], ("b", "/jointset/b_joint"))
    with pytest.raises(TypeError):
        framework.couple_weld(frames[0], "b:/bodyset/b_body")
    with pytest.raises(ValueError, match="three finite"):
        framework.couple_point(*frames, first_location=(0.0, 0.0))

    framework.uncouple("hinge")
    assert list(framework.couplings) == ["weld_1", "coordinate_1"]
    with pytest.raises(KeyError):
        framework.uncouple("hinge")
    framework.uncouple()
    assert not framework.couplings


def test_weld_and_point_couplings_exist_only_in_the_composite():
    framework = make_pair()
    frames = (("a", "/bodyset/a_body"), ("b", "/bodyset/b_body"))
    framework.couple_weld(*frames)
    framework.couple_point(*frames, second_location=(0.0, 0.5, 0.0))

    composite = framework.composite

    assert constraint_names(composite) == ["weld_1", "point_1"]
    weld = opensim.WeldConstraint.safeDownCast(composite.getConstraintSet().get(0))
    assert weld.getSocket("frame1").getConnecteePath() == "/bodyset/a_body"
    assert weld.getSocket("frame2").getConnecteePath() == "/bodyset/b_body"
    assert framework.a.getConstraintSet().getSize() == 0
    assert framework.b.getConstraintSet().getSize() == 0

    framework.uncouple("weld_1")
    assert constraint_names(framework.composite) == ["point_1"]


def test_coordinate_coupling_builds_a_native_coupler_constraint():
    framework = make_pair()
    framework.couple_coordinate(
        ("a", "/jointset/a_joint/a_q"),
        ("b", "/jointset/b_joint/b_q"),
        function=opensim.LinearFunction(2.0, 0.0),
    )

    composite = framework.composite

    coupler = opensim.CoordinateCouplerConstraint.safeDownCast(
        composite.getConstraintSet().get(0)
    )
    assert coupler.getDependentCoordinateName() == "b_q"
    assert coupler.getIndependentCoordinateNames().get(0) == "a_q"
    assert coupler.getFunction().calcValue(opensim.Vector(1, 0.5)) == pytest.approx(1.0)
    assert framework.a.getConstraintSet().getSize() == 0


def test_joint_coupling_follows_the_parent_model_during_simulation():
    parent = make_pendulum("a")
    child = make_pendulum("b", parent_offset=(0.0, 1.0, 0.0))
    parent.getCoordinateSet().get(0).setSpeedValue(parent.state, 1.0)
    framework = Framework({"a": parent, "b": child})
    framework.set_gravity((0.0, 0.0, 0.0))
    framework.couple_joint(("a", "/bodyset/a_body"), ("b", "/jointset/b_joint"))

    state = framework.forward(0.1)
    framework.composite.realizePosition(state)
    angle = parent.getCoordinateSet().get(0).getValue(parent.state)
    position = framework.composite.getBodySet().get("b_body").getPositionInGround(state)
    assert angle == pytest.approx(0.1, abs=1e-3)
    assert position.get(0) == pytest.approx(-math.sin(angle), abs=1e-3)
    assert position.get(1) == pytest.approx(math.cos(angle), abs=1e-3)
    assert child.getCoordinateSet().get(0).getValue(child.state) == pytest.approx(
        0.0, abs=1e-6
    )


def test_coordinate_coupling_rejects_ambiguous_coordinate_names_on_rebuild():
    first = make_pendulum("a")
    second = make_pendulum("b")
    second.getJointSet().get(0).upd_coordinates(0).setName("a_q")
    second.state = second.initSystem()
    framework = Framework({"a": first, "b": second})
    framework.couple_coordinate(
        ("a", "/jointset/a_joint/a_q"), ("b", "/jointset/b_joint/a_q")
    )

    with pytest.raises(ValueError, match="not unique"):
        framework.initSystem()


def test_joint_coupling_attaches_a_ground_rooted_joint_to_another_model():
    parent = make_pendulum("a")
    child = make_pendulum("b", parent_offset=(0.0, 1.0, 0.0))
    parent.getCoordinateSet().get(0).setValue(parent.state, math.pi / 2, True)
    framework = Framework({"a": parent, "b": child})
    coupling = framework.couple_joint(
        ("a", "/bodyset/a_body"), ("b", "/jointset/b_joint")
    )

    composite = framework.composite
    state = framework.realize()
    position = composite.getBodySet().get("b_body").getPositionInGround(state)

    assert coupling.kind == "joint"
    assert [position.get(i) for i in range(3)] == pytest.approx(
        [-1.0, 0.0, 0.0], abs=1e-9
    )
    child_state = child.getWorkingState() if not hasattr(child, "state") else child.state
    child_position = child.getBodySet().get("b_body").getPositionInGround(child_state)
    assert child_position.get(1) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="already coupled"):
        framework.couple_joint(("a", "/bodyset/a_body"), ("b", "/jointset/b_joint"))

    chain = make_pendulum("c")
    second_body = opensim.Body("c2_body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    chain.addBody(second_body)
    chain.addJoint(
        opensim.PinJoint(
            "c2_joint",
            chain.getBodySet().get("c_body"),
            opensim.Vec3(0),
            opensim.Vec3(0),
            second_body,
            opensim.Vec3(0),
            opensim.Vec3(0),
        )
    )
    chain.state = chain.initSystem()
    framework["c"] = chain
    with pytest.raises(ValueError, match="attached to ground"):
        framework.couple_joint(("a", "/bodyset/a_body"), ("c", "/jointset/c2_joint"))


def test_add_force_constraint_and_controller_validate_member_references():
    framework = make_pair()
    spring = opensim.PointToPointSpring(
        framework.a.getBodySet().get("a_body"),
        opensim.Vec3(0),
        framework.b.getBodySet().get("b_body"),
        opensim.Vec3(0),
        10.0,
        0.5,
    )
    spring.setName("link")

    force = framework.add_force(
        spring,
        {"body1": ("a", "/bodyset/a_body"), "body2": ("b", "/bodyset/b_body")},
    )
    weld = framework.add_constraint(
        opensim.WeldConstraint(),
        {"frame1": ("a", "/bodyset/a_body"), "frame2": ("b", "/bodyset/b_body")},
        name="locked",
    )

    composite = framework.composite
    assert force.name == "link" and weld.kind == "constraint"
    forces = composite.getForceSet()
    assert [forces.get(i).getName() for i in range(forces.getSize())] == ["link"]
    assert constraint_names(composite) == ["locked"]
    assert framework.a.getForceSet().getSize() == 0

    with pytest.raises(ValueError, match="no socket"):
        framework.add_force(spring, {"nope": ("a", "/bodyset/a_body")})
    with pytest.raises(ValueError, match="does not exist"):
        framework.add_force(spring, {"body1": ("a", "/bodyset/x")}, name="other")
    with pytest.raises(ValueError, match="single component"):
        framework.add_force(
            spring,
            {"body1": [("a", "/bodyset/a_body"), ("b", "/bodyset/b_body")]},
            name="other",
        )
    with pytest.raises(TypeError, match="Controller"):
        framework.add_controller(spring, {"actuators": ("a", "/bodyset/a_body")})
    with pytest.raises(KeyError):
        framework.add_force(spring, {"body1": ("zzz", "/bodyset/a_body")}, name="x")


def test_couplings_are_recreated_on_rebuild_and_copied_with_the_framework():
    framework = make_pair()
    framework.couple_weld(("a", "/bodyset/a_body"), ("b", "/bodyset/b_body"))
    assert constraint_names(framework.composite) == ["weld_1"]

    framework.set_gravity((0.0, -1.0, 0.0))
    assert constraint_names(framework.composite) == ["weld_1"]

    shallow = copy.copy(framework)
    deep = copy.deepcopy(framework)
    shallow.uncouple()
    assert list(deep.couplings) == ["weld_1"] and list(framework.couplings) == ["weld_1"]
    assert constraint_names(deep.composite) == ["weld_1"]


def test_stale_coupling_after_member_removal_fails_clearly_until_uncoupled():
    framework = make_pair()
    framework.couple_weld(("a", "/bodyset/a_body"), ("b", "/bodyset/b_body"))
    del framework["b"]

    with pytest.raises(ValueError, match="uncouple"):
        framework.initSystem()
    framework.uncouple()
    assert framework.composite.getBodySet().getSize() == 1


def test_merge_to_model_selects_members_names_mapping_and_independence():
    framework = make_pair()
    framework.couple_weld(("a", "/bodyset/a_body"), ("b", "/bodyset/b_body"))

    full = framework.merge_to_model()
    bare = framework.merge_to_model(include_couplings=False)
    only_a = framework.merge_to_model(only=["a"])
    without_a, mapping = framework.merge_to_model(
        name="rest", exclude=["a"], return_mapping=True
    )

    assert full.getName() == "framework" and constraint_names(full) == ["weld_1"]
    assert constraint_names(bare) == []
    assert only_a.getBodySet().getSize() == 1 and constraint_names(only_a) == []
    assert without_a.getName() == "rest" and without_a.getBodySet().getSize() == 1
    assert mapping[("b", "/bodyset/b_body")] == "/bodyset/b_body"
    assert not any(member == "a" for member, _ in mapping)

    full.getBodySet().get("a_body").setMass(5.0)
    full.getConstraintSet().get(0).setName("renamed")
    assert framework.a.getBodySet().get("a_body").getMass() == 1.0
    assert list(framework.couplings) == ["weld_1"]
    assert constraint_names(framework.merge_to_model()) == ["weld_1"]

    with pytest.raises(KeyError):
        framework.merge_to_model(only=["missing"])
    with pytest.raises(ValueError, match="no framework models"):
        framework.merge_to_model(exclude=["a", "b"])
    with pytest.raises(ValueError, match="non-empty"):
        framework.merge_to_model(name="")
