"""Coerenza dello schema ``scripts/_opensim_schema.json`` estratto da OpenSim."""

import json
from pathlib import Path

import pytest

SCHEMA = Path(__file__).resolve().parent.parent / "scripts" / "_opensim_schema.json"


@pytest.fixture(scope="module")
def classes():
    return json.loads(SCHEMA.read_text(encoding="utf-8"))["classes"]


def test_moco_is_excluded(classes):
    assert not [n for n in classes if n.startswith("Moco")]


def test_expected_classes_and_hierarchy(classes):
    for name in ("Model", "Body", "Joint", "PinJoint", "Coordinate", "Muscle", "Force", "Mesh"):
        assert name in classes, name
    assert "Joint" in classes["PinJoint"]["bases"]
    assert classes["Joint"]["concrete"] is False
    assert classes["PinJoint"]["concrete"] is True


def test_body_properties(classes):
    props = {p["name"]: p for p in classes["Body"]["properties"]}
    assert props["mass"]["type"] == "double"
    assert props["mass_center"]["type"] == "Vec3"
    assert props["mass"]["comment"]


def test_every_property_has_name_and_type(classes):
    for name, cls in classes.items():
        for prop in cls.get("properties") or []:
            assert prop["name"] and prop["type"], (name, prop)


def test_sockets_for_concrete_joint(classes):
    assert set(classes["PinJoint"]["sockets"]) >= {"parent_frame", "child_frame"}


def test_most_object_classes_have_docs(classes):
    objects = [c for c in classes.values() if c["is_object"]]
    documented = sum(1 for c in objects if c["doc"])
    assert documented / len(objects) > 0.9
