"""Freshness tests for partial OpenSim stubs."""

import subprocess
import sys
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUB = ROOT / "typings" / "opensim" / "__init__.pyi"


def test_generated_opensim_stubs_are_fresh():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_opensim_stubs.py"),
            "--all",
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_sample_stubs_include_expected_classes_and_exclude_moco():
    content = STUB.read_text(encoding="utf-8")
    for name in ("Model", "Body", "Joint", "Coordinate", "Force"):
        assert f"class {name}" in content
    assert "class Moco" not in content


def test_stub_package_is_marked_partial():
    marker = STUB.parent / "py.typed"
    assert marker.read_text(encoding="utf-8").strip() == "partial"


def test_full_stub_surface_matches_non_moco_schema():
    schema = json.loads(
        (ROOT / "scripts" / "_opensim_schema.json").read_text(encoding="utf-8")
    )["classes"]
    expected = set(schema)
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    declared = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert expected == set(exports) == declared
    assert not any(name.startswith("Moco") for name in declared)
