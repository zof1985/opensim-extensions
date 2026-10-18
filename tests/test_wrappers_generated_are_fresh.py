"""Freshness and inheritance checks for statically generated wrappers."""

import subprocess
import sys
from pathlib import Path

import opensim
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_generated_wrapper_files_are_fresh():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_wrappers.py"), "--check"],
        cwd=ROOT,
        check=True,
    )


def test_generated_body_subclasses_native_body():
    from opensim_models._generated import Body

    assert issubclass(Body, opensim.Body)
    body = Body("test", 2.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1))
    assert body.mass == 2.0
    body.set_mass(3.5)
    assert body.mass == 3.5
    assert "Parameters" in Body.__doc__
    assert "Returns" in Body.mass.__doc__


def test_native_body_proxy_can_be_reclassed():
    from opensim_models._generated._registry import wrap_component

    model = opensim.Model()
    native_body = opensim.Body("wrapped", 1.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1))
    model.addBody(native_body)
    proxy = model.getBodySet().get("wrapped")
    wrapped = wrap_component(proxy)
    assert isinstance(wrapped, opensim.Body)
    assert type(wrapped).__name__ == "Body"
    assert wrapped.mass == pytest.approx(1.0)
