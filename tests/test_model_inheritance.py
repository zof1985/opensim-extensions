"""Tests for the OpenSim-native base class contract."""

import opensim

from opensim_models import OpenSimModel


def test_open_sim_model_is_native_model_subclass():
    assert issubclass(OpenSimModel, opensim.Model)
    model = OpenSimModel()

    assert isinstance(model, opensim.Model)
    assert model.getName()
    assert model.state is not None


def test_open_sim_model_exposes_native_model_methods_directly():
    model = OpenSimModel()
    body = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1))
    model.addBody(body)

    assert model.getBodySet().get("body").getName() == "body"


def test_copy_returns_independent_native_model_subclass():
    source = OpenSimModel()
    source.setName("source")

    duplicate = source.copy()
    assert isinstance(duplicate, OpenSimModel)
    assert int(duplicate.this) != int(source.this)

    duplicate.setName("duplicate")
    assert source.getName() == "source"
