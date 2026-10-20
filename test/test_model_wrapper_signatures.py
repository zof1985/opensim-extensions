"""Every add_*/remove_* convenience method on OpenSimModel is a thin, hand-written
pass-through to the same-named function in opensim_models.operators (see each
method's own "A thin wrapper equivalent to ``operators.<name>(self, ...)``"
docstring). Nothing enforces that the two signatures stay in sync -- this test
does, so a future edit to one side alone (as nearly happened this session,
when operators.add_weld_joint's real keyword arguments were confused with
attach_component's) fails loudly instead of silently drifting.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from opensim_models import OpenSimModel, operators

#: Every OpenSimModel convenience method that is a pure pass-through to the
#: same-named operators.py function (confirmed by reading model.py: each one's
#: body is exactly ``from . import operators; return operators.<name>(self, ...)``
#: -- or a bare call, for the remove_* methods, which return None).
DELEGATED_METHOD_NAMES = [
    "add_body",
    "add_free_joint",
    "add_pin_joint",
    "add_ball_joint",
    "add_slider_joint",
    "add_weld_joint",
    "attach_component",
    "add_offset_frame",
    "add_force",
    "add_muscle",
    "add_marker",
    "add_constraint",
    "add_weld_constraint",
    "add_point_constraint",
    "add_coordinate_coupler_constraint",
    "add_controller",
    "add_contact_geometry",
    "add_contact_sphere",
    "add_contact_half_space",
    "add_contact_mesh",
    "add_probe",
    "remove_body",
    "remove_joint",
    "remove_force",
    "remove_muscle",
    "remove_marker",
    "remove_constraint",
    "remove_controller",
    "remove_contact_geometry",
    "remove_probe",
]


@pytest.mark.parametrize("name", DELEGATED_METHOD_NAMES)
def test_wrapper_signature_matches_operator(name):
    wrapper_params = list(inspect.signature(getattr(OpenSimModel, name)).parameters.values())
    operator_params = list(inspect.signature(getattr(operators, name)).parameters.values())

    # Drop each side's own leading positional-or-keyword receiver
    # (``self`` for the method, ``model`` for the free function) before
    # comparing -- everything after that must line up exactly: same names,
    # same order, same kinds (positional/keyword-only), same defaults.
    assert wrapper_params[1:] == operator_params[1:]


def test_every_delegated_method_actually_exists_on_both_sides():
    for name in DELEGATED_METHOD_NAMES:
        assert hasattr(OpenSimModel, name), f"OpenSimModel.{name} no longer exists"
        assert hasattr(operators, name), f"operators.{name} no longer exists"
