"""opensim_models.operators used to be a single flat module; it is now a
package (operators/__init__.py re-exporting from bodies.py, joints.py,
attachment.py, primitives.py, forces.py, markers.py, constraints.py,
auxiliary.py, rotation.py, translation.py, geometry.py, solving.py, with
shared helpers in _shared.py/_spatial.py). This test guards the one thing
that split must never change: opensim_models.operators.<name> (and
opensim_models.operators.__all__) resolving exactly as before.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import operators

EXPECTED_ALL = [
    "add_component",
    "remove_component",
    "add_body",
    "remove_body",
    "add_joint",
    "remove_joint",
    "add_free_joint",
    "add_pin_joint",
    "add_ball_joint",
    "add_slider_joint",
    "add_weld_joint",
    "attach_component",
    "add_offset_frame",
    "add_box_body",
    "add_cylinder_body",
    "add_sphere_body",
    "add_force",
    "remove_force",
    "add_muscle",
    "remove_muscle",
    "add_marker",
    "remove_marker",
    "add_constraint",
    "remove_constraint",
    "add_weld_constraint",
    "add_point_constraint",
    "add_coordinate_coupler_constraint",
    "add_point_on_plane_constraint",
    "add_controller",
    "remove_controller",
    "add_contact_geometry",
    "remove_contact_geometry",
    "add_probe",
    "remove_probe",
    "add_contact_sphere",
    "add_contact_half_space",
    "add_contact_mesh",
    "add_sliding_point_contact",
    "rotate_object",
    "translate_object",
    "euclidean_distance",
    "from_global_to_local",
    "from_local_to_global",
    "solve_coordinates",
    "solve_point_coincidence",
]


def test_all_matches_expected_names():
    assert operators.__all__ == EXPECTED_ALL


def test_every_public_name_resolves_to_a_callable():
    for name in EXPECTED_ALL:
        assert callable(getattr(operators, name)), f"operators.{name} is not callable"


def test_from_operators_import_star_still_works():
    namespace: dict = {}
    exec("from opensim_models.operators import *", namespace)
    for name in EXPECTED_ALL:
        assert name in namespace
