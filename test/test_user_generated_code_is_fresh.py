"""User's posture setters/getters and joint-center properties are generated
from src/opensim_models/models/user/_posture_table.py and
_joint_center_table.py via scripts/generate_user_code.py. This test catches
the "edited the table, forgot to re-run the generator" mistake: the
committed _posture_generated.py/_joint_centers_generated.py must always
match what the generator would produce right now.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.generate_user_code import (
    JOINT_CENTERS_GENERATED_PATH,
    POSTURE_GENERATED_PATH,
    render_joint_centers_module,
    render_posture_module,
)


def test_posture_generated_file_matches_the_table():
    assert POSTURE_GENERATED_PATH.read_text() == render_posture_module()


def test_joint_centers_generated_file_matches_the_table():
    assert JOINT_CENTERS_GENERATED_PATH.read_text() == render_joint_centers_module()


def test_mixins_expose_exactly_the_attributes_the_tables_expect():
    from opensim_models.models.user._joint_center_table import JOINT_CENTER_ENTRIES
    from opensim_models.models.user._joint_centers_generated import _JointCenterMixin
    from opensim_models.models.user._posture_generated import _PostureMixin
    from opensim_models.models.user._posture_table import POSTURE_ENTRIES

    expected_posture_names = set()
    for entry in POSTURE_ENTRIES:
        expected_posture_names.add(f"set_{entry.attr_suffix}")
        expected_posture_names.add(entry.attr_suffix)
    actual_posture_names = {
        name for name in vars(_PostureMixin) if not name.startswith("__")
    }
    assert actual_posture_names == expected_posture_names

    expected_joint_center_names = {entry.attr_name for entry in JOINT_CENTER_ENTRIES}
    actual_joint_center_names = {
        name for name in vars(_JointCenterMixin) if not name.startswith("__")
    }
    assert actual_joint_center_names == expected_joint_center_names
