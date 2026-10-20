"""Public modules explicitly declare and expose their package API."""

import importlib
import pkgutil

import opensim_extensions


def public_modules():
    for info in pkgutil.walk_packages(
        opensim_extensions.__path__, "opensim_extensions."
    ):
        if any(part.startswith("_") for part in info.name.split(".")[1:]):
            continue
        yield info.name


def test_every_public_module_has_an_explicit_clean_all():
    for name in public_modules():
        module = importlib.import_module(name)
        assert hasattr(module, "__all__"), name
        names = list(module.__all__)
        assert len(names) == len(set(names)), name
        assert all(not item.startswith("_") for item in names), name
        assert all(hasattr(module, item) for item in names), name


def test_top_level_api_contains_only_requested_public_elements():
    assert set(opensim_extensions.__all__) == {
        "Framework",
        "Bike",
        "IRON_DENSITY_KG_M3",
        "User",
        "save_animation",
        "VTKViewer",
        "VTKVisualizer",
        "DEFAULT_SIZE",
        "VIEW_NAMES",
        "MotionData",
        "MotionPlayer",
        "PlayerWindow",
    }
