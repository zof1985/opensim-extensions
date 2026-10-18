"""Regole di esportazione del pacchetto (piano §5bis).

Verifica, per ogni modulo/sotto-pacchetto pubblico di ``opensim_models``:
``__all__`` presente, senza duplicati, senza nomi privati, con nomi esistenti;
che i nomi di ogni ``__all__`` siano raggiungibili dal livello più alto; che non
ci siano collisioni tra moduli aggregati.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import opensim_models

#: Violazioni preesistenti ancora da sanare (si svuota man mano che il piano avanza).
KNOWN_EXCEPTIONS: set[str] = set()


def _public_modules():
    for info in pkgutil.walk_packages(opensim_models.__path__, "opensim_models."):
        parts = info.name.split(".")
        if any(p.startswith("_") for p in parts[1:]):
            continue
        yield info.name


PUBLIC_MODULES = sorted(_public_modules())


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_public_module_has_clean_all(name):
    if name in KNOWN_EXCEPTIONS:
        pytest.skip("eccezione nota")
    module = importlib.import_module(name)
    assert hasattr(module, "__all__"), f"{name} non definisce __all__"
    exported = list(module.__all__)
    assert len(exported) == len(set(exported)), f"{name}: nomi duplicati in __all__"
    for item in exported:
        assert not item.startswith("_"), f"{name}: nome privato {item!r} in __all__"
        assert hasattr(module, item), f"{name}: {item!r} in __all__ ma non definito"


def test_top_level_all_is_aggregated_and_complete():
    top = opensim_models
    assert hasattr(top, "__all__"), "opensim_models non definisce __all__"
    names = list(top.__all__)
    assert len(names) == len(set(names)), "duplicati in opensim_models.__all__"
    for item in names:
        assert not item.startswith("_"), f"nome privato {item!r} in opensim_models.__all__"
        assert hasattr(top, item), f"{item!r} non raggiungibile da opensim_models"


def test_subpackage_exports_reachable_from_top_level():
    top = opensim_models
    for name in PUBLIC_MODULES:
        module = importlib.import_module(name)
        if not hasattr(module, "__path__"):
            continue  # solo i sotto-pacchetti sono aggregati dal livello superiore
        for item in getattr(module, "__all__", []):
            assert getattr(top, item, None) is getattr(module, item), (
                f"{name}.{item} non coincide con opensim_models.{item}"
            )


def test_no_private_names_star_exported():
    leaked = [n for n in getattr(opensim_models, "__all__", []) if n.startswith("_")]
    assert not leaked, leaked
