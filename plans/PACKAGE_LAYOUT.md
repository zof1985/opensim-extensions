# PACKAGE_LAYOUT – mappa «modulo attuale → modulo target» (S05)

Riferimento: piano §5bis. Strati (nessun ciclo): `_internal` ← `core` ← `wrappers` ← `operators` ← `framework`/`models`/`viz`.

## Mappa

| Attuale | Target | Note |
|---|---|---|
| `model.py::import_opensim`, `_ensure_visualizer_dll_path` | `_internal/opensim_import.py` (+ `core` riesporta `import_opensim`, pubblico) | `import_opensim` è pubblico (`__all__`), il resto privato |
| `model.py::_register_geometry_search_path`, `_unique_*`, `_patch_sockets` | `_internal/naming.py`, `_internal/geometry_paths.py` | privati |
| `model.py::OpenSimModel` (3244 righe) | `core/` spezzata in mixin: `state.py` (state/update/reinitialize/structural_change/settle), `io.py` (from_osim/to_osim/mesh), `editing.py` (add_*/remove_*, delega a `operators`), `access.py` (bodies/joints/...), `viewing.py` (show/player/visualizer), `composition.py` (copy/add_model/__add__) + `model.py` (classe finale `OpenSimModel(opensim.Model)`) | Mixin privati in `core/_mixins/` |
| `_registry.py` | eliminato (S39) | sostituito da re-class |
| `_geometry.py`, `_cad_import.py`, `_primitives.py` | `_internal/` (`_primitives` eliminato con S37 se orfano) | privati |
| `_animation.py::save_animation` | `viz/animation.py` | pubblico |
| `_gui/*` | `viz/_impl/*` (privato) + `viz/__init__` con API minima | |
| `components/__init__.py` (`_ComponentWrapper`, `Body`, `Joint`, ...) | `wrappers/<area>/` generati (S20+); eliminato alla fine di S39 | |
| `components/box.py`, `cylinder.py`, `screen.py`, `assets/` | eliminati (S37) | |
| `operators/*` | `operators/*` invariato nei temi; import da `wrappers` invece di `components`; `_shared`, `_spatial` privati | già con `__all__` letterale nel `__init__` |
| `models/user/*` | `models/user/*` invariato; `_posture_generated.py`/`_joint_centers_generated.py` restano privati | `User` unica sottoclasse |
| (nuovo) | `framework/` (S130+) | `OpenSimFramework`, `Coupling` |

## Cicli preesistenti da risolvere
- `operators` ↔ `components` (components.box importa operators; operators importa components) → scompare con S37.
- `components` → `model` (`OpenSimModel`) e `model` → `components` (annotazioni, import locali) → con `core` sotto `wrappers` e `TYPE_CHECKING`.
- `operators._spatial`/`rotation`/`translation` importano `..model` → ok (core sotto operators).

## Stato export attuale (da correggere)
- Top-level `__init__` usa star-import ma `model.py` ha `__all__ = ["import_opensim", "OpenSimModel"]`, `components/__init__.py` ha `__all__` e `operators/__init__.py` ha `__all__` letterale; `models/__init__.py` e `models/user/__init__.py` ok.
- `_animation.save_animation` importato esplicitamente (non in `__all__` aggregato).
- `__init__.py` principale non definisce `__all__` aggregato.
