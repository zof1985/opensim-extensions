# PROGRESS – piano `01_opensim_wrapper_framework.md`

Per riprendere: prendere il primo step non spuntato con dipendenze completate. Un commit per step (`step Sxx: titolo`).

Esecuzione test (Windows, tmp di pytest non accessibile):
`$env:PYTEST_DEBUG_TEMPROOT="$env:TEMP\pt"; mkdir $env:PYTEST_DEBUG_TEMPROOT; python -m pytest -q -p no:cacheprovider`

## Baseline (S00)
- 530 test passati in ~286 s su `main` (OpenSim 4.6, Python 3.13).

## Decisioni / note
- **S01–S03 (spike, `scripts/spikes/`)**: strategia *re-class in place* VALIDATA.
  - `obj.__class__ = Wrapper` su proxy SWIG funziona (getter/setter, `initSystem` ok); 300 cicli add/del/gc senza crash.
  - Ogni `get(i)`/`getComponent` crea un **nuovo proxy** (identità Python non conservata, attributi Python aggiunti si perdono) → i wrapper devono essere **senza stato di istanza** (tutto derivato dalle `Property`/modello); lo stato extra va nel modello (`OpenSimModel`) o in registri per path.
  - I getter ritornano la classe base (`Joint`, `Body`): serve downcast `getattr(opensim, obj.getConcreteClassName()).safeDownCast(obj)` e poi `__class__ = WRAPPERS[nome]` (mappa generata, S23).
  - `clone()` ritorna il tipo nativo; `Model(subclass_instance)` (copy-ctor) mantiene la sottoclasse → `copy()` di `OpenSimModel` via copy-ctor.
  - Fallback factory non necessario.

## Fase 0
- [x] S00 baseline + PROGRESS
- [x] S01 spike re-class
- [x] S02 spike ownership/GC
- [x] S03 spike polimorfismo + decisione
- [x] S04 tabella conflitti nomi (`scripts/check_name_conflicts.py`: nessun conflitto tra API attuale di OpenSimModel/User e opensim.Model)
- [x] S05 layout pacchetto (`plans/PACKAGE_LAYOUT.md`)
- [x] S06 `tests/test_package_structure.py` verde (26); `__all__` aggregato nel top-level e in `models/`. Pylance su `[*mod.__all__]` da verificare in S112 (fallback: forma letterale generata); `scripts/check_exports.py` sostituito dal test.

## Fase 1 – Schema
- [x] S10 classi e gerarchia: 639 classi pubbliche non-Moco in `scripts/_opensim_schema.json` (OpenSim 4.6).
- [x] S11 proprietà, tipi, lista e commenti; `Body.mass` verificata.
- [x] S12 metodi dichiarati e socket/input/output; `PinJoint` verificata.
- [x] S13 docstring disponibili da SWIG e commenti Property (Doxygen non presente nell'ambiente).
- [x] S14 `tests/test_schema.py`: 6 passati.

## Fase 2 – Generazione wrappers
- [x] S20-S22 generatore deterministico, wrapper `Body`, property getter e `set_*` per proprietà non-lista in staging privato `src/opensim_models/_generated/` (le liste attendono accessor dedicati; niente reinitialize automatico).
- [x] S23 registry concreto + downcast/re-class per `Body`.
- [x] S24 freschezza e test docstring/property; suite mirata 35 passati.
- [x] S25 stub parziali campione + `stubPath`/marker `partial`.
- [x] S26 stubs di 639 classi non-Moco in `typings/opensim/__init__.pyi`; rigenerazione, sintassi e copertura verificati. Firme native generiche `Any` perché SWIG non espone `__text_signature__`; verifica Pylance/pyright rimane da fare (Pylance MCP transport failure, pyright CLI assente).

## Fase 3 – modello nativo
- [ ] S30 struttura modulare in corso: creati i namespace `core/`, `wrappers/`, `framework/`, `viz/`; `__init__.py` top-level re-esporta i nomi pubblici con `*` e `__all__`; `_geometry.py` trasferito in `_internal/geometry.py`. `model.py`/`_cad_import.py` non sono ancora stati fisicamente suddivisi/spostati.
- [x] S30A `OpenSimModel` eredita direttamente da `opensim.Model`; tre test nuovi mirati passano (istanza, metodi nativi, copia indipendente).
- Test mirati dopo i cambi S30/S30A: `test_model_inheritance.py` + `test_package_structure.py`: 35 passati. I test delle classi legacy `Box/Cylinder/Screen` non sono gate (pianificata rimozione S37).
- [ ] S31 in corso: i 51 riferimenti `self.model.<native_api>` in `model.py` sono stati migrati a `self.<native_api>`; l'alias temporaneo `model` resta finché operatori e GUI non sono migrati.

## Fase 4 – I/O
- [x] S40 `OpenSimModel.from_osim(file, meshes=...)`; registra modello/Geometry/percorsi espliciti prima di caricare.
- [x] S41 validazione file e mesh mancanti / tipi errati.
- [x] S42 `to_osim(file, with_meshes=True, mesh_dir="Geometry")`; test scrittura e copia mesh.
- [ ] S43 round-trip `User` e rimozione completa di `export()`/`__init__(model_path)` da completare; il caricamento di modelli base è coperto da S40-S42.
- Test mirati I/O + ereditarietà + struttura: 40 passati.

## Note operative
- I wrapper sono temporaneamente nel package privato `_generated` per evitare collisioni con i wrapper legacy in `components`; vanno spostati/esposti quando la migrazione (S39) li sostituisce.
- I metodi setter generati chiamano il metodo SWIG non sovrascritto esplicitamente (`opensim.Body.set_mass(self, value)`) per evitare ricorsione.
