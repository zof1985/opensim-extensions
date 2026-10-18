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
- [ ] S25 stub campione (tool Pylance MCP irraggiungibile; generatore stub alternativo da predisporre).
- [ ] S26 stub completi.

## Fase 3 – modello nativo
- [ ] S30 struttura fisica modulare (`core/`, `_internal/`, `viz/`); top-level esistente resta stabile finché la migrazione non è completa.
- [x] S30A `OpenSimModel` eredita direttamente da `opensim.Model`; 2 test nuovi mirati verdi.

## Note operative
- I wrapper sono temporaneamente nel package privato `_generated` per evitare collisioni con i wrapper legacy in `components`; vanno spostati/esposti quando la migrazione (S39) li sostituisce.
- I metodi setter generati chiamano il metodo SWIG non sovrascritto esplicitamente (`opensim.Body.set_mass(self, value)`) per evitare ricorsione.
