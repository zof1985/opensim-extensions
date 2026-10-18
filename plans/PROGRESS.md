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
- [ ] S04 tabella conflitti nomi
- [ ] S05 layout pacchetto
- [ ] S06 test struttura/esportazioni

## Fasi successive
Vedi piano §7.2 (S10…S13A). Aggiungere qui le righe man mano che gli step diventano attivi.
