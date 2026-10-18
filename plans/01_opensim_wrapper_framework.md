# Piano: framework wrapper OpenSim (tutti i wrapper sono sottoclassi delle classi `opensim`)

## 1. Obiettivo

1. Per **ogni** oggetto OpenSim esiste un wrapper che ne è **sottoclasse** (`class Body(opensim.Body)`), con docstring in **stile NumPy**, setter `set_<prop>(...)` e property `obj.<prop>` visibili all'IDE (Pylance).
2. `OpenSimModel(opensim.Model)` espone tutte le funzionalità native, analizzabili dall'IDE.
3. `OpenSimModel.from_osim(file, meshes=...)` (classmethod) e `to_osim(file)` importano/esportano modelli `.osim` con le mesh collegate.

## 2. Vincoli decisi

- **Fuori scope: Moco.** Nessun wrapper per Moco*/opensim.moco; sarà un tool sviluppato successivamente sopra questo framework (generatori e schema devono restare estendibili a nuove aree).

- **Nessuna retrocompatibilità**: si può rompere l'API attuale (`.model`, `export`, `components.*` basati su composizione, `_ComponentWrapper`, `_registry` di ownership ecc.). Niente alias, shim o `DeprecationWarning`; il codice legacy viene riscritto o eliminato.
- **Ereditarietà ovunque**: nessun wrapper per composizione. Ogni wrapper eredita dalla classe nativa corrispondente e ne conserva intatta l'API.
- **Docstring NumPy** (sezioni `Parameters`, `Returns`, `Raises`, `Notes`, `See Also`, `Examples`), anche nel codice generato e nei test di qualità.
- **Generazione statica** (codice committato + test di freschezza, come `generate_user_code.py`), senza `__getattr__`/metaclassi: Pylance deve vedere ogni attributo.

## 3. Stato attuale (analisi)

| Aspetto | Situazione |
|---|---|
| `model.py` (3244 righe) | `OpenSimModel` è una facade per composizione: `self.model` + `self.state`, ~63 usi di `self.model`. Manca `from_osim`/`to_osim`; esistono `__init__(model_path)`, `from_step`, `copy`, `export`, `add_model/__add__`. |
| API di editing | ~40 `add_*`/`remove_*` pass-through a `operators/*` (test `test_model_wrapper_signatures.py`). |
| `components/__init__.py` (1547 righe) | ~20 wrapper per composizione (`Body, Joint, Coordinate, Marker, Muscle, Force, Constraint, ...`) + `Box/Cylinder/Screen` (da eliminare, vedi S37). |
| `models/user/user.py` | `User(OpenSimModel, _PostureMixin, _JointCenterMixin)`; mixin generati da `scripts/generate_user_code.py`: modello del pattern da generalizzare. |
| Binding OpenSim 4.6 (SWIG) | 724 classi, **nessun `.pyi`/`py.typed`**: l'IDE non vede firme né docstring native. |

Dimensione dell'albero delle classi (misurata su `opensim` 4.6):
`OpenSimObject` 405 · `Component` 182 · `ModelComponent` 152 · `Force` 42 · `Function` 20 · `Actuator` 19 · `Analysis` 15 · `Joint` 13 · `Geometry` 12 · `Muscle` 8 · `Probe` 8 · `Point` 8 · `Constraint` 7 · `ContactGeometry` 7 · `Frame` 7 · `Controller` 6 · `Tool` 6 · `WrapObject` 5 (le categorie si sovrappongono).

## 4. Problemi tecnici chiave (da risolvere per primi)

1. **Oggetti restituiti dal modello sono nativi.** `model.getBodySet().get(0)` ritorna un `opensim.Body`, non il nostro `Body`. Soluzioni da validare in uno spike (Fase 0):
   - *Re-class in place*: assegnare `obj.__class__ = Wrapper` sul proxy SWIG (possibile se il wrapper non aggiunge `__slots__`/stato di istanza incompatibile; lo stato aggiuntivo va tenuto in attributi dinamici o derivato dalle `Property`).
   - Override nei wrapper dei getter di collezioni (`getBodySet`, `getJointSet`, `get_<x>`, `updComponent`, `getComponent`, `get<Type>`...) che restituiscono il wrapper tramite re-class; `clone()`/`__copy__` ridefiniti per restituire il tipo Python.
   - Fallback se il re-class non è affidabile: factory `wrap(obj)` che costruisce il wrapper dal clone, con regole esplicite di ownership.
2. **Ownership SWIG**: `addBody`, `addJoint`, `addComponent` ecc. prendono ownership (`__disown__`); i wrapper creati in Python devono rispettarlo.
3. **Polimorfismo C++**: `Model.getComponent(path)` ritorna il tipo dinamico nativo → serve una mappa `classe nativa → wrapper` (generata) per il re-class.
4. **Conflitti di nome** tra property Python generate (`name`, `mass`, `parent`...) e metodi nativi (`getName`, ...): le property nuove non devono sovrascrivere nomi nativi esistenti; tabella dei conflitti generata e verificata.
5. **Stato di `OpenSimModel`** (`state`, `reinitialize`, `structural_change`): vive in attributi dell'istanza Python.

## 5. Decisioni di design

- **D1** `OpenSimModel(opensim.Model)`; `self.model` rimosso: tutto `self.model.X` → `self.X`.
- **D2** Wrapper generati da schema di introspezione (JSON versionato) → rendering in `wrappers/<area>/` (struttura e regole di esportazione in §5bis). **`User` è l'unica sottoclasse di `OpenSimModel` mantenuta**; `Box`, `Cylinder`, `Screen` (e `_primitives.py`, `components/assets/meshes`) vengono eliminati per ora, e reintrodotti in futuro solo se serviranno (come sottoclassi dei wrapper generati).
- **D3** Per ogni `Property` nativa: `@property <name>` (getter tipizzato; `Vec3`→`np.ndarray`, liste→`list`) e `set_<name>(value, *, reinitialize=False) -> Self`; per le proprietà-lista anche `append_<name>`, `clear_<name>`, `set_<name>_at`. Metodi nativi non wrappati restano accessibili per eredità (+ stub `.pyi`).
- **D4** Docstring NumPy ricavate da: Doxygen opensim-core 4.6 (se reperibile) → `__doc__` SWIG → `Property.getComment()` → fallback (nome + firma). Qualità verificata da test.
- **D5** Stub `.pyi` parziali generati per tutte le classi native pubbliche non-Moco trovate dallo schema (`typings/opensim/`); i metodi SWIG senza firme introspezionabili usano inizialmente `*args/**kwargs: Any` con docstring conservate e vengono raffinati a gruppi per area. Il gate finale richiede firme esplicite sulle API più usate di `OpenSimModel`.

## 5bis. Struttura modulare del pacchetto e regole di esportazione

### Layout target (`src/opensim_models/`)

```text
opensim_models/
├── __init__.py            # solo star-import dei sotto-pacchetti pubblici + __all__ aggregato
├── py.typed
├── _internal/             # PRIVATO: dll/PATH, naming, geometria VTK, import CAD, utilità condivise
├── core/                  # OpenSimModel(opensim.Model), import_opensim, stato, I/O (from_osim/to_osim)
├── wrappers/              # PUBBLICO, GENERATO: un sotto-pacchetto per area OpenSim
│   ├── __init__.py
│   ├── base/              # Object, Component, ModelComponent, Set/liste
│   ├── frames/            # Frame, PhysicalOffsetFrame, Ground, Station, Point, Marker, Body
│   ├── joints/            # Joint (13) + Coordinate
│   ├── constraints/
│   ├── forces/            # Force, ExternalForce, contact forces...
│   ├── actuators/         # Actuator, Muscle (+ famiglie)
│   ├── controllers/
│   ├── probes/
│   ├── geometry/          # Geometry, Appearance, ContactGeometry, WrapObject, GeometryPath/PathPoint
│   ├── functions/
│   ├── data/              # Storage, Table, TimeSeriesTable, StatesTrajectory, helper numpy
│   ├── tools/             # AbstractTool/Tool (IK, ID, RRA, CMC, Forward, Scale)
│   ├── analyses/
│   └── misc/              # coda lunga (Moco escluso)
├── operators/             # add_*/remove_*/solve_* (già modulare per tema)
├── framework/             # OpenSimFramework, Coupling, composito, sync stato
├── models/                # modelli specifici
│   └── user/              # User (unica sottoclasse di OpenSimModel mantenuta)
└── viz/                   # viewer/player (VTK/Tk); implementazione in viz/_impl, API pubblica minima
```

Principi: un modulo = una responsabilità; moduli ≤ ~400–500 righe (i grandi file attuali — `model.py` 3244, `user.py` 1062, `_posture_*` — vengono spezzati, p.es. `OpenSimModel` composto da mixin per tema in `core/_mixins/` o moduli `core/state.py`, `core/io.py`, `core/editing.py`, `core/viewing.py`, ereditati dalla classe finale); gerarchia di dipendenze **a strati, senza cicli**: `_internal` ← `core` ← `wrappers` ← `operators` ← `framework`/`models`/`viz`.

### Regole di esportazione (obbligatorie, verificate da test)

1. **Ogni modulo pubblico** (`body.py`, `joints.py`, …) definisce `__all__` con **solo** i nomi pubblici (senza underscore) che vuole esportare; i nomi privati (`_x`) non compaiono mai in `__all__`.
2. **Ogni `__init__.py` di sotto-pacchetto** fa il richiamo con `*` dei propri moduli pubblici e aggrega `__all__`:
   ```python
   from . import body, coordinate
   from .body import *
   from .coordinate import *

   __all__ = [*body.__all__, *coordinate.__all__]
   ```
   (l'import del modulo come nome è necessario per poter leggere `module.__all__`; i nomi dei moduli stessi non vanno in `__all__`).
3. **`opensim_models/__init__.py`** ripete lo schema sui sotto-pacchetti pubblici (`core`, `wrappers`, `operators`, `framework`, `models`, `viz`): ogni oggetto pubblico è importabile dal livello più alto (`from opensim_models import OpenSimModel, Body, add_body, OpenSimFramework, User`).
4. **Elementi privati**: moduli/sotto-pacchetti con prefisso `_` (`_internal`, `_impl`, `_mixins`), funzioni e classi `_x` **non** sono mai importati con `*`, non entrano in `__all__` e non sono esposti al livello alto; si importano solo da codice interno, con import espliciti.
5. **Nessun nome duplicato** tra i moduli aggregati allo stesso livello (collisione = errore del test); i wrapper omonimi delle classi native (`Body`, `Joint`…) vivono nel namespace `opensim_models` e **non** sostituiscono `opensim.*` (l'utente li usa come `opensim_models.Body` o `import opensim_models as om`); nome non ambiguo documentato in README.
6. **Nessuna importazione circolare**: ogni modulo importa solo da strati inferiori; le annotazioni di tipo cross-strato usano `TYPE_CHECKING`.
7. Il codice **generato** (`wrappers/**`) emette automaticamente `__all__` e gli `__init__.py` con star-import secondo lo schema sopra (il generatore scrive anche i file `__init__` e verifica che non rimangano file orfani).
8. Stub e IDE: poiché `__all__` è statico e letterale (liste di stringhe/concatenazioni risolvibili da Pylance), l'autocomplete dei nomi star-importati funziona; `[*mod.__all__]` è supportato da Pylance, ma se un test di verifica (S06) mostrasse nomi non risolti si ricade sulla forma letterale generata dallo script.

### Test di struttura (`tests/test_package_structure.py`)
- Per ogni modulo/sotto-pacchetto pubblico: `__all__` esiste, non ha duplicati, ogni nome esiste, nessun nome inizia con `_`.
- Ogni nome pubblico definito a livello di modulo (classi/funzioni/costanti non `_`) è in `__all__` (o in un elenco esplicito di eccezioni), per evitare omissioni.
- Tutti i nomi di ogni `__all__` sono raggiungibili da `opensim_models` e corrispondono allo stesso oggetto.
- Nessun nome privato (`_x`) esposto in `dir(opensim_models)` oltre ai dunder e ai nomi di sotto-pacchetti.
- Nessuna collisione di nomi tra moduli aggregati; nessun ciclo di import (import di ogni modulo in un processo pulito); tempo di import entro un budget (es. < 2 s senza `opensim` caricato dal test, misurato) con eventuale caricamento lazy dei soli sotto-pacchetti rari (`tools`, `analyses`, `misc`) tramite `__getattr__` di modulo **solo se** il budget è superato.

## 6. Fasi (panoramica)

Ogni fase è scomposta in **step brevi e interrompibili** nel §7: lavorare e committare per step, non per fase.

### Fase 0 – Baseline e spike tecnico (1 gg)
- [ ] `pytest` completo come riferimento; branch `feature/inheritance-wrappers`.
- [ ] **Spike**: `class Body(opensim.Body)` + re-class di un body preso da un modello caricato; verificare getter/setter, `clone()`, ownership con `addBody`, garbage collection, assenza di crash su `initSystem`.
- [ ] Tabella conflitti nomi tra API attuale/property previste e `dir(opensim.<Classe>)`.
- [ ] Decisione finale su strategia §4.1 (re-class vs factory) → aggiornare questo piano.

### Fase 1 – Introspezione e schema (1–2 gg)
- [ ] `scripts/introspect_opensim.py` → `scripts/_opensim_schema.json`: per ogni classe base/gerarchia, proprietà (tipo, list/non-list, default, commento), metodi pubblici con overload, socket/input/output, flag astratta/concreta.
- [ ] Raccolta docstring (D4) e normalizzazione in formato NumPy.
- [ ] Test: schema coerente (classi attese, nessuna proprietà senza tipo).

### Fase 2 – Infrastruttura di generazione e stub (2 gg)
- [ ] `scripts/generate_wrappers.py` con `render_*()` pure e `__main__` (schema di `generate_user_code.py`); template per classe, property, `set_*`, liste, docstring NumPy.
- [ ] Mappa generata `classe nativa → wrapper` e funzione di re-class (§4.1).
- [ ] `scripts/generate_opensim_stubs.py` → `typings/opensim/*.pyi` + configurazione `stubPath`.
- [ ] Test di freschezza (`tests/test_wrappers_generated_are_fresh.py`) e test di qualità docstring (sezioni NumPy, `Parameters` coerenti con la firma).

### Fase 3 – `OpenSimModel(opensim.Model)` (3–4 gg, fase critica)
- [ ] Riscrittura di `model.py`: sottoclasse di `opensim.Model`, `self.model` eliminato, `state`/`reinitialize`/`structural_change`/`settle_under_gravity`/`update_state` mantenuti come metodi dell'istanza.
- [ ] `copy()`/`clone()` che restituiscono `type(self)`; `__add__/add_model/remove_model`, `from_step`, `scale_bodies`, `show/player` adattati.
- [ ] `operators/*`, `_gui/*`, `_animation.py`, `User` riscritti per operare direttamente su `OpenSimModel`/wrapper ereditari (senza `_ComponentWrapper`). `Box/Cylinder/Screen` eliminati (S37).
- [ ] Eliminare `_registry.py` e `components/_ComponentWrapper` se non più necessari.
- [ ] Test: `isinstance(m, opensim.Model)`, `getName()`, `initSystem()`, `printToXML()`, regressione sui test esistenti riadattati.

### Fase 4 – `from_osim` / `to_osim` (1–1.5 gg)
- [ ] `@classmethod from_osim(cls, file: Path, meshes: Path | Sequence[Path] | None = None, *, copy_meshes=False) -> Self`: registra le directory mesh **prima** del caricamento (`ModelVisualizer.addDirToGeometrySearchPaths`); default = cartella del file + `Geometry/`; errori chiari per mesh mancanti.
- [ ] `to_osim(self, file: Path, *, with_meshes=True, mesh_dir="Geometry", overwrite=True) -> Path`: `printToXML` + copia mesh referenziate con percorsi relativi coerenti.
- [ ] `__init__(model_path)` ed `export()` rimossi/sostituiti da questi due metodi.
- [ ] Test: round-trip (corpi/giunti/mesh uguali, mesh su disco), directory mesh custom, mesh mancanti, `User` e un `OpenSimModel` costruito da `.osim` di test (cls-aware).

### Fasi 5–10 – Generazione di tutti gli oggetti, per ondate
Ogni ondata: (a) estendere lo schema, (b) generare wrapper + stub, (c) test di freschezza/qualità docstring, (d) smoke test per classe concreta (istanzia, legge tutte le property, round-trip `set_*` ↔ property, re-class da modello), (e) esposizione in `opensim_models.<area>` con `__all__`.

| Fase | Area | Contenuto (classi approx.) | Stima |
|---|---|---|---|
| **5** | **Core modello** | `Object`, `Component`, `ModelComponent`, `Model`(→F3), `Body`, `Coordinate`, `Marker`, `Station`, `Point`, `Frame` (+`PhysicalOffsetFrame`, `Ground`), Set/liste (`BodySet`, `JointSet`, `MarkerSet`, ...), `Joint` (13), `Constraint` (7) | 3–4 gg |
| **6** | **Forze e attuatori** | `Force` (42), `Actuator` (19), `Muscle` (8: Millard, Thelen, DeGrooteFregly, ...), `Controller` (6), `Probe` (8), `ExternalLoads`/`ExternalForce` | 3 gg |
| **7** | **Geometria e contatto** | `Geometry` (12: Mesh, Brick, Sphere, ...), `Appearance`, `ContactGeometry` (7), `WrapObject` (5), `GeometryPath`/`PathPoint`/`PathSpring`, `ContactHalfSpace/Sphere/Mesh` | 2 gg |
| **8** | **Funzioni e dati** | `Function` (20: Constant, LinearFunction, PiecewiseLinear, Spline, MultiplierFunction, ...), `Storage`, `TimeSeriesTable`, `Table`, `StatesTrajectory`, `Vec3/Mat33/Transform` helper (conversioni numpy) | 2 gg |
| **9** | **Strumenti di analisi** | `Tool`/`AbstractTool` (IK, ID, RRA, CMC, Forward, Scale), `Analysis` (15), `Reporter`, `InverseKinematicsTool`, `MarkerPlacer`, `ModelScaler` | 2–3 gg |
| **10** | **Coda lunga** | Tutte le classi rimanenti, **escluso Moco** (`Moco*`, `opensim.moco`): `examplecomponents`, `simulation`/`simbody` helper, utility; report di copertura automatico (classi native senza wrapper = 0) | 1–2 gg |

Regola di completezza: lo script di copertura elenca le classi native pubbliche prive di wrapper; la Fase 10 termina quando l'elenco è vuoto (o contiene solo esclusioni esplicite e motivate in `scripts/wrapper_exclusions.toml`, dove Moco è escluso per prefisso).

### Fase 11 – IDE-friendliness e qualità (1 gg)
- [ ] `py.typed`, `__all__`, `Self` nei setter.
- [ ] Verifica Pylance/pyright senza errori; 10 casi di hover/autocomplete campione.
- [ ] Test automatici: ogni property/metodo pubblico ha docstring NumPy; ogni `set_x` ha la property `x`.

### Fase 12 – Documentazione e rilascio (1 gg)
- [ ] README riscritto (nuovo modello a oggetti, `from_osim/to_osim`, guida alle property/`set_*`), bump major (`pyproject.toml`), `package-data` per stub/schema.
- [ ] Generatori e controlli di freschezza nella suite di test.

## 7. Step di sviluppo interrompibili

### 7.1 Protocollo di ripresa (valido per ogni step)

- **Dimensione**: ogni step è ≤ ~2–3 ore, ha un risultato verificabile e termina con **un commit** (`step Sxx: <titolo>`) a suite di test verde (o con i soli test dello step, se indicato).
- **Atomicità**: uno step non lascia il repository in stato rotto; se interrotto a metà, `git restore`/`git stash` e si riparte dall'inizio dello step.
- **Tracciamento**: `plans/PROGRESS.md` contiene una riga per step (`[ ] S00`, `[x] S00 – commit abc123 – note`). Si aggiorna **dentro** il commit dello step. Per riprendere: leggere `PROGRESS.md`, prendere il primo step non spuntato le cui dipendenze sono completate.
- **Idempotenza**: i generatori sono rieseguibili (`python scripts/generate_*.py`) e producono output deterministico; lo step "genera" si può ripetere senza effetti collaterali.
- **Note di ripresa**: ogni step lascia in `PROGRESS.md` eventuali decisioni prese/aperte (es. esito dello spike) invece di tenerle solo in chat.
- **Formato**: `Sxx` – titolo · *Dip.*: dipendenze · *Output*: artefatti · *Verifica*: criterio di done.

### 7.2 Step

**Fase 0 – Baseline e spike**
- **S00** – Baseline: `pytest` completo, salvare esito in `PROGRESS.md`; creare branch e `plans/PROGRESS.md` con tutti gli step. · *Dip.*: – · *Verifica*: file creato, baseline registrata.
- **S01** – Spike re-class: script `scripts/spikes/reclass.py` con `class Body(opensim.Body)` e `obj.__class__ = Body` su un body caricato. · *Dip.*: S00 · *Verifica*: getter/setter e `clone()` funzionanti.
- **S02** – Spike ownership/GC: `addBody` con wrapper creato in Python, `del`/`gc.collect()`, `initSystem`, ciclo ripetuto 1000×. · *Dip.*: S01 · *Verifica*: nessun crash.
- **S03** – Spike polimorfismo: `model.getComponent(path)`/`getBodySet().get(i)` + re-class tramite mappa; decisione finale re-class vs factory scritta in `PROGRESS.md` e nel §4.1. · *Dip.*: S02 · *Verifica*: decisione documentata.
- **S04** – Tabella conflitti nomi (script che confronta property previste con `dir(opensim.<Classe>)`), output `scripts/_name_conflicts.json`. · *Dip.*: S00.
- **S05** – Progettazione struttura pacchetto (§5bis): mappa «modulo attuale → modulo target» (`model.py`, `components/`, `operators/`, `_gui/`, `models/user/`, file privati) in `plans/PACKAGE_LAYOUT.md`, con elenco dei nomi pubblici/privati attuali e verifica degli strati (nessun ciclo). · *Dip.*: S00 · *Verifica*: documento approvato.
- **S06** – Infrastruttura di esportazione: `tests/test_package_structure.py` (regole §5bis) + piccolo script `scripts/check_exports.py`; spike Pylance su `__all__ = [*mod.__all__]` con star-import (autocomplete e hover OK, oppure scelta della forma letterale). · *Dip.*: S05 · *Verifica*: test eseguibile sulla struttura attuale (può segnalare le violazioni preesistenti, elencate come to-do).

**Fase 1 – Schema**
- **S10** – `introspect_opensim.py`: elenco classi + gerarchia + flag astratta/concreta → JSON. · *Dip.*: S00 · *Verifica*: 639 classi pubbliche rilevate da OpenSim 4.6, Moco escluso per prefisso (numero version-specific).
- **S11** – Estensione schema: proprietà (nome, tipo, list, default, commento). · *Dip.*: S10 · *Verifica*: `Body.mass`, `PinJoint` presenti.
- **S12** – Estensione schema: metodi pubblici con overload, socket/input/output. · *Dip.*: S11.
- **S13** – Raccolta docstring (Doxygen → SWIG → `Property` comment → fallback) e normalizzazione NumPy nello schema. · *Dip.*: S12 · *Verifica*: report % di classi con docstring.
- **S14** – Test schema (`tests/test_schema.py`). · *Dip.*: S13.

**Fase 2 – Generatore e stub**
- **S20** – Generatore: template di **una** classe (`Body`) con docstring NumPy; l'output viene tenuto in `src/opensim_models/_generated/` (staging privato, per non collidere con `components.Body` legacy) e spostato in `wrappers/<area>/` alla migrazione S39. Genera `__all__` e `__init__.py` (S06). · *Dip.*: S14, S03, S06.
- **S21** – Template property per proprietà scalari e oggetto non-lista (getter tipizzato); tipi OpenSim sono mantenuti come tipi nativi per evitare conversioni implicite con perdita di semantica. Proprietà-lista e conversioni NumPy vengono abilitate solo dopo test mirati per tipo. · *Dip.*: S20.
- **S22** – Template `set_*` letterali; il setter chiama il metodo SWIG non sovrascritto tramite la classe nativa (evita ricorsione). Nessun `reinitialize` automatico: un componente re-class non conserva in modo affidabile il proprietario modello e il chiamante re-inizializza il modello esplicitamente se necessario. Proprietà-lista escluse finché non sono implementati/testati gli accessor indicizzati (`append_`, `clear_`, `set_<name>_at`). · *Dip.*: S21, S04.
- **S23** – Mappa `classe nativa → wrapper` + funzione di re-class (secondo S03). · *Dip.*: S22.
- **S24** – Test di freschezza dei generati + test qualità docstring NumPy. · *Dip.*: S23.
- **S25** – `generate_opensim_stubs.py`: stub parziali di 5 classi campione (`Model`, `Body`, `Joint`, `Coordinate`, `Force`) + `stubPath` e freshness test. · *Dip.*: S14.
- **S26** – Stub per tutte le classi native pubbliche non-Moco; verifica Pylance/pyright e affina le firme delle API centrali quando disponibili. · *Dip.*: S25.

**Fase 3 – `OpenSimModel(opensim.Model)`** (ogni step lascia i test del suo ambito verdi)
- **S30** – Creazione scheletro pacchetto (§5bis): cartelle `_internal/`, `core/`, `wrappers/`, `framework/`, `viz/` con `__init__.py` secondo le regole di esportazione; spostamento dei file privati in `_internal/` (`_geometry`, `_cad_import`, helper DLL/naming) con import interni aggiornati; `opensim_models/__init__.py` con star-import e `__all__` aggregato; `test_package_structure` verde sulle parti già migrate. · *Dip.*: S06, S03.
- **S30A** – Nuova classe minima `OpenSimModel(opensim.Model)` (`__init__`, `import_opensim`, `state`) in `model.py` durante la transizione, senza rimuovere altro; successivamente trasferita in `core/` con S30. · *Dip.*: S03.
- **S31** – Migrazione lettura: property `bodies/joints/...`, `body()/joint()/...` su `self`; `core/` suddiviso per tema (`state`, `io`, `editing`, `viewing`) invece di un unico `model.py` da 3000+ righe. · *Dip.*: S30A, S23.
- **S32** – Migrazione `update_state`, `reinitialize`, `structural_change`, `settle_under_gravity`. · *Dip.*: S31.
- **S33** – Migrazione `copy()/clone()` → `type(self)`; `__add__/add_model/remove_model`. · *Dip.*: S32.
- **S34** – Migrazione `operators/*` (un modulo per commit: `bodies`, `joints`, `frames`, `markers` / `forces`, `constraints` / `contact`, `geometry`, `auxiliary`, `attachment` / `rotation`, `translation`, `solving`, `primitives`) – **un sotto-step per gruppo**, S34a–S34e. · *Dip.*: S33.
- **S35** – `from_step`, `scale_bodies`, `_cad_import`. · *Dip.*: S34.
- **S36** – `_gui/*`, `_animation.py`, `show/player`. · *Dip.*: S34.
- **S37** – **Eliminazione** di `components/box.py`, `cylinder.py`, `screen.py`, `components/assets/`, `_primitives.py` (se usato solo da essi), dei relativi test (`test_box.py`, `test_cylinder.py`, `test_screen.py`, `test_primitives` se presente) e dei riferimenti in docstring/`__init__`/`operators/attachment.py`/`model.py`/`pyproject.toml` (`package-data`). Verificare con `grep` che `User` non ne dipenda (altrimenti spostare il minimo indispensabile). `User` resta l'unica sottoclasse di `OpenSimModel`. · *Dip.*: S34 · *Verifica*: nessun residuo, suite verde.
- **S38** – `User` e mixin generati adattati. · *Dip.*: S34.
- **S39** – Rimozione di `self.model`, `_registry.py`, `_ComponentWrapper`; `grep` senza residui; suite completa verde. · *Dip.*: S35–S38.

**Fase 4 – I/O osim**
- **S40** – `from_osim` (registrazione directory mesh prima del load, default cartella + `Geometry/`) + test base. · *Dip.*: S39.
- **S41** – Errori chiari per mesh mancanti + parametro `meshes` (path singolo/sequenza) + test. · *Dip.*: S40.
- **S42** – `to_osim` (`printToXML`, copia mesh, percorsi relativi) + test. · *Dip.*: S40.
- **S43** – Test round-trip incl. `User`; rimozione di `export()` e `__init__(model_path)`. · *Dip.*: S41, S42.

**Fasi 5–10 – Ondate di generazione.** Ogni ondata si spezza in step da 1–3 sotto-aree, ciascuno con: schema → generazione → smoke test (istanzia, legge property, round-trip `set_*`, re-class da modello) → esposizione in `__all__` → commit. Ordine delle ondate: S5x prima; S6x–S9x indipendenti (qualsiasi ordine); S10x per ultima.
- **S50** `Object`/`Component`/`ModelComponent` · **S51** `Body`, `Station`, `Point`, `Marker` · **S52** `Frame`/`PhysicalOffsetFrame`/`Ground` · **S53** `Coordinate` + Set/liste · **S54** `Joint` (13, 2 sotto-step: pin/slider/weld/ball… e custom/free/universal…) · **S55** `Constraint` (7).
- **S60** `Force` di base (`ForceAdapter`, `PrescribedForce`, `ExternalForce`…) · **S61** `Force` restanti (42 totali, 2–3 sotto-step per famiglia) · **S62** `Actuator` · **S63** `Muscle` · **S64** `Controller` · **S65** `Probe`.
- **S70** `Geometry`/`Appearance` · **S71** `ContactGeometry` + forze di contatto · **S72** `WrapObject` · **S73** `GeometryPath`/`PathPoint`/`PathSpring`.
- **S80** `Function` (20, 2 sotto-step) · **S81** `Storage`/`Table`/`TimeSeriesTable` · **S82** `StatesTrajectory` e helper `Vec3/Mat33/Transform`.
- **S90** `AbstractTool`/`Tool` base · **S91** IK/ID/Scale/MarkerPlacer · **S92** RRA/CMC/Forward · **S93** `Analysis` (15, 2 sotto-step) · **S94** `Reporter`.
- **S100** Script di copertura + `wrapper_exclusions.toml` (Moco per prefisso) · **S101–S10n** classi residue a gruppi di ~20 fino a copertura 100%.

**Fase 11 – Qualità**
- **S110** Verifica finale di `__all__`/star-import su tutto il pacchetto (`test_package_structure`, `scripts/check_exports.py` senza violazioni), `py.typed`. · **S111** Test docstring/`set_x`↔`x`. · **S112** Verifica Pylance/pyright e 10 casi hover/autocomplete (inclusi nomi importati da `opensim_models` di primo livello).

**Fase 12 – Rilascio**
- **S120** README riscritto. · **S121** bump versione, `package-data`. · **S122** Controlli di freschezza in CI e pulizia finale.

**Fase 13 – `OpenSimFramework`** (dopo la Fase 12; richiede S39 + S43 come minimo, vedi §7.5)
Principio: il framework è un contenitore di **modelli** (`OpenSimModel`) che mantengono identità separata (oggetto Python, nome, stato proprio), ma ogni simulazione/analisi opera su un **modello composito unitario** costruito dal framework. Il framework non è sottoclasse di `opensim.Model`, però ne espone l'API di simulazione (vedi S132–S135). Un `Body` (anche generato) si aggiunge a un **modello**, non direttamente al framework; i test usano `User` e modelli minimali `OpenSimModel` costruiti con `add_body`/`add_*_joint` al posto di un modello `Bike`, che non esiste ancora (era solo un esempio).

Architettura (da validare con lo spike S130):
- **Modello composito** (`F.composite`, sola lettura): un `opensim.Model` costruito dal framework con la logica già presente in `add_model`/`__add__` (clone dei membri, nomi dei componenti prefissati con il nome nel framework, es. `/bodyset/user/pelvis`), **ricostruito pigramente** quando un membro cambia struttura (flag "dirty" sui modelli, agganciato a `structural_change`/`reinitialize`). I modelli originali **non vengono modificati né fusi**.
- **Mappa di identità**: tabella `nome membro ↔ percorsi nel composito`, usata per tradurre percorsi (`F.path('user', '/jointset/hip_r')` → percorso nel composito) e per sincronizzare gli stati.
- **Sincronizzazione dello stato** bidirezionale tramite i valori di coordinate/velocità/attuatori per percorso: membro → composito prima di simulare/valutare; composito → membri dopo (`update_state`, `simulate`, `settle`). Un membro modificato da solo (`F.user.set_...`) viene propagato al composito alla prossima operazione del framework.
- **Accoppiamenti** (`couplings`): oggetti dichiarativi tra componenti di membri diversi, aggiunti **solo al composito** (non ai membri), persistiti nel framework e ricreati a ogni ricostruzione: `F.couple_weld(...)`, `F.couple_point(...)`, `F.couple_coordinate(...)`, `F.couple_joint(parent=('bike','/bodyset/seat'), child=('user','/bodyset/pelvis'), joint=...)`, `F.add_force(...)` tra elementi di modelli diversi. Un accoppiamento si rimuove senza toccare i modelli.

- **S130** – Spike del modello composito: costruire a mano da due modelli un composito con prefissi, un'unica `initSystem()`, forward dynamics di pochi step, sincronizzazione coordinate in entrambe le direzioni, accoppiamento con un `WeldConstraint`/`PointConstraint` tra body di due modelli diversi; misurare costo di ricostruzione. Decisione documentata in `PROGRESS.md` (e §7.5). · *Dip.*: S39 · *Verifica*: simulazione unitaria corretta, stati sincronizzati.
- **S131** – Contenitore `OpenSimFramework(dict[str, OpenSimModel])`: **sottoclasse di `dict`** che ammette solo chiavi `str` e valori `OpenSimModel`.
  - Inserimento: `F['user'] = U`, `F.add(model=U, name='user')` (alias di `__setitem__`, restituisce il modello), `F.update(...)`, `F.setdefault(...)`, `F |= {...}`, costruttore `OpenSimFramework(user=U, bike=B)`/`OpenSimFramework({...})`: tutti instradati su un'unica funzione di validazione, perché i metodi `dict` non si richiamano tra loro (vanno **sovrascritti tutti**: `__init__`, `__setitem__`, `update`, `setdefault`, `__ior__`, `copy`, `fromkeys`; `__delitem__`, `pop`, `popitem`, `clear` per gli hook di rimozione).
  - Validazione: chiave `str` e identificatore Python valido (serve per `F.user`), valore `OpenSimModel` (`TypeError` chiaro; per un `Body`/componente suggerisce di aggiungerlo prima a un modello), nessuna stessa istanza sotto due nomi, nomi riservati rifiutati (`keys`, `items`, `values`, `get`, `pop`, `update`, `composite`, `simulate`, ... = tutti i membri di `dict` e di `OpenSimFramework`).
  - Sostituire/rimuovere un modello (`F['user'] = U2`, `del F['user']`) segna il composito come da ricostruire e invalida/rimuove gli accoppiamenti che lo referenziano (con avviso esplicito).
  - Accesso: `F['user']` e `F.user` (`__getattr__` **solo** per le chiavi presenti, `__setattr__` per le chiavi → `F.user = U` equivale a `F['user'] = U`; `__delattr__` analogo; `__dir__` per l'autocomplete a runtime).
  - Limite noto di `dict` e mitigazione: le funzioni C che leggono il dict direttamente (`dict.update(F, ...)`, `{**x}` su altri dict, `copy.copy`, `pickle`) bypassano gli override; si documenta e si copre con test (`copy.copy`/`deepcopy` esplicitamente definiti: copia superficiale = stessi modelli, `deepcopy` = clona i modelli).
  - Ordine di inserimento = ordine di broadcast. Docstring NumPy. · *Dip.*: S39 · *Verifica*: test di tutte le vie di inserimento/rimozione (valide e non valide), `isinstance(F, dict)`, `F.user is F['user']`.
- **S132** – Gestione del composito: costruzione pigra, flag dirty (hook su `structural_change`/`reinitialize` dei membri), tabella di identità, `F.composite`, `F.path(model, component_path)` e inverso. · *Dip.*: S130, S131, S33.
- **S133** – Sincronizzazione dello stato e **broadcast di stato**: `F.update_state()`, `F.reinitialize()`, `F.initSystem()`, `F.realize(stage)` agiscono sul composito (quindi su tutti i modelli, con accoppiamenti) e riscrivono lo stato nei membri; `F['user'].update_state()`/`F.user.update_state()` aggiornano **solo** quel modello, e il composito se ne accorge alla prossima operazione. Opzioni `only=[...]`/`exclude=[...]` e errori riportati con il nome del modello. · *Dip.*: S132, S32.
- **S134** – **API di simulazione unitaria**: `F.simulate(...)` / `F.forward(t_end, ...)` (Manager/integratore sul composito), `F.settle_under_gravity(...)`, `F.set_gravity`, `F.get_state()`/`F.state`, controller e forze aggiunti a livello di framework, accesso a coordinate/forze/output di qualunque membro tramite percorsi (`F.coordinate('user', 'hip_flexion_r')`); i risultati (traiettorie) sono riscritti sui membri e salvabili per membro o combinati (`TimeSeriesTable` del composito). L'API replica i nomi OpenSim di uso comune del modello (`initSystem`, `realizeVelocity`, `getWorkingState`, `equilibrateMuscles`...) con docstring NumPy; ciò che non è replicato è raggiungibile da `F.composite`. · *Dip.*: S133.
- **S135** – **Accoppiamenti**: classe `Coupling` (dataclass NumPy-doc) e metodi `F.couple_weld/point/coordinate/joint`, `F.add_force/add_constraint/add_controller` a livello framework con riferimenti `(nome_modello, percorso)`; elenco/rimozione (`F.couplings`, `F.uncouple(id)`); validazione dei riferimenti; test: due modelli accoppiati si muovono insieme in simulazione e restano indipendenti quando l'accoppiamento è rimosso. · *Dip.*: S134.
- **S135A** – **`F.merge_to_model(name=None) -> OpenSimModel`**: restituisce un `OpenSimModel` **indipendente** (nuova istanza, della classe base `OpenSimModel`) che contiene tutti i componenti di tutti i membri e **tutte le relazioni** tra essi: giunti, vincoli, forze, controller, marker, contact geometry, mesh/directory di geometria (unione, senza collisioni di nomi di file), gravità, e gli accoppiamenti del framework (S135) tradotti in componenti nativi (weld/point constraint, giunti, forze) con i percorsi risolti. Si basa sul composito (S132) ma ne restituisce una **copia profonda** non legata al framework né ai membri: modificare `K` non cambia `F` né i modelli e viceversa.
  - Nomi: prefissi per membro come nel composito (`<nome>/…`), con mappa `K` ↔ membri (opzione `return_mapping=True` → dizionario `percorso_in_K → (nome_modello, percorso_originale)`).
  - Stato: coordinate/velocità correnti dei membri copiati in `K` (posture coerenti con `F`); `K.initSystem()` pronto.
  - Opzioni: `include_couplings=True` (se `False`, i membri sono uniti senza relazioni inter-modello), `only=[...]`/`exclude=[...]` per unire un sottoinsieme.
  - Verifica equivalenza: simulazione di `K` e di `F.simulate` con stesse condizioni iniziali producono traiettorie equivalenti (tolleranza numerica); `K.to_osim(file)` ricarica correttamente con le mesh (`from_osim`); i sottomodelli di `K` conservano massa totale, numero di corpi/giunti/vincoli = somma dei membri + accoppiamenti. Sostituisce/estende `__add__`/`add_model` (che restano come caso a due modelli, riscritti sopra la stessa logica). · *Dip.*: S135, S43 · *Verifica*: test di equivalenza, indipendenza di `K` da `F`, round-trip su file. `VTKVisualizer`/`MotionData`/`PlayerWindow` accettano una lista di `(nome, modello)`; un actor group per modello, tooltip `"<nome_modello>-<componente>"`; `OpenSimModel.show()` diventa il caso a un solo elemento. · *Dip.*: S36, S131 · *Verifica*: rendering offscreen con 2 modelli.
- **S137** – `F.show(...)`: finestra unica con tutti i modelli; refresh coerente con `F.update_state()` e `F[name].update_state()` (solo i gruppi interessati); modelli/accoppiamenti aggiunti o rimossi a finestra aperta aggiornano la scena; i modelli accoppiati appaiono nella posizione risultante dal composito. · *Dip.*: S136, S133.
- **S138** – Motion multi-modello: `F.show(motion=...)` con un motion per modello o una traiettoria del composito (`F.simulate` → animazione diretta), timeline comune documentata, export animazione. · *Dip.*: S137, S134.
- **S139** – Posizionamento iniziale in scena: `F.place(name, translation, rotation)` (modifica la posa iniziale del membro nel composito via giunto di base/offset del ground, senza alterare l'identità del modello) e persistenza: `F.to_osim(directory)` → `<nome>.osim` per modello con mesh + `framework.json` (nomi, posizionamenti, accoppiamenti serializzati, classi Python); `OpenSimFramework.from_osim(directory)`; opzione `F.composite.printToXML` per esportare anche il modello unitario. · *Dip.*: S135, S43.
- **S13A** – Test d'integrazione dell'esempio d'uso (`F.add(U, "user")`, `F.add(B, "bike")` con modelli reali o minimali), `F.show` offscreen, `F.user.update_state()` solo user, `F.update_state()` tutti, simulazione di due modelli accoppiati; README (sezione Framework). · *Dip.*: S138, S139.

> **Nota**: `Bike` non esiste ancora ed era solo un esempio: i test usano `User` e modelli minimali `OpenSimModel` costruiti con `add_body`/giunti.

### 7.3 Punti di sosta naturali (milestone utilizzabili)

| Dopo | Stato del progetto |
|---|---|
| S04 | Rischi tecnici chiariti, decisione re-class/factory presa |
| S26 | Schema + stub: IDE già mostra firme/docstring native |
| S39 | `OpenSimModel` sottoclasse di `opensim.Model`, suite verde |
| S43 | `from_osim`/`to_osim` completi (obiettivo 2 e 3 raggiunti) |
| S55 | Wrapper del core modello con property e `set_*` |
| S100 | Copertura completa |
| S133 | `OpenSimFramework` con modello composito e broadcast di stato (senza viewer) |
| S135 | Simulazione unitaria e accoppiamenti tra modelli |
| S135A | `F.merge_to_model()`: modello unico indipendente con tutte le relazioni |
| S137 | `F.show()` con più modelli in scena |
| S13A | Framework completo, con persistenza e test d'integrazione |

### 7.4 Dipendenze tra fasi e stima

`F0 → F1 → F2 → F3 → F4 → F5 → (F6, F7, F8, F9 in qualsiasi ordine) → F10 → F11 → F12 → F13`. Totale ≈ 27–35 giorni-persona (F13 ≈ 5–7 gg), circa 100–120 step da ≤ 3 ore.

### 7.5 Nota sul rapporto tra F13 e il resto

`OpenSimFramework` richiede solo F3 (modello come sottoclasse) e F4 (I/O); non dipende dai wrapper generati (F5–F10), ma ne beneficia (accoppiamenti tipizzati). Per volontà del progetto resta **ultima** fase: da anticipare solo su richiesta. In tal caso S130–S135 (composito, simulazione, accoppiamenti) possono partire subito dopo S43.

**Scelte di design del framework (da confermare)**
- `OpenSimFramework` **non** è sottoclasse di `opensim.Model`: è una sottoclasse di `dict[str, OpenSimModel]` (chiavi `str`, valori solo `OpenSimModel`) che, per simulare, produce un **modello composito** derivato (clone con prefissi) e ne espone l'API di simulazione. L'identità dei membri è sempre preservata (oggetti, nomi e stati propri).
- Alternativa scartata: modello unico fuso in modo definitivo (perde l'identità dei membri) e simulazione indipendente per membro (impossibile accoppiare).
- Gli accoppiamenti vivono nel framework, non nei membri; il composito li reinserisce a ogni ricostruzione.
- `F.<nome>` tramite `__getattr__` è l'unica eccezione al divieto di accesso dinamico (nomi scelti dall'utente); i membri propri della classe sono letterali per l'IDE e `F['nome']` resta l'accesso tipizzato.
- Rischi specifici: costo di ricostruzione del composito (mitigato da ricostruzione pigra e dirty flag), coerenza della sincronizzazione stati (test dedicati), prefissi dei percorsi che cambiano i nomi dei componenti nel composito (mitigato dalla mappa `F.path`).

## 8. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Re-class/ownership SWIG instabile | Spike in F0; test di stress GC e `initSystem`; strategia fallback (factory). |
| Conflitti di nome con l'API nativa | Tabella generata; il generatore non sovrascrive mai nomi nativi (salta o prefissa). |
| Volume di codice generato | Moduli per area, import lazy per le aree rare (Tool). |
| Docstring native mancanti/povere | Fallback gerarchico (D4) e report di qualità. |
| Classi non istanziabili o con costruttori speciali | Flag nello schema; smoke test limitati alle concrete. |

## 9. Criteri di accettazione

- Ogni wrapper è `issubclass` della corrispondente classe `opensim`; `isinstance(OpenSimModel(), opensim.Model)`.
- Oggetti ottenuti dal modello (`model.getBodySet().get(i)`, `getComponent`) sono istanze dei wrapper.
- `from_osim(f, meshes=...)` / `to_osim(f)` round-trip senza perdita di mesh, anche per `User`.
- Property + `set_*` per ogni proprietà nativa, con docstring NumPy, verificate da test automatici.
- Copertura classi native = 100% (meno esclusioni motivate); generatori con test di freschezza; suite verde.
- `OpenSimFramework`: `F['user'] = U` (o `F.add(model=U, name="user")`) accetta solo modelli (non `Body`); `F.show()` mostra tutti i modelli; `F.user.update_state()`/`F['user'].update_state()` aggiorna solo quel modello, `F.update_state()` li aggiorna tutti; i modelli restano oggetti separati.
- Simulazione unitaria: `F.simulate(...)` tratta tutti i modelli come un unico modello, rispetta gli accoppiamenti tra modelli diversi e riscrive i risultati sui membri; rimuovere un accoppiamento ripristina l'indipendenza.
- `K = F.merge_to_model()` restituisce un `OpenSimModel` indipendente con tutti gli elementi e le relazioni (accoppiamenti inclusi), equivalente in simulazione a `F`.
