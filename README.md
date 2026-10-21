# opensim-extensions

`opensim-extensions` è un package Python che estende [OpenSim](https://opensim.stanford.edu/)
con modelli antropometrici, una bike da indoor cycling regolabile, composizione
di modelli e strumenti di visualizzazione ed esportazione.

## Funzionalità

- **`User`**: modello full-body OpenSim basato sul modello
  Rajagopal-Lai-Uhlrich 2023, scalato con riferimenti antropometrici ANSUR II.
  Espone misure, centri articolari, marker e setter espliciti per le misure e
  la postura.
- **`Bike`**: modello OpenSim di una bike regolabile, con fitting di un utente,
  pedali, manubrio, schermo, supporto e sella arrotondata e rastremata.
- **`Framework`**: contenitore nominativo di modelli OpenSim distinti, con
  snapshot composito, accoppiamenti e strumenti per simulare o creare un
  modello OpenSim indipendente.
- **Visualizzazione**: `VTKViewer` e `VTKVisualizer` mostrano i modelli; il
  player interattivo gestisce le motion `.mot`/`.sto`; l'esportazione salva
  snapshot SVG e animazioni MP4.

## Architettura

La struttura attuale del package separa modelli, composizione e
visualizzazione:

```text
src/opensim_extensions/
├── models/
│   ├── user/       User, riferimenti ANSUR II, scaling e asset anatomici
│   └── bike/       Bike e fitting cinematico dell'utente
├── bodies/         primitive geometriche e builder per corpi OpenSim
├── framework/      composizione di modelli e accoppiamenti
├── viewer/         scena VTK, player, lettura geometrie ed export SVG
├── animation.py    rendering off-screen delle motion in MP4
└── opensim.py      caricamento binding OpenSim e risoluzione mesh
```

`User` e `Bike` sono modelli nativi OpenSim. `Framework` li mantiene come
modelli distinti e costruisce uno snapshot composito quando serve simulare,
accoppiare componenti o visualizzare l'insieme. I vincoli registrati nel
framework appartengono allo snapshot, non modificano i modelli sorgente.
`VTKViewer` è la facciata pubblica per modelli o framework; `VTKVisualizer`
gestisce la scena VTK e il player interattivo offre controlli di riproduzione.

## Requisiti e installazione

Il progetto dichiara **Python 3.10 o successivo**. Per usare i modelli occorre
anche disporre dei binding Python di OpenSim compatibili con l'interprete
selezionato. La versione OpenSim va verificata nell'ambiente di installazione:
il package non definisce una matrice di compatibilità Python/OpenSim.

Il nome della distribuzione è `opensim-extensions`; il namespace Python da
importare è `opensim_extensions`.

Da una copia locale del repository, installare il package con l'extra OpenSim:

```powershell
python -m pip install -e ".[opensim]"
```

Per installare anche le dipendenze di test:

```powershell
python -m pip install -e ".[opensim,test]"
```

L'extra `opensim` richiede un binding installabile con `pip` nell'ambiente
corrente. Se OpenSim è già gestito da un ambiente Conda o da un'installazione
separata, usare quell'ambiente e assicurarsi che `import opensim` funzioni
prima di importare `opensim_extensions`. Su Windows il package prepara il
percorso delle librerie native dell'ambiente Python prima di caricare OpenSim.

Il modello OpenSim, i dati ANSUR II e le mesh usate da `User` sono inclusi nel
package. La geometria di `Bike` usa primitive e mesh incluse nel package.

## Esempio rapido

```python
from pathlib import Path

from opensim_extensions import Bike, VTKViewer

# Crea e adatta la bike al rider di riferimento.
bike = Bike()
framework = bike.fit_user()
user = framework["user"]

print(user.height_cm, user.mass_kg)
print(bike.screen_center)  # coordinate mondo in mm

viewer = VTKViewer()
viewer.save_snapshot(framework, path=Path("output") / "bike_user.svg")
```

Se l'utente non viene passato a `fit_user()`, viene creato un riferimento
maschile di **170 cm e 70 kg**. Il percorso fornito a `save_snapshot()` può
essere una stringa o un `pathlib.Path`; le directory di destinazione vengono
create automaticamente.

## Modello antropometrico `User`

```python
from pathlib import Path

from opensim_extensions import User

user = User("F", height_cm=170.0, mass_kg=65.0)
user.set_left_arm_length(user.left_arm_length * 1.05)
user.set_neck_flexion(15.0)
output = Path("output")
output.mkdir(exist_ok=True)
user.to_osim(output / "user.osim")
```

Il costruttore accetta `gender` (`"M"` o `"F"`), `height_cm` oppure
`percentile` (default 50), un eventuale `mass_kg` e, facoltativamente, il
percorso a un dataset ANSUR II alternativo. Quando viene specificata
`height_cm`, le misure antropometriche vengono risolte in funzione
dell'altezza; altrimenti si usa il percentile indicato. Se `mass_kg` non è
fornito, il modello calibra la massa totale sul peso risolto dai riferimenti
ANSUR II. Un'altezza esterna all'intervallo osservato può essere estrapolata e
genera un `UserWarning`.

La property `height_cm` è espressa in centimetri; `mass_kg` è in
chilogrammi. Le lunghezze e le circonferenze antropometriche esposte dalle
property, i marker e le coordinate spaziali del modello seguono le unità SI di
OpenSim (metri). Le misure destra e sinistra sono indipendenti: ad esempio,
`set_left_arm_length()` modifica il lato sinistro senza scalare quello destro.

Le articolazioni cervicali includono flesso-estensione e flessione laterale
della base del collo (`neck_base_flexion`,
`neck_base_lateral_flexion`) e flesso-estensione, flessione laterale e
rotazione del cranio (`neck_flexion`, `neck_lateral_flexion`,
`neck_rotation`). I polsi consentono flesso-estensione e deviazione
radiale-ulnare. I marker `REYE`/`LEYE` individuano le orbite, `RPALM`/`LPALM`
il centro dei palmi e i marker del piede descrivono tallone, primo metatarso e
quinto metatarso. Il joint subtalare fornisce un grado di libertà di
inversione/eversione per ciascun piede.

`to_osim()` esporta il modello scalato nella posa corrente. Per default copia
anche le mesh referenziate in una cartella `Geometry` accanto al file `.osim`.
Il metodo `show()` apre il visualizzatore interattivo per il modello.

## Bike e fitting

La bike usa il riferimento mondo con `x` in avanti, `y` verso l'alto e `z`
verso destra. Le lunghezze e le posizioni dell'API pubblica di `Bike` sono in
**millimetri**, gli angoli in **gradi**, le forze in newton e le densità in
kg/m³; i calcoli interni OpenSim restano in unità SI. Telaio, sella, attacchi e
manubrio sono fissi; ruotano la guarnitura e i due pedali.

`Bike(...)` consente di configurare gli intervalli e gli indici di sella,
attacco e scorrimento del manubrio, la posizione dell'asse della guarnitura,
la lunghezza delle pedivelle, le dimensioni dei pedali, le dimensioni del
manubrio, le densità dei componenti e la forza frenante.
Per esempio, `seat_tube_steps=20` definisce 21 posizioni (indici 0-20), mentre
`seat_tube_step` seleziona una posizione; se omesso, parte dall'indice centrale.

Per avviare il fitting:

```python
from opensim_extensions import Bike, User

bike = Bike()
user = User("M", height_cm=178.0, mass_kg=75.0)
framework = bike.fit_user(
    user,
    shoe_height_mm=30.0,
    trunk_angle_deg=45.0,
    shoulder_flexion_deg=75.0,
    elbow_flexion_deg=15.0,
)
```

`fit_user()` modifica la bike e l'utente e restituisce un `Framework` con i
due modelli e i piedi accoppiati ai pedali. I valori predefiniti della posa
sono busto a 45°, spalle a 75° e gomiti a 15°. L'utente predefinito è il
riferimento maschile di 170 cm e 70 kg.
Durante l'IK, flessione delle spalle e dei gomiti sono condivise tra i due
lati; l'adduzione della spalla resta regolabile indipendentemente per lato.

Gli angoli dei piantoni di `Bike` non si specificano: per piantone sella e
piantone manubrio si indicano la tupla di inizio e fine dell'escursione
(`seat_tube_range_mm`, `handlebar_tube_range_mm`, ciascuna `((x0, y0), (x1, y1))`
in mm, coordinate globali, z sempre 0) e il numero di step (`seat_tube_steps`,
`handlebar_tube_steps`). Le guide di sella e manubrio sono relative alla cima del
rispettivo piantone: `seat_slide_range_mm` e `handlebar_slide_range_mm` sono
intervalli 1D `(inizio, fine)` in mm, con `*_slide_steps` e `*_slide_angle_deg`
(angolo della guida nel piano XY). `seat_tube_angle_deg` e
`handlebar_tube_angle_deg` sono proprietà di sola lettura calcolate dai punti.
`frame_tube_diameter_mm` è il diametro dei tubi del telaio.
Gli angoli geometrici orientati nel piano XY sono misurati da +X
verso +Y secondo la regola della mano destra (rotazione positiva attorno a +Z):
un piantone della sella che sale all'indietro ha un angolo maggiore di 90°
(default circa 104°); la parte alta
dello schermo si inclina con `screen_inclination_deg` (default 40).
`horn_pitch_deg`
inclina i corni verso l'alto quando positivo; `horn_roll_deg` positivo li
ruota medialmente, verso il piano centrale della bike. Questi angoli orientano
i due corni individualmente: non è previsto un tilt globale del manubrio.

Il fitting centra il punto medio tra i marker `TOE` e `MT5` sul centro del
pedale e ne imposta la distanza dalla superficie del pedale, misurata lungo
la normale, a `shoe_height_mm`. In coordinate locali del pedale il punto medio
è centrato lungo la superficie e la sua distanza dall'asse del pedale è pari a
metà dello spessore del pedale più `shoe_height_mm`. I piedi sono
accoppiati rigidamente ai rispettivi pedali: il punto medio mantiene quindi
questa distanza mentre pedivella e pedali ruotano. Il piede destro e il
relativo pedale restano orizzontali; il pedale sinistro segue l'inclinazione
del piede con caviglia sinistra neutra. L'IK minimizza inoltre l'errore tra il
piano della suola e la faccia superiore del pedale, e tra gli assi longitudinali
di piede e pedale. La rotazione subtalare è una delle coordinate disponibili
per correggere l'inclinazione della suola; questi obiettivi di orientamento
restano subordinati al posizionamento dei marker sul pedale. Il risultato e gli
scostamenti raggiunti sono disponibili in `bike.fit_report`.

Con `allow_ik_adjustment=False`, gli angoli richiesti di busto, spalle e
gomiti restano fissi; il fitting può continuare a regolare gli altri gradi di
libertà. Passare `trunk_angle_deg=None` usa la postura lombare iniziale come
riferimento, ma consente all'IK di adattarla per raggiungere le corna del
manubrio. In questa modalità, la posizione verticale del manubrio viene scelta
per avvicinarsi all'altezza della sella.

### Ergonomia visiva

`VisualErgonomicsFramework`, esportata da `opensim_extensions.framework`, estende
`Framework` con un cono visivo a base rettangolare ancorato al punto medio
degli occhi e con le superfici dello spazio di raggiungimento delle mani.
Ogni mano è un corpo articolato rispetto al radio da un giunto che consente
flessione-estensione e deviazione radiale-ulnare del polso. I marker `RPALM` e
`LPALM`, vincolati ai rispettivi corpi della mano, identificano il centro del
palmo e vengono aggiornati in base alla lunghezza antropometrica del palmo. Il
cono si apre in avanti; il suo angolo verticale è misurato verso il basso. Gli
overlay sono inclusi nel framework e appaiono quando viene mostrato nel viewer.

```python
from opensim_extensions import Bike, User
from opensim_extensions.framework import VisualErgonomicsFramework

user = User("M", height_cm=178.0, mass_kg=75.0)
bike = Bike()
bike.fit_user(user)
ergonomics = VisualErgonomicsFramework(
    user,
    bike,
)

optimized = ergonomics.optimize_screen_position(
    incidence_weight=0.33,
    observed_screen_weight=0.33,
    reachability_weight=0.34,
    inplace=False,
    screen_horizontal_offset_mm=(150.0, 250.0),
    screen_vertical_offset_mm=32.0,
    screen_inclination_deg=(10.0, 90.0),
)
view_state = optimized.get_view_state()

optimized_user = ergonomics.optimize_user_position(
    neck_base_flexion_deg=(-30.0, 30.0),
    lumbar_extension_deg=0.0,
    arm_flex_r_deg=(20.0, 90.0),
)
```

Per i parametri di posa dello schermo e per le coordinate di `User`, `None`
mantiene il valore corrente, un numero fissa il valore e una tupla di due
numeri definisce il range ottimizzabile. `optimize_user_position` espone
esplicitamente le articolazioni di collo, braccia, gomiti, avambracci e polsi,
oltre alla flesso-estensione lombare; mantiene fisso lo schermo e, per
default, ottimizza solo la flesso-estensione del collo (`neck_base_flexion`,
da -30° a 30°). Se un valore o un range supera i limiti attuali di un
parametro, i limiti vengono ampliati e viene emesso un warning; i vincoli
geometrici di fattibilità dello schermo restano attivi.

Per impostazione predefinita, il cono copre 30° in orizzontale (15° per lato)
e 30° verso il basso in verticale. Il limite verticale rappresenta la zona
visiva preferita indicata dal CCOHS, centrata circa 15° sotto l'orizzontale;
gli studi di Turville et al. (1998) e Burgess-Limerick et al. (2000)
supportano il posizionamento del display al di sotto del livello degli occhi e
segnalano i compromessi posturali a inclinazioni maggiori. Le fonti non
definiscono un limite numerico orizzontale equivalente: i 30° totali sono
un'approssimazione simmetrica conservativa del modello, non una soglia
ergonomica universale.

Riferimenti: Turville et al., *Applied Ergonomics*, 29(4), 239-246 (1998),
DOI: 10.1016/S0003-6870(97)00048-3
([PubMed/NLM](https://pubmed.ncbi.nlm.nih.gov/9701537/)); Burgess-Limerick
et al., *Human Factors*, 42(1), 140-150 (2000),
DOI: 10.1518/001872000779656598
([PubMed/NLM](https://pubmed.ncbi.nlm.nih.gov/10917150/)); Canadian Centre
for Occupational Health and Safety, [Monitor Positioning](https://www.ccohs.ca/oshanswers/ergonomics/office/monitor_positioning.html).

I tre pesi devono essere compresi tra 0 e 1 e sommare a 1. `inplace=False`
restituisce un nuovo framework con copie dei modelli; `inplace=True` modifica
il framework corrente e restituisce `None`. `get_view_state()` restituisce
l'angolo di incidenza (0° è perpendicolare allo schermo) e le percentuali di
schermo visibile nel cono e raggiungibile. Le percentuali sono stimate su una
griglia regolare. Per stimare la raggiungibilità si campionano le posizioni dei
marker al centro dei palmi, variando la cinematica di spalla, avambraccio e
polso e mantenendo i gomiti completamente estesi. Gli inviluppi convessi
risultanti definiscono le superfici mostrate nel viewer e la parte dello
schermo raggiungibile. I limiti effettivi delle coordinate OpenSim sono
inclusi combinando sistematicamente i valori estremi dei range con campioni
casuali interni. Le coordinate articolari aggiuntive si impostano, per
esempio, con
`coordinate_ranges_deg={"arm_flex_r": (60.0, 90.0), "elbow_flex_r": (10.0, 35.0)}`.
Questi limiti modificano la postura ottimizzata per l'impugnatura ma non
l'inviluppo di raggiungibilità, calcolato indipendentemente con le braccia
distese; busto e bacino restano invece nella postura corrente.
Gli offset dello schermo si limitano con `screen_horizontal_offset_range_mm` e
`screen_vertical_offset_range_mm`. Il vincolo è verificato sull'intera
superficie: lo schermo non può avvicinarsi al rider oltre il centro del
manubrio né scendere sotto la sua altezza.

Altri parametri e proprietà:

- `crank_spacing_mm` è la distanza tra i piani centrali delle pedivelle.
- `pedal_spacing_mm` è la distanza tra le facce interne dei pedali; il default
  è 185 mm ed è indipendente da `crank_spacing_mm`.
- `screen_width_mm`, `screen_height_mm` e `screen_inclination_deg` definiscono
  dimensioni e inclinazione dello schermo.
- `screen_vertical_offset_mm` e `screen_horizontal_offset_mm` definiscono la
  posizione della base dello schermo rispetto al centro del manubrio; devono
  essere positivi e spostano lo schermo verso l'alto e in avanti.
- `screen_base` e `screen_center` restituiscono le coordinate mondo della
  base e del centro dello schermo, in millimetri. Schermo e supporto sono
  collegati.
- La sella è formata da mesh ellissoidali arrotondate: la coda è larga e la
  punta anteriore è rastremata. `seat_length_mm`, `seat_width_mm` e
  `seat_height_mm` ne controllano l'ingombro.

## Composizione con `Framework`

`Framework` è un contenitore simile a un dizionario, con modelli OpenSim
distinti associati a nomi. I modelli possono essere raggiunti sia con la
sintassi `framework["user"]` sia, quando il nome non è riservato, con
`framework.user`.

```python
from opensim_extensions import Bike

framework = Bike().fit_user()
state = framework.initSystem()
composite = framework.composite
standalone_model = framework.merge_to_model()
```

`initSystem()` crea uno snapshot composito: i modelli conservati nel framework
restano separati. Modifiche applicate direttamente a un membro, ad esempio
`framework["user"].set_neck_flexion(30)`, aggiornano quel modello; una nuova
chiamata a `merge_to_model()` legge lo stato corrente dei membri e crea un
nuovo `opensim.Model` indipendente. Uno snapshot composito già inizializzato
non riceve invece notifiche automatiche: chiamare nuovamente `initSystem()`
per ricostruirlo. Dopo aver modificato lo stato del composito, `update_state()`
trasferisce coordinate e velocità ai modelli membri. I parametri `only`,
`exclude` e `include_couplings` di `merge_to_model()` permettono di selezionare
i membri e gli accoppiamenti da includere.

Gli accoppiamenti registrati con il framework vengono ricreati negli snapshot
compositi e non modificano direttamente i modelli membri. Le operazioni
principali sono:

- `couple_weld()` per saldare due frame;
- `couple_point()` per coincidere due punti di frame;
- `couple_coordinate()` per pilotare una coordinata tramite altre coordinate;
- `couple_joint()` per collegare a un frame una giunzione di un modello
  altrimenti radicato al suolo;
- `add_force()`, `add_constraint()` e `add_controller()` per aggiungere
  componenti OpenSim al composito;
- `uncouple()` per rimuovere un accoppiamento.

I riferimenti ai componenti sono tuple `(nome_modello, percorso_OpenSim)`, ad
esempio `("user", "/bodyset/hand_r")`. I modelli che compongono un framework
devono avere gravità coerente. `set_gravity((0.0, -9.81, 0.0))` imposta una
gravità comune, espressa in m/s².

`forward(final_time)` integra fino al tempo assoluto indicato in secondi;
`settle_under_gravity(duration_s)` integra per la durata specificata a partire
dal tempo corrente, ma non verifica che il sistema abbia raggiunto
l'equilibrio. L'integrazione tramite `opensim.Manager` rifiuta esplicitamente
i modelli che contengono un `opensim.FreeJoint`.

## Visualizzazione ed esportazione

```python
from opensim_extensions import VTKViewer

viewer = VTKViewer()
window = viewer.show(framework, motion="motion.sto", loop=True, fps=30.0)
viewer.save_snapshot(
    framework,
    path="output/pose.svg",
    motion="motion.sto",
    time_s=1.5,
)
viewer.save_animation(framework, "output/motion.mp4", motion="motion.sto")
```

`show()` apre il visualizzatore interattivo e restituisce una finestra di
controllo della riproduzione. Sono supportati i modelli OpenSim e gli oggetti
`Framework`; un framework viene visualizzato come snapshot composito. Le
tabelle di movimento possono essere passate come percorso `.mot`/`.sto` o
come `opensim.TimeSeriesTable`. Nel gruppo View, i controlli `Vision cone +
axis` e `Reachability` mostrano o nascondono i rispettivi overlay; risultano
disabilitati quando il modello non contiene quegli elementi.

`save_snapshot()` esporta un'immagine **SVG** o **PNG** senza aprire una
finestra, in base all'estensione del percorso. Per i PNG, `dpi` imposta i DPI
registrati nei metadati (default 300); non cambia le dimensioni in pixel,
configurabili con `size`. `frame` seleziona una riga della tabella di movimento
e `time_s` interpola la posa al tempo indicato. Senza una tabella, viene usata
la posa corrente. Le altre estensioni non sono supportate.
`save_animation()` esporta un video **MP4** off-screen e ripristina i valori
delle coordinate del modello dopo il rendering. Anche `User.show()` e
`Framework.show()` offrono l'apertura rapida del visualizzatore.

## Test

Il repository configura pytest tramite `pyproject.toml` e raccoglie i test
dalla directory `test/`. Dopo aver installato l'extra di test, eseguire:

```powershell
python -m pytest
```

In VS Code selezionare l'interprete dell'ambiente che contiene OpenSim e
impostare pytest come framework di test Python; la discovery è configurata
anche in `.vscode/settings.json`.

## API pubblica

Gli elementi principali esportati da `opensim_extensions` sono:
`User`, `Bike`, `Framework`, `save_animation`, `VTKViewer`, `VTKVisualizer`,
`MotionData`, `MotionPlayer`, `PlayerWindow`, `DEFAULT_SIZE` e `VIEW_NAMES`.
Le primitive geometriche e gli helper per creare corpi OpenSim sono esposti
nel sottopackage `opensim_extensions.bodies`, ad esempio
`Box`, `Cylinder`, `Disk`, `Ellipsoid`, `Sphere`, `Tube` e
`mass_properties`.
