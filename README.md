# opensim-extensions

`opensim-extensions` è un package Python che aggiunge a OpenSim due strumenti
di composizione e un visualizzatore:

- **`Framework`** raccoglie modelli nativi `opensim.Model` con chiavi
  nominative. `composite` espone uno snapshot OpenSim sincronizzato;
  `merge_to_model()` crea invece un modello indipendente.
- **`User`** è una sottoclasse diretta di `opensim.Model`. Carica il modello
  full-body Rajagopal-Lai-Uhlrich, lo scala usando i riferimenti antropometrici
  ANSUR II e rende accessibili misure, postura e centri articolari tramite
  properties documentate e setter espliciti. Le misure laterali, ad esempio
  `left_arm_length` e `right_arm_length`, sono indipendenti.
- **`VTKViewer`** (`viewer = VTKViewer()`) visualizza un `opensim.Model` o un
  `Framework` con `viewer.show(obj)`; per un framework la scena usa uno
  snapshot unificato senza modificare i modelli contenuti. Esporta immagini
  vettoriali con `viewer.save_snapshot(obj, frame=None, time_s=0.0, path="x.svg")`
  (solo `.svg`; `frame`/`time_s` si riferiscono alla `motion=` indicata) e
  video MP4 con `viewer.save_animation(obj, "x.mp4", motion=...)`.

## Installazione

```powershell
python -m pip install -e ".[opensim]"
```

## Compatibilità Python/OpenSim

Il package dichiara `Python >=3.10`. Questo requisito non implica che i
binding OpenSim siano verificati su ogni versione Python: la verifica completa
dell'integrazione OpenSim arriva a Python 3.11. Python 3.12 e successivi non
sono nella matrice di compatibilità verificata; in particolare, Python 3.13 ha
mostrato problemi in passato. Eventuali test mirati superati su una versione
più recente non costituiscono da soli una validazione completa né una
dichiarazione di supporto.

## Esempio

```python
from pathlib import Path

from opensim_extensions import Framework, User, VTKViewer

user = User("F", height_cm=170.0)
user.set_left_arm_length(user.left_arm_length * 1.05)
user.to_osim(Path("output") / "user.osim")

framework = Framework({"user": user})
state = framework.initSystem()
framework.forward(1.0)  # integra fino al tempo assoluto 1 s
merged_model = framework.merge_to_model()
viewer = VTKViewer()
viewer.show(framework)
viewer.save_snapshot(framework, path=Path("output") / "pose.svg")
```

`framework.composite` espone il modello composito attivo e `framework.state`
il suo stato. Dopo modifiche dirette a un modello membro, chiamare
`framework.initSystem()` per ricostruire lo snapshot; dopo modifiche dirette
allo stato composito, `framework.update_state()` propaga coordinate e velocità
ai membri. `forward(final_time)` integra fino al tempo indicato, mentre
`settle_under_gravity(duration_s)` integra per una durata a partire dal tempo
corrente. L'integrazione richiede gravità uguale fra i membri e rifiuta
esplicitamente `opensim.FreeJoint`, che può causare un crash nativo di OpenSim.
`framework.set_gravity((0.0, -9.81, 0.0))` aggiorna la gravità di tutti i
modelli membri.

### Accoppiamenti

I membri restano modelli indipendenti; gli accoppiamenti esistono solo nel
composito e vengono ricreati a ogni `initSystem()`. I riferimenti sono tuple
`(membro, percorso)`:
`couple_weld`, `couple_point`, `couple_coordinate`, `couple_joint`,
`add_force`, `add_constraint`, `add_controller`; `framework.couplings` li
elenca e `uncouple(name=None)` li rimuove. Un accoppiamento che riferisce un
membro rimosso solleva `ValueError` finché non viene tolto con `uncouple()`.

`merge_to_model(name=None, only=None, exclude=None, include_couplings=True,
return_mapping=False)` produce un modello indipendente: `only`/`exclude`
selezionano i membri, gli accoppiamenti con membri esclusi sono saltati e
`return_mapping=True` restituisce anche la mappa dei percorsi.

Nota: in alcune installazioni OpenSim l'integrazione con vincoli (es.
`CoordinateCouplerConstraint`) fallisce con errori LAPACK; preferire
`couple_joint` per le simulazioni.

`User.to_osim()` esporta la geometria scalata e la posa corrente; per default
copia anche le mesh referenziate in una cartella `Geometry` accanto al file.

## Struttura

```text
src/opensim_extensions/
    __init__.py
    animation.py
    opensim.py
    framework/
        framework.py
        couplings.py
    models/user/
        user.py
        data.py
        mapping.py
        measurement_generated.py
        posture_table.py
        posture_generated.py
        joint_center_table.py
        joint_centers_generated.py
        assets/
    viewer/
        __init__.py
        visualizer.py
        viewer.py
        svg.py        player.py
        geometry.py
        export.py
        tooltip.py
        win32_embed.py
```

Ogni modulo definisce `__all__`; i package aggregano soltanto gli elementi
pubblici richiesti.
