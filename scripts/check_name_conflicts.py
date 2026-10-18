"""S04: conflitti di nome tra l'API pubblica di OpenSimModel e i membri di opensim.Model."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import opensim  # noqa: E402

from opensim_models import OpenSimModel, User  # noqa: E402


def public_members(cls):
    return {n for n in dir(cls) if not n.startswith("_")}


native = public_members(opensim.Model)
report = {}
for cls in (OpenSimModel, User):
    own = {n for n in public_members(cls) if any(n in vars(k) for k in cls.__mro__ if k.__module__.startswith("opensim_models"))}
    report[cls.__name__] = sorted(own & native)

out = Path(__file__).with_name("_name_conflicts.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
