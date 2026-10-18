"""Estrae da ``opensim`` (SWIG) lo schema delle classi in ``scripts/_opensim_schema.json``.

Per ogni classe pubblica (Moco escluso): modulo, basi dirette, astratta/concreta,
proprietà dichiarate (nome, tipo, lista, commento), metodi dichiarati con
docstring, socket/input/output (solo classi concrete), docstring di classe.

Uso::

    python scripts/introspect_opensim.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("_opensim_schema.json")
EXCLUDED_PREFIXES = ("Moco",)
EXCLUDED_MODULES = ("opensim.moco",)


def _excluded(cls: type) -> bool:
    return cls.__name__.startswith(EXCLUDED_PREFIXES) or cls.__module__ in EXCLUDED_MODULES


def _instantiate(cls: type):
    """Return a default-constructed instance of ``cls`` or ``None`` if impossible."""
    try:
        return cls()
    except Exception:
        return None


def _properties(obj) -> list[dict]:
    props = []
    for index in range(obj.getNumProperties()):
        prop = obj.getPropertyByIndex(index)
        props.append(
            {
                "name": prop.getName(),
                "type": prop.getTypeName(),
                "is_list": bool(prop.isListProperty()),
                "is_object": bool(prop.isObjectProperty()),
                "comment": prop.getComment() or "",
            }
        )
    return props


def _doc(obj) -> str:
    return (getattr(obj, "__doc__", None) or "").strip()


def _declared_methods(cls: type) -> list[dict]:
    methods = []
    for name, member in vars(cls).items():
        if name.startswith("_") or not callable(member) or isinstance(member, type):
            continue
        methods.append({"name": name, "doc": _doc(member)})
    return sorted(methods, key=lambda m: m["name"])


def build_schema() -> dict:
    import opensim

    classes = {
        name: obj
        for name in sorted(dir(opensim))
        if isinstance(obj := getattr(opensim, name), type) and not _excluded(obj)
    }
    object_cls = getattr(opensim, "OpenSimObject", None) or opensim.Object

    instances = {name: _instantiate(cls) for name, cls in classes.items() if issubclass(cls, object_cls)}
    full_props: dict[str, list[dict]] = {}
    for name, inst in instances.items():
        if inst is not None:
            full_props[name] = _properties(inst)

    def props_of(name: str) -> list[dict] | None:
        """Full property list; for abstract classes, those shared by all concrete subclasses."""
        if name in full_props:
            return full_props[name]
        cls = classes[name]
        descendants = [
            full_props[other]
            for other, other_cls in classes.items()
            if other in full_props and other != name and issubclass(other_cls, cls)
        ]
        if not descendants:
            return None
        common = {p["name"] for p in descendants[0]}
        for plist in descendants[1:]:
            common &= {p["name"] for p in plist}
        return [p for p in descendants[0] if p["name"] in common]

    schema: dict[str, dict] = {}
    for name, cls in classes.items():
        bases = [
            b.__name__ for b in cls.__bases__ if b.__name__ in classes and isinstance(b, type)
        ]
        inst = instances.get(name)
        entry: dict = {
            "module": cls.__module__,
            "bases": bases,
            "is_object": issubclass(cls, object_cls),
            "concrete": inst is not None,
            "doc": _doc(cls),
            "methods": _declared_methods(cls),
        }
        if entry["is_object"]:
            mine = props_of(name)
            inherited: set[str] = set()
            for base in bases:
                base_props = props_of(base)
                if base_props:
                    inherited |= {p["name"] for p in base_props}
            entry["properties"] = (
                [p for p in mine if p["name"] not in inherited] if mine is not None else None
            )
        if inst is not None:
            for key, getter in (
                ("sockets", "getSocketNames"),
                ("inputs", "getInputNames"),
                ("outputs", "getOutputNames"),
            ):
                try:
                    entry[key] = list(getattr(inst, getter)())
                except Exception:
                    pass
        schema[name] = entry
    return {"opensim_version": opensim.__version__, "classes": schema}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    schema = build_schema()
    SCHEMA_PATH.write_text(
        json.dumps(schema, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    classes = schema["classes"]
    objects = [c for c in classes.values() if c["is_object"]]
    print(
        f"{len(classes)} classes, {len(objects)} OpenSim objects, "
        f"{sum(c['concrete'] for c in objects)} concrete"
    )


if __name__ == "__main__":
    main()
