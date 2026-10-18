"""Generate partial type stubs for the non-Moco OpenSim Python bindings.

The SWIG package exposes very few useful Python annotations. This generator
uses the introspection schema to publish class hierarchies, properties,
documented method names, and a conservative ``Any`` fallback for native
overloads whose signatures are not available through SWIG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "scripts" / "_opensim_schema.json"
OUTPUT = ROOT / "typings" / "opensim" / "__init__.pyi"
SAMPLE_CLASSES = {"Model", "Body", "Joint", "Coordinate", "Force"}

PRIMITIVE_TYPES = {
    "bool": "bool",
    "double": "float",
    "float": "float",
    "int": "int",
    "string": "str",
}


def _class_closure(classes: dict, requested: set[str] | None) -> set[str]:
    if requested is None:
        return set(classes)
    missing = requested - classes.keys()
    if missing:
        raise ValueError(f"Unknown class names: {', '.join(sorted(missing))}")
    selected = set(requested)
    pending = list(requested)
    while pending:
        name = pending.pop()
        for base in classes[name]["bases"]:
            if base not in selected:
                selected.add(base)
                pending.append(base)
    return selected


def _ordered_classes(classes: dict, selected: set[str]) -> list[str]:
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        for base in classes[name]["bases"]:
            if base in selected:
                visit(base)
        ordered.append(name)

    for name in sorted(selected):
        visit(name)
    return ordered


def _annotation(prop: dict) -> str:
    if prop["is_list"]:
        return "Sequence[Any]"
    return PRIMITIVE_TYPES.get(prop["type"], "Any")


def render_stub(schema: dict, requested: set[str] | None) -> str:
    classes = schema["classes"]
    selected = _class_closure(classes, requested)
    lines = [
        '"""Partial OpenSim stubs generated from the installed SWIG bindings."""',
        "",
        "from typing import Any, Sequence",
        "",
    ]
    for name in _ordered_classes(classes, selected):
        entry = classes[name]
        bases = [base for base in entry["bases"] if base in selected]
        declaration = f"class {name}({', '.join(bases)}):" if bases else f"class {name}:"
        lines.append(declaration)
        doc = entry["doc"].strip()
        lines.append(f"    {doc!r}" if doc else "    ...")
        if doc:
            lines.append("    ...")
        members = 0
        for prop in entry.get("properties") or []:
            prop_name = prop["name"]
            if not prop_name.isidentifier() or prop_name.startswith("_"):
                continue
            lines.extend(
                [
                    "",
                    "    @property",
                    f"    def {prop_name}(self) -> {_annotation(prop)}: ...",
                ]
            )
            members += 1
        for method in entry["methods"]:
            method_name = method["name"]
            if not method_name.isidentifier():
                continue
            if method_name == "safeDownCast":
                lines.extend(
                    [
                        "",
                        "    @staticmethod",
                        f"    def safeDownCast(obj: Any) -> {name}:",
                    ]
                )
            elif method_name == "__init__":
                lines.extend(
                    [
                        "",
                        "    def __init__(self, *args: Any, **kwargs: Any) -> None:",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        f"    def {method_name}(self, *args: Any, **kwargs: Any) -> Any:",
                    ]
                )
            if method["doc"]:
                lines.append(f"        {method['doc'].strip()!r}")
                lines.append("        ...")
            else:
                lines[-1] += " ..."
            members += 1
        if members == 0:
            lines.append("    ...")
        lines.append("")
    exports = ", ".join(repr(name) for name in sorted(selected))
    lines.append(f"__all__ = [{exports}]")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="generate stubs for every public non-Moco OpenSim class",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated stub is stale",
    )
    args = parser.parse_args()
    requested = None if args.all else SAMPLE_CLASSES
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    content = render_stub(schema, requested)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Generated stub is stale: {OUTPUT.relative_to(ROOT)}")
        print(f"Stub is up to date: {OUTPUT.relative_to(ROOT)}")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8", newline="\n")
    print(f"Generated {len(_class_closure(schema['classes'], requested))} class stubs.")


if __name__ == "__main__":
    main()
