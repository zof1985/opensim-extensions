"""Generate static OpenSim wrapper classes from ``_opensim_schema.json``.

The first migration slice deliberately writes to the private ``_generated``
package. It can be moved to the public ``wrappers`` package once the legacy
composition wrappers are removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "scripts" / "_opensim_schema.json"
OUTPUT = ROOT / "src" / "opensim_models" / "_generated"

TYPE_MAP = {
    "bool": "bool",
    "double": "float",
    "float": "float",
    "int": "int",
    "string": "str",
    "Vec3": "opensim.Vec3",
    "Vec6": "opensim.Vec6",
}


def _identifier(name: str) -> str:
    return name if name.isidentifier() else ""


def render_body(schema: dict) -> str:
    body = schema["classes"]["Body"]
    properties = body.get("properties") or []
    lines = [
        '"""Generated wrapper for :class:`opensim.Body`."""',
        "",
        "from __future__ import annotations",
        "",
        "import opensim",
        "",
        "",
        "class Body(opensim.Body):",
        f'    """OpenSim rigid body.\\n\\n    {body["doc"].splitlines()[0]}',
        "",
        "    Parameters",
        "    ----------",
        "    See the native :class:`opensim.Body` constructor.",
        "",
        "    Notes",
        "    -----",
        "    This generated subclass adds statically discoverable properties.",
        '    """',
    ]
    for prop in properties:
        name = prop["name"]
        if not _identifier(name) or not prop["type"] or prop["is_list"]:
            continue
        annotation = TYPE_MAP.get(prop["type"], "object")
        getter = f"get_{name}"
        setter = f"set_{name}"
        if name in {"components"}:
            continue
        lines.extend(
            [
                "",
                "    @property",
                f"    def {name}(self) -> {annotation}:",
                f'        """{prop["comment"] or f"Get {name}."}',
                "",
                "        Returns",
                "        -------",
                f"        {annotation}",
                f"            Value of the ``{name}`` property.",
                '        """',
                f"        return self.{getter}()",
            ]
        )
        lines.extend(
            [
                "",
                f"    def set_{name}(self, value: {annotation}) -> None:",
                f'        """Set the ``{name}`` property.',
                "",
                "        Parameters",
                "        ----------",
                f"        value : {annotation}",
                f"            New value for ``{name}``.",
                '        """',
                f"        opensim.Body.{setter}(self, value)",
            ]
        )
    lines.append("")
    lines.append('__all__ = ["Body"]')
    lines.append("")
    return "\n".join(lines)


def render_init() -> str:
    return '"""Private staging package for generated OpenSim wrappers."""\n\nfrom .body import *\n\n__all__ = ["Body"]\n'


def render_registry() -> str:
    return '''"""Private registry used to re-class native SWIG component proxies."""

import opensim

from .body import Body

WRAPPERS = {"Body": Body}


def wrap_component(component):
    """Return a statically typed wrapper for a native component proxy."""
    concrete_name = component.getConcreteClassName()
    native_type = getattr(opensim, concrete_name, None)
    if native_type is None or not hasattr(native_type, "safeDownCast"):
        return component
    concrete = native_type.safeDownCast(component)
    wrapper_type = WRAPPERS.get(concrete_name)
    if wrapper_type is None:
        return concrete
    concrete.__class__ = wrapper_type
    return concrete


__all__ = ["wrap_component"]
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    outputs = {
        OUTPUT / "body.py": render_body(schema),
        OUTPUT / "__init__.py": render_init(),
        OUTPUT / "_registry.py": render_registry(),
    }
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        raise SystemExit("Generated wrappers are stale: " + ", ".join(stale))
    if args.check:
        print("Generated wrappers are up to date.")
    else:
        print(f"Generated {len(outputs)} files in {OUTPUT.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
