"""Private registry used to re-class native SWIG component proxies."""

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
