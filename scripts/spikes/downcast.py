"""Spike S03: downcast al tipo concreto + re-class, mappa per nome di classe concreta."""

import opensim as osim

SRC = r"src/opensim_models/models/user/assets/rajagopalaiulrich2023.osim"


class PinJoint(osim.PinJoint):
    def hello(self):
        return "pin:" + self.getName()


class CustomJoint(osim.CustomJoint):
    def hello(self):
        return "custom:" + self.getName()


WRAPPERS = {"PinJoint": PinJoint, "CustomJoint": CustomJoint}


def wrap(obj):
    name = obj.getConcreteClassName()
    wrapper = WRAPPERS.get(name)
    native = getattr(osim, name, None)
    if native is None:
        return obj
    down = native.safeDownCast(obj)
    if wrapper is not None:
        down.__class__ = wrapper
    return down


m = osim.Model(SRC)
js = m.getJointSet()
seen = {}
for i in range(js.getSize()):
    w = wrap(js.get(i))
    seen[type(w).__name__] = seen.get(type(w).__name__, 0) + 1
    if hasattr(w, "hello") and i < 3:
        print(w.hello())
print(seen)
w = wrap(js.get(0))
w.upd_reference_frame if False else None
print("mutation through wrapper:", end=" ")
if isinstance(w, osim.CustomJoint):
    cm = w.get_coordinates(0)
    print(type(cm))
m.initSystem()
print("initSystem ok")
# nome gia' presente in modello: i wrapper non conservano attributi Python
w.foo = 1
w2 = wrap(js.get(0))
print("attr persists across re-get:", hasattr(w2, "foo"))
