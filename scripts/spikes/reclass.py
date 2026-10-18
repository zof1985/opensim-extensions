"""Spike S01-S03: sottoclassi Python di classi SWIG di OpenSim, re-class, ownership, GC."""

import gc
import opensim as osim

SRC = r"src/opensim_models/models/user/assets/rajagopalaiulrich2023.osim"


class Body(osim.Body):
    @property
    def mass_kg(self):
        return self.get_mass()


class Model(osim.Model):
    def hello(self):
        return "hi"


# S01 -- re-class in place di un body nativo
m = osim.Model(SRC)
b = m.getBodySet().get(0)
print("native type:", type(b))
b.__class__ = Body
print("reclassed:", type(b), b.mass_kg, b.getName())
c = b.clone()
print("clone type:", type(c))
m.initSystem()
print("initSystem after reclass OK")

# la stessa istanza python e' restituita di nuovo?
b2 = m.getBodySet().get(0)
print("same python obj on re-get:", b2 is b, type(b2))

# S02 -- sottoclasse di Model + addBody con wrapper creato in Python
for i in range(300):
    mm = Model()
    bb = Body("b%d" % i, 1.0, osim.Vec3(0), osim.Inertia(1, 1, 1))
    j = osim.WeldJoint("j%d" % i, mm.getGround(), osim.Vec3(0), osim.Vec3(0), bb, osim.Vec3(0), osim.Vec3(0))
    mm.addBody(bb)
    mm.addJoint(j)
    st = mm.initSystem()
    del mm, bb, j, st
    gc.collect()
print("loop ok; hello:", Model().hello())
mm = Model()
bb = Body("x", 1.0, osim.Vec3(0), osim.Inertia(1, 1, 1))
mm.addBody(bb)
print("after addBody get:", type(mm.getBodySet().get("x")), mm.getBodySet().get("x") is bb)

# S03 -- polimorfismo: getComponent / updComponent
m = osim.Model(SRC)
comp = m.getComponent("/bodyset/pelvis") if m.hasComponent("/bodyset/pelvis") else None
print("getComponent:", type(comp))
j = m.getJointSet().get(0)
print("joint type:", type(j))
print("copy-ctor of subclass:", type(Model(m)))
print("clone of subclass:", type(Model(m).clone()))
