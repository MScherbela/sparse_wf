#%%
import numpy as np
from sparse_wf.geometry import Geometry, save_geometries
from ase.visualize import view
import copy

geoms = []
for ox_state in ["red", "ox"]:
    fname = f"ferrocene_{ox_state}.xyz"
    name = f"FerroceneCl_{ox_state}_geom"
    geom = Geometry.from_xyz(fname, comment=name, name=name)
    geoms.append(geom)

geoms_charged = []
for geom in geoms:
    geom_charged = copy.deepcopy(geom)
    geom_charged.charge = 1
    geom_charged.spin = 1
    geom_charged.comment += "_charged"
    geoms_charged.append(geom_charged)

save_geometries(geoms + geoms_charged)


