#%%
import numpy as np
from sparse_wf.geometry import Geometry, save_geometries
import copy

fnames = [
    ("/home/scherbelam20/tmp/ferrocene/16_red.xyz", "ClCN_ferrocene", "Ferrocene_Toma16_red"),
    ("/home/scherbelam20/tmp/ferrocene/19_red.xyz", "S_ferrocene", "Ferrocene_Toma19_red"),
]

geoms = []
for fname, name, comment in fnames:
    geoms.append(Geometry.from_xyz(fname, comment=comment, name=name))

geoms_charged = []
for geom in geoms:
    geom_charged = copy.deepcopy(geom)
    geom_charged.charge = 1
    geom_charged.spin = 1
    geom_charged.comment += "_charged"
    geoms_charged.append(geom_charged)

save_geometries(geoms + geoms_charged)


