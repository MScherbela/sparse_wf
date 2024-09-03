#%%
from sparse_wf.geometry import load_geometries
from ase.visualize import view
from ase.units import Bohr
import json
import numpy as np

all_geoms = load_geometries()
comment = "WG_09"
geoms = [g for g in all_geoms.values() if g.comment == comment]
g = geoms[0]

view(g.as_ase())


