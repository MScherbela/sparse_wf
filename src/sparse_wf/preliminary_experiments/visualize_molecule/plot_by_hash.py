#%%
import ase
from ase.visualize import view
from ase.units import Bohr
import json
import numpy as np

geom_db_fname = "../../../../data/geometries.json"
with open(geom_db_fname) as f:
    all_geoms = json.load(f)

# geom_hash = "ddc45f81176bdb177efef0fe3e87c4eb"
geom_hash = "156ee3365b98696abc280b34aef5d7ca"
g = all_geoms[geom_hash]

atoms = ase.Atoms(positions=np.array(g["R"]), numbers=all_geoms[geom_hash]["Z"])
view(atoms)


