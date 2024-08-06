#%%
import pandas as pd
from sparse_wf.geometry import Geometry, save_geometries
import re


df = pd.read_csv('/home/mscherbela/tmp/mpconf196/total_energies.csv')

energy_data = []
geometries = {}
for i, row in df.iterrows():
    name = row["Geometry"]
    geom_fname = re.sub(r"_(\d)", lambda m: m[1], name) # remove underscore before number
    geom_fname = "/home/mscherbela/tmp/mpconf196/geometries/" + geom_fname + ".xyz"
    geom = Geometry.from_xyz(geom_fname)
    geom.name = name.split("_")[0]
    geom.comment = name
    geometries[geom.hash] = geom

    methods = [m for m in row.index if m != "Geometry"]
    for method in methods:
        energy = row[method]
        if pd.isna(energy):
            continue
        energy_data.append(dict(
            geom_hash=geom.hash,
            geom_comment=geom.comment,
            n_el=geom.n_el,
            model=method.split("/")[0],
            model_comment=method.split("/")[1],
            E=energy,
        ))

save_geometries(geometries)

df_energies = pd.read_csv("/home/mscherbela/develop/sparse_wf/data/energies.csv")
df_energies = pd.concat([df_energies, pd.DataFrame(energy_data)])
df_energies.to_csv("/home/mscherbela/develop/sparse_wf/data/energies.csv", index=False)
