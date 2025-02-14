#%%
from sparse_wf.geometry import save_geometries, load_geometries

all_geoms = load_geometries()
all_geoms = list(all_geoms.values())

geoms_to_save = []
for n in range(2, 25, 2):
    name_triplet = f"cumulene_C{n}H4_90deg_triplet"
    name_quintuplet = f"cumulene_C{n}H4_90deg_quintuplet"
    geom = [g for g in all_geoms if g.comment == name_triplet]
    assert len(geom) == 1
    geom = geom[0]
    geom.spin = 4
    geom.comment = name_quintuplet
    geom.name = name_quintuplet
    geoms_to_save.append(geom)

save_geometries(geoms_to_save)

