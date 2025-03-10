# %%
from sparse_wf.scf import run_hf
from sparse_wf.system import get_molecule
import yaml
import pandas as pd

DEFAULT_CACHE_DIR = "~/runs/pyscf_cache"


def load_yaml(fname):
    with open(fname, "r") as f:
        return yaml.safe_load(f)


# default_config = load_yaml(pathlib.Path(__file__).parent / "../../config/default.yaml")
default_config = load_yaml("/home/scherbelam20/develop/sparse_wf/config/default.yaml")
mol_config = default_config["molecule_args"]
# mol_config["basis"] = "cc-pVDZ"
# mol_config["pseudopotentials"] = []
hf_config = default_config["pretraining"]["hf"]
hf_config["cache_dir"] = DEFAULT_CACHE_DIR
hf_config["newton"] = True
# hf_config["restricted"] = True

spin_data = []
n_values = [2, 4, 6, 8, 12, 16, 20, 24, 36]
states = ["0deg_singlet", "90deg_triplet"]
geom_strings = [f"cumulene_C{n}H4_{state}" for n in n_values for state in states]
# geom_strings = ["cumulene_C24H4_0deg_singlet", "cumulene_C36H4_0deg_singlet"]
for geom_str in geom_strings:
    mol_config["database_args"]["comment"] = geom_str
    mol = get_molecule(mol_config)
    hf = run_hf(mol, hf_config)

    hf.max_cycle = 5
    energy = hf.kernel()
    s2, mult = hf.spin_square()
    print(f"Geom: {geom_str}, E: {energy}")
    print(f"S2: {s2}, mult: {mult}")
    spin_data.append(
        {
            "geom": geom_str,
            "s2": s2,
            "mult": mult,
            "E": energy,
        }
    )
df = pd.DataFrame(spin_data)
df.to_csv("cumulene_spin_data.csv", index=False)
