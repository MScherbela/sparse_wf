# %%
from sparse_wf.scf import run_hf
from sparse_wf.system import get_molecule
import yaml
import pathlib

DEFAULT_CACHE_DIR = "~/runs/pyscf_cache2"


def load_yaml(fname):
    with open(fname, "r") as f:
        return yaml.safe_load(f)


default_config = load_yaml(pathlib.Path(__file__).parent / "../../config/default.yaml")
molecule_args = default_config["molecule_args"]
hf_args = default_config["pretraining"]["hf"]
hf_args["cache_dir"] = DEFAULT_CACHE_DIR
hf_args["newton"] = True

geom_names = [
    "corannulene_dimer",
    # "Ferrocene_Toma16_red",
    # "Ferrocene_Toma19_red",
    # "Ferrocene_Toma16_red_charged",
    # "Ferrocene_Toma19_red_charged",
]

for geom_str in geom_names:
    print(geom_str)
    molecule_args["database_args"]["comment"] = geom_str
    mol = get_molecule(molecule_args)
    hf = run_hf(mol, hf_args)
    s2, mult = hf.spin_square()
    with open("energies.csv", "a") as f:
        f.write(f"{geom_str},{hf.e_tot},{s2},{mult}\n")
