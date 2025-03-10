# %%
from sparse_wf.scf import run_hf
from sparse_wf.system import get_molecule
import pathlib
import yaml

DEFAULT_CACHE_DIR = "~/runs/pyscf_cache"


def load_yaml(fname):
    with open(fname, "r") as f:
        return yaml.safe_load(f)


default_config = load_yaml(pathlib.Path(__file__).parent / "../../config/default.yaml")
config = load_yaml("config.yaml")
config["molecule_args"] = default_config["molecule_args"] | config.get("molecule_args", {})
config["hf"] = default_config["pretraining"]["hf"] | config.get("hf", {})

if config["hf"]["cache_dir"] is None:
    print(f"Setting pyscf cache dir as: {DEFAULT_CACHE_DIR}")
    config["hf"]["cache_dir"] = DEFAULT_CACHE_DIR

geom_names = [
    "01_Water_dimer",
    "01_Water_dimer_Dissociated",
    "02_Formic_acid_dimer",
    "02_Formic_acid_dimer_Dissociated",
    "03_Formamide_dimer",
    "03_Formamide_dimer_Dissociated",
    "04_Uracil_dimer_h-bonded",
    "04_Uracil_dimer_h-bonded_Dissociated",
    "05_Methane_dimer",
    "05_Methane_dimer_Dissociated",
    "06_Ethene_dimer",
    "06_Ethene_dimer_Dissociated",
    "07_Uracil_dimer_stack",
    "07_Uracil_dimer_stack_Dissociated",
    "08_Ethene-ethyne_complex",
    "08_Ethene-ethyne_complex_Dissociated",
    "09_Benzene-water_complex",
    "09_Benzene-water_complex_Dissociated",
    "11_Phenol_dimer",
    "11_Phenol_dimer_Dissociated",
]

for geom_str in geom_names:
    print(geom_str)
    config["molecule_args"]["database_args"]["comment"] = geom_str
    mol = get_molecule(config["molecule_args"])
    hf = run_hf(mol, config["hf"])
    s2, mult = hf.spin_square()
    with open("energies.csv", "a") as f:
        f.write(f"{geom_str},{hf.e_tot},{s2},{mult}\n")
