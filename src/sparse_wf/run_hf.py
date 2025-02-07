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

mol = get_molecule(config["molecule_args"])
hf = run_hf(mol, config["hf"])
