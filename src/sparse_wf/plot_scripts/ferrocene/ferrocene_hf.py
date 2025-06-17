# %%
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import pyscf.gto
from sparse_wf.system import get_molecule


mol = get_molecule(
    dict(  # type: ignore
        method="database",
        database_args=dict(
            comment="FerroceneCl_red_geom",
        ),
        # pseudopotentials=[],
        # basis="STO-6G",
        # basis="ccpvdz"
        pseudopotentials=["Fe", "Cl", "C"],
        basis="ccecp-ccpvdz",
    )
)

hf = pyscf.scf.RHF(mol)
hf = hf.newton()
hf.verbose = 4
hf.max_cycle = 200
hf.chkfile = "ferrocene_hf.chk"
hf.kernel()
print(hf.e_tot)
s2, mult = hf.spin_square()
print(f"S^2 = {s2}, mult = 2S+1 = {mult}, E={hf.e_tot:.6f}")
