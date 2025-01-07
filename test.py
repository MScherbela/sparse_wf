from copy import copy, deepcopy
import pyscf
from seml import Experiment
from sparse_wf.system import database

mol = database(name="zhai_et_al_2023_HC_df2-svp")
mol = pyscf.gto.M(atom="H 0 0 0; H 0 0 1.1", unit="Bohr")
mol.basis = "def2-svp"
mol.ecp = {s: "ccecp" for s in ["C", "S", "Fe"]}
mol.build()
import pickle
import hashlib

mol_vars = copy(vars(mol))
del mol_vars["stdout"]
for k in list(mol_vars.keys()):
    if k.startswith("_"):
        del mol_vars[k]
print(hashlib.sha256(pickle.dumps(mol_vars)).hexdigest())
# print(vars(mol))
# print(mol.nao)
mf = mol.RHF()
mf.verbose = 4
# mf.init_guess = "sap"
mf.max_cycle = 2
mf = mf.newton()
print(mf.chkfile)
mf.chkfile = "zhai_et_al_2023_HC_df2-svp.h5"
mf.init_guess = "chk"
mf.kernel()
import numpy as np

print(np.array(mf.mo_coeff).shape)
