# %%
import pyscf
import numpy as np

Z = 30

mol = pyscf.gto.M(
    atom=[(Z, np.zeros(3))],
    # ecp="stuttgart",
    # ecp="def2-svp",
    ecp="cc-pvdz-pp",

)

n_core, pp_params = list(mol._ecp.values())[0]
nl_values = np.array([nl[1][2] for nl in pp_params[1:]])
print(nl_values)
