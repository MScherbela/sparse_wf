#%%
from sparse_wf.scf import run_hf, run_cas, get_most_important_determinants
from sparse_wf.system import get_molecule
import numpy as np



DEFAULT_CACHE_DIR = "~/runs/pyscf_cache"

molecule_args = dict(
    method="database",
    database_args=dict(comment="cumulene_C8H4_90deg_singlet"),
    basis="ccecp-ccpvtz",
    pseudopotentials=["C"]
)
hf_args = dict(
    x2c=False,
    newton=True,
    restricted=False,
    restart=False,
    smearing=0.0,
    cache_dir=DEFAULT_CACHE_DIR,
    antiferromagnetic_broken_symmetry=False,
    init_calc=None,
    from_all_electron=False,
    xc=None,
    require_converged=True)

print("Running HF")
mol = get_molecule(molecule_args)
hf = run_hf(mol, hf_args)

print("Running CAS")
cas = run_cas(hf, 8, 8, s2=2)

#%%
idx_orbitals, ci_coeffs = get_most_important_determinants(cas, n_dets=10, threshold=0.01)

def remove_trivial_orbitals(idx_orbitals, spin):
    n_el = len(idx_orbitals[0])
    n_up = (n_el + spin) // 2
    print(n_up)

    is_trivial = np.zeros(np.max(idx_orbitals) + 1, dtype=bool)
    for idx in np.arange(len(is_trivial)):
        n_occurance = (idx_orbitals == idx).sum()
        if n_occurance == (len(idx_orbitals) * 2):
            is_trivial[idx] = True

    orbitals_filt_up = [[idx for idx in idx_orb[:n_up] if not is_trivial[idx]] for idx_orb in idx_orbitals]
    orbitals_filt_dn = [[idx for idx in idx_orb[n_up:] if not is_trivial[idx]] for idx_orb in idx_orbitals]
    return orbitals_filt_up, orbitals_filt_dn

idx_orb_up, idx_orb_dn = remove_trivial_orbitals(idx_orbitals, spin=0)

for idx_up, idx_dn, ci_coeff in zip(idx_orb_up, idx_orb_dn, ci_coeffs):
    print(f"{ci_coeff:+.3f}, {ci_coeff**2:.4f}: {idx_up} | {idx_dn}")
# logging.info(f"Selected {len(idx_orbitals)} determinants; sum of ci^2: {np.sum(ci_coeffs**2)}")
# self.idx_orbitals, self.ci_coeffs = split_large_determinants(idx_orbitals, ci_coeffs, n_determinants)
# logging.info("Final determinants:")
# for i, (idx, ci) in enumerate(zip(self.idx_orbitals, self.ci_coeffs)):
#     logging.info(f"CAS determinant {i:2d}: {ci:.5f}: {list(idx)}")
# self.idx_orbitals = jnp.asarray(self.idx_orbitals, dtype=jnp.int32)
# self.ci_coeffs = jnp.asarray(self.ci_coeffs, dtype=jnp.float32)
