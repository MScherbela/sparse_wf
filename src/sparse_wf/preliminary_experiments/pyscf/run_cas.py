#%%
import pyscf
from pyscf.fci.cistring import _gen_occslst
import pyscf.ci
import pyscf.scf
import pyscf.mcscf
import numpy as np
from typing import NamedTuple

class CASResult(NamedTuple):
    mo_coeff: np.ndarray # [n_basis x n_orbitals]
    ci_coeffs: np.ndarray # [n_dets_up x n_dets_dn]
    idx_orbitals_up: np.ndarray # [n_dets_up x n_el]
    idx_orbitals_dn: np.ndarray # [n_dets_dn x n_el]
    energy: float
    s2: float # spin operator S^2


def run_hf(mol):
    hf = pyscf.scf.RHF(mol)
    hf.verbose = 0
    hf = pyscf.scf.addons.smearing_(hf, sigma=0.05, method='fermi')
    hf.max_cycle = 5
    hf.kernel()

    hf.sigma = 1e-6
    hf.max_cycle = 50
    hf.kernel()
    return hf

def run_cas(hf, n_orbitals, n_electrons, s2=None):
    # Run CAS
    cas = pyscf.mcscf.CASCI(hf, n_orbitals, n_electrons)
    if s2 is not None:
        cas.fix_spin_(ss=s2)
    cas.kernel()

    # Get list of orbitals
    n_core_orbitals = cas.ncore
    n_active_orbitals = cas.ncas
    n_active_el_up, n_active_el_dn = cas.nelecas

    idx_core_orb = np.arange(n_core_orbitals)
    orb_list_up = _gen_occslst(np.arange(n_active_orbitals) + n_core_orbitals, n_active_el_up)
    orb_list_up = np.array([np.concatenate([idx_core_orb, orb]) for orb in orb_list_up], int)
    orb_list_dn = _gen_occslst(np.arange(n_active_orbitals) + n_core_orbitals, n_active_el_dn)
    orb_list_dn = np.array([np.concatenate([idx_core_orb, orb]) for orb in orb_list_dn])

    ci_coeffs = np.array(cas.ci)
    assert ci_coeffs.shape == (len(orb_list_up), len(orb_list_dn))
    return CASResult(
        mo_coeff=hf.mo_coeff,
        ci_coeffs=ci_coeffs,
        idx_orbitals_up=orb_list_up,
        idx_orbitals_dn=orb_list_dn,
        energy=cas.e_tot,
        s2=pyscf.mcscf.spin_square(cas)[0]
    ), cas


def get_most_important_determinants(cas: CASResult, n_dets, threshold=0.05):
    is_large = (cas.ci_coeffs**2) > threshold
    idx_large = np.array(np.where(is_large)).T
    ci_coeffs_large = cas.ci_coeffs[idx_large[:, 0], idx_large[:, 1]]
    idx_sort = np.argsort(ci_coeffs_large**2)[::-1]
    idx_large = idx_large[idx_sort]
    if len(idx_large) > n_dets:
        idx_large = idx_large[:n_dets]
        ci_coeffs_large = ci_coeffs_large[:n_dets]
    idx_orbitals = np.concatenate([cas.idx_orbitals_up[idx_large[:, 0]],
                                   cas.idx_orbitals_dn[idx_large[:, 1]]], axis=1)
    return idx_orbitals, ci_coeffs_large

def split_large_determinants(idx_orbitals, ci_coeffs, n_dets):
    while len(ci_coeffs) < n_dets:
        idx_max = np.argmax(ci_coeffs**2)
        ci_coeffs[idx_max] /= 2
        ci_coeffs = np.append(ci_coeffs, ci_coeffs[idx_max])
        idx_orbitals = np.concatenate([idx_orbitals, idx_orbitals[idx_max, None]], axis=0)
    return idx_orbitals, ci_coeffs


if __name__ == "__main__":
    from sparse_wf.geometry import load_geometries
    all_geoms = load_geometries()
    geoms = [g for g in all_geoms.values() if g.comment == 'cumulene_C10H4_90deg']
    assert len(geoms) == 1
    geom = geoms[0]
    basis_set = "sto-6g"
    mol = geom.as_pyscf_molecule(basis_set)

    hf = run_hf(mol)
    cas_result, cas = run_cas(hf, 8, (4, 4))


    n_dets = 16
    idx_orbitals, ci_coeffs_large = get_most_important_determinants(cas_result, n_dets)
    print(idx_orbitals)

    print("Splitting large determinants")
    idx_orbitals, ci_coeffs_large = split_large_determinants(idx_orbitals, ci_coeffs_large, n_dets)
    print(idx_orbitals)


