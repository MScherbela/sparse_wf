from copy import copy
import hashlib
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import pyscf
import pyscf.scf
import rich
from sparse_wf.api import Electrons, HFArgs, HFOrbitals
from sparse_wf.jax_utils import replicate, copy_from_main, only_on_main_process
from typing import NamedTuple
from pyscf.fci.cistring import _gen_occslst
import pyscf.mcscf
import logging


def hash_mol(mol: pyscf.gto.Mole):
    # Returns a unique hash for each pyscf molecule.
    # This is not robust to permutations of the atoms.
    mol_vars = copy(vars(mol))
    del mol_vars["stdout"]
    for k in list(mol_vars.keys()):
        if k.startswith("_"):
            del mol_vars[k]
    return hashlib.sha256(pickle.dumps(mol_vars)).hexdigest()


def mean_field_chkfile(mol: pyscf.gto.Mole, args: HFArgs):
    method = "RHF" if args["restricted"] else "UHF"
    x2c = "_x2c" if args["x2c"] else ""
    file_name = Path(f"{hash_mol(mol)}_{method}{x2c}.h5")
    cache_dir = Path(args["cache_dir"]).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / file_name


class CASResult(NamedTuple):
    mo_coeff: np.ndarray  # [n_basis x n_orbitals]
    ci_coeffs: np.ndarray  # [n_dets_up x n_dets_dn]
    idx_orbitals_up: np.ndarray  # [n_dets_up x n_el]
    idx_orbitals_dn: np.ndarray  # [n_dets_dn x n_el]
    energy: float
    s2: float  # spin operator S^2


def run_hf(mol: pyscf.gto.Mole, args: HFArgs):
    logging.info("Running HF with args:")
    rich.print_json(data=args)
    if args["restricted"]:
        hf = pyscf.scf.RHF(mol)
    else:
        hf = pyscf.scf.UHF(mol)
    # output energy at each iteration
    hf.verbose = 4

    # Relativistic effects
    if args["x2c"]:
        hf = hf.x2c()

    # Load from checkpoint if available
    if args["cache_dir"]:
        hf.chkfile = mean_field_chkfile(mol, args)
        logging.info(f"Using chkfile: {hf.chkfile}")
        if not args["restart"]:
            hf.init_guess = "chk"

    # HF convergence settings
    if args["newton"]:
        hf = hf.newton()
    if args["smearing"] > 0:
        hf = pyscf.scf.addons.smearing_(hf, sigma=args["smearing"], method="fermi")
        hf.max_cycle = 5
        hf.kernel()
        logging.info("HF smearing completed. Starting main SCF.")
        hf.sigma = 1e-6

    # Run HF
    hf.max_cycle = 100
    try:
        hf.kernel()
    except TypeError:
        logging.info("HF failed loading.")
        hf.init_guess = "minao"
        hf.kernel()
    logging.info(f"HF energy: {hf.e_tot}")
    return hf


def run_cas(hf, n_orbitals, n_electrons, s2=None):
    # Run CAS
    cas = pyscf.mcscf.CASCI(hf, n_orbitals, n_electrons)
    if (s2 is not None) and (s2 >= 0):
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
        s2=pyscf.mcscf.spin_square(cas)[0],
    )


def get_most_important_determinants(cas: CASResult, n_dets, threshold=0.05):
    is_large = (cas.ci_coeffs**2) > threshold
    idx_large = np.array(np.where(is_large)).T
    ci_coeffs_large = cas.ci_coeffs[idx_large[:, 0], idx_large[:, 1]]
    idx_sort = np.argsort(ci_coeffs_large**2)[::-1]
    idx_large = idx_large[idx_sort]
    ci_coeffs_large = ci_coeffs_large[idx_sort]
    if len(idx_large) > n_dets:
        idx_large = idx_large[:n_dets]
        ci_coeffs_large = ci_coeffs_large[:n_dets]
    idx_orbitals = np.concatenate(
        [cas.idx_orbitals_up[idx_large[:, 0]], cas.idx_orbitals_dn[idx_large[:, 1]]],
        axis=1,
    )
    return idx_orbitals, ci_coeffs_large


def split_large_determinants(idx_orbitals, ci_coeffs, n_dets):
    multiplicities = np.ones(len(ci_coeffs), dtype=int)
    while np.sum(multiplicities) < n_dets:
        idx_max = np.argmax(np.abs(ci_coeffs))
        ci_coeffs[idx_max] *= multiplicities[idx_max] / (multiplicities[idx_max] + 1)
        multiplicities[idx_max] += 1
    ci_coeffs = np.repeat(ci_coeffs, multiplicities)
    idx_orbitals = np.repeat(idx_orbitals, multiplicities, axis=0)
    return idx_orbitals, ci_coeffs


def _cpu_atomic_orbitals(mol, electrons: np.ndarray):
    batch_shape = electrons.shape[:-1]
    ao_values = mol.eval_gto("GTOval_sph", electrons.reshape(-1, 3)).astype(electrons.dtype)
    return ao_values.reshape(*batch_shape, mol.nao)


def _eval_atomic_orbitals(mol, electrons: Electrons):
    return jax.pure_callback(  # type: ignore
        lambda r: _cpu_atomic_orbitals(mol, r),
        jax.ShapeDtypeStruct((*electrons.shape[:-1], mol.nao), electrons.dtype),
        electrons,
        vectorized=True,
    )


def eval_molecular_orbitals(mol, coeffs, electrons: Electrons):
    n_up, n_dn = mol.nelec
    ao = _eval_atomic_orbitals(mol, electrons)
    if coeffs.ndim == 2:
        mo_values = jnp.array(ao @ coeffs, electrons.dtype)
        up_orbitals = mo_values[..., :n_up, :]
        dn_orbitals = mo_values[..., n_up:, :]
    else:
        up_orbitals = jnp.array(ao[..., :n_up, :] @ coeffs[0], electrons.dtype)
        dn_orbitals = jnp.array(ao[..., n_up:, :] @ coeffs[1], electrons.dtype)
    return up_orbitals, dn_orbitals


class HFWavefunction:
    def __init__(self, mol: pyscf.gto.Mole, args: HFArgs):
        self.mol = mol
        self.n_up, self.n_dn = mol.nelec
        self.n_el = self.n_up + self.n_dn
        if args["restricted"]:
            self.mo_coeffs = jnp.zeros((mol.nao, mol.nao))
        else:
            self.mo_coeffs = jnp.zeros((2, mol.nao, mol.nao))
        with only_on_main_process():
            self.hf = run_hf(mol, args)
            self.mo_coeffs = jnp.asarray(self.hf.mo_coeff)  # type: ignore
        # We first copy for each local device and then synchronize across processes
        self.mo_coeffs = copy_from_main(replicate(self.mo_coeffs))[0]

    def orbitals(self, electrons: Electrons) -> HFOrbitals:
        mo_up, mo_dn = eval_molecular_orbitals(self.mol, self.mo_coeffs, electrons)
        mo_up = mo_up[..., None, :, : self.n_up]  # [batch x det x el x orb]
        mo_dn = mo_dn[..., None, :, : self.n_dn]
        return mo_up, mo_dn

    def excited_signed_logpsi(self, mo_indices: jnp.ndarray, electrons: Electrons):
        n_states, n_orb = mo_indices.shape
        assert n_orb == electrons.shape[-2]

        mo_up, mo_dn = eval_molecular_orbitals(self.mol, self.mo_coeffs, electrons)
        mo_up = mo_up[..., mo_indices[:, : self.n_up]]  # [batch x el x state x orb]
        mo_dn = mo_dn[..., mo_indices[:, self.n_up :]]
        mo_up = jnp.moveaxis(mo_up, -2, -3)  # [batch x state x el x orb]
        mo_dn = jnp.moveaxis(mo_dn, -2, -3)

        sign_up, logdet_up = jnp.linalg.slogdet(mo_up)
        sign_dn, logdet_dn = jnp.linalg.slogdet(mo_dn)
        logpsi = logdet_up + logdet_dn
        sign = sign_up * sign_dn
        return sign, logpsi

    def __call__(self, params, electrons: Electrons, static):
        """Compute log|psi| for the Hartree-Fock wavefunction."""
        up_orbitals, dn_orbitals = self.orbitals(electrons)
        up_logdet = jnp.linalg.slogdet(up_orbitals)[1]
        dn_logdet = jnp.linalg.slogdet(dn_orbitals)[1]
        logpsi = up_logdet + dn_logdet
        return logpsi


class CASWavefunction(HFWavefunction):
    def __init__(
        self,
        mol: pyscf.gto.Mole,
        args: HFArgs,
        n_determinants: int,
        active_orbitals: int,
        active_electrons: int,
        det_threshold: float,
        s2: float,
    ):
        super().__init__(mol, args)
        self.mol = mol
        self.idx_orbitals = jnp.zeros([n_determinants, self.n_el], dtype=jnp.int32)
        self.ci_coeffs = jnp.zeros([n_determinants], dtype=jnp.float32)

        with only_on_main_process():
            active_orbitals, active_electrons = self.sanity_check_active_space(active_orbitals, active_electrons)
            cas_result = run_cas(self.hf, active_orbitals, active_electrons, s2)
            self.cas_result = cas_result
            logging.info(f"CASCI energy: {cas_result.energy}")
            idx_orbitals, ci_coeffs = get_most_important_determinants(cas_result, n_determinants, det_threshold)
            logging.info(f"Selected {len(idx_orbitals)} determinants; sum of ci^2: {np.sum(ci_coeffs**2)}")
            self.idx_orbitals, self.ci_coeffs = split_large_determinants(idx_orbitals, ci_coeffs, n_determinants)
            logging.info("Final determinants:")
            for i, (idx, ci) in enumerate(zip(self.idx_orbitals, self.ci_coeffs)):
                logging.info(f"CAS determinant {i:2d}: {ci:.5f}: {list(idx)}")
            self.idx_orbitals = jnp.asarray(self.idx_orbitals, dtype=jnp.int32)
            self.ci_coeffs = jnp.asarray(self.ci_coeffs, dtype=jnp.float32)
        self.idx_orbitals = copy_from_main(replicate(self.idx_orbitals))[0]
        self.ci_coeffs = copy_from_main(replicate(self.ci_coeffs))[0]

    def sanity_check_active_space(self, active_orb, active_el):
        # All nrs of electrons/orbitals in this function refer to the number per spin
        assert (
            self.mol.nelec[0] == self.mol.nelec[1]
        ), "CASCI currently only supports equal number of spin-up and spin-down electrons"
        n_el = self.mol.nelec[0]
        active_el = min(active_el, n_el)
        n_core = n_el - active_el
        active_orb = min(active_orb, self.mol.nao - n_core)
        return active_orb, active_el

    def orbitals(self, electrons: Electrons):
        mo_up, mo_dn = eval_molecular_orbitals(self.mol, self.mo_coeffs, electrons)
        n_el = self.n_up + self.n_dn
        idx_up = self.idx_orbitals[:, : self.n_up]  # [n_dets x n_up]
        idx_dn = self.idx_orbitals[:, self.n_up :]
        mo_up = jnp.moveaxis(mo_up[..., idx_up], -2, -3)  # [el x det x orb] -> [det x el x orb]
        mo_dn = jnp.moveaxis(mo_dn[..., idx_dn], -2, -3)

        ci_weights = np.abs(self.ci_coeffs) ** (1 / n_el)
        mo_up = mo_up * ci_weights[:, None, None]
        mo_dn = mo_dn * ci_weights[:, None, None]

        # adjust the sign of the first orbital to yield the correct sign of the determinant
        ci_signs = np.sign(self.ci_coeffs)
        mo_up = mo_up.at[..., 0].multiply(ci_signs[:, None])
        # do NOT adjust the sign of the down orbitals as well, because the sign would then cancel
        return mo_up, mo_dn
