import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from sparse_wf.api import Electrons, HFOrbitalFn, HFOrbitals
from sparse_wf.jax_utils import replicate, copy_from_main, only_on_main_process


def make_hf_orbitals(mol: pyscf.gto.Mole) -> HFOrbitalFn:
    coeffs = jnp.zeros((mol.nao, mol.nao))
    with only_on_main_process():
        best_mf = mol.RHF()
        best_mf.kernel()
        best_energy = best_mf.e_tot
        for _ in range(30):
            mf = mol.RHF()
            mf.kernel()
            if mf.e_tot < best_energy:
                best_mf = mf
                best_energy = mf.e_tot
        mf = best_mf
        coeffs = jnp.asarray(mf.mo_coeff)
    # We first copy for each local device and then synchronize across processes
    coeffs = copy_from_main(replicate(coeffs))[0]
    n_up, n_down = mol.nelec

    def cpu_atomic_orbitals(electrons: np.ndarray):
        batch_shape = electrons.shape[:-1]
        ao_values = mol.eval_gto("GTOval_sph", electrons.reshape(-1, 3)).astype(electrons.dtype)
        return ao_values.reshape(*batch_shape, mol.nao)

    def hf_orbitals(electrons: Electrons) -> HFOrbitals:
        ao_orbitals = jax.pure_callback(  # type: ignore
            cpu_atomic_orbitals,
            jax.ShapeDtypeStruct((*electrons.shape[:-1], mol.nao), electrons.dtype),
            electrons,
            vectorized=True,
        )
        mo_values = jnp.array(ao_orbitals @ coeffs, electrons.dtype)

        orbitals = np.arange(n_up)
        up_orbitals = mo_values[..., None, :n_up, orbitals]
        down_orbitals = mo_values[..., None, n_up:, orbitals]
        up_orbitals_excited = mo_values[..., None, :n_up, orbitals]
        down_orbitals_excited = mo_values[..., None, n_up:, orbitals]
        up_orbitals = (up_orbitals + up_orbitals_excited) / jnp.sqrt(2)
        down_orbitals = (down_orbitals + down_orbitals_excited) / jnp.sqrt(2)
        return up_orbitals, down_orbitals

    return hf_orbitals


def make_hf_logpsi(hf_orbitals: HFOrbitalFn):
    def logpsi(params, electrons: Electrons, static):
        up_orbitals, dn_orbitals = hf_orbitals(electrons)
        up_logdet = jnp.linalg.slogdet(up_orbitals)[1]
        dn_logdet = jnp.linalg.slogdet(dn_orbitals)[1]
        logpsi = up_logdet + dn_logdet
        return logpsi

    return logpsi
