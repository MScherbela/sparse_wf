import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from sparse_wf.api import Electrons, HFOrbitals
from sparse_wf.jax_utils import replicate, copy_from_main, only_on_main_process


class HFWavefunction:
    def __init__(self, mol: pyscf.gto.Mole):
        self.mol = mol
        self.n_up, self.n_down = mol.nelec
        self.coeffs = jnp.zeros((mol.nao, mol.nao))
        with only_on_main_process():
            mf = mol.RHF()
            mf.kernel()
            self.coeffs = jnp.asarray(mf.mo_coeff)
        # We first copy for each local device and then synchronize across processes
        self.coeffs = copy_from_main(replicate(self.coeffs))[0]

    def _cpu_atomic_orbitals(self, electrons: np.ndarray):
        batch_shape = electrons.shape[:-1]
        ao_values = self.mol.eval_gto("GTOval_sph", electrons.reshape(-1, 3)).astype(electrons.dtype)
        return ao_values.reshape(*batch_shape, self.mol.nao)

    def hf_orbitals(self, electrons: Electrons) -> HFOrbitals:
        ao_orbitals = jax.pure_callback(  # type: ignore
            self._cpu_atomic_orbitals,
            jax.ShapeDtypeStruct((*electrons.shape[:-1], self.mol.nao), electrons.dtype),
            electrons,
            vectorized=True,
        )
        mo_values = jnp.array(ao_orbitals @ self.coeffs, electrons.dtype)

        up_orbitals = mo_values[..., : self.n_up, : self.n_up]
        down_orbitals = mo_values[..., self.n_up :, : self.n_down]
        return up_orbitals, down_orbitals

    def __call__(self, params, electrons: Electrons, static):
        """Compute log|psi| for the Hartree-Fock wavefunction."""
        up_orbitals, dn_orbitals = self.hf_orbitals(electrons)
        up_logdet = jnp.linalg.slogdet(up_orbitals)[1]
        dn_logdet = jnp.linalg.slogdet(dn_orbitals)[1]
        logpsi = up_logdet + dn_logdet
        return logpsi
