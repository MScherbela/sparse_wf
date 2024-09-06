import jax
import jax.numpy as jnp
import numpy as np
import pyscf
import pyscf.scf
from sparse_wf.api import Electrons, HFOrbitals
from sparse_wf.jax_utils import replicate, copy_from_main, only_on_main_process


class HFWavefunction:
    def __init__(self, mol: pyscf.gto.Mole):
        self.mol = mol
        self.n_up, self.n_down = mol.nelec
        self.coeffs = jnp.zeros((mol.nao, mol.nao))
        with only_on_main_process():
            mf = mol.RHF()
            # Run a few steps with smeared orbital occupation to avoid local minima
            mf = pyscf.scf.addons.smearing_(mf, sigma=0.1, method="fermi")
            mf.max_cycle = 10
            mf.kernel()
            # Run without smearing to get actual ground state
            mf.sigma = 1e-9
            mf.max_cycle = 50
            mf.kernel()
            self.coeffs = jnp.asarray(mf.mo_coeff)
        # We first copy for each local device and then synchronize across processes
        self.coeffs = copy_from_main(replicate(self.coeffs))[0]

    def _cpu_atomic_orbitals(self, electrons: np.ndarray):
        batch_shape = electrons.shape[:-1]
        ao_values = self.mol.eval_gto("GTOval_sph", electrons.reshape(-1, 3)).astype(electrons.dtype)
        return ao_values.reshape(*batch_shape, self.mol.nao)

    def _eval_ao_orbitals(self, electrons: Electrons):
        return jax.pure_callback(  # type: ignore
            self._cpu_atomic_orbitals,
            jax.ShapeDtypeStruct((*electrons.shape[:-1], self.mol.nao), electrons.dtype),
            electrons,
            vectorized=True,
        )

    def hf_orbitals(self, electrons: Electrons) -> HFOrbitals:
        ao_orbitals = self._eval_ao_orbitals(electrons)
        coeffs_occ = self.coeffs[:, : max(self.n_up, self.n_down)]
        mo_values = jnp.array(ao_orbitals @ coeffs_occ, electrons.dtype)

        # TODO: revert!!
        idx_occ = np.arange(self.n_up)
        idx_occ[-1] += 1  # use LUMO instad of HOMO

        # up_orbitals = mo_values[..., : self.n_up, : self.n_up]
        # down_orbitals = mo_values[..., self.n_up :, : self.n_down]
        up_orbitals = mo_values[..., : self.n_up, idx_occ]
        dn_orbitals = mo_values[..., self.n_up :, idx_occ]
        return up_orbitals, dn_orbitals

    def excited_signed_logpsi(self, mo_indices: jnp.ndarray, electrons: Electrons):
        n_states, n_orb = mo_indices.shape
        assert n_orb == electrons.shape[-2]

        ao_orbitals = self._eval_ao_orbitals(electrons)
        mos = jnp.array(ao_orbitals @ self.coeffs, electrons.dtype)
        mo_up = mos[..., : self.n_up, mo_indices[:, : self.n_up]]  # [batch x el x state x orb]
        mo_dn = mos[..., self.n_up :, mo_indices[:, self.n_up :]]
        mo_up = jnp.moveaxis(mo_up, -2, -3)  # [batch x state x el x orb]
        mo_dn = jnp.moveaxis(mo_dn, -2, -3)

        sign_up, logdet_up = jnp.linalg.slogdet(mo_up)
        sign_dn, logdet_dn = jnp.linalg.slogdet(mo_dn)
        logpsi = logdet_up + logdet_dn
        sign = sign_up * sign_dn
        return sign, logpsi

    def __call__(self, params, electrons: Electrons, static):
        """Compute log|psi| for the Hartree-Fock wavefunction."""
        up_orbitals, dn_orbitals = self.hf_orbitals(electrons)
        up_logdet = jnp.linalg.slogdet(up_orbitals)[1]
        dn_logdet = jnp.linalg.slogdet(dn_orbitals)[1]
        logpsi = up_logdet + dn_logdet
        return logpsi
