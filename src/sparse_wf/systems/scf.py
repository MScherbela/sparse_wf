import jax
import numpy as np
import pyscf
from sparse_wf.api import Electrons, HFOrbitalFn, HFOrbitals
from sparse_wf.systems.molecule import Molecule


def make_hf_orbitals(molecule: Molecule | pyscf.gto.Mole, basis: str) -> HFOrbitalFn:
    if isinstance(molecule, Molecule):
        mol = molecule.to_pyscf(basis=basis)
    else:
        mol = molecule
        mol.basis = basis
        mol.build()
    mf = mol.RHF()
    mf.kernel()

    num_ao = len(mf.ao_labels())
    coeffs = mf.mo_coeff

    spin_up, spin_down = molecule.spins

    def cpu_atomic_orbitals(electrons: np.ndarray):
        batch_shape = electrons.shape[:-1]
        ao_values = mol.eval_gto("GTOval_sph", electrons.reshape(-1, 3)).astype(electrons.dtype)
        return ao_values.reshape(*batch_shape, num_ao)

    def hf_orbitals(electrons: Electrons) -> HFOrbitals:
        ao_orbitals = jax.pure_callback(  # type: ignore
            cpu_atomic_orbitals,
            jax.ShapeDtypeStruct((electrons.shape[:-1], num_ao), electrons.dtype),
            electrons,
            vectorized=True,
        )
        mo_values = ao_orbitals @ coeffs

        up_orbitals = mo_values[..., :spin_up, :spin_up]
        down_orbitals = mo_values[..., spin_up:, :spin_down]
        return up_orbitals, down_orbitals

    return hf_orbitals
