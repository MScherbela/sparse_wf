from typing import TypeVar

import folx
import jax.numpy as jnp

from sparse_wf.api import Charges, Electrons, LocalEnergy, Nuclei, ParameterizedWaveFunction
from sparse_wf.jax_utils import vectorize

P, S, MS = TypeVar("P"), TypeVar("S"), TypeVar("MS")


@vectorize(signature="(n,d),(m,d),(m)->()")
def potential_energy(r: Electrons, R: Nuclei, Z: Charges):
    """Compute the potential energy of the system"""
    dist_ee = jnp.triu(jnp.linalg.norm(r[:, None] - r, axis=-1), k=1)
    dist_en = jnp.linalg.norm(r[:, None] - R, axis=-1)
    dist_nn = jnp.linalg.norm(R[:, None, :] - R, axis=-1)

    E_ee = jnp.sum(jnp.triu(1 / dist_ee, k=1))
    E_en = -jnp.sum(Z / dist_en)
    E_nn = jnp.sum(jnp.triu(Z[:, None] * Z / dist_nn, k=1))

    return E_ee + E_en + E_nn


def make_local_energy(wf: ParameterizedWaveFunction[P, S, MS], R: Nuclei, Z: Charges, use_fwd_lap=True):
    """Create a local energy function from a wave function"""

    @vectorize(signature="(n,d)->()", excluded=frozenset({0, 2}))
    def local_energy(params: P, electrons: Electrons, static: S) -> LocalEnergy:
        """Compute the local energy of the system"""

        def closed_wf(electrons):
            return wf(params, electrons, static)

        if use_fwd_lap:
            laplacian, jacobian = folx.ForwardLaplacianOperator(0.6)(closed_wf)(electrons)
        else:
            laplacian, jacobian = folx.LoopLaplacianOperator()(closed_wf)(electrons)
        kinetic_energy = -0.5 * (laplacian.sum() + jnp.vdot(jacobian, jacobian))
        potential = potential_energy(electrons, R, Z)
        return kinetic_energy + potential

    return local_energy
