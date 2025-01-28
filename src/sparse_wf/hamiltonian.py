from typing import Literal, TypeVar, Sequence

import folx
import jax
import jax.numpy as jnp

from sparse_wf.api import Charges, Electrons, LocalEnergy, Nuclei, ParameterizedWaveFunction, StaticInput
from sparse_wf.jax_utils import vectorize
from sparse_wf.pseudopotentials import make_pseudopotential

P, MS = TypeVar("P"), TypeVar("MS")


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


def make_kinetic_energy(wf: ParameterizedWaveFunction[P, MS], use_fwd_lap=True):
    """Create a kinetic energy function from a wave function"""

    @vectorize(signature="(n,d)->()", excluded=frozenset({0, 2}))
    def kinetic(params: P, electrons: Electrons, static: StaticInput) -> LocalEnergy:
        """Compute the local energy of the system"""

        def closed_wf(electrons):
            return wf(params, electrons, static)

        if use_fwd_lap:
            laplacian, jacobian = folx.ForwardLaplacianOperator(0.6)(closed_wf)(electrons)
        else:
            laplacian, jacobian = folx.LoopLaplacianOperator()(closed_wf)(electrons)
        kinetic_energy = -0.5 * (laplacian.sum() + jnp.vdot(jacobian, jacobian))
        return kinetic_energy

    return kinetic


def make_local_energy(
    wf: ParameterizedWaveFunction[P, MS],
    energy_operator: Literal["sparse", "dense"],
    pseudopotentials: Sequence[str],  # list of atoms for which to use pseudopotentials
    pp_grid_points: dict[str, int],
):
    """Create a local energy function from a wave function"""
    match energy_operator.lower():
        case "dense":
            kin_fn = wf.kinetic_energy_dense
        case "sparse":
            kin_fn = wf.kinetic_energy
        case _:
            raise ValueError(f"Unknown energy operator: {energy_operator}")

    eff_charges, pp_local, pp_nonlocal = make_pseudopotential(wf.Z, pseudopotentials, pp_grid_points)

    def local_energy(
        key: jax.Array, params: P, electrons: Electrons, static: StaticInput
    ) -> tuple[LocalEnergy, StaticInput]:
        """Compute the local energy of the system"""
        kinetic_energy = kin_fn(params, electrons, static)
        potential = potential_energy(electrons, wf.R, eff_charges)
        potential += pp_local(electrons, wf.R)
        nl_pp, new_static = pp_nonlocal(key, wf, params, electrons, static)
        potential += nl_pp
        return kinetic_energy + potential, new_static

    return local_energy
