from jaxtyping import Float, Array
from typing import Callable, Optional, Sequence, cast
import flax.linen as nn
import jax.numpy as jnp
import jax
from sparse_wf.api import Electrons, Nuclei, Charges

FilterKernel = Float[Array, "neighbour features"]
Embedding = Float[Array, "features"]
NeighbourEmbeddings = Float[Embedding, "neighbours"]


def nuclear_potential_energy(R: Nuclei, Z: Charges) -> Float[Array, ""]:
    """Compute the nuclear potential energy of the system"""
    dist = jnp.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    E_pot = Z[:, None] * Z[None, :] / dist
    E_pot = jnp.triu(E_pot, k=1)
    return jnp.sum(E_pot)


def potential_energy(r: Electrons, R: Nuclei, Z: Charges, E_nuc_nuc: Optional[Float[Array, ""]] = None):
    """Compute the potential energy of the system"""

    dist_ee = jnp.triu(jnp.linalg.norm(r[:, :, None] - r[:, None, :], axis=-1), k=1)
    dist_en = jnp.linalg.norm(r[..., :, None, :] - R, axis=-1)

    E_ee = jnp.sum(jnp.triu(1 / dist_ee, k=1))
    E_en = jnp.sum(Z / dist_en)
    if E_nuc_nuc is None:
        E_nuc_nuc = nuclear_potential_energy(R, Z)

    return E_ee + E_en + E_nuc_nuc


class MLP(nn.Module):
    widths: Sequence[int]
    activate_final: bool = False
    activation: Callable = jax.nn.silu

    @nn.compact
    def __call__(self, x: Float[Array, "*batch_dims _"]) -> Float[Array, "*batch_dims _"]:
        depth = len(self.widths)
        for ind_layer, out_width in enumerate(self.widths):
            x = nn.Dense(out_width)(x)
            if (ind_layer < depth - 1) or self.activate_final:
                x = self.activation(x)
        return x


def cutoff_function(d: Float[Array, "*dims"], p=4) -> Float[Array, "*dims"]:  # noqa: F821
    a = -(p + 1) * (p + 2) * 0.5
    b = p * (p + 2)
    c = -p * (p + 1) * 0.5
    cutoff = 1 + a * d**p + b * d ** (p + 1) + c * d ** (p + 2)
    # Heavyside is only required to enforce cutoff in fully connected implementation
    # and when evaluating the cutoff to a padding/placeholder "neighbour" which is far away
    cutoff *= jax.numpy.heaviside(1 - d, 0.0)
    return cutoff


def contract(h_nb: NeighbourEmbeddings, Gamma_nb: FilterKernel, h_center: Optional[Embedding] = None):
    """Helper function implementing the message passing step"""

    # n = neighbour, f = feature
    msg = jnp.einsum("...nf,...nf->...f", h_nb, Gamma_nb)
    if h_center is not None:
        msg += h_center
    return cast(Embedding, jax.nn.silu(msg))
