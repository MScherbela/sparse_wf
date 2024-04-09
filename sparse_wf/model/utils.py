from jaxtyping import Float, Array
from typing import Callable, Optional, Sequence, cast
import flax.linen as nn
import jax.numpy as jnp
import jax

FilterKernel = Float[Array, "neighbour features"]
Embedding = Float[Array, "features"]


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


def contract(h_nb: Embedding, Gamma_nb: FilterKernel, h_center: Optional[FilterKernel] = None):
    """Helper function implementing the message passing step"""
    msg = jnp.einsum("...ij,...ij->...j", h_nb, Gamma_nb)
    if h_center is not None:
        msg += h_center
    return cast(Embedding, jax.nn.silu(msg))
