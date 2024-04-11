import jax.nn as jnn
from jaxtyping import Float, Array
from typing import Callable, Optional, Sequence, cast
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np
import functools

from sparse_wf.api import HFOrbitals, SlaterMatrices, SignedLogAmplitude
from sparse_wf.model.dense_ferminet import ElecInp, ElecNucDistances

FilterKernel = Float[Array, "neighbour features"]
Embedding = Float[Array, "features"]
NeighbourEmbeddings = Float[Embedding, "neighbours"]


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


def swap_bottom_blocks(matrix: Float[Array, "... n m"], n: int, m: int | None = None) -> Float[Array, "... n m"]:
    if m is None:
        m = n
    return jnp.concatenate(
        [
            matrix[..., :n, :],
            jnp.concatenate(
                [
                    matrix[..., n:, m:],
                    matrix[..., n:, :m],
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def signed_logpsi_from_orbitals(orbitals: SlaterMatrices) -> SignedLogAmplitude:
    slogdets = [jnp.linalg.slogdet(orbs) for orbs in orbitals]
    # For block-diagonal determinants, orbitals is a tuple of length 2. The following line is a fancy way to write
    # logdet, sign = logdet_up + logdet_dn, sign_up * sign_dn
    sign, logdet = functools.reduce(lambda x, y: (x[0] * y[0], x[1] + y[1]), slogdets, (1, 0))
    logpsi, signpsi = jnn.logsumexp(logdet, b=sign, return_sign=True)
    return (signpsi, logpsi)


class IsotropicEnvelope(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, dists: ElecNucDistances) -> jax.Array:
        sigma = nn.softplus(self.param("sigma", jnn.initializers.ones, (dists.shape[-2], self.out_dim), dists.dtype))
        pi = self.param("pi", jnn.initializers.ones, (dists.shape[-2], self.out_dim), dists.dtype)
        dists = dists[..., -1:]
        scaled_dists = dists * sigma
        env = jnp.exp(-scaled_dists)
        out = env * pi
        return out.sum(-2)  # sum over atom positions


class SlaterOrbitals(nn.Module):
    n_determinants: int
    spins: tuple[int, int]

    @nn.compact
    def __call__(self, h_one: ElecInp, dists: ElecNucDistances) -> SlaterMatrices:
        n = h_one.shape[0]
        spins = np.array(self.spins)
        orbitals = nn.Dense(self.n_determinants * n)(h_one)
        orbitals *= IsotropicEnvelope(self.n_determinants * n)(dists)
        orbitals = orbitals.reshape(n, self.n_determinants, n)
        orbitals = jnp.transpose(orbitals, (1, 0, 2))
        # reverse bottom two blocks
        orbitals = swap_bottom_blocks(orbitals, spins[0])
        return (orbitals,)

    @staticmethod
    def transform_hf_orbitals(hf_orbitals: HFOrbitals) -> SlaterMatrices:
        leading_shape = hf_orbitals[0].shape[:-2]
        n_up = hf_orbitals[0].shape[-1]
        n_down = hf_orbitals[1].shape[-1]
        full_det = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        hf_orbitals[0],
                        jnp.zeros((*leading_shape, n_up, n_down)),
                    ],
                    axis=-1,
                ),
                jnp.concatenate(
                    [
                        jnp.zeros((*leading_shape, n_down, n_up)),
                        hf_orbitals[1],
                    ],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
        # Add broadcast dimension for many determinants
        return (full_det[..., None, :, :],)
