import jax.nn as jnn
from jaxtyping import Float, Array
from typing import Callable, Optional, Sequence, cast
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np
import functools
from sparse_wf.api import HFOrbitals, SlaterMatrices, SignedLogAmplitude
import einops

ElecInp = Float[Array, "*batch n_electrons n_in"]
ElecNucDistances = Float[Array, "*batch n_electrons n_nuclei"]
ElecElecDistances = Float[Array, "*batch n_electrons n_electrons"]


FilterKernel = Float[Array, "neighbour features"]
Embedding = Float[Array, "features"]
NeighbourEmbeddings = Float[Embedding, "neighbours"]


class MLP(nn.Module):
    widths: Sequence[int]
    activate_final: bool = False
    activation: Callable = jax.nn.silu
    residual: bool = False

    @nn.compact
    def __call__(self, x: Float[Array, "*batch_dims _"]) -> Float[Array, "*batch_dims _"]:
        depth = len(self.widths)
        for ind_layer, out_width in enumerate(self.widths):
            y = nn.Dense(out_width)(x)
            if (ind_layer < depth - 1) or self.activate_final:
                y = self.activation(y)
                if self.residual and (x.shape[-1] == out_width):
                    x = x + y
                else:
                    x = y
        return x


def cutoff_function(d: Float[Array, "*dims"], p=4) -> Float[Array, "*dims"]:  # noqa: F821
    a = -(p + 1) * (p + 2) * 0.5
    b = p * (p + 2)
    c = -p * (p + 1) * 0.5
    cutoff = 1 + a * d**p + b * d ** (p + 1) + c * d ** (p + 2)
    # Heavyside is only required to enforce cutoff in fully connected implementation
    # and when evaluating the cutoff to a padding/placeholder "neighbour" which is far away
    # cutoff *= jax.numpy.heaviside(1 - d, 0.0)
    cutoff = jnp.where(d < 1, cutoff, 0.0)
    return cutoff


def contract(h_nb: NeighbourEmbeddings, Gamma_nb: FilterKernel, h_center: Optional[Embedding] = None):
    """Helper function implementing the message passing step"""

    # n = neighbour, f = feature
    msg = jnp.einsum("...nf,...nf->...f", h_nb, Gamma_nb)
    if h_center is not None:
        msg += h_center
    emb_out = jax.nn.silu(msg)
    return cast(Embedding, emb_out)


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
    return signpsi, logpsi

def get_dist_same_diff(electrons: ElecElecDistances, n_up):
    # Compute distances
    diff = electrons[..., None, :, :] - electrons[..., :, None, :]
    dists = jnp.linalg.norm(diff, axis=-1)

    # Get one copy of the distances between all electrons with the same spin
    upper_tri_indices = jnp.triu_indices(n_up, 1)
    dist_same_up = dists[..., :n_up, :n_up][..., upper_tri_indices[0], upper_tri_indices[1]]
    upper_tri_indices = jnp.triu_indices(dists.shape[-1] - n_up, 1)
    dist_same_down = dists[..., n_up:, n_up:][..., upper_tri_indices[0], upper_tri_indices[1]]
    dist_same = jnp.concatenate([dist_same_up, dist_same_down], axis=-1)
    # Get the distance between all electrons of different spin
    flat_shape = dists.shape[:-2] + (-1,)
    dist_diff = dists[..., :n_up, n_up:].reshape(flat_shape)

    jax.debug.print("electrons: {e}", e=electrons)
    jax.debug.print("dists: {d}", d=dists)
    jax.debug.print("dist_same: {d}", d=dist_same)
    jax.debug.print("dist_diff: {d}", d=dist_diff)

    return dist_same, dist_diff


class IsotropicEnvelope(nn.Module):
    out_dim: int
    cutoff: Optional[float] = None

    @nn.compact
    def __call__(self, dists: ElecNucDistances) -> jax.Array:
        n_nuc = dists.shape[-1]
        sigma = nn.softplus(self.param("sigma", jnn.initializers.ones, (n_nuc, self.out_dim), jnp.float32))
        pi = self.param("pi", jnn.initializers.ones, (n_nuc, self.out_dim), jnp.float32)
        scaled_dists = dists[..., None] * sigma
        env = jnp.exp(-scaled_dists)
        if self.cutoff is not None:
            env *= cutoff_function(dists / self.cutoff)
        out = env * pi
        return out.sum(axis=-2)  # sum over atom positions


class SlaterOrbitals(nn.Module):
    n_determinants: int
    spins: tuple[int, int]

    @nn.compact
    def __call__(self, h_one: ElecInp, dists: ElecNucDistances) -> SlaterMatrices:
        n_el = h_one.shape[-2]
        spins = np.array(self.spins)
        orbitals = nn.Dense(self.n_determinants * n_el)(h_one)
        orbitals *= IsotropicEnvelope(self.n_determinants * n_el)(dists)
        orbitals = einops.rearrange(
            orbitals, "... el (det orb) -> ... det el orb", el=n_el, orb=n_el, det=self.n_determinants
        )
        orbitals = swap_bottom_blocks(orbitals, spins[0])  # reverse bottom two blocks
        return (orbitals,)


def hf_orbitals_to_fulldet_orbitals(hf_orbitals: HFOrbitals) -> SlaterMatrices:
    dtype = hf_orbitals[0].dtype
    leading_shape = hf_orbitals[0].shape[:-2]
    n_up = hf_orbitals[0].shape[-1]
    n_down = hf_orbitals[1].shape[-1]
    full_det = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    hf_orbitals[0],
                    jnp.zeros((*leading_shape, n_up, n_down), dtype),
                ],
                axis=-1,
            ),
            jnp.concatenate(
                [
                    jnp.zeros((*leading_shape, n_down, n_up), dtype),
                    hf_orbitals[1],
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    # Add broadcast dimension for many determinants
    return (full_det[..., None, :, :],)
