import jax.nn as jnn
from jaxtyping import Float, Array, ArrayLike
from typing import Callable, Optional, Sequence, cast, NamedTuple
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np
import functools
from sparse_wf.api import Electrons, HFOrbitals, Int, SlaterMatrices, SignedLogAmplitude
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
    use_bias: bool = True
    output_bias: bool = True

    @nn.compact
    def __call__(self, x: Float[Array, "*batch_dims _"]) -> Float[Array, "*batch_dims _"]:
        depth = len(self.widths)
        for ind_layer, out_width in enumerate(self.widths):
            is_output_layer = ind_layer == (depth - 1)

            if is_output_layer:
                y = nn.Dense(out_width, use_bias=self.output_bias)(x)
            else:
                y = nn.Dense(out_width, use_bias=self.use_bias)(x)

            if (not is_output_layer) or self.activate_final:
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

    return dist_same, dist_diff


def param_initializer(value: float):
    def init(key, shape, dtype) -> Array:
        return value * jnp.ones(shape, dtype)

    return init


def truncated_normal_with_mean_initializer(mean: float, stddev=0.01):
    def init(key, shape, dtype) -> Array:
        return mean + nn.initializers.truncated_normal(stddev)(key, shape, dtype)

    return init


class JastrowFactor(nn.Module):
    embedding_n_hidden: Optional[Sequence[int]]
    soe_n_hidden: Optional[Sequence[int]]
    init_with_zero: bool = False

    @nn.compact
    def __call__(self, embeddings):
        """
        There are three options here (i is the electron index):
        (1) J_i = MLP_[embedding_n_hidden, 1](embeddings) , J=sum(J_i)
        (2) J_i = MLP_[embedding_n_hidden](embeddings), J=MLP_[soe_n_hidden, 1](sum(J_i))
        (3) J=MLP_[soe_n_hidden, 1](sum(embeddings_i))
        """
        if self.embedding_n_hidden is None and self.soe_n_hidden is None:
            raise KeyError("Either embedding_n_hidden or soe_n_hidden must be specified when using mlp jastrow.")

        if self.embedding_n_hidden is not None:
            if self.soe_n_hidden is None:  # Option (1)
                jastrow = jnp.squeeze(
                    MLP([*self.embedding_n_hidden, 1], activate_final=False, residual=False, output_bias=False)(
                        embeddings
                    ),
                    axis=-1,
                )
                jastrow = jnp.sum(jastrow, axis=-1)
            else:  # Option (2) part 1
                jastrow = MLP(self.embedding_n_hidden, activate_final=False, residual=False, output_bias=False)(
                    embeddings
                )
        else:  # Option (3) part 2
            jastrow = embeddings

        if self.soe_n_hidden is not None:  # Option (2 or 3)
            jastrow = jnp.sum(jastrow, axis=-2)  # Sum over electrons.
            jastrow = jnp.squeeze(
                MLP([*self.soe_n_hidden, 1], activate_final=False, residual=False, output_bias=False)(jastrow), axis=-1
            )

        return jastrow


class IsotropicEnvelope(nn.Module):
    n_determinants: int
    n_orbitals: int
    envelope_size: int
    cutoff: Optional[float] = None

    @nn.compact
    def __call__(self, dists: ElecNucDistances) -> jax.Array:
        n_nuc = dists.shape[-1]
        sigma = self.param(
            "sigma", truncated_normal_with_mean_initializer(1, 0.2), (n_nuc, self.envelope_size), jnp.float32
        )
        sigma = nn.softplus(sigma)
        # pi = self.param(
        #     "pi", jnn.initializers.lecun_normal, (n_nuc, self.n_orbitals * self.n_determinants, self.envelope_size), jnp.float32
        # )
        scaled_dists = dists[..., None] * sigma
        env = jnp.exp(-scaled_dists)
        if self.cutoff is not None:
            env *= cutoff_function(dists / self.cutoff)
        env = env.reshape(-1)  # [atom x envelopes] => [atom*envelopes]
        out = nn.Dense(
            self.n_orbitals * self.n_determinants,
            use_bias=False,
        )(env)
        return out


class SlaterOrbitals(nn.Module):
    n_determinants: int
    envelope_size: int
    spins: tuple[int, int]

    @nn.compact
    def __call__(self, h_one: ElecInp, dists: ElecNucDistances) -> SlaterMatrices:
        n_el = h_one.shape[-2]
        spins = np.array(self.spins)
        orbitals = nn.Dense(self.n_determinants * n_el)(h_one)
        orbitals *= IsotropicEnvelope(self.n_determinants, n_el, self.envelope_size)(dists)
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


class DynamicFilterParams(NamedTuple):
    scales: jax.Array
    kernel: jax.Array
    bias: jax.Array


def scale_initializer(cutoff, rng, shape, dtype=jnp.float32):
    n_scales = shape[-1]
    scale = jnp.linspace(0, cutoff, n_scales, dtype=dtype)
    scale *= 1 + 0.1 * jax.random.normal(rng, shape, dtype)
    return scale


class PairwiseFilter(nn.Module):
    cutoff: float
    pair_dim: int

    @nn.compact
    def __call__(
        self, dist_diff: Float[Array, "*batch_dims features_in"], dynamic_params: DynamicFilterParams
    ) -> Float[Array, "*batch_dims features_out"]:
        """Compute the pairwise filters between two particles.

        Args:
            dist_diff: The distance, 3D difference and optional spin difference between two particles [n_el x n_nb x 4(5)].
                The 0th feature dimension must contain the distance, the remaining dimensions can contain arbitrary
                features that are used to compute the pairwise filters, e.g. product of spins.
        """
        # Direction- (and spin-) dependent MLP
        directional_features = jax.nn.silu(dist_diff @ dynamic_params.kernel + dynamic_params.bias)
        directional_features = nn.Dense(self.pair_dim)(directional_features)

        # Distance-dependenet radial filters
        dist = dist_diff[..., 0]
        # scales = self.param("scales", self.scale_initializer, (self.n_envelopes,), jnp.float32)
        scales = jax.nn.softplus(dynamic_params.scales)
        envelopes = jnp.exp(-((dist[..., None] / scales) ** 2))
        envelopes *= cutoff_function(dist / self.cutoff)[..., None]
        envelopes = nn.Dense(self.pair_dim, use_bias=False)(envelopes)
        beta = directional_features * envelopes
        return beta


class ElElCusp(nn.Module):
    n_up: int

    @nn.compact
    def __call__(self, electrons: Electrons) -> Float[Array, " *batch_dims"]:
        dist_same, dist_diff = get_dist_same_diff(electrons, self.n_up)

        alpha_same = self.param("alpha_same", nn.initializers.ones, (), jnp.float32)
        alpha_diff = self.param("alpha_diff", nn.initializers.ones, (), jnp.float32)
        # factor_same = self.param("factor_same", param_initializer(-0.25), (), jnp.float32)
        # factor_diff = self.param("factor_diff", param_initializer(-0.5), (), jnp.float32)
        factor_same, factor_diff = -0.25, -0.5

        cusp_same = jnp.sum(alpha_same**2 / (alpha_same + dist_same), axis=-1)
        cusp_diff = jnp.sum(alpha_diff**2 / (alpha_diff + dist_diff), axis=-1)

        return factor_same * cusp_same + factor_diff * cusp_diff


def get_diff_features(
    r: Float[ArrayLike, "dim=3"],
    r_nb: Float[Array, "dim=3"],
    s: Optional[Int] = None,
    s_nb: Optional[Int] = None,
):
    diff = r - r_nb
    dist = jnp.linalg.norm(diff, keepdims=True)
    features = [dist, diff]
    if s is not None:
        features.append(s[None])
    if s_nb is not None:
        features.append(s_nb[None])
    return jnp.concatenate(features)


get_diff_features_vmapped = jax.vmap(get_diff_features, in_axes=(None, 0, None, 0))
