import functools
from typing import Callable, Literal, NamedTuple, Optional, Sequence, cast, overload

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from sparse_wf.api import ElectronIdx, Electrons, HFOrbitals, Int, SignedLogAmplitude, SlaterMatrices
from sparse_wf.jax_utils import vectorize

ElecInp = Float[Array, "*batch n_electrons n_in"]
ElecNucDistances = Float[Array, "*batch n_electrons n_nuclei"]
ElecNucDifferences = Float[Array, "*batch n_electron n_nuclei dim=3"]
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


class GatedLinearUnit(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x: Float[Array, "*batch_dims inp_dim"]) -> Float[Array, "*batch_dims out_dim"]:
        x = nn.Dense(2 * self.out_dim, use_bias=False)(x)
        x, gate = jnp.split(x, 2, axis=-1)
        return x * nn.silu(gate)


def init_glu_feedforward(rng, width: int, depth: int, input_dim: int, out_dim: Optional[int] = None):
    layers = []
    for layer in range(depth):
        d_out = 2 * width if (layer < (depth - 1)) else (out_dim or width)
        rng, key = jax.random.split(rng)
        W = lecun_normal(key, [input_dim, d_out])
        b = jnp.zeros(d_out, jnp.float32)
        layers.append([W, b])
        input_dim = d_out // 2
    return layers


def apply_glu_feedforward(params, x):
    for _, (W, b) in enumerate(params[:-1]):
        x = x @ W + b
        x, gate = jnp.split(x, 2, axis=-1)
        x = x * jax.nn.silu(gate)
    W, b = params[-1]
    x = x @ W + b
    return x


class GLUFeedForward(nn.Module):
    width: int
    depth: int
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x: Float[Array, "*batch_dims inp_dim"]) -> Float[Array, "*batch_dims out_dim"]:
        out_dim = self.out_dim or self.width
        for _ in range(self.depth - 1):
            x = nn.Dense(2 * self.width, use_bias=True)(x)
            x, gate = jnp.split(x, 2, axis=-1)
            x = x * nn.silu(gate)
        x = nn.Dense(out_dim, use_bias=True)(x)
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


def get_inverse_from_lu(lu, permutation):
    n = lu.shape[0]
    b = jnp.eye(n, dtype=lu.dtype)[permutation]
    # The following lines trigger mypy (private usage?)
    x = jax.lax.linalg.triangular_solve(lu, b, left_side=True, lower=True, unit_diagonal=True)  # type: ignore
    x = jax.lax.linalg.triangular_solve(lu, x, left_side=True, lower=False)  # type: ignore
    return x


def slogdet_from_lu(lu, pivot):
    assert (lu.ndim == 2) and (lu.shape[0] == lu.shape[1])
    n = lu.shape[0]
    diag = jnp.diag(lu)
    logdet = jnp.sum(jnp.log(jnp.abs(diag)))
    parity = jnp.count_nonzero(pivot != jnp.arange(n))  # sign flip for each permutation
    parity += jnp.count_nonzero(diag < 0)  # sign flip for each negative diagonal element
    sign = jnp.where(parity % 2 == 0, 1.0, -1.0).astype(lu.dtype)
    return sign, logdet


def slog_and_inverse(A: Float[Array, "*batch_dims N N"]):
    # We have this wrapper to package the output properly
    @vectorize(signature="(n,n)->(),(),(n,n)")
    def inner(A):
        # mypy complains about lu not being exported by jax.lax.linalg
        lu, pivot, permutation = jax.lax.linalg.lu(A)  # type: ignore
        inverse = get_inverse_from_lu(lu, permutation)
        sign, logdet = slogdet_from_lu(lu, pivot)
        return sign, logdet, inverse

    sign, logdet, inverse = inner(A)
    return (sign, logdet), inverse


class LogPsiState(NamedTuple):
    matrices: Sequence[jax.Array]
    inverses: Sequence[jax.Array]
    slogdets: Sequence[tuple[jax.Array, jax.Array]]


@overload
def signed_logpsi_from_orbitals(
    orbitals: SlaterMatrices, return_state: Literal[False] = False
) -> SignedLogAmplitude: ...


@overload
def signed_logpsi_from_orbitals(
    orbitals: SlaterMatrices, return_state: Literal[True]
) -> tuple[SignedLogAmplitude, LogPsiState]: ...


def signed_logpsi_from_orbitals(orbitals: SlaterMatrices, return_state: bool = False):
    dtype = orbitals[0].dtype
    if return_state:
        slog_inv = [slog_and_inverse(orbs) for orbs in orbitals]
        slogdets, inverses = map(list, zip(*slog_inv))  # list of tuples to tuple of lists
    else:
        slogdets = [jnp.linalg.slogdet(orbs) for orbs in orbitals]
    # For block-diagonal determinants, orbitals is a tuple of length 2. The following line is a fancy way to write
    # logdet, sign = logdet_up + logdet_dn, sign_up * sign_dn
    sign, logdet = functools.reduce(
        lambda x, y: (x[0] * y[0], x[1] + y[1]), slogdets, (jnp.ones((), dtype=dtype), jnp.zeros((), dtype=dtype))
    )
    logpsi, signpsi = jnn.logsumexp(logdet, b=sign, return_sign=True)
    if return_state:
        return (signpsi, logpsi), LogPsiState(orbitals, inverses, slogdets)
    return (signpsi, logpsi)


def signed_log_psi_from_orbitals_low_rank(orbitals: SlaterMatrices, changed_electrons: ElectronIdx, state: LogPsiState):
    # Here we apply the matrix determinant lemma to compute low rank updates on the slogdet
    # det(A + UV^T) = det(A) * det(I + V^T A^-1 U) where A is the original matrix, U and V are the low rank updates
    # Since we update n-rows, U will a matrix of n x k basis vectors and V will be a matrix of k x n coefficients
    # To update the inverses, we use the Woodburry matrix identity
    # (A + UV^T)^-1 = A^-1 - A^-1 U (I + V^T A^-1 U)^-1 V^T A^-1
    k = len(changed_electrons)
    dtype = orbitals[0].dtype
    slogdets = []
    inverses = []
    for A, A_inv, orb, (s_psi, log_psi) in zip(state.matrices, state.inverses, orbitals, state.slogdets):
        # orb is K x N x N (current step orbitals)
        # A is K x N x N (previous orbitals)
        # A_inv is K x N x N (previous inverse)
        # s_psi, log_psi are K (previous slogdet)
        V = orb.at[:, changed_electrons].get(mode="fill", fill_value=0) - A.at[:, changed_electrons].get(
            mode="fill", fill_value=0
        )
        Ainv_U = A_inv.at[..., changed_electrons].get(mode="fill", fill_value=0)
        V_Ainv_U = V @ Ainv_U
        (s_delta, log_delta), inv_delta = slog_and_inverse(jnp.eye(k, dtype=dtype) + V_Ainv_U)
        # update slog det
        s_psi, log_psi = s_psi * s_delta, log_psi + log_delta
        slogdets.append((s_psi, log_psi))
        # update inverse - the optimal contraction would be first do the outer contractions, then the inner.
        new_inv = A_inv - jnp.einsum("...ab,...bc,...cd,...de->...ae", Ainv_U, inv_delta, V, A_inv)
        inverses.append(new_inv)
    # For block-diagonal determinants, orbitals is a tuple of length 2. The following line is a fancy way to write
    # logdet, sign = logdet_up + logdet_dn, sign_up * sign_dn
    sign, logdet = functools.reduce(
        lambda x, y: (x[0] * y[0], x[1] + y[1]), slogdets, (jnp.ones((), dtype), jnp.zeros((), dtype))
    )
    logpsi, signpsi = jnn.logsumexp(logdet, b=sign, return_sign=True)
    return (signpsi, logpsi), LogPsiState(orbitals, inverses, slogdets)


def get_dist(electrons: Electrons) -> ElecElecDistances:
    diff = electrons[..., None, :, :] - electrons[..., :, None, :]
    return jnp.linalg.norm(diff, axis=-1)


def get_dist_same_diff(electrons: Electrons, n_up):
    dists = get_dist(electrons)

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


def lecun_normal(rng, shape):
    fan_in = shape[0]
    scale = 1 / jnp.sqrt(fan_in)
    return jax.random.truncated_normal(rng, -1, 1, shape, jnp.float32) * scale


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


def scale_initializer(rng, cutoff, shape, dtype=jnp.float32):
    n_scales = shape[-1]
    max_length_scale = min(20, cutoff)
    scale = jnp.linspace(0, max_length_scale, n_scales, dtype=dtype)
    scale *= 1 + 0.1 * jax.random.normal(rng, shape, dtype)
    return scale.astype(jnp.float32)


def zeros_initializer(dtype=jnp.float32):
    def init_function(rng, shape):
        return jnp.zeros(shape, dtype=dtype)

    return init_function


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


ScalingParam = Float[Array, ""]


def normalize(x, scale: ScalingParam | None, return_scale=False):
    if scale is None:
        scale = 1.0 / jnp.std(x)
        scale = jnp.where(jnp.isfinite(scale), scale, 1.0)
    x = x * scale
    if return_scale:
        return x, scale
    return x


class FixedScalingFactor(nn.Module):
    """
    A Flax module that scales the input tensor by a fixed factor to achieve a target standard deviation.

    Attributes:
        target_std (float): The target standard deviation to achieve.
        element_wise (bool): Whether to scale each element of the input tensor independently.
    """

    target_std: float = 1.0
    element_wise: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, weighting: jax.Array | None = None) -> jax.Array:
        """
        Scales the input tensor by a fixed factor to achieve a target standard deviation.

        Args:
            x (jax.Array): The input tensor.
            weighting (Optional[jax.Array]): The weighting to apply to the input tensor.

        Returns:
            jax.Array: The scaled input tensor.
        """
        is_initialized = self.has_variable("scaling_factors", "scale")
        if self.element_wise:
            scaling = self.variable("scaling_factors", "scale", jnp.ones, (x.shape[-1],), jnp.float32)
        else:
            scaling = self.variable("scaling_factors", "scale", jnp.ones, (), jnp.float32)
        if not is_initialized:
            if x.size > 1:
                # Sum over all but the last dim
                axes = tuple(range(x.ndim - 1)) if self.element_wise else tuple(range(x.ndim))
                if weighting is None:
                    weighting = jnp.ones((), dtype=x.dtype)
                weighting = jnp.broadcast_to(weighting, x.shape)
                weighting /= weighting.sum(axes)

                # Weighted Std computation
                x_mean = (x * weighting).sum(axes)

                x_std = ((x - x_mean) ** 2 * weighting).sum(axes) ** 0.5
                value = self.target_std / x_std
                value = jnp.where(jnp.logical_or(jnp.isnan(value), jnp.isinf(value)), 1, value)
                scaling.value = value.astype(jnp.float32)
        return x * scaling.value


def get_relative_tolerance(dtype):
    return 1e-12 if (dtype == jnp.float64) else 1e-6
