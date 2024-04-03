# %%
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import folx
from folx.api import FwdLaplArray, FwdJacobian
import functools
import einops
from get_neighbours import merge_dependencies, get_with_fill
import chex


def cutoff_function(d, p=4):
    a = -(p + 1) * (p + 2) * 0.5
    b = p * (p + 2)
    c = -p * (p + 1) * 0.5
    cutoff = 1 + a * d**p + b * d ** (p + 1) + c * d ** (p + 2)
    # Only required to enforce cutoff in fully connected implementation
    cutoff *= jax.numpy.heaviside(1 - d, 0.0)
    return cutoff


class MLP(nn.Module):
    width: int
    depth: int
    activate_final: bool

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = nn.Dense(self.width)(x)
            if (i < self.depth - 1) or self.activate_final:
                x = jax.nn.silu(x)
        return x


class MessagePassingLayer(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, h_center, h_neighbors, diff_features):
        h_center = nn.Dense(self.out_dim, name="proj_i")(h_center)
        filter_kernel = nn.Dense(self.out_dim, use_bias=False, name="Gamma")(diff_features)

        h_out = jax.nn.silu(h_center + jnp.sum(filter_kernel * h_neighbors, axis=-2))
        return h_out


class InitialEmbeddings(nn.Module):
    out_dim: int
    depth: int = 2

    @nn.compact
    def __call__(self, diff, diff_features):
        dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        g = jnp.concatenate([diff, dist], axis=-1)
        g = MLP(self.out_dim, self.depth, activate_final=True)(g)

        filter_kernel = nn.Dense(self.out_dim, use_bias=False, name="Gamma")(diff_features)
        return jnp.einsum("...ij,...ij->...j", g, filter_kernel)


class PairwiseFeatures(nn.Module):
    width_hidden: int
    width_out: int
    n_envelopes: int
    cutoff: float

    def scale_initializer(self, rng, shape):
        noise = 1 + 0.25 * jax.random.normal(rng, shape)
        return self.cutoff * noise * 0.5

    @nn.compact
    def __call__(self, diff):
        dist = jnp.linalg.norm(diff, axis=-1)
        scales = self.param("scales", self.scale_initializer, (self.n_envelopes,))
        scales = jax.nn.softplus(scales)
        envelopes = jnp.exp(-(dist[..., None] ** 2) / scales)
        envelopes = nn.Dense(self.width_out)(envelopes)

        diff = nn.Dense(self.width_hidden)(diff)
        diff = jax.nn.silu(diff)
        diff = nn.Dense(self.width_out)(diff)

        return envelopes * diff * cutoff_function(dist / self.cutoff)[..., None]


class OrbitalLayer(nn.Module):
    R_orb: np.ndarray

    @nn.compact
    def __call__(self, r, h_out):
        n_orb = len(self.R_orb)
        dist_el_orb = jnp.linalg.norm(r[:, None, :] - self.R_orb[None, :, :], axis=-1)
        orbital_envelope = jnp.exp(-dist_el_orb * 0.2)
        phi = nn.Dense(n_orb)(h_out) * orbital_envelope
        return phi


def get_neighbours(x, ind_neighbour=None, fill_value=0.0):
    assert x.ndim == 2
    assert (ind_neighbour is None) or (ind_neighbour.ndim == 2)
    if ind_neighbour is None:
        return x[None, :, :]
    else:
        return x.at[ind_neighbour].get(mode="fill", fill_value=fill_value)


# vmap over center electron
@functools.partial(jax.vmap, in_axes=(None, 0, 0, None, None))
def get_neighbour_with_FwdLapArray(h: FwdLaplArray, ind_neighbour, fixed_deps, ind_dep, n_dep_out):
    # Get and assert shapes
    n_neighbour = ind_neighbour.shape[-1]
    feature_dims = h.x.shape[1:]
    n_dep_in = ind_dep.shape[-1]
    assert h.jacobian.data.shape[1] == 3 * n_dep_in

    # Get neighbour data by indexing into the input data and padding with 0 any out of bounds indices
    h_neighbour = get_with_fill(h.x, ind_neighbour, 0.0)
    jac_neighbour = get_with_fill(h.jacobian.data, ind_neighbour, 0.0)
    lap_h_neighbour = get_with_fill(h.laplacian, ind_neighbour, 0.0)

    # Remaining issue: The jacobians for each embedding can depend on different input coordinates
    # 1) Get a joint set of dependencies for each neighbour embedding
    dependencies_neighbours = get_with_fill(ind_dep, ind_neighbour)
    # inp_dep_out is of shape (total_neighbors_t = level0 neighbors + level1 neighbors + level...)
    # dep_map mis a of shape (total_neighbors_t-1, levelt neighbors) maps to total_neighbors_t
    ind_dep_out, dep_map = merge_dependencies(dependencies_neighbours, fixed_deps, n_dep_out)

    # 2) Split jacobian input dim into electrons x xyz
    jac_neighbour = einops.rearrange(
        jac_neighbour,
        "n_neighbour (n_dep_in dim) D -> n_neighbour n_dep_in dim D",
        n_neighbour=n_neighbour,
        n_dep_in=n_dep_in,
        dim=3,
    )

    # 3) Combine the jacobians into a larger jacobian, that depends on the joint dependencies
    @functools.partial(jax.vmap, in_axes=(0, 0), out_axes=2)
    def _jac_for_neighbour(J, dep_map_):
        jac_out = jnp.zeros([n_dep_out, 3, *feature_dims])
        jac_out = jac_out.at[dep_map_].set(J, mode="drop")
        return jac_out

    # total_neighbors_t
    jac_neighbour = _jac_for_neighbour(jac_neighbour, dep_map)

    # 4) Merge electron and xyz dim back together to jacobian input dim
    jac_neighbour = einops.rearrange(
        jac_neighbour,
        "n_dep_out dim n_neighbour D -> (n_dep_out dim) n_neighbour D",
        n_dep_out=n_dep_out,
        dim=3,
        n_neighbour=n_neighbour,
    )
    return FwdLaplArray(h_neighbour, FwdJacobian(data=jac_neighbour), lap_h_neighbour), ind_dep_out


def forward_lap_with_frozen_x0_idx(f, idx0_values, sparsity_threshold=0):
    def replace_mask(lap_arr: FwdLaplArray, idx):
        return FwdLaplArray(lap_arr.x, FwdJacobian(data=lap_arr.jacobian.data, x0_idx=idx), lap_arr.laplacian)

    def transformed(*args):
        args = [replace_mask(arg, idx) for arg, idx in zip(args, idx0_values)]
        return folx.forward_laplacian(f, sparsity_threshold=sparsity_threshold, disable_jit=True)(*args)

    return transformed


@chex.dataclass
class SparseWavefunctionParams:
    pair_features: jax.Array = None
    initial_embeddings: jax.Array = None
    mlp: jax.Array = None
    message_passing: jax.Array = None
    orbital_layer: jax.Array = None


class SparseWavefunctionWithFwdLap:
    def __init__(self, R_orb, cutoff, width=64, depth=3, beta_width_hidden=16, beta_width_out=8, beta_n_envelopes=16):
        self.R_orb = R_orb
        self.cutoff = cutoff
        self.width = width
        self.depth = depth
        self.beta_width_hidden = beta_width_hidden
        self.beta_width_out = beta_width_out
        self.beta_n_envelopes = beta_n_envelopes

        self.pair_features = PairwiseFeatures(
            self.beta_width_hidden, self.beta_width_out, self.beta_n_envelopes, self.cutoff
        )
        self.initial_embeddings = InitialEmbeddings(self.width)
        self.mlp = MLP(self.width, self.depth, activate_final=False)
        self.message_passing = MessagePassingLayer(self.width)
        self.orbital_layer = OrbitalLayer(self.R_orb)

    def get_beta_h0_h(self, params: SparseWavefunctionParams, r, r_neighbour):
        diff = r - r_neighbour
        beta = self.pair_features.apply(params.pair_features, diff)
        h0 = self.initial_embeddings.apply(params.initial_embeddings, diff, beta)
        h = self.mlp.apply(params.mlp, h0)
        return beta, h0, h

    def init(self, rng, r, ind_neighbour, max_n_dependencies=None):
        params = SparseWavefunctionParams()
        rngs = jax.random.split(rng, 5)

        r_neighbour = get_neighbours(r, ind_neighbour, 1e6)
        diff = r[:, None, :] - r_neighbour
        beta, params.pair_features = self.pair_features.init_with_output(rngs[0], diff)
        h0, params.initial_embeddings = self.initial_embeddings.init_with_output(rngs[1], diff, beta)
        h, params.mlp = self.mlp.init_with_output(rngs[2], h0)
        h_neighbour = get_neighbours(h, ind_neighbour, 0.0)
        h_out, params.message_passing = self.message_passing.init_with_output(rngs[3], h0, h_neighbour, beta)
        phi, params.orbital_layer = self.orbital_layer.init_with_output(rngs[4], r, h_out)
        return params

    def apply(self, params: SparseWavefunctionParams, r, ind_neighbour):
        r_neighbour = get_neighbours(r, ind_neighbour, 1e6)
        beta, h0, h = jax.vmap(self.get_beta_h0_h, in_axes=(None, 0, 0))(params, r, r_neighbour)
        h_neighbour = get_neighbours(h, ind_neighbour, 0.0)
        h_out = jax.vmap(self.message_passing.apply, in_axes=(None, 0, 0, 0))(
            params.message_passing, h0, h_neighbour, beta
        )
        return h_out

    def apply_with_fwd_lap(self, params, r, ind_neighbour, max_n_dependencies):
        n_el, n_neighbours = ind_neighbour.shape

        # Step 0: Get neighbours
        r_neighbour = get_neighbours(r, ind_neighbour, 1e6)

        # Step 1:
        # These steps contain no dynamic indexing => can use compile-time sparsity of folx
        beta, h0, h = jax.vmap(
            folx.forward_laplacian(
                lambda *args: self.get_beta_h0_h(params, *args), sparsity_threshold=0.6, disable_jit=True
            )
        )(r, r_neighbour)

        # Step 2: These steps contain dynamic indexing => cannot use compile-time sparsity of folx, but can use local sparsity
        # Every diff/beta/h0/h depends on the center electron and its neighbours
        ind_dep = jnp.concatenate([np.arange(n_el)[:, None], ind_neighbour], axis=-1)
        h_neighbour, ind_dep_out = get_neighbour_with_FwdLapArray(
            h, ind_neighbour, ind_dep, ind_dep, max_n_dependencies[1]
        )

        # folx doesn't nicely work together with transformations such as vmap, because vmapping over a FwdLaplArray
        # also vmaps over the x0_idx array. This turns x0_idx into a jnp.array which breaks the compile-time constant requirement
        # Hacky solution: Build the x0_idx array manually and replace it in the FwdLaplArray
        # TODO: could extract these x0_idx from a compile-time pass through forward_laplacian instead of rebuilding manually
        x0_idx_h0 = np.tile(np.arange(3 * (n_neighbours + 1))[:, None], self.width)
        x0_idx_beta = np.zeros([6, n_neighbours, self.beta_width_out], dtype=int)
        for j in np.arange(n_neighbours):
            x0_idx_beta[:3, j, :] = np.arange(3)[:, None]
            x0_idx_beta[3:, j, :] = np.arange(3 * (j + 1), 3 * (j + 2))[:, None]

        message_passing = functools.partial(self.message_passing.apply, params.message_passing)
        message_passing = forward_lap_with_frozen_x0_idx(
            message_passing, [x0_idx_h0, None, x0_idx_beta], sparsity_threshold=0.6
        )
        message_passing = jax.vmap(message_passing)
        h_out = message_passing(h0, h_neighbour, beta)
        return h_out
