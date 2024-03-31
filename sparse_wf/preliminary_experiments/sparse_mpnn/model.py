# %%
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import folx
from folx.api import FwdLaplArray, FwdJacobian
from collections import namedtuple
import functools
import einops
from get_neighbours import merge_dependencies, get_with_fill


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
        return jnp.sum(g * filter_kernel, axis=-2)


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


def multi_vmap(f, n):
    for _ in range(n):
        f = jax.vmap(f)
    return f


def pairwise_vmap(f):
    """Double vmap for a function of type f([n x ...], [n x m x ...]) -> [n, x m x ....]"""
    return jax.vmap(jax.vmap(f, in_axes=(None, 0)))


# vmap over center electron
@functools.partial(jax.vmap, in_axes=(None, 0, None, None))
def get_neighbour_with_fwd_lap(h: FwdLaplArray, ind_neighbour, ind_dep, n_dep_out):
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
    ind_dep_out, dep_map, _ = merge_dependencies(get_with_fill(ind_dep, ind_neighbour), n_dep_out)

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


def SparseWavefunctionWithFwdLap(
    R_orb: np.ndarray,
    cutoff: float,
    width: int = 64,
    depth: int = 3,
    beta_width_hidden: int = 16,
    beta_width_out: int = 8,
    beta_n_envelpoes: int = 16,
):
    pair_features_model = PairwiseFeatures(beta_width_hidden, beta_width_out, beta_n_envelpoes, cutoff)
    initial_embeddings_model = InitialEmbeddings(width)
    mlp_model = MLP(width, depth, activate_final=False)
    message_passing_model = MessagePassingLayer(width)
    orbital_layer_model = OrbitalLayer(R_orb)

    def init(rng, r, ind_neighbour, max_n_dependencies=None):
        params = {}
        rngs = jax.random.split(rng, 5)

        r_neighbour = get_neighbours(r, ind_neighbour, 1e6)
        diff = pairwise_vmap(lambda r1, r2: r1 - r2)(r, r_neighbour)
        beta, params["pair_features"] = pair_features_model.init_with_output(rngs[0], diff)
        h0, params["initial_embeddings"] = initial_embeddings_model.init_with_output(rngs[1], diff, beta)
        h, params["mlp"] = mlp_model.init_with_output(rngs[2], h0)
        h_neighbour = get_neighbours(h, ind_neighbour, 0.0)
        h_out, params["message_passing"] = message_passing_model.init_with_output(rngs[3], h0, h_neighbour, beta)
        phi, params["orbital_layer"] = orbital_layer_model.init_with_output(rngs[4], r, h_out)
        return params

    def apply(params, r, ind_neighbour):
        pair_features = functools.partial(pair_features_model.apply, params["pair_features"])
        initial_embeddings = functools.partial(initial_embeddings_model.apply, params["initial_embeddings"])
        mlp = functools.partial(mlp_model.apply, params["mlp"])

        r_neighbour = get_neighbours(r, ind_neighbour, 1e6)
        diff = r[:, None, :] - r_neighbour
        beta = pair_features(diff)
        h0 = initial_embeddings(diff, beta)
        h = mlp(h0)
        h_neighbour = get_neighbours(h, ind_neighbour, 0.0)
        h_out = jnp.sum(h_neighbour, axis=-2)
        return h_out

    def apply_with_fwd_lap(params, r, ind_neighbour, max_n_dependencies):
        pair_features = functools.partial(pair_features_model.apply, params["pair_features"])
        initial_embeddings = functools.partial(initial_embeddings_model.apply, params["initial_embeddings"])
        mlp = functools.partial(mlp_model.apply, params["mlp"])
        # message_passing = functools.partial(message_passing_model.apply, params["message_passing"])
        # orbital_layer = functools.partial(orbital_layer_model.apply, params["orbital_layer"])

        n_el = r.shape[-2]

        def get_diff_beta_h0_h(r, r_neighbour):
            diff = r - r_neighbour
            beta = pair_features(diff)
            h0 = initial_embeddings(diff, beta)
            h = mlp(h0)
            return diff, beta, h0, h

        # Step 0: Get neighbours
        r_neighbour = get_neighbours(r, ind_neighbour, 1e6)

        # Step 1:
        # These steps contain no dynamic indexing => can use compile-time sparsity of folx
        diff, beta, h0, h = jax.vmap(folx.forward_laplacian(get_diff_beta_h0_h, sparsity_threshold=0.6))(r, r_neighbour)

        # Every diff/beta/h0/h depends on the center electron and its neighbours
        ind_dep = jnp.concatenate([np.arange(n_el)[:, None], ind_neighbour], axis=-1)

        # Step 2: These steps contain dynamic indexing => cannot use compile-time sparsity of folx, but can use local sparsity
        h_neighbour, ind_dep_out = get_neighbour_with_fwd_lap(h, ind_neighbour, ind_dep, max_n_dependencies[1])
        h_out = jax.vmap(folx.forward_laplacian(lambda hn: jnp.sum(hn, axis=-2)))(h_neighbour)

        return h_out

    return namedtuple("SparseWavefunctionWithFwdLap", ["init", "apply", "apply_with_fwd_lap"])(
        init, apply, apply_with_fwd_lap
    )
