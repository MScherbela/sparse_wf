# %%
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import folx
from folx.api import FwdLaplArray
from collections import namedtuple
import functools
import einops



def cutoff_function(d, p=4):
    a = -(p + 1) * (p + 2) * 0.5
    b = p * (p + 2)
    c = -p * (p + 1) * 0.5
    cutoff = 1 + a * d**p + b * d ** (p + 1) + c * d ** (p + 2)
    return jnp.where(d < 1, cutoff, 0.0)


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


def get_neighbours(x, ind_neighbour=None):
    assert x.ndim == 2
    assert (ind_neighbour is None) or (ind_neighbour.ndim == 2)
    if ind_neighbour is None:
        return x[None, :, :]
    else:
        return x[ind_neighbour]


class SparseWavefunction(nn.Module):
    R_orb: np.ndarray
    cutoff: float
    width: int = 64
    depth: int = 3
    beta_width_hidden: int = 16
    beta_width_out: int = 8
    beta_n_envelpoes: int = 16

    @nn.compact
    def __call__(self, r, ind_neighbour=None, weight_neighbour=None):
        r_neighbour = get_neighbours(r, ind_neighbour)
        diff = r[:, None, :] - r_neighbour
        beta = PairwiseFeatures(self.beta_width_hidden, self.beta_width_out, self.beta_n_envelpoes, self.cutoff)(diff)
        if weight_neighbour is not None:
            beta *= weight_neighbour[:, :, None]

        h0 = InitialEmbeddings(self.width)(diff, beta)
        h = MLP(self.width, self.depth, activate_final=False)(h0)
        h_neighbour = get_neighbours(h, ind_neighbour)
        h_out = MessagePassingLayer(self.width)(h0, h_neighbour, beta)
        phi = OrbitalLayer(self.R_orb)(r, h_out)
        return phi
        # return jnp.slogdet(phi)[1]


def get_diff(r1, r2):
    return r1 - r2


def multi_vmap(f, n):
    for _ in range(n):
        f = jax.vmap(f)
    return f


def pairwise_vmap(f):
    """Double vmap for a function of type f([n x ...], [n x m x ...]) -> [n, x m x ....]"""
    return jax.vmap(jax.vmap(f, in_axes=(None, 0)))


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

    def init(rng, r, ind_neighbour, map_reverse):
        params = {}
        rngs = jax.random.split(rng, 5)

        r_neighbour = get_neighbours(r, ind_neighbour[0])
        diff = pairwise_vmap(get_diff)(r, r_neighbour)
        beta, params["pair_features"] = pair_features_model.init_with_output(rngs[0], diff)
        h0, params["initial_embeddings"] = initial_embeddings_model.init_with_output(rngs[1], diff, beta)
        h, params["mlp"] = mlp_model.init_with_output(rngs[2], h0)
        h_neighbour = get_neighbours(h, ind_neighbour[0])
        h_out, params["message_passing"] = message_passing_model.init_with_output(rngs[3], h0, h_neighbour, beta)
        phi, params["orbital_layer"] = orbital_layer_model.init_with_output(rngs[4], r, h_out)
        return params
    
    def apply(params, r, ind_neighbour, map_reverse):
        pair_features = functools.partial(pair_features_model.apply, params["pair_features"])
        initial_embeddings = functools.partial(initial_embeddings_model.apply, params["initial_embeddings"])
        mlp = functools.partial(mlp_model.apply, params["mlp"])
        message_passing = functools.partial(message_passing_model.apply, params["message_passing"])
        orbital_layer = functools.partial(orbital_layer_model.apply, params["orbital_layer"])

        n_neighbours_1 = ind_neighbour[0].shape[-1]
        n_neighbours_2 = ind_neighbour[1].shape[-1]

        
        @jax.vmap      # vmap over center electron i           
        @jax.vmap      # vmap over output neighbours 
        def build_jacobian_h_neighbour(J_h, map_r):
            jac = jnp.zeros([n_neighbours_2, 3, h.shape[-1]])
            jac = jac.at[map_r].set(J_h)
            return jac
        
        def get_diff_beta_h0_h(r, r_neighbour):
            diff = get_diff(r, r_neighbour)
            beta = pair_features(diff)
            h0 = initial_embeddings(diff, beta)
            h = mlp(h0)
            return diff, beta, h0, h


        # Step 0: Get neighbours
        r_neighbour = get_neighbours(r, ind_neighbour[0])

        # Step 1:
        # These steps contain no dynamic indexing => can use compile-time sparsity of folx
        diff, beta, h0, h = jax.vmap(folx.forward_laplacian(get_diff_beta_h0_h, sparsity_threshold=0.6))(r, r_neighbour)

        # Step 2: These steps contain dynamic indexing => cannot use compile-time sparsity of folx, but can use local sparsity
        Jh = einops.rearrange(h.jacobian.data, "nel (m1 dim) D -> nel m1 dim D", dim=3)
        Jh_neighbour = build_jacobian_h_neighbour(Jh, map_reverse)
        Jh_neighbour = einops.rearrange(Jh_neighbour, "nel m1o m2 3 D -> nel (m2 3) m1o D")
        h_neighbour = get_neighbours(h.x, ind_neighbour[0])
        lap_h_neighbour = get_neighbours(h.laplacian, ind_neighbour[0])
        h_neighbour = FwdLaplArray(h_neighbour, Jh_neighbour, lap_h_neighbour)
        return beta

    return namedtuple("SparseWavefunctionWithFwdLap", ["init", "apply"])(init, apply)


# class SparseWavefunctionWithFwdLap(SparseWavefunction):
#     def setup(self):
#         self.pair_features = PairwiseFeatures(self.beta_width_hidden, self.beta_width_out, self.beta_n_envelpoes, self.cutoff)
#         self.initial_embeddings = InitialEmbeddings(self.width)
#         self.mlp = MLP(self.width, self.depth, activate_final=False)
#         self.message_passing = MessagePassingLayer(self.width)
#         self.orbital_layer = OrbitalLayer(self.R_orb)

#     def __call__(self, r, ind_neighbour=None, weight_neighbour=None):
#         r_neighbour = get_neighbours(r, ind_neighbour)

#         diff = pairwise_vmap(folx.forward_laplacian(get_diff))(r, r_neighbour)
#         beta = multi_vmap(folx.forward_laplacian(self.pair_features), 2)(diff)

#         # TODO: re-add this part
#         # if weight_neighbour is not None:
#         #     beta *= weight_neighbour[:, :, None]

#         h0 = InitialEmbeddings(self.width)(diff, beta)
#         h = MLP(self.width, self.depth, activate_final=False)(h0)
#         h_neighbour = get_neighbours(h, ind_neighbour)
#         h_out = MessagePassingLayer(self.width)(h0, h_neighbour, beta)
#         phi = OrbitalLayer(self.R_orb)(r, h_out)
#         return phi
#         # return jnp.slogdet(phi)[1]
