# %%
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import folx
from folx.api import FwdLaplArray, FwdJacobian
import functools
import einops
from get_neighbours import get_with_fill, NeighbourIndicies, NO_NEIGHBOUR
import chex
from typing import Callable
from dataclasses import dataclass, field
from utils import fwd_lap


def cutoff_function(d, p=4):
    a = -(p + 1) * (p + 2) * 0.5
    b = p * (p + 2)
    c = -p * (p + 1) * 0.5
    cutoff = 1 + a * d**p + b * d ** (p + 1) + c * d ** (p + 2)
    # Heavyside is only required to enforce cutoff in fully connected implementation
    # and when evaluating the cutoff to a padding/placeholder "neighbour" which is far away
    cutoff *= jax.numpy.heaviside(1 - d, 0.0)
    return cutoff


def get_diff_features(r1, r2, s1=None, s2=None):
    diff = r1 - r2
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    features = [dist, diff]
    if (s1 is not None) and (s2 is not None):
        s_prod = s1 * s2
        features.append(s_prod[..., None])
    return jnp.concatenate(features, axis=-1)


class MLP(nn.Module):
    width: int = None
    depth: int = None
    widths: tuple[int] = None
    activate_final: bool = False
    activation: Callable = jax.nn.silu

    @nn.compact
    def __call__(self, x):
        output_widths = self.widths or [self.width] * self.depth
        depth = len(output_widths)
        for ind_layer, out_width in enumerate(output_widths):
            x = nn.Dense(out_width)(x)
            if (ind_layer < depth - 1) or self.activate_final:
                x = self.activation(x)
        return x


class PairwiseFilter(nn.Module):
    cutoff: float
    directional_mlp_widths: tuple[int]
    n_envelopes: int
    out_dim: int

    def scale_initializer(self, rng, shape):
        return self.cutoff + 0.5 * self.cutoff * jax.random.normal(rng, shape)

    @nn.compact
    def __call__(self, dist_diff):
        """Compute the pairwise filters between two particles.

        Args:
            dist_diff: The distance and spin difference between two particles [n_el x n_nb x 4].
                The 0th feature dimension must contain the distance, the remaining dimensions can contain arbitrary
                features that are used to compute the pairwise filters, e.g. product of spins.
        """
        # Direction- (and spin-) dependent MLP
        directional_features = MLP(widths=self.directional_mlp_widths)(dist_diff)

        # Distance-dependenet radial filters
        dist = dist_diff[..., 0]
        scales = self.param("scales", self.scale_initializer, (self.n_envelopes,))
        scales = jax.nn.softplus(scales)
        envelopes = jnp.exp(-(dist[..., None] ** 2) / scales)
        envelopes = nn.Dense(directional_features.shape[-1], use_bias=False)(envelopes)
        envelopes *= cutoff_function(dist / self.cutoff)[..., None]
        beta = directional_features * envelopes
        return nn.Dense(self.out_dim, use_bias=False)(beta)


class OrbitalLayer(nn.Module):
    R_orb: np.ndarray

    @nn.compact
    def __call__(self, r, h_out):
        n_orb = len(self.R_orb)
        dist_el_orb = jnp.linalg.norm(r[:, None, :] - self.R_orb[None, :, :], axis=-1)
        orbital_envelope = jnp.exp(-dist_el_orb * 0.2)
        phi = nn.Dense(n_orb)(h_out) * orbital_envelope
        return phi


@functools.partial(jax.vmap, in_axes=(0, None))
def densify_jacobian_by_zero_padding(h: folx.api.FwdLaplArray, n_deps_out):
    jac = h.jacobian.data
    n_deps_sparse = jac.shape[0]
    padding = jnp.zeros([n_deps_out - n_deps_sparse, *jac.shape[1:]])
    return folx.api.FwdLaplArray(
        x=h.x, jacobian=folx.api.FwdJacobian(jnp.concatenate([jac, padding], axis=0)), laplacian=h.laplacian
    )


@jax.vmap
def densify_jacobian_diagonally(h: folx.api.FwdLaplArray):
    jac = h.jacobian.data
    assert jac.shape[0] == 3
    n_neighbours = jac.shape[1]

    idx_neighbour = np.arange(n_neighbours)
    jac_out = jnp.zeros([n_neighbours, 3, n_neighbours, *jac.shape[2:]])
    jac_out = jac_out.at[idx_neighbour, :, idx_neighbour, ...].set(jac.swapaxes(0, 1))
    jac_out = jax.lax.collapse(jac_out, 0, 2)  # merge (n_deps, 3) into (n_deps*3)
    return folx.api.FwdLaplArray(x=h.x, jacobian=folx.api.FwdJacobian(jac_out), laplacian=h.laplacian)


@chex.dataclass
class SparseMoonParams:
    ee_filter: jax.Array = None
    ne_filter: jax.Array = None
    en_filter: jax.Array = None
    mlp_nuc: jax.Array = None
    lin_h0: jax.Array = None
    lin_orbitals: jax.Array = None


@dataclass
class SparseMoonWavefunction:
    n_orbitals: int
    cutoff: float
    feature_dim: int
    pair_mlp_widths: tuple[int]
    pair_n_envelopes: int
    ee_filter: PairwiseFilter = field(init=False)
    ne_filter: PairwiseFilter = field(init=False)
    en_filter: PairwiseFilter = field(init=False)
    lin_h0: nn.Dense = field(init=False)
    lin_orbitals: nn.Dense = field(init=False)
    mlp_nuc: MLP = field(init=False)

    def __post_init__(self):
        self.ee_filter = PairwiseFilter(
            self.cutoff, self.pair_mlp_widths, self.pair_n_envelopes, self.feature_dim, name="Gamma_ee"
        )
        self.ne_filter = PairwiseFilter(
            self.cutoff, self.pair_mlp_widths, self.pair_n_envelopes, self.feature_dim, name="Gamma_ne"
        )
        self.en_filter = PairwiseFilter(
            self.cutoff, self.pair_mlp_widths, self.pair_n_envelopes, self.feature_dim, name="Gamma_en"
        )
        self.mlp_nuc = MLP(width=self.feature_dim, depth=3, name="mlp_nuc")
        self.lin_h0 = nn.Dense(self.feature_dim, use_bias=True, name="lin_h0")
        self.lin_orbitals = nn.Dense(self.n_orbitals, use_bias=False, name="lin_orbitals")

    def init(self, rng):
        rngs = jax.random.split(rng, 6)
        params = SparseMoonParams(
            ee_filter=self.ee_filter.init(rngs[0], np.zeros([5])),  # dist + 3 * diff + spin
            ne_filter=self.ne_filter.init(rngs[1], np.zeros([4])),  # dist + 3 * diff
            en_filter=self.en_filter.init(rngs[2], np.zeros([4])),  # dist + 3 * diff
            lin_h0=self.lin_h0.init(rngs[3], np.zeros([self.feature_dim])),
            mlp_nuc=self.mlp_nuc.init(rngs[5], np.zeros([self.feature_dim])),
            lin_orbitals=self.lin_orbitals.init(rngs[4], np.zeros([self.feature_dim])),
        )
        return params

    def apply(self, params: SparseMoonParams, r, spin, R, idx_nb: NeighbourIndicies, deps=None, dep_maps=None):
        with_fwd_lap = (deps is not None) and (dep_maps is not None)

        # Apply fwd_lap when with_fwd_lap is True, else just acts as identity
        def maybe_fwd_lap(func, argnums=None, sparsity_threshold=0.6):
            if with_fwd_lap:
                return fwd_lap(func, argnums, sparsity_threshold=sparsity_threshold)
            return func

        # Helper function implementing the message passing step
        @jax.vmap
        @maybe_fwd_lap
        def contract(h_nb, Gamma_nb, h_center=None):
            msg = jnp.sum(h_nb * Gamma_nb, axis=-2)
            if h_center is not None:
                msg += h_center
            return jax.nn.silu(msg)

        # Step 0: Get all neighbour coordinates
        spin_nb_ee = get_with_fill(spin, idx_nb.ee, 0.0)
        r_nb_ee = get_with_fill(r, idx_nb.ee, NO_NEIGHBOUR)  # [n_el  x n_neighbouring_electrons x 3]
        r_nb_ne = get_with_fill(r, idx_nb.ne, NO_NEIGHBOUR)  # [n_nuc x n_neighbouring_electrons x 3]
        R_nb_en = get_with_fill(R, idx_nb.en, NO_NEIGHBOUR)  # [n_el  x n_neighbouring_nuclei    x 3]

        # Step 1: Contract ee to get electron embeedings h0
        @jax.vmap
        @functools.partial(maybe_fwd_lap, argnums=(0, 1))
        def get_h0(r_, r_nb_ee_, spin_, spin_nb_ee_):
            # vmap over neighbours
            features_ee = jax.vmap(get_diff_features, in_axes=(None, 0, None, 0))(r_, r_nb_ee_, spin_, spin_nb_ee_)
            Gamma_ee = self.ee_filter.apply(params.ee_filter, features_ee)
            h0 = jnp.sum(Gamma_ee, axis=-2)
            h0 = self.lin_h0.apply(params.lin_h0, h0)
            return h0

        # Shapes: h0: [nel x feature_dim]; h0.jac: [n_el x 3*n_deps1 x feature_dim] (dense)
        h0 = get_h0(r, r_nb_ee, spin, spin_nb_ee)

        # Step 2: Contract to nuclei + MLP on nuclei => nuclear embedding H
        # 2a: Get the spatial filter between nuclei and neighbouring electrons
        @jax.vmap  # vmap over center nuclei
        @functools.partial(jax.vmap, in_axes=(None, 0), out_axes=-2)  # vmap over neighbouring electrons
        @functools.partial(maybe_fwd_lap, argnums=(1,), sparsity_threshold=0)
        def get_Gamma_ne(R, r_nb_ne_):
            features_ne = get_diff_features(R, r_nb_ne_)
            return self.ne_filter.apply(params.ne_filter, features_ne)

        # Gamma_ne: [n_nuc x n_neighbour x feature_dim]; Gamma_ne.jac: [n_nuc x 3 x n_neighbour x feature_dim] (sparse)
        Gamma_ne = get_Gamma_ne(R, r_nb_ne)

        # 2b: Get the neighbouring electron embeddings
        if with_fwd_lap:
            n_deps_out = deps[1].shape[-1]
            # Shapes: h0_nb_ne: [n_nuc x n_neighbour x feature_dim]; h0_nb_ne.jac: [n_nuc x 3*n_deps2 x feature_dim] (dense)
            h0_nb_ne = get_neighbour_with_FwdLapArray(h0, idx_nb.ne, n_deps_out, dep_maps[0])
            Gamma_ne = densify_jacobian_diagonally(Gamma_ne)  # densify from deps == 3 --> deps==3*n_neighbours
            Gamma_ne = densify_jacobian_by_zero_padding(
                Gamma_ne, 3 * n_deps_out
            )  # densify from deps==3*n_neighbours --> deps==3*n_deps2
        else:
            h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0.0)

        # 2c: Contract and apply the MLP
        H0 = contract(h0_nb_ne, Gamma_ne)
        H = jax.vmap(maybe_fwd_lap(lambda H0_: self.mlp_nuc.apply(params.mlp_nuc, H0_)))(H0)

        # Step 3: Contract back to electrons => Final electron embeddings h_out
        @jax.vmap  # vmap over center electrons
        @functools.partial(jax.vmap, in_axes=(None, 0), out_axes=-2)  # vmap over neighbouring nuclei
        @functools.partial(maybe_fwd_lap, argnums=(0,), sparsity_threshold=0)
        def get_Gamma_en(r_, R_nb_en_):
            features_en = get_diff_features(r_, R_nb_en_)
            return self.en_filter.apply(params.en_filter, features_en)

        # Shapes: Gamma_en: [n_el x n_neighbouring_nuc x feature_dim]; Gamma_en.jac: [n_el x 3 x n_neighbouring_nuc x feature_dim] (dense)
        Gamma_en = get_Gamma_en(r, R_nb_en)

        if with_fwd_lap:
            n_deps_out = deps[2].shape[-1]
            H_nb_en = get_neighbour_with_FwdLapArray(H, idx_nb.en, n_deps_out, dep_maps[1])
            Gamma_en = densify_jacobian_by_zero_padding(
                Gamma_en, 3 * n_deps_out
            )  # densify from deps==3 --> deps==3*n_deps3
            h0 = densify_jacobian_by_zero_padding(
                h0, 3 * n_deps_out
            )  # densify from deps==3*n_deps1 --> deps==3*n_deps3
        else:
            H_nb_en = get_with_fill(H, idx_nb.en, 0.0)
        h_out = contract(H_nb_en, Gamma_en, h0)
        return h_out


# vmap over center particle
@functools.partial(jax.vmap, in_axes=(None, 0, None, 0))
def get_neighbour_with_FwdLapArray(h: FwdLaplArray, ind_neighbour, n_deps_out, dep_map):
    # Get and assert shapes
    n_neighbour = ind_neighbour.shape[-1]
    feature_dims = h.x.shape[1:]

    # Get neighbour data by indexing into the input data and padding with 0 any out of bounds indices
    h_neighbour = get_with_fill(h.x, ind_neighbour, 0.0)
    jac_neighbour = get_with_fill(h.jacobian.data, ind_neighbour, 0.0)
    lap_h_neighbour = get_with_fill(h.laplacian, ind_neighbour, 0.0)

    # Remaining issue: The jacobians for each embedding can depend on different input coordinates
    # 1) Split jacobian input dim into electrons x xyz
    jac_neighbour = einops.rearrange(
        jac_neighbour,
        "n_neighbour (n_dep_in dim) D -> n_neighbour n_dep_in dim D",
        n_neighbour=n_neighbour,
        dim=3,
    )

    # 2) Combine the jacobians into a larger jacobian, that depends on the joint dependencies
    @functools.partial(jax.vmap, in_axes=(0, 0), out_axes=2)
    def _jac_for_neighbour(J, dep_map_):
        jac_out = jnp.zeros([n_deps_out, 3, *feature_dims])
        jac_out = jac_out.at[dep_map_].set(J, mode="drop")
        return jac_out

    jac_neighbour = _jac_for_neighbour(jac_neighbour, dep_map)

    # 3) Merge electron and xyz dim back together to jacobian input dim
    jac_neighbour = einops.rearrange(
        jac_neighbour,
        "n_dep_out dim n_neighbour D -> (n_dep_out dim) n_neighbour D",
        n_dep_out=n_deps_out,
        dim=3,
        n_neighbour=n_neighbour,
    )
    return FwdLaplArray(h_neighbour, FwdJacobian(data=jac_neighbour), lap_h_neighbour)


# def forward_lap_with_frozen_x0_idx(f, idx0_values, sparsity_threshold=0):
#     def replace_mask(lap_arr: FwdLaplArray, idx):
#         return FwdLaplArray(lap_arr.x, FwdJacobian(data=lap_arr.jacobian.data, x0_idx=idx), lap_arr.laplacian)

#     def transformed(*args):
#         args = [replace_mask(arg, idx) for arg, idx in zip(args, idx0_values)]
#         return folx.forward_laplacian(f, sparsity_threshold=sparsity_threshold, disable_jit=True)(*args)

#     return transformed


# @chex.dataclass
# class SparseWavefunctionParams:
#     pair_features: jax.Array = None
#     initial_embeddings: jax.Array = None
#     mlp: jax.Array = None
#     message_passing: jax.Array = None
#     orbital_layer: jax.Array = None


# class SparseWavefunctionWithFwdLap:
#     def __init__(self, R_orb, cutoff, width=64, depth=3, beta_width_hidden=16, beta_width_out=8, beta_n_envelopes=16):
#         self.R_orb = R_orb
#         self.cutoff = cutoff
#         self.width = width
#         self.depth = depth
#         self.beta_width_hidden = beta_width_hidden
#         self.beta_width_out = beta_width_out
#         self.beta_n_envelopes = beta_n_envelopes

#         self.pair_features = PairwiseFilter(
#             self.beta_width_hidden, self.beta_width_out, self.beta_n_envelopes, self.cutoff
#         )
#         self.initial_embeddings = InitialEmbeddings(self.width)
#         self.mlp = MLP(self.width, self.depth, activate_final=False)
#         self.message_passing = MessagePassingLayer(self.width)
#         self.orbital_layer = OrbitalLayer(self.R_orb)

#     def get_beta_h0_h(self, params: SparseWavefunctionParams, r, r_neighbour):
#         diff = r - r_neighbour
#         beta = self.pair_features.apply(params.pair_features, diff)
#         h0 = self.initial_embeddings.apply(params.initial_embeddings, diff, beta)
#         h = self.mlp.apply(params.mlp, h0)
#         return beta, h0, h

#     def init(self, rng, r, ind_neighbour, max_n_dependencies=None):
#         params = SparseWavefunctionParams()
#         rngs = jax.random.split(rng, 5)

#         r_neighbour = get_neighbours(r, ind_neighbour, 1e6)
#         diff = r[:, None, :] - r_neighbour
#         beta, params.pair_features = self.pair_features.init_with_output(rngs[0], diff)
#         h0, params.initial_embeddings = self.initial_embeddings.init_with_output(rngs[1], diff, beta)
#         h, params.mlp = self.mlp.init_with_output(rngs[2], h0)
#         h_neighbour = get_neighbours(h, ind_neighbour, 0.0)
#         h_out, params.message_passing = self.message_passing.init_with_output(rngs[3], h0, h_neighbour, beta)
#         phi, params.orbital_layer = self.orbital_layer.init_with_output(rngs[4], r, h_out)
#         return params

#     def apply(self, params: SparseWavefunctionParams, r, ind_neighbour):
#         r_neighbour = get_neighbours(r, ind_neighbour, 1e6)
#         beta, h0, h = jax.vmap(self.get_beta_h0_h, in_axes=(None, 0, 0))(params, r, r_neighbour)
#         h_neighbour = get_neighbours(h, ind_neighbour, 0.0)
#         h_out = jax.vmap(self.message_passing.apply, in_axes=(None, 0, 0, 0))(
#             params.message_passing, h0, h_neighbour, beta
#         )
#         return h_out

#     def apply_with_fwd_lap(self, params, r, ind_neighbour, max_n_dependencies):
#         n_el, n_neighbours = ind_neighbour.shape

#         # Step 0: Get neighbours
#         r_neighbour = get_neighbours(r, ind_neighbour, 1e6)

#         # Step 1:
#         # These steps contain no dynamic indexing => can use compile-time sparsity of folx
#         beta, h0, h = jax.vmap(
#             folx.forward_laplacian(
#                 lambda *args: self.get_beta_h0_h(params, *args), sparsity_threshold=0.6, disable_jit=True
#             )
#         )(r, r_neighbour)

#         # Step 2: These steps contain dynamic indexing => cannot use compile-time sparsity of folx, but can use local sparsity
#         # Every diff/beta/h0/h depends on the center electron and its neighbours
#         ind_dep = jnp.concatenate([np.arange(n_el)[:, None], ind_neighbour], axis=-1)
#         h_neighbour, ind_dep_out = get_neighbour_with_FwdLapArray(
#             h, ind_neighbour, ind_dep, ind_dep, max_n_dependencies[1]
#         )

#         # folx doesn't nicely work together with transformations such as vmap, because vmapping over a FwdLaplArray
#         # also vmaps over the x0_idx array. This turns x0_idx into a jnp.array which breaks the compile-time constant requirement
#         # Hacky solution: Build the x0_idx array manually and replace it in the FwdLaplArray
#         # TODO: could extract these x0_idx from a compile-time pass through forward_laplacian instead of rebuilding manually
#         x0_idx_h0 = np.tile(np.arange(3 * (n_neighbours + 1))[:, None], self.width)
#         x0_idx_beta = np.zeros([6, n_neighbours, self.beta_width_out], dtype=int)
#         for j in np.arange(n_neighbours):
#             x0_idx_beta[:3, j, :] = np.arange(3)[:, None]
#             x0_idx_beta[3:, j, :] = np.arange(3 * (j + 1), 3 * (j + 2))[:, None]

#         message_passing = functools.partial(self.message_passing.apply, params.message_passing)
#         message_passing = forward_lap_with_frozen_x0_idx(
#             message_passing, [x0_idx_h0, None, x0_idx_beta], sparsity_threshold=0.6
#         )
#         message_passing = jax.vmap(message_passing)
#         h_out = message_passing(h0, h_neighbour, beta)
#         return h_out
