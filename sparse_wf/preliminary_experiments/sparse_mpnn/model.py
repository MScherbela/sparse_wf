# %%
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from folx.api import FwdLaplArray, FwdJacobian
import functools
from get_neighbours import (
    get_with_fill,
    get_neighbour_with_FwdLapArray,
    NeighbourIndicies,
    NrOfDependencies,
    NO_NEIGHBOUR,
)
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
def densify_jacobian_by_zero_padding(h: FwdLaplArray, n_deps_out):
    jac = h.jacobian.data
    n_deps_sparse = jac.shape[0]
    padding = jnp.zeros([n_deps_out - n_deps_sparse, *jac.shape[1:]])
    return FwdLaplArray(x=h.x, jacobian=FwdJacobian(jnp.concatenate([jac, padding], axis=0)), laplacian=h.laplacian)


@jax.vmap
def densify_jacobian_diagonally(h: FwdLaplArray):
    jac = h.jacobian.data
    assert jac.shape[0] == 3
    n_neighbours = jac.shape[1]

    idx_neighbour = np.arange(n_neighbours)
    jac_out = jnp.zeros([n_neighbours, 3, n_neighbours, *jac.shape[2:]])
    jac_out = jac_out.at[idx_neighbour, :, idx_neighbour, ...].set(jac.swapaxes(0, 1))
    jac_out = jax.lax.collapse(jac_out, 0, 2)  # merge (n_deps, 3) into (n_deps*3)
    return FwdLaplArray(x=h.x, jacobian=FwdJacobian(jac_out), laplacian=h.laplacian)


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

    def apply(
        self,
        params: SparseMoonParams,
        r,
        spin,
        R,
        idx_nb: NeighbourIndicies,
        n_deps: NrOfDependencies = None,
        dep_maps=None,
    ):
        with_fwd_lap = (n_deps is not None) and (dep_maps is not None)

        # Define local heper functions
        def maybe_fwd_lap(func, argnums=None, sparsity_threshold=0.6):
            """Helper function which applies fwd_lap when with_fwd_lap is True, else just acts as identity."""
            if with_fwd_lap:
                return fwd_lap(func, argnums, sparsity_threshold=sparsity_threshold)
            return func

        @jax.vmap
        @maybe_fwd_lap
        def contract(h_nb, Gamma_nb, h_center=None):
            """Helper function implementing the message passing step"""
            msg = jnp.einsum("...ij,...ij->...j", h_nb, Gamma_nb)
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
            # Shapes: h0_nb_ne: [n_nuc x n_neighbour x feature_dim]; h0_nb_ne.jac: [n_nuc x 3*n_deps2 x feature_dim] (dense)
            h0_nb_ne = get_neighbour_with_FwdLapArray(h0, idx_nb.ne, n_deps.Hnuc, dep_maps[0])
            # densify from deps == 3 --> deps==3*n_neighbours
            Gamma_ne = densify_jacobian_diagonally(Gamma_ne)
            # densify from deps==3*n_neighbours --> deps==3*n_deps2
            Gamma_ne = densify_jacobian_by_zero_padding(Gamma_ne, 3 * n_deps.Hnuc)
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
            H_nb_en = get_neighbour_with_FwdLapArray(H, idx_nb.en, n_deps.hout, dep_maps[1])
            # densify Gamma from deps==3 --> deps==3*n_deps3
            Gamma_en = densify_jacobian_by_zero_padding(Gamma_en, 3 * n_deps.hout)
            # densify h0 from deps==3*n_deps1 --> deps==3*n_deps3
            h0 = densify_jacobian_by_zero_padding(h0, 3 * n_deps.hout)
        else:
            H_nb_en = get_with_fill(H, idx_nb.en, 0.0)
        h_out = contract(H_nb_en, Gamma_en, h0)
        return h_out
