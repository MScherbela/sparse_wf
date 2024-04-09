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
    NeighbourIndices,
    NrOfDependenciesMoon,
    NO_NEIGHBOUR,
)
import chex
from typing import Callable
from dataclasses import dataclass, field
from utils import fwd_lap
from sparse_wf.api import Electrons, Nuclei, Charges, Parameters, DependencyMap
from typing import Optional, Sequence, cast
from jaxtyping import Float, Array, PRNGKeyArray

FilterKernel = Float[Array, "neighbour features"]
Embedding = Float[Array, "features"]




def cutoff_function(d: Float[Array, "*dims"], p=4) -> Float[Array, "*dims"]: # noqa: F821
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


class PairwiseFilter(nn.Module):
    cutoff: float
    directional_mlp_widths: Sequence[int]
    n_envelopes: int
    out_dim: int

    def scale_initializer(self, rng, shape):
        return self.cutoff + 0.5 * self.cutoff * jax.random.normal(rng, shape)

    @nn.compact
    def __call__(self, dist_diff: Float[Array, "*batch_dims features_in"]) -> Float[Array, "*batch_dims features_out"]:
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


def contract(h_nb: Embedding, Gamma_nb: FilterKernel, h_center: Optional[FilterKernel]=None):
    """Helper function implementing the message passing step"""
    msg = jnp.einsum("...ij,...ij->...j", h_nb, Gamma_nb)
    if h_center is not None:
        msg += h_center
    return cast(Embedding, jax.nn.silu(msg))


@chex.dataclass
class SparseMoonParams(Parameters):
    ee_filter: Parameters
    ne_filter: Parameters
    en_filter: Parameters
    mlp_nuc: Parameters
    lin_h0: Parameters
    lin_orbitals: Parameters


@dataclass
class SparseMoonWavefunction:
    n_orbitals: int
    R: Nuclei
    Z: Charges
    cutoff: float
    feature_dim: int
    nuc_mlp_depth: int
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
        self.mlp_nuc = MLP(widths=[self.feature_dim] * self.nuc_mlp_depth, name="mlp_nuc")
        self.lin_h0 = nn.Dense(self.feature_dim, use_bias=True, name="lin_h0")
        self.lin_orbitals = nn.Dense(self.n_orbitals, use_bias=False, name="lin_orbitals")

    def init(self, rng: PRNGKeyArray):
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

    def get_neighbour_coordinates(self, r: Electrons, spin, idx_nb: NeighbourIndices):
        spin_nb_ee = get_with_fill(spin, idx_nb.ee, 0.0)
        r_nb_ee = get_with_fill(r, idx_nb.ee, NO_NEIGHBOUR)  # [n_el  x n_neighbouring_electrons x 3]
        r_nb_ne = get_with_fill(r, idx_nb.ne, NO_NEIGHBOUR)  # [n_nuc x n_neighbouring_electrons x 3]
        R_nb_en = get_with_fill(self.R, idx_nb.en, NO_NEIGHBOUR)  # [n_el  x n_neighbouring_nuclei    x 3]
        return spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en
    
    def _get_h0(self, params, r_, r_nb_ee_, spin_, spin_nb_ee_):
        # vmap over neighbours
        features_ee = jax.vmap(get_diff_features, in_axes=(None, 0, None, 0))(r_, r_nb_ee_, spin_, spin_nb_ee_)
        Gamma_ee = cast(FilterKernel, self.ee_filter.apply(params.ee_filter, features_ee))
        h0 = jnp.sum(Gamma_ee, axis=-2)
        h0 = cast(Embedding, self.lin_h0.apply(params.lin_h0, h0))
        return h0
    
    def _get_Gamma_ne(self, params, R, r_nb_ne):
        features_ne = get_diff_features(R, r_nb_ne)
        return cast(FilterKernel, self.ne_filter.apply(params.ne_filter, features_ne))

    def _get_Gamma_en(self, params, r, R_nb_en_):
        features_en = get_diff_features(r, R_nb_en_)
        return cast(FilterKernel, self.en_filter.apply(params.en_filter, features_en))

    def _get_Gamma_ne_vmapped(self, params, R, r_nb_ne):
        _get_Gamma = jax.vmap(self._get_Gamma_ne, in_axes=(None, None, 0)) # vmap over neighbours
        _get_Gamma = jax.vmap(_get_Gamma, in_axes=(None, 0, 0))            # vmap over center
        return _get_Gamma(params, R, r_nb_ne)
    
    def _get_Gamma_en_vmapped(self, params, r, R_nb_en_):
        _get_Gamma = jax.vmap(self._get_Gamma_en, in_axes=(None, None, 0)) # vmap over neighbours
        _get_Gamma = jax.vmap(_get_Gamma, in_axes=(None, 0, 0))            # vmap over center
        return _get_Gamma(params, r, R_nb_en_)

    def orbitals(self,
        params: SparseMoonParams,
        r: Electrons,
        spin,
        idx_nb: NeighbourIndices):

        # Step 0: Get neighbours
        spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en = self.get_neighbour_coordinates(r, spin, idx_nb)

        # Step 1: Get h0
        h0 = jax.vmap(self._get_h0, in_axes=(None, 0, 0, 0, 0))(params, r, r_nb_ee, spin, spin_nb_ee)

        # Step 2: Contract to nuclei + MLP
        Gamma_ne = self._get_Gamma_ne_vmapped(params, self.R, r_nb_ne)
        h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0.0)
        H0 = contract(h0_nb_ne, Gamma_ne)
        H = cast(Embedding, self.mlp_nuc.apply(params.mlp_nuc, H0))

        # Step 3: Contract back to electrons => Final electron embeddings h_out
        Gamma_en = self._get_Gamma_en_vmapped(params, r, R_nb_en)
        H_nb_en = get_with_fill(H, idx_nb.en, 0.0)
        h_out = contract(H_nb_en, Gamma_en, h0)
        return h_out


    def orbitals_with_fwd_lap(
        self,
        params: SparseMoonParams,
        r: Electrons,
        spin,
        R: Nuclei,
        idx_nb: NeighbourIndices,
        n_deps: NrOfDependenciesMoon,
        dep_maps: Sequence[DependencyMap],
    ):
         # Step 0: Get neighbours
        spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en = self.get_neighbour_coordinates(r, spin, idx_nb)


        # Step 1: Contract ee to get electron embeedings h0
        get_h0 = fwd_lap(self._get_h0, argnums=(1, 2))(params, r, r_nb_ee, spin, spin_nb_ee)
        get_h0 = jax.vmap(get_h0, in_axes=(None, 0, 0, 0, 0))
        # Shapes: h0: [nel x feature_dim]; h0.jac: [n_el x 3*n_deps1 x feature_dim] (dense)
        h0 = get_h0(r, r_nb_ee, spin, spin_nb_ee)

        # Step 2: Contract to nuclei + MLP on nuclei => nuclear embedding H
        # 2a: Get the spatial filter between nuclei and neighbouring electrons
        get_Gamma_ne = fwd_lap(self._get_Gamma_ne, argnums=(2,))(params, R, r_nb_ne)
        get_Gamma_ne = jax.vmap(get_Gamma_ne, in_axes=(None, None, 0))
        get_Gamma_ne = jax.vmap(get_Gamma_ne, in_axes=(None, 0, 0))
        # Gamma_ne: [n_nuc x n_neighbour x feature_dim]; Gamma_ne.jac: [n_nuc x 3 x n_neighbour x feature_dim] (sparse)
        Gamma_ne = get_Gamma_ne(R, r_nb_ne)
        Gamma_ne = densify_jacobian_diagonally(Gamma_ne) # deps: 3 --> 3*n_neighbours
        Gamma_ne = densify_jacobian_by_zero_padding(Gamma_ne, 3 * n_deps.Hnuc) # deps: 3*n_neighbours --> 3*n_deps2

        # 2b: Get the neighbouring electron embeddings
        h0_nb_ne = get_neighbour_with_FwdLapArray(h0, idx_nb.ne, n_deps.Hnuc, dep_maps[0])

        # 2c: Contract and apply the MLP
        H0 = fwd_lap(contract)(h0_nb_ne, Gamma_ne)
        H = jax.vmap(fwd_lap(self.mlp_nuc.apply, argnums=(1,)), in_axes=(None, 0))(H0)

        # Step 3: Contract back to electrons => Final electron embeddings h_out
        # 3a: Get Filters
        get_Gama_en = fwd_lap(self._get_Gamma_en, argnums=(1,))(params, r, R_nb_en)
        get_Gama_en = jax.vmap(get_Gama_en, in_axes=(None, None, 0))
        get_Gama_en = jax.vmap(get_Gama_en, in_axes=(None, 0, 0))
        # Shapes: Gamma_en: [n_el x n_neighbouring_nuc x feature_dim]; Gamma_en.jac: [n_el x 3 x n_neighbouring_nuc x feature_dim] (dense)
        Gamma_en = get_Gama_en(params, r, R_nb_en)
        Gamma_en = densify_jacobian_by_zero_padding(Gamma_en, 3 * n_deps.hout) # deps: 3 --> 3*n_deps3


        H_nb_en = get_neighbour_with_FwdLapArray(H, idx_nb.en, n_deps.hout, dep_maps[1])
        h0 = densify_jacobian_by_zero_padding(h0, 3 * n_deps.hout) # deps: 3*n_deps1 --> 3*n_deps3
        h_out = fwd_lap(contract)(H_nb_en, Gamma_en, h0)
        return h_out
