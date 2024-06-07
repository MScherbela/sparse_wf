from typing import Callable, Optional, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from sparse_wf.api import Electrons, Int, Nuclei
from sparse_wf.jax_utils import jit, nn_multi_vmap, nn_vmap, fwd_lap
from sparse_wf.model.graph_utils import (
    DistanceMatrix,
    get_full_distance_matrices,
    get_neighbour_coordinates,
    get_neighbour_indices,
    get_nr_of_neighbours,
    get_with_fill,
    round_to_next_step,
    NO_NEIGHBOUR,
    NeighbourIndices,
    Dependency,
    DependencyMap,
    merge_dependencies,
    get_dependency_map,
)
from sparse_wf.model.utils import (
    DynamicFilterParams,
    FixedScalingFactor,
    PairwiseFilter,
    contract,
    get_diff_features,
    get_diff_features_vmapped,
    scale_initializer,
    zeros_initializer,
)
from sparse_wf.model.wave_function import MoonLikeWaveFunction, NrOfDependencies, StaticInput
from sparse_wf.tree_utils import tree_idx
from folx.api import FwdLaplArray
import functools


@jit
def _get_max_nr_of_dependencies(dist_ee: DistanceMatrix, dist_ne: DistanceMatrix, cutoff: float):
    # Thest first electron message passing step can depend at most on electrons within 1 * cutoff
    n_deps_max_h0 = jnp.max(jnp.sum(dist_ee < cutoff, axis=-1))

    # The nuclear embeddings are computed with 2 message passing steps and can therefore depend at most on electrons within 2 * cutoff
    n_deps_max_H = jnp.max(jnp.sum(dist_ne < cutoff * 2, axis=-1))

    # The output electron embeddings are computed with 3 message passing step and can therefore depend at most on electrons within 3 * cutoff
    n_deps_max_h_out = jnp.max(jnp.sum(dist_ee < cutoff * 3, axis=-1))
    return n_deps_max_h0, n_deps_max_H, n_deps_max_h_out


class MoonElecEmb(nn.Module):
    R: Nuclei
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    activation: Callable = nn.silu

    @nn.compact
    def __call__(
        self,
        r: Float[Array, "dim=3"],
        r_nb: Float[Array, "*neighbours dim=3"],
        s: Optional[Int] = None,
        s_nb: Optional[Integer[Array, " *neighbors"]] = None,
    ):
        features_ee = get_diff_features_vmapped(r, r_nb, s, s_nb)
        beta = PairwiseFilter(self.cutoff, self.filter_dims[1], name="beta_ee")
        dynamic_params_ee = DynamicFilterParams(
            scales=self.param("ee_scales", scale_initializer, self.cutoff, (self.n_envelopes,)),
            kernel=self.param(
                "ee_kernel",
                jax.nn.initializers.lecun_normal(dtype=jnp.float32),
                (features_ee.shape[-1], self.filter_dims[0]),
            ),
            bias=self.param("ee_bias", jax.nn.initializers.normal(2, dtype=jnp.float32), (self.filter_dims[0],)),
        )
        beta_ee = beta(features_ee, dynamic_params_ee)
        gamma_ee = nn.Dense(self.feature_dim, use_bias=False)(beta_ee)

        # logarithmic rescaling
        inp_ee = features_ee / features_ee[..., :1] * jnp.log1p(features_ee[..., :1])
        feat_ee = self.activation(nn.Dense(self.feature_dim)(inp_ee))
        result = jnp.einsum("...id,...id->...d", feat_ee, gamma_ee)
        result = nn.Dense(self.feature_dim)(result)
        result = nn.silu(result)
        return result


class MoonElecToNucGamma(nn.Module):
    R: Nuclei
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int

    @nn.compact
    def __call__(self, r_nb_ne: Float[Array, "n_nuc neighbours dim=3"]):
        n_nuc = len(self.R)

        features_ne = jax.vmap(jax.vmap(get_diff_features, in_axes=(None, 0)))(self.R, r_nb_ne)

        filter_ne = PairwiseFilter(self.cutoff, self.filter_dims[1], name="beta_ne")
        # vmap over neighbors (electrons), vmap over center (nuclei)
        filter_ne = nn_multi_vmap(filter_ne, in_axes=[(0, None), (0, 0)])
        dynamic_params_ne = DynamicFilterParams(
            scales=self.param(
                "ne_scales",
                scale_initializer,
                self.cutoff,
                (n_nuc, self.n_envelopes),
            ),
            kernel=self.param(
                "ne_kernel",
                jax.nn.initializers.lecun_normal(dtype=jnp.float32),
                (n_nuc, features_ne.shape[-1], self.filter_dims[0]),
            ),
            bias=self.param("ne_bias", jax.nn.initializers.normal(2, dtype=jnp.float32), (n_nuc, self.filter_dims[0])),
        )
        beta_ne = filter_ne(features_ne, dynamic_params_ne)
        gamma_ne = nn.Dense(self.feature_dim, use_bias=False)(beta_ne)

        z_n = self.param("z_n", jax.nn.initializers.normal(1.0, dtype=jnp.float32), (n_nuc, self.feature_dim))
        # logarithmic rescaling
        inp_ne = features_ne / features_ne[..., :1] * jnp.log1p(features_ne[..., :1])
        edge_ne = nn.Dense(self.feature_dim)(inp_ne) + z_n[:, None]
        return gamma_ne, edge_ne


class MoonNucToElecGamma(nn.Module):
    R: Nuclei
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int

    @nn.compact
    def __call__(
        self,
        r: Electrons,
        R_nb_en: Float[Array, "n_elec neighbours dim=3"],
        idx_en: Integer[Array, "n_elec neighbours"],
    ):
        n_nuc = len(self.R)

        features_en = jax.vmap(jax.vmap(get_diff_features, in_axes=(None, 0)))(r, R_nb_en)

        filter_en = PairwiseFilter(self.cutoff, self.filter_dims[1], name="beta_en")
        # vmap over neighbors (nuclei),  vmap over center (electrons)
        filter_en = nn_multi_vmap(filter_en, in_axes=[0, 0])
        dynamic_params_en = DynamicFilterParams(
            scales=self.param(
                "en_scales",
                scale_initializer,
                self.cutoff,
                (n_nuc, self.n_envelopes),
            ),
            kernel=self.param(
                "en_kernel",
                jax.nn.initializers.lecun_normal(dtype=jnp.float32),
                (n_nuc, features_en.shape[-1], self.filter_dims[0]),
            ),
            bias=self.param(
                "en_bias", jax.nn.initializers.normal(2, dtype=jnp.float32), (n_nuc, self.filter_dims[0])
            ).astype(jnp.float32),
        )
        dynamic_params_en = jax.vmap(tree_idx, in_axes=(None, 0))(dynamic_params_en, idx_en)
        beta_en = filter_en(features_en, dynamic_params_en)

        gamma_en_init = nn.Dense(self.feature_dim, use_bias=False)(beta_en)
        gamma_en_out = nn.Dense(self.feature_dim, use_bias=False)(beta_en)

        # logarithmic rescaling
        inp_en = features_en / features_en[..., :1] * jnp.log1p(features_en[..., :1])
        edge_en = nn.Dense(self.feature_dim)(inp_en)
        nuc_emb = self.param("z_n", jax.nn.initializers.normal(1.0, dtype=jnp.float32), (n_nuc, self.feature_dim))
        edge_en += nuc_emb[idx_en]

        return gamma_en_init, gamma_en_out, edge_en


class MoonNucLayer(nn.Module):
    @nn.compact
    def __call__(self, H_up, H_down):
        dim = H_up.shape[-1]
        same_dense = nn.Dense(dim)
        diff_dense = nn.Dense(dim, use_bias=False)
        return (
            (nn.silu(same_dense(H_up) + diff_dense(H_down)) + H_up) / jnp.sqrt(2),
            (nn.silu(same_dense(H_down) + diff_dense(H_up)) + H_down) / jnp.sqrt(2),
        )


class MoonNucMLP(nn.Module):
    n_layers: int

    @nn.compact
    def __call__(self, H_up, H_down):
        for _ in range(self.n_layers):
            H_up, H_down = MoonNucLayer()(H_up, H_down)
        return H_up, H_down


class MoonElecOut(nn.Module):
    @nn.compact
    def __call__(self, elec, msg):
        dim = elec.shape[-1]
        elec = nn.silu(nn.Dense(dim)(elec))
        out = nn.silu(nn.Dense(dim)(elec) + msg)
        out = nn.silu(nn.Dense(dim)(out))
        return FixedScalingFactor()(out + elec)


class DependenciesMoon(NamedTuple):
    h0: Dependency
    H_nuc: Dependency
    h_el_out: Dependency


class DependencyMaps(NamedTuple):
    h0_to_Hnuc: DependencyMap
    Gamma_ne_to_Hnuc: DependencyMap
    Hnuc_to_hout: DependencyMap
    h0_to_hout: DependencyMap


@jit(static_argnames=("n_deps_max",))
def get_all_dependencies(idx_nb: NeighbourIndices, n_deps_max: NrOfDependencies):
    """Get the indices of electrons on which each embedding will depend on.

    Args:
        idx_nb: NeighbourIndices, named tuple containing the indices of the neighbours of each electron and nucleus.
        n_deps_max: maximum_nr_of electrons that each embedding can depend on.
            - n_deps_max[0]: maximum number of dependencies for the electron embeddings at the first step.
            - n_deps_max[1]: maximum number of dependencies for the nuclear embeddings.
            - n_deps_max[2]: maximum number of dependencies for the output electron embeddings.

    Returns:
        deps: tuple of jnp.ndarray, dependencies for the electron embeddings at each step.
            deps_h0: [n_el  x nr_of_deps_level_1]
            deps_H:  [n_nuc x nr_of deps_level_2]
            deps_hout: [n_el x nr_of_deps_level_3]
        dep_maps: tuple of jnp.ndarray, maps the dependencies between the levels:
            h0_to_Hnuc: [n_nuc x n_neighbouring_el x nr_of_deps_level_1]; values are in [0 ... deps_level_2]
            Gamma_ne_to_Hnuc: [n_nuc x n_neighbouring_el x 1]; values are in [0 ... deps_level_2]
            Hnuc_to_hout: [n_el x n_neighbouring_nuc x nr_of_deps_level_2]; values are in [0 ... deps_level_3]
            h0_to_hout: [n_el x n_neighbouring_el x nr_of_deps_level_1]; values are in [0 ... deps_level_3]
    """
    n_el = idx_nb.ee.shape[-2]
    batch_dims = idx_nb.ee.shape[:-2]
    self_dependency = jnp.arange(n_el)[:, None]
    self_dependency = jnp.tile(self_dependency, batch_dims + (1, 1))

    @functools.partial(jnp.vectorize, signature="(center1,deps),(center2,neigbours)->(center2,neigbours,deps)")
    def get_deps_nb(deps, idx_nb):
        return get_with_fill(deps, idx_nb, NO_NEIGHBOUR)

    get_dep_map_for_all_centers = jax.vmap(jax.vmap(get_dependency_map, in_axes=(0, None)))

    # Step 1: Initial electron embeddings depend on themselves and their neighbours
    deps_h0: Dependency = jnp.concatenate([self_dependency, idx_nb.ee], axis=-1)

    # Step 2: Nuclear embeddings depend on all dependencies of their neighbouring electrons
    deps_neighbours = get_deps_nb(deps_h0, idx_nb.ne)
    deps_H = merge_dependencies(deps_neighbours, idx_nb.ne, None, n_deps_max.H_nuc)
    dep_map_h0_to_H = get_dep_map_for_all_centers(deps_neighbours, deps_H)
    dep_map_Gamma_ne_to_H = get_dep_map_for_all_centers(idx_nb.ne[..., None], deps_H)

    # Step 3: Output electron embeddings depend on themselves, their neighbouring electrons and all dependencies of their neighbouring nuclei
    deps_neighbours = get_deps_nb(deps_H, idx_nb.en)
    deps_hout = merge_dependencies(deps_neighbours, deps_h0, jnp.arange(n_el)[:, None], n_deps_max.h_el_out)
    dep_map_H_to_hout = get_dep_map_for_all_centers(deps_neighbours, deps_hout)
    dep_map_h0_to_hout = jax.vmap(get_dependency_map)(deps_h0, deps_hout)

    return DependenciesMoon(deps_h0, deps_H, deps_hout), DependencyMaps(
        dep_map_h0_to_H, dep_map_Gamma_ne_to_H, dep_map_H_to_hout, dep_map_h0_to_hout
    )


class Moon(MoonLikeWaveFunction):
    def setup(self):
        super().setup()

        n_nuc = len(self.R)
        self.dynamic_params_en = DynamicFilterParams(
            scales=self.param(
                "en_scales",
                scale_initializer,
                self.cutoff,
                (n_nuc, self.pair_n_envelopes),
            ),
            kernel=self.param(
                "en_kernel", jax.nn.initializers.lecun_normal(dtype=jnp.float32), (n_nuc, 5, self.pair_mlp_widths[0])
            ),
            bias=self.param("en_bias", zeros_initializer(jnp.float32), (n_nuc, self.pair_mlp_widths[0])),
        )
        self.filter_en = PairwiseFilter(
            self.cutoff, self.pair_mlp_widths[1], name="beta_en"
        )  # TODO: make this cutoff larger?
        self.gamma_ne = MoonElecToNucGamma(
            R=self.R,
            cutoff=self.cutoff,
            filter_dims=self.pair_mlp_widths,
            feature_dim=self.feature_dim,
            n_envelopes=self.pair_n_envelopes,
        )
        self.gamma_en = MoonNucToElecGamma(
            R=self.R,
            cutoff=self.cutoff,
            filter_dims=self.pair_mlp_widths,
            feature_dim=self.feature_dim,
            n_envelopes=self.pair_n_envelopes,
        )
        self.elec_elec_emb = MoonElecEmb(
            R=self.R,
            cutoff=self.cutoff,
            filter_dims=self.pair_mlp_widths,
            feature_dim=self.feature_dim,
            n_envelopes=self.pair_n_envelopes,
        )
        self.nuc_mlp = MoonNucMLP(self.nuc_mlp_depth)
        self.elec_out = MoonElecOut()

        # scalings
        self.scales = dict(
            h0=FixedScalingFactor(),
            H1_up=FixedScalingFactor(),
            H1_down=FixedScalingFactor(),
            h1=FixedScalingFactor(),
            msg=FixedScalingFactor(),
            nuc=FixedScalingFactor(),
        )

    def _apply_elec_elec_emb(self, r, r_nb_ee, spin, spin_nb_ee):
        return self.elec_elec_emb(r, r_nb_ee, spin, spin_nb_ee)

    def _apply_Gamma_ne(self, r_nb_ne):
        return self.gamma_ne(r_nb_ne)

    def _apply_scale(self, submodule, value):
        return self.scales[submodule](value)

    def _embedding(self, electrons: Electrons, static: StaticInput) -> Electrons:
        idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en = get_neighbour_coordinates(
            electrons, self.R, idx_nb, self.spins
        )

        # initial electron embedding
        h0 = nn_vmap(self.elec_elec_emb)(electrons, r_nb_ee, self.spins, spin_nb_ee)
        h0 = self._apply_scale("h0", h0)

        # construct nuclei embeddings
        Gamma_ne, edge_ne_emb = self.gamma_ne(r_nb_ne)
        h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0)
        edge_ne_emb = nn.silu(h0_nb_ne + edge_ne_emb)
        edge_ne_up = jnp.where(spin_nb_ne[..., None] > 0, edge_ne_emb, 0)
        edge_ne_down = jnp.where(spin_nb_ne[..., None] < 0, edge_ne_emb, 0)
        H1_up = contract(edge_ne_up, Gamma_ne)
        H1_down = contract(edge_ne_down, Gamma_ne)
        H1_up = self._apply_scale("H1_up", H1_up)
        H1_down = self._apply_scale("H1_down", H1_down)

        # construct electron embedding
        gamma_en_init, gamma_en_out, edge_en_emb = self.gamma_en(electrons, R_nb_en, idx_nb.en)
        edge_en_emb = nn.silu(h0[:, None] + edge_en_emb)
        h1 = contract(edge_en_emb, gamma_en_init)
        h1 = self._apply_scale("h1", h1)

        # update electron embedding
        HL_up, HL_down = self.nuc_mlp(H1_up, H1_down)
        HL_up_nb_en = get_with_fill(HL_up, idx_nb.en, 0)
        HL_down_nb_en = get_with_fill(HL_down, idx_nb.en, 0)
        HL_nb_en = jnp.where(self.spins[..., None, None] > 0, HL_up_nb_en, HL_down_nb_en)
        m = contract(HL_nb_en, gamma_en_out)
        m = self._apply_scale("msg", m)
        # readout
        hL = self.elec_out(h1, m)
        return hL

    def _embedding_with_fwd_lap(
        self, params, electrons: Electrons, static: StaticInput
    ) -> tuple[FwdLaplArray, Dependency]:
        spins = self.get_spins()
        idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en = get_neighbour_coordinates(electrons, self.R, idx_nb, spins)
        deps, dep_maps = get_all_dependencies(idx_nb, static.n_deps)

        @functools.partial(jax.vmap, in_axes=0, out_axes=-2)  # vmap over center electrons
        @functools.partial(fwd_lap, argnums=(0, 1))
        def h0_apply(r, r_nb, s, s_nb):
            h0 = self.apply(params, r, r_nb, s, s_nb, method=self._apply_elec_elec_emb)
            h0 = self.apply(params, "h0", h0, method=self._apply_scale)
            return h0

        h0 = h0_apply(electrons, r_nb_ee, spins, spin_nb_ee)

        @functools.partial(jax.vmap, in_axes=0, out_axes=-3)  # vmap over center nuclei
        @functools.partial(jax.vmap, in_axes=0, out_axes=-2)  # vmap over neighbouring electrons
        @fwd_lap
        def get_Gamma_ne(r_ne):
            return self.apply(params, r_ne, method=self._apply_Gamma_ne)

        Gamma_ne, edge_ne_emb = get_Gamma_ne(r_nb_ne)

        h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0)
        edge_ne_emb = nn.silu(h0_nb_ne + edge_ne_emb)
        # edge_ne_up = jnp.where(spin_nb_ne[..., None] > 0, edge_ne_emb, 0)
        # edge_ne_down = jnp.where(spin_nb_ne[..., None] < 0, edge_ne_emb, 0)

        return h0, deps.h0

    def get_static_input(self, electrons: Array) -> StaticInput:
        def round_fn(x):
            return int(round_to_next_step(x, 1.2, 1, self.n_electrons))

        dist_ee, dist_ne = get_full_distance_matrices(electrons, self.R)
        n_neighbours = get_nr_of_neighbours(dist_ee, dist_ne, self.cutoff, 1.2, 1)
        n_deps_h0, n_deps_H, n_deps_hout = _get_max_nr_of_dependencies(dist_ee, dist_ne, self.cutoff)  # noqa: F821

        n_deps_h0_padded = round_fn(n_deps_h0)
        n_deps_H_padded = round_fn(n_deps_H)
        n_deps_hout_padded = round_fn(n_deps_hout)
        n_deps = NrOfDependencies(n_deps_h0_padded, n_deps_H_padded, n_deps_hout_padded)
        return StaticInput(n_neighbours=n_neighbours, n_deps=n_deps)
