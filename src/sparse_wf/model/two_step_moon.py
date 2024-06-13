from sparse_wf.api import Electrons, StaticInput
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.model.utils import (
    PairwiseFilter,
    DynamicFilterParams,
    scale_initializer,
    get_diff_features,
    MLP,
)
from sparse_wf.model.graph_utils import (
    get_full_distance_matrices,
    get_neighbour_indices,
    get_neighbour_coordinates,
    get_nr_of_neighbours,
    get_with_fill,
)
from sparse_wf.jax_utils import nn_vmap, nn_multi_vmap
import flax.linen as nn
import jax
from typing import Optional, NamedTuple
import jax.tree_util as jtu
import jax.numpy as jnp


def contract(Gamma, edge, neighbour=None):
    if neighbour is not None:
        edge += neighbour
    return jnp.einsum("jf,jf->f", Gamma, jax.nn.silu(edge))


def zeros_initializer(rng, shape):
    return jnp.zeros(shape, dtype=jnp.float32)


class DynamicParams(NamedTuple):
    filter: DynamicFilterParams
    edge_kernel: jax.Array
    edge_bias: jax.Array


class EdgeFeatures(nn.Module):
    feature_dim: int
    edge_feature_dim: int
    cutoff: float

    @nn.compact
    def __call__(self, r, r_nb, s, s_nb, params: DynamicParams):
        features = get_diff_features(r, r_nb, s, s_nb)
        beta = PairwiseFilter(self.cutoff, self.edge_feature_dim, name="beta")(features, params.filter)
        Gamma = nn.Dense(self.feature_dim, use_bias=False, name="Gamma")(beta)
        edge_features = features @ params.edge_kernel + params.edge_bias
        return Gamma, edge_features


class ElectronOutputMLP(nn.Module):
    @nn.compact
    def __call__(self, h0, msg_from_nuc, msg_from_el):
        feature_dim = h0.shape[-1]
        msg_from_nuc = nn.Dense(feature_dim, use_bias=True)(msg_from_nuc)
        msg_from_el = nn.Dense(feature_dim, use_bias=False)(msg_from_el)
        msg_from_residual = nn.Dense(feature_dim, use_bias=False)(h0)
        h_out = jax.nn.silu(msg_from_nuc + msg_from_el + msg_from_residual)
        h_out = nn.Dense(feature_dim)(h_out) + h0
        return h_out


class TwoStepMoon(MoonLikeWaveFunction):
    def build_dynamic_params(self, name: str, input_dim: int, cutoff: float, n_nuc: Optional[int] = None):
        hidden_dim = self.pair_mlp_widths[0]
        shape_filter_scales = (n_nuc, self.pair_n_envelopes) if n_nuc else (self.pair_n_envelopes,)
        shape_filter_kernel = (n_nuc, input_dim, hidden_dim) if n_nuc else (input_dim, hidden_dim)
        shape_filter_bias = (n_nuc, hidden_dim) if n_nuc else (hidden_dim,)
        shape_edge_kernel = (n_nuc, input_dim, self.feature_dim) if n_nuc else (input_dim, self.feature_dim)
        shape_edge_bias = (n_nuc, self.feature_dim) if n_nuc else (self.feature_dim,)

        return DynamicParams(
            filter=DynamicFilterParams(
                scales=self.param(f"{name}_scales", scale_initializer, cutoff, shape_filter_scales),
                kernel=self.param(
                    f"{name}_kernel", jax.nn.initializers.lecun_normal(dtype=jnp.float32), shape_filter_kernel
                ),
                bias=self.param(f"{name}_bias", zeros_initializer, shape_filter_bias),
            ),
            edge_kernel=self.param(
                f"{name}_edge_kernel", jax.nn.initializers.lecun_normal(dtype=jnp.float32), shape_edge_kernel
            ),
            edge_bias=self.param(f"{name}_edge_bias", zeros_initializer, shape_edge_bias),
        )

    def setup(self):
        super().setup()
        n_nuc = len(self.R)

        # Edge features and dynamic edge params
        self.edges_en_in = EdgeFeatures(self.feature_dim, self.pair_mlp_widths[1], self.cutoff, name="en_in")
        self.edges_en_out = EdgeFeatures(self.feature_dim, self.pair_mlp_widths[1], self.cutoff, name="en_out")
        self.edges_ne = EdgeFeatures(self.feature_dim, self.pair_mlp_widths[1], self.cutoff, name="ne")
        self.edges_ee = EdgeFeatures(self.feature_dim, self.pair_mlp_widths[1], self.cutoff, name="ee")
        self.params_en_in = self.build_dynamic_params("en_in", 5, self.cutoff, n_nuc)
        self.params_en_out = self.build_dynamic_params("en_out", 5, self.cutoff, n_nuc)
        self.params_ne = self.build_dynamic_params("ne", 5, self.cutoff, n_nuc)
        self.params_ee = self.build_dynamic_params("ee", 6, self.cutoff)

        # Element-wise MLPs
        self.lin_h0 = nn.Dense(self.feature_dim)
        self.mlp_nuc = MLP([self.feature_dim] * self.nuc_mlp_depth, activate_final=True, residual=True)
        self.hout_block = ElectronOutputMLP()

    def get_static_input(self, electrons: Electrons) -> StaticInput:
        dist_ee, dist_ne = get_full_distance_matrices(electrons, self.R)
        n_neighbours = get_nr_of_neighbours(dist_ee, dist_ne, self.cutoff, 1.2, 1)  # TODO: different cutoffs
        return StaticInput(n_neighbours=n_neighbours, n_deps=None)  # type: ignore # TODO

    def _embedding(self, electrons: jax.Array, static: StaticInput) -> jax.Array:
        idx_nb = get_neighbour_indices(
            electrons, self.R, static.n_neighbours, cutoff_en=self.cutoff, cutoff_ee=2 * self.cutoff
        )

        # # Step 0: Get neighbours
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en = get_neighbour_coordinates(
            electrons, self.R, idx_nb, self.spins
        )
        params_en_in, params_en_out = jtu.tree_map(
            lambda p: get_with_fill(p, idx_nb.en, 0.0), (self.params_en_in, self.params_en_out)
        )

        # Compute all edge features / filters which only depend on raw coordinates
        Gamma_en_in, edge_en_in = nn_multi_vmap(
            self.edges_en_in, in_axes=[(None, 0, None, None, 0), (0, 0, 0, None, 0)]
        )(electrons, R_nb_en, self.spins, None, params_en_in)
        Gamma_en_out, edge_en_out = nn_multi_vmap(
            self.edges_en_out, in_axes=[(None, 0, None, None, 0), (0, 0, 0, None, 0)]
        )(electrons, R_nb_en, self.spins, None, params_en_out)
        Gamma_ne, edge_ne = nn_multi_vmap(self.edges_ne, in_axes=[(None, 0, None, 0, None), 0])(
            self.R, r_nb_ne, None, spin_nb_ne, self.params_ne
        )
        Gamma_ee, edge_ee = nn_multi_vmap(self.edges_ee, in_axes=[(None, 0, None, 0, None), (0, 0, 0, 0, None)])(
            electrons, r_nb_ee, self.spins, spin_nb_ee, self.params_ee
        )

        # Step 1: Contract to electrons; vmap over electrons
        h0 = jax.vmap(contract)(Gamma_en_in, edge_en_in)
        h0 = self.lin_h0(h0)

        # Step 2: Contract to nuclei; vmap over nuclei
        h0_ne = get_with_fill(h0, idx_nb.ne, 0.0)
        H = jax.vmap(contract)(Gamma_ne, edge_ne, h0_ne)
        H = nn_vmap(self.mlp_nuc)(H)

        # Step 3: Contract to electrons; vmap over electrons
        H_en = get_with_fill(H, idx_nb.en, 0.0)
        h0_ee = get_with_fill(h0, idx_nb.ee, 0.0)
        msg_from_el = jax.vmap(contract)(Gamma_ee, edge_ee, h0_ee)
        msg_from_nuc = jax.vmap(contract)(Gamma_en_out, edge_en_out, H_en)
        h_out = nn_vmap(self.hout_block)(h0, msg_from_nuc, msg_from_el)
        return h_out
