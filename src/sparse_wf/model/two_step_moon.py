from sparse_wf.api import Electrons, Nuclei
from sparse_wf.model.wave_function import MoonLikeWaveFunction, StaticInput
from sparse_wf.model.utils import PairwiseFilter, DynamicFilterParams, scale_initializer, get_diff_features, MLP
from sparse_wf.model.graph_utils import get_neighbour_indices, get_neighbour_coordinates, NO_NEIGHBOUR, get_with_fill
import flax.linen as nn
import jax
from jaxtyping import Float, Array, Int, Integer
from typing import Callable, Optional
import jax.tree_util as jtu
import jax.numpy as jnp


# class InitialElectronEmbedding(nn.Module):
#     R: Nuclei
#     cutoff: float
#     filter_dims: tuple[int, int]
#     feature_dim: int
#     activation: Callable = nn.silu

#     @nn.compact
#     def __call__(
#         self,
#         r: Float[Array, "dim=3"],
#         R_nb: Float[Array, "*neighbours dim=3"],
#         s: Optional[Int] = None,
#     ):
#         n_nuclei = len(self.R)

#         features_en = get_diff_features(r, R_nb, s, None)
#         beta = PairwiseFilter(self.cutoff, self.filter_dims[1], name="beta_ee")
#         dynamic_params_en = DynamicFilterParams(
#             scales=self.param(
#                 "en_scales",
#                 scale_initializer,
#                 (
#                     n_nuclei,
#                     self.n_envelopes,
#                 ),
#             ),
#             kernel=self.param(
#                 "en_kernel", jax.nn.initializers.lecun_normal(), (n_nuclei, features_ee.shape[-1], self.filter_dims[0])
#             ),
#             bias=self.param("en_bias", jax.nn.initializers.zeros, (n_nuclei, self.filter_dims[0])),
#         )
#         beta_en = beta(features_en, dynamic_params_en)
#         gamma_en = nn.Dense(self.feature_dim, use_bias=False)(beta_en)

#         W_dynamic = self.param("W_en_g",  jax.nn.initializers.lecun_normal(), (n_nuclei, features_en.shape[-1], self.filter_dims[0])),
#         feat_ee = self.activation(nn.Dense(self.feature_dim)(features_en))
#         return jnp.einsum("...id,...id->...d", feat_ee, gamma_ee)


class TwoStepMoon(MoonLikeWaveFunction):
    def setup(self):
        super().setup()

        n_nuc = len(self.R)
        self.dynamic_params_en = DynamicFilterParams(
            scales=self.param(
                "en_scales",
                scale_initializer,
                (n_nuc, self.n_envelopes),
            ),
            kernel=self.param("en_kernel", jax.nn.initializers.lecun_normal(), (n_nuc, 5, self.pair_mlp_widths[0])),
            bias=self.param("en_bias", jax.nn.initializers.zeros, (n_nuc, self.pair_mlp_widths[0])),
        )
        self.dynamic_params_ne = DynamicFilterParams(
            scales=self.param(
                "ne_scales",
                scale_initializer,
                (n_nuc, self.n_envelopes),
            ),
            kernel=self.param("ne_kernel", jax.nn.initializers.lecun_normal(), (n_nuc, 5, self.pair_mlp_widths[0])),
            bias=self.param("ne_bias", jax.nn.initializers.zeros, (n_nuc, self.pair_mlp_widths[0])),
        )
        self.dynamic_params_ee = DynamicFilterParams(
            scales=self.param(
                "ee_scales",
                scale_initializer,
                (self.n_envelopes,),
            ),
            kernel=self.param("ee_kernel", jax.nn.initializers.lecun_normal(), (6, self.pair_mlp_widths[0])),
            bias=self.param("ee_bias", jax.nn.initializers.zeros, (self.pair_mlp_widths[0])),
        )
        self.filter_ee = PairwiseFilter(2 * self.cutoff, self.pair_mlp_widths[1], name="beta_ee")
        self.filter_en = PairwiseFilter(
            self.cutoff, self.pair_mlp_widths[1], name="beta_en"
        )  # TODO: make this cutoff larger?
        self.filter_ne = PairwiseFilter(self.cutoff, self.pair_mlp_widths[1], name="beta_ne")
        self.lin_Gamma_ee = nn.Dense(self.feature_dim, use_bias=False, name="Gamma_ee")
        self.lin_Gamma_en_in = nn.Dense(self.feature_dim, use_bias=False, name="Gamma_en_in")
        self.lin_Gamma_en_out = nn.Dense(self.feature_dim, use_bias=False, name="Gamma_en_out")
        self.mlp_nuc = MLP([self.feature_dim] * self.nuc_mlp_depth, activate_final=True, residual=True)

        self.kernel_en_in = self.param(
            "kernel_en_in", jax.nn.initializers.lecun_normal(), (n_nuc, 5, self.feature_dim)
        )  # diff (3) + dist (1) + spin (1) = 5
        self.bias_en_in = self.param(
            "bias_en_in",
            jax.nn.initializers.zeros,
            (
                n_nuc,
                self.feature_dim,
            ),
        )

        self.kernel_ne = self.param(
            "kernel_ne", jax.nn.initializers.lecun_normal(), (n_nuc, 5, self.feature_dim)
        )  # diff (3) + dist (1) + spin (1) = 5
        self.bias_ne = self.param(
            "bias_ne",
            jax.nn.initializers.zeros,
            (
                n_nuc,
                self.feature_dim,
            ),
        )

    def _get_filters_en(self, electrons, R_nb_en, params_en):
        features_en = get_diff_features(electrons, R_nb_en, self.spins)
        beta_en = self.filter_en(features_en, params_en)
        Gamma_en_in = self.lin_Gamma_en_in(beta_en)
        Gamma_en_out = self.lin_Gamma_en_out(beta_en)
        return features_en, Gamma_en_in, Gamma_en_out

    def _get_initial_electron_embeddings(self, features_en, Gamma_en_in, kernel_en, bias_en):
        msg_en = jax.nn.silu(kernel_en @ features_en + bias_en)
        h0 = jnp.einsum("Jf,Jf->f", Gamma_en_in, msg_en)
        return h0

    def _get_nuclear_embeddings(self, R, r_ne, spins_ne, h0_ne, dynamic_params_ne, kernel_ne, bias_ne):
        features_ne = get_diff_features(R, r_ne, spins_ne)
        Gamma_ne = self.filter_ne(features_ne, dynamic_params_ne)
        msg_ne = jax.nn.silu(kernel_ne @ features_ne + bias_ne + h0_ne)
        H0 = jnp.einsum("Ijf,Ijf->If", Gamma_ne, msg_ne)
        H = self.mlp_nuc(H0)
        return H

    def _get_electron_output_embedding(
        self, electrons, h0, r_nb_ee, spins, spin_nb_ee, Gamma_en_out, H_en, Gamma_ee, h0_ee
    ):
        msg_from_nuc = jnp.einsum("Jf,Jf->If", Gamma_en_out, H_en)

        features_ee = get_diff_features(electrons, r_nb_ee, spins, spin_nb_ee)
        beta_ee = self.filter_ee(features_ee, self.dynamic_params_ee)
        Gamma_ee = self.lin_Gamma_ee(beta_ee)
        msg_ee = nn.Dense(self.feature_dim)(features_ee)
        msg_from_el = jnp.einsum("jf,jf->f", Gamma_ee, msg_ee)

        residual = nn.Dense(self.feature_dim)(h0)
        h_out = jax.nn.silu(msg_from_nuc + msg_from_el + residual)
        h_out = jax.nn.silu(nn.Dense(h_out)) + h0
        return h_out

    def _embedding(self, electrons: jax.Array, static: StaticInput) -> jax.Array:
        idx_nb = get_neighbour_indices(
            electrons, self.R, static.n_neighbours, cutoff_en=self.cutoff, cutoff_ee=2 * self.cutoff
        )

        # # Step 0: Get neighbours
        spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en = get_neighbour_coordinates(electrons, self.R, idx_nb, self.spins)

        # Step 1: Get h0 by contracting information from nuclei to electrons
        # Get pairwise features and filters
        params_en = jtu.tree_map(lambda p: get_with_fill(p, idx_nb.en, 0.0), self.dynamic_params_en)
        kernel_en = get_with_fill(self.kernel_en_in, idx_nb.en, 0.0)
        bias_en = get_with_fill(self.bias_en_in, idx_nb.en, 0.0)
        features_en, Gamma_en_in, Gamma_en_out = nn.vmap(self._get_filters_en)(electrons, R_nb_en, params_en)

        # Contract nuclei information to electrons
        # vmap over electrons
        h0 = nn.vmap(self._get_initial_electron_embeddings)(features_en, Gamma_en_in, kernel_en, bias_en)

        # Step 2: Get H0 by contracting information from electron to nuclei
        h0_ne = get_with_fill(h0, idx_nb.ne, 0.0)
        # vmap over nuclei
        H = nn.vmap(self._get_nuclear_embeddings)(
            self.R, r_nb_ne, spin_nb_ee, h0_ne, self.dynamic_params_ne, self.kernel_ne, self.bias_ne
        )

        # Step 3: Get electron embeddings by contracting information to electrons:
        # Sources ar:
        #   - Embeddings H of neighbouring nuclei
        #   - Embeddings h0 of neighbouring electrons
        #   - Projection of own embedding h0
        H_en = get_with_fill(H, idx_nb.en, 0.0)
        h0_ee = get_with_fill(h0, idx_nb.ee, 0.0)
        h_out = jax.vmap(self._get_electron_output_embedding)(
            electrons, h0, r_nb_ee, self.spins, spin_nb_ee, Gamma_en_out, H_en, self.lin_Gamma_ee, h0_ee
        )
        return h_out
