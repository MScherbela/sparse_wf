# from typing import Callable, Optional
# from sparse_wf.api import Electrons, Nuclei, Int
# from sparse_wf.model.graph_utils import get_neighbour_indices, get_with_fill
# from sparse_wf.model.graph_utils import get_neighbour_coordinates
# from sparse_wf.model.wave_function import MoonLikeWaveFunction, StaticInput
# from sparse_wf.model.utils import (
#     MLP,
#     PairwiseFilter,
#     DynamicFilterParams,
#     contract,
#     get_diff_features_vmapped,
#     scale_initializer,
# )
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# from jaxtyping import Float, Array, Integer
# from sparse_wf.tree_utils import tree_idx


# class MoonElecEmb(nn.Module):
#     R: Nuclei
#     cutoff: float
#     filter_dims: tuple[int, int]
#     feature_dim: int
#     activation: Callable = nn.silu

#     @nn.compact
#     def __call__(
#         self,
#         r: Float[Array, "dim=3"],
#         r_nb: Float[Array, "*neighbours dim=3"],
#         s: Optional[Int] = None,
#         s_nb: Optional[Integer[Array, "*neighbors"]] = None,
#     ):
#         features_ee = get_diff_features_vmapped(r, r_nb, s, s_nb)
#         beta = PairwiseFilter(self.cutoff, self.filter_dims[1], name="beta_ee")
#         dynamic_params_ee = DynamicFilterParams(
#             scales=self.param("ee_scales", scale_initializer, (self.n_envelopes,)),
#             kernel=self.param(
#                 "ee_kernel", jax.nn.initializers.lecun_normal(), (features_ee.shape[-1], self.filter_dims[0])
#             ),
#             bias=self.param("ee_bias", jax.nn.initializers.zeros, (self.filter_dims[0])),
#         )
#         beta_ee = beta(features_ee, dynamic_params_ee)
#         gamma_ee = nn.Dense(self.feature_dim, use_bias=False)(beta_ee)

#         feat_ee = self.activation(nn.Dense(self.feature_dim)(features_ee))
#         return jnp.einsum("...id,...id->...d", feat_ee, gamma_ee)


# class MoonElecToNucGamma(nn.Module):
#     R: Nuclei
#     cutoff: float
#     filter_dims: tuple[int, int]
#     feature_dim: int

#     @nn.compact
#     def __call__(self, r_nb_ne: Float[Array, "n_nuc neighbours dim=3"]):
#         n_nuc = len(self.R)

#         features_ne = jax.vmap(get_diff_features_vmapped)(self.R, r_nb_ne)

#         filter_ne = PairwiseFilter(self.cutoff, self.filter_dims[1], name="beta_ne")
#         filter_ne = nn.vmap(filter_ne, in_axes=(None, 0))  # vmap over neighbors (electrons)
#         filter_ne = nn.vmap(filter_ne, in_axes=(0, 0))  # vmap over center (nuclei)
#         dynamic_params_ne = DynamicFilterParams(
#             scales=self.param(
#                 "ne_scales",
#                 scale_initializer,
#                 (n_nuc, self.n_envelopes),
#             ),
#             kernel=self.param(
#                 "ne_kernel", jax.nn.initializers.lecun_normal(), (n_nuc, features_ne.shape[-1], self.filter_dims[0])
#             ),
#             bias=self.param("ne_bias", jax.nn.initializers.zeros, (n_nuc, self.filter_dims[0])),
#         )
#         beta_ne = filter_ne(features_ne, dynamic_params_ne)
#         gamma_ne = nn.Dense(self.feature_dim, use_bias=False)(beta_ne)

#         z_n = self.param("z_n", jax.nn.initializers.normal(1.0), (n_nuc, self.feature_dim))
#         edge_ne = nn.Dense(self.feature_dim)(features_ne) + z_n[:, None]
#         return gamma_ne, edge_ne


# class MoonNucToElecGamma(nn.Module):
#     R: Nuclei
#     cutoff: float
#     filter_dims: tuple[int, int]
#     feature_dim: int

#     @nn.compact
#     def __call__(
#         self,
#         r: Electrons,
#         R_nb_en: Float[Array, "n_elec neighbours dim=3"],
#         idx_en: Integer[Array, "n_elec neighbours"],
#     ):
#         n_nuc = len(self.R)

#         features_en = jax.vmap(get_diff_features_vmapped)(r, R_nb_en)

#         filter_en = PairwiseFilter(self.cutoff, self.filter_dims[1], name="beta_en")
#         filter_en = nn.vmap(filter_en, in_axes=(0, 0))  # vmap over neighbors (nuclei)
#         filter_en = nn.vmap(filter_en, in_axes=(0, None))  # vmap over center (electrons)
#         dynamic_params_en = DynamicFilterParams(
#             scales=self.param(
#                 "en_scales",
#                 scale_initializer,
#                 (n_nuc, self.n_envelopes),
#             ),
#             kernel=self.param(
#                 "en_kernel", jax.nn.initializers.lecun_normal(), (n_nuc, features_en.shape[-1], self.filter_dims[0])
#             ),
#             bias=self.param("en_bias", jax.nn.initializers.zeros, (n_nuc, self.filter_dims[0])),
#         )
#         dynamic_params_en = jax.vmap(tree_idx, in_axes=(None, 0))(dynamic_params_en, idx_en)
#         beta_en = filter_en(features_en, dynamic_params_en)

#         gamma_en_init = nn.Dense(self.feature_dim, use_bias=False)(beta_en)
#         gamma_en_out = nn.Dense(self.feature_dim, use_bias=False)(beta_en)

#         edge_en = nn.Dense(self.feature_dim)(features_en)
#         nuc_emb = self.param("z_n", jax.nn.initializers.normal(1.0), (n_nuc, self.feature_dim))

#         return gamma_en_init, gamma_en_out, edge_en


# class Moon(MoonLikeWaveFunction):
#     def setup(self):
#         super().setup()

#         n_nuc = len(self.R)
#         self.dynamic_params_en = DynamicFilterParams(
#             scales=self.param(
#                 "en_scales",
#                 scale_initializer,
#                 (n_nuc, self.n_envelopes),
#             ),
#             kernel=self.param("en_kernel", jax.nn.initializers.lecun_normal(), (n_nuc, 5, self.pair_mlp_widths[0])),
#             bias=self.param("en_bias", jax.nn.initializers.zeros, (n_nuc, self.pair_mlp_widths[0])),
#         )
#         self.filter_en = PairwiseFilter(
#             self.cutoff, self.pair_mlp_widths[1], name="beta_en"
#         )  # TODO: make this cutoff larger?
#         self.gamma_ne = MoonElecToNucGamma(
#             R=self.R,
#             cutoff=self.cutoff,
#             filter_dims=self.pair_mlp_widths,
#             feature_dim=self.feature_dim,
#         )
#         self.gamma_en = MoonNucToElecGamma(
#             R=self.R,
#             cutoff=self.cutoff,
#             filter_dims=self.pair_mlp_widths,
#             feature_dim=self.feature_dim,
#         )
#         self.elec_elec_emb = MoonElecEmb(
#             R=self.R,
#             cutoff=self.cutoff,
#             filter_dims=self.pair_mlp_widths,
#             feature_dim=self.feature_dim,
#         )
#         self.elec_elec_emb = nn.vmap(self.elec_elec_emb)
#         self.nuc_mlp = MLP([self.feature_dim] * self.nuc_mlp_depth, True)

#     def _embedding(self, electrons: Electrons, static: StaticInput) -> Electrons:
#         idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)
#         spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en = get_neighbour_coordinates(electrons, self.R, idx_nb, self.spins)

#         # initial electron embedding
#         h0 = self.elec_elec_emb(r_nb_ee, r_nb_ee, spin_nb_ee, spin_nb_ee)

#         # construct nuclei embeddings
#         Gamma_ne, edge_ne_emb = self.gamma_ne(r_nb_ne)
#         h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0)
#         edge_ne_emb = h0_nb_ne + edge_ne_emb
#         H0 = contract(edge_ne_emb, Gamma_ne)

#         # construct electron embedding
#         gamma_en_init, gamma_en_out = self.gamma_en(electrons, R_nb_en, idx_nb.en)

#         # update nuclei embedding
#         H = self.nuc_mlp(H0)
#         return super()._embedding(electrons, static)
