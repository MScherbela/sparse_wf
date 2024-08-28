import functools
from typing import Callable, NamedTuple, Optional, TypedDict, cast, Generic, TypeVar


import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util as jtu
from flax.struct import PyTreeNode
import flax.struct
from jaxtyping import Array, Float, Integer

from folx.api import FwdLaplArray
from sparse_wf.api import (
    Charges,
    ElectronIdx,
    Electrons,
    Int,
    Nuclei,
    NucleiIdx,
    Parameters,
    Spins,
    Embedding,
    StaticInput,
)
from sparse_wf.jax_utils import fwd_lap, jit, nn_vmap
from sparse_wf.static_args import round_with_padding
from sparse_wf.model.graph_utils import (
    NO_NEIGHBOUR,
    Dependency,
    DependencyMap,
    ElectronElectronEdges,
    ElectronNucleiEdges,
    NucleiElectronEdges,
    affected_particles,
    get_dependency_map,
    get_full_distance_matrices,
    get_neighbour_features,
    get_with_fill,
    merge_dependencies,
    pad_jacobian,
    pad_pairwise_jacobian,
    zeropad_jacobian,
)
from sparse_wf.model.utils import (
    DynamicFilterParams,
    FixedScalingFactor,
    GatedLinearUnit,
    PairwiseFilter,
    ScalingParam,
    contract,
    get_diff_features,
    lecun_normal,
    normalize,
    scale_initializer,
)
from sparse_wf.tree_utils import tree_idx

T = TypeVar("T")


class NrOfNeighbours(NamedTuple, Generic[T]):
    ee: T
    ee_out: T
    en: T
    ne: T
    en_1el: T


class NrOfDependencies(NamedTuple, Generic[T]):
    h_el_initial: T
    H_nuc: T
    h_el_out: T


class NrOfChanges(NamedTuple, Generic[T]):
    # Maximum number of changes per moved electrons
    h0: T
    nuclei: T
    out: T


@flax.struct.dataclass
class StaticInputMoon(StaticInput, Generic[T]):
    n_deps: NrOfDependencies[T]
    n_neighbours: NrOfNeighbours[T]
    n_changes: NrOfChanges[T]

    def round_with_padding(self, padding_factor, n_el, n_up, n_nuc):
        n_neighbours = NrOfNeighbours(
            ee=round_with_padding(self.n_neighbours.ee, padding_factor, n_el - 1),
            ee_out=round_with_padding(self.n_neighbours.ee_out, padding_factor, n_el - 1),
            en=round_with_padding(self.n_neighbours.en, padding_factor, n_nuc),
            ne=round_with_padding(self.n_neighbours.ne, padding_factor, n_el),
            en_1el=round_with_padding(self.n_neighbours.en_1el, padding_factor, n_nuc),
        )
        n_deps = NrOfDependencies(
            h_el_initial=n_neighbours.ee + 1,
            H_nuc=round_with_padding(self.n_deps.H_nuc, padding_factor, n_el),
            h_el_out=round_with_padding(self.n_deps.h_el_out, padding_factor, n_el),
        )
        n_changes = NrOfChanges(
            h0=round_with_padding(self.n_changes.h0, padding_factor, n_el),
            nuclei=round_with_padding(self.n_changes.nuclei, padding_factor, n_nuc),
            out=round_with_padding(self.n_changes.out, padding_factor, n_el),
        )
        return StaticInputMoon(n_deps, n_neighbours, n_changes)


class NeighbourIndices(NamedTuple):
    ee: ElectronElectronEdges
    en: ElectronNucleiEdges
    ne: NucleiElectronEdges
    ee_out: ElectronElectronEdges
    en_1el: ElectronNucleiEdges


class DependenciesMoon(NamedTuple):
    h0: Dependency
    H_nuc: Dependency
    h_el_out: Dependency


class DependencyMaps(NamedTuple):
    hinit_to_h0: DependencyMap
    h0_to_Hnuc: DependencyMap
    Gamma_ne_to_Hnuc: DependencyMap
    Hnuc_to_hout: DependencyMap
    h0_to_hout: DependencyMap
    hinit_to_hout: DependencyMap


class NucleusDependentParams(NamedTuple):
    filter: DynamicFilterParams
    nuc_embedding: Optional[jax.Array]


class MoonElecEmb(nn.Module):
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
        h: Float[Array, " feature_dim"],
        h_nb: Float[Array, "*neighbours feature_dim"],
        s: Int,
        s_nb: Integer[Array, " *neighbors"],
    ):
        # TODO: Computing the pairwise terms outside in another module would be more efficient since don't have to carry the full jacobian.
        features_ee = jax.vmap(get_diff_features, in_axes=(None, 0))(r, r_nb)
        spin_mask = s == s_nb
        beta = PairwiseFilter(self.cutoff, self.filter_dims, name="beta_ee")
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
        gamma_ee_same = nn.Dense(self.feature_dim, use_bias=False)(beta_ee)
        gamma_ee_diff = nn.Dense(self.feature_dim, use_bias=False)(beta_ee)
        gamma_ee = jnp.where(spin_mask[:, None], gamma_ee_same, gamma_ee_diff)

        # logarithmic rescaling
        inp_ee = features_ee / features_ee[..., :1] * jnp.log1p(features_ee[..., :1])
        feat_ee_same = nn.Dense(self.feature_dim)(inp_ee)
        feat_ee_diff = nn.Dense(self.feature_dim)(inp_ee)
        feat_ee = jnp.where(spin_mask[:, None], feat_ee_same, feat_ee_diff)

        # using h^init
        feat_ee += h + h_nb
        feat_ee = self.activation(feat_ee)

        # contraction
        result = jnp.einsum("...id,...id->...d", feat_ee, gamma_ee)
        result = nn.Dense(self.feature_dim)(result) + nn.Dense(self.feature_dim, use_bias=False)(h)
        result = self.activation(result)
        # result = self.activation(nn.Dense(self.feature_dim)(result) + result)
        return result


class MoonEdgeFeatures(nn.Module):
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    return_edge_embedding: bool = True

    @nn.compact
    def __call__(
        self,
        r_center: Float[Array, "dim=3"],
        r_neighbour: Float[Array, "dim=3"],
        dynamic_params: NucleusDependentParams,
    ):
        features = get_diff_features(r_center, r_neighbour)
        beta = PairwiseFilter(self.cutoff, self.filter_dims)(features, dynamic_params.filter)
        gamma = nn.Dense(self.feature_dim, use_bias=False)(beta)
        if not self.return_edge_embedding:
            return gamma
        scaled_features = features / features[..., :1] * jnp.log1p(features[..., :1])
        edge_embedding = nn.Dense(self.feature_dim, use_bias=False)(scaled_features) + dynamic_params.nuc_embedding
        return gamma, edge_embedding


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


class MoonElecInit(nn.Module):
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    activation: Callable = nn.silu

    def setup(self):
        self.edge = MoonEdgeFeatures(self.cutoff, self.filter_dims, self.feature_dim, self.n_envelopes)
        self.glu = GatedLinearUnit(self.feature_dim)
        self.dense = nn.Dense(self.feature_dim)
        self.dense_same = nn.Dense(self.feature_dim, use_bias=False)
        self.dense_diff = nn.Dense(self.feature_dim, use_bias=False)

    @nn.compact
    def __call__(
        self,
        r: Float[Array, " dim=3"],
        R_nb: Float[Array, "n_neighbours dim=3"],
        dynamic_params: NucleusDependentParams,
    ):
        features, Gamma = nn_vmap(self.edge, in_axes=(None, 0, 0))(r, R_nb, dynamic_params)  # vmap over nuclei
        result = jnp.einsum("...Jd,...Jd->...d", features, Gamma)
        h_init = self.dense(self.glu(result))
        h_init_same = self.dense_same(self.activation(h_init))
        h_init_diff = self.dense_diff(self.activation(h_init))
        return h_init, h_init_same, h_init_diff


class MoonElecOut(nn.Module):
    cutoff: float
    filter_dims: tuple[int, int]
    n_envelopes: int

    @nn.compact
    def __call__(self, elec, msg, r, r_nb, s, s_nb, hinit_nb):
        dim = elec.shape[-1]
        elec = nn.silu(nn.Dense(dim)(elec))  # TODO: pulling this layer outside would improve laplacian computations

        # EE - msg passing
        # TODO: One should probably pull this out of here to make the computation efficient and only require 6-dimensional jacobians
        features_ee = jax.vmap(get_diff_features, in_axes=(None, 0))(r, r_nb)
        spin_mask = s == s_nb
        beta = PairwiseFilter(self.cutoff, self.filter_dims, name="beta_ee")
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
        gamma_ee_same = nn.Dense(dim, use_bias=False)(beta_ee)
        gamma_ee_diff = nn.Dense(dim, use_bias=False)(beta_ee)
        gamma_ee = jnp.where(spin_mask[:, None], gamma_ee_same, gamma_ee_diff)
        hinit_msg = jnp.einsum("...id,...id->...d", nn.silu(elec + hinit_nb), gamma_ee)

        out = nn.silu(nn.Dense(dim)(elec) + hinit_msg)  # TODO: The first layer here can also be pulled outside
        out = nn.silu(nn.Dense(dim)(out) + msg)
        out = nn.silu(nn.Dense(dim)(out))
        return FixedScalingFactor()(out + elec)


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
            dep_map_hinit_to_h0: [n_el x n_neighbouring_el x 1]
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
    deps_neighbours = get_deps_nb(jnp.arange(n_el)[:, None], idx_nb.ee)
    dep_map_hinit_to_h0 = get_dep_map_for_all_centers(deps_neighbours, deps_h0)

    # Step 2: Nuclear embeddings depend on all dependencies of their neighbouring electrons
    deps_neighbours = get_deps_nb(deps_h0, idx_nb.ne)
    deps_H = merge_dependencies(deps_neighbours, idx_nb.ne, None, n_deps_max.H_nuc)
    dep_map_h0_to_H = get_dep_map_for_all_centers(deps_neighbours, deps_H)
    dep_map_Gamma_ne_to_H = get_dep_map_for_all_centers(idx_nb.ne[..., None], deps_H)

    # Step 3: Output electron embeddings depend on themselves, their neighbouring electrons and all dependencies of their neighbouring nuclei
    # Step 4: Initial electron embeddings to output electron embeddings
    deps_neighbouring_nuc = get_deps_nb(deps_H, idx_nb.en)
    deps_neighbouring_hinit = get_deps_nb(self_dependency, idx_nb.ee_out)
    deps_hout = merge_dependencies(deps_neighbouring_nuc, deps_h0, jnp.arange(n_el)[:, None], n_deps_max.h_el_out)
    deps_hout = merge_dependencies(deps_neighbouring_hinit, deps_hout, jnp.arange(n_el)[:, None], n_deps_max.h_el_out)
    dep_map_H_to_hout = get_dep_map_for_all_centers(deps_neighbouring_nuc, deps_hout)
    dep_map_h0_to_hout = jax.vmap(get_dependency_map)(deps_h0, deps_hout)
    dep_map_hinit_to_hout = get_dep_map_for_all_centers(deps_neighbouring_hinit, deps_hout)

    # Assert that dependencies are consistent with static dims
    assert deps_h0.shape[-1] == n_deps_max.h_el_initial
    assert deps_H.shape[-1] == n_deps_max.H_nuc
    assert deps_hout.shape[-1] == n_deps_max.h_el_out
    assert dep_map_Gamma_ne_to_H.shape[-1] == 1
    assert dep_map_hinit_to_h0.shape[-1] == 1
    assert dep_map_h0_to_H.shape[-1] == n_deps_max.h_el_initial
    assert dep_map_H_to_hout.shape[-1] == n_deps_max.H_nuc
    assert dep_map_h0_to_hout.shape[-1] == n_deps_max.h_el_initial
    assert dep_map_hinit_to_hout.shape[-1] == 1

    return DependenciesMoon(deps_h0, deps_H, deps_hout), DependencyMaps(
        dep_map_hinit_to_h0,
        dep_map_h0_to_H,
        dep_map_Gamma_ne_to_H,
        dep_map_H_to_hout,
        dep_map_h0_to_hout,
        dep_map_hinit_to_hout,
    )


class EmbeddingChanges(NamedTuple):
    h0: ElectronIdx
    nuclei: NucleiIdx
    msg: ElectronIdx
    out: ElectronIdx


@jit(static_argnames="static")
def get_changed_embeddings(
    electrons: Electrons,
    previous_electrons: Electrons,
    changed_electrons: ElectronIdx,
    nuclei: Nuclei,
    static: StaticInputMoon[int],
    cutoff: float,
):
    idx_changed_h0 = affected_particles(
        previous_electrons[changed_electrons],
        previous_electrons,
        electrons[changed_electrons],
        electrons,
        static.n_changes.h0,
        cutoff,
    )
    idx_changed_nuclei = affected_particles(
        previous_electrons[idx_changed_h0], nuclei, electrons[idx_changed_h0], nuclei, static.n_changes.nuclei, cutoff
    )
    idx_changed_msg = affected_particles(
        nuclei[idx_changed_nuclei],
        previous_electrons,
        nuclei[idx_changed_nuclei],
        electrons,
        static.n_changes.out,  # TODO: one could bound this tighter
        cutoff,
        changed_electrons,
    )
    idx_changed_out = affected_particles(
        previous_electrons[changed_electrons],
        previous_electrons,
        electrons[changed_electrons],
        electrons,
        static.n_changes.out,
        3 * cutoff,
        idx_changed_h0,
    )
    return EmbeddingChanges(h0=idx_changed_h0, nuclei=idx_changed_nuclei, msg=idx_changed_msg, out=idx_changed_out)


class MoonScales(TypedDict):
    h0: Optional[ScalingParam]
    H1_up: Optional[ScalingParam]
    H1_dn: Optional[ScalingParam]
    msg: Optional[ScalingParam]


class MoonEmbeddingParams(PyTreeNode):
    elec_init: Parameters
    elec_elec_emb: Parameters
    Gamma_ne: Parameters
    Gamma_en: Parameters
    nuc_mlp: Parameters
    elec_out: Parameters
    dynamic_params_en: NucleusDependentParams
    dynamic_params_ne: NucleusDependentParams
    dynamic_params_elec_init: NucleusDependentParams
    scales: MoonScales


class MoonState(PyTreeNode):
    electrons: Electrons
    h_init: Array
    h_init_same: Array
    h_init_diff: Array
    h0: Array
    HL_up: Array
    HL_dn: Array
    msg: Array
    h_out: Array


def get_neighbour_coordinates(electrons: Electrons, R: Nuclei, idx_nb: NeighbourIndices, spins: Spins):
    # [n_el  x n_neighbouring_electrons] - spin of each adjacent electron for each electron (within 1*cutoff and 3*cutoff)
    spin_nb_ee = get_with_fill(spins, idx_nb.ee, 0.0)
    spin_nb_ee_out = get_with_fill(spins, idx_nb.ee_out, 0.0)

    # [n_el  x n_neighbouring_electrons x 3] - position of each adjacent electron for each electron (within 1*cutoff and 3*cutoff)
    r_nb_ee = get_with_fill(electrons, idx_nb.ee, NO_NEIGHBOUR)
    r_nb_ee_out = get_with_fill(electrons, idx_nb.ee_out, NO_NEIGHBOUR)

    # [n_nuc  x n_neighbouring_electrons] - spin of each adjacent electron for each nucleus
    spin_nb_ne = get_with_fill(spins, idx_nb.ne, 0.0)

    # [n_nuc x n_neighbouring_electrons x 3] - position of each adjacent electron for each nuclei
    r_nb_ne = get_with_fill(electrons, idx_nb.ne, NO_NEIGHBOUR)

    # [n_el  x n_neighbouring_nuclei    x 3] - position of each adjacent nuclei for each electron
    R_nb_en = get_with_fill(R, idx_nb.en, NO_NEIGHBOUR)

    # [n_el  x n_neighbouring_nuclei    x 3] - position of each adjacent nuclei for each electron (but with larger cutoff)
    R_nb_en_1el = get_with_fill(R, idx_nb.en_1el, NO_NEIGHBOUR)

    return spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en, R_nb_en_1el, r_nb_ee_out, spin_nb_ee_out


class MoonEmbedding(PyTreeNode, Embedding[MoonEmbeddingParams, StaticInputMoon, MoonState]):
    # Molecule
    R: Nuclei
    Z: Charges
    n_electrons: int
    n_up: int

    # Hyperparams
    cutoff: float
    cutoff_1el: float
    feature_dim: int
    pair_mlp_widths: tuple[int, int]
    pair_n_envelopes: int
    nuc_mlp_depth: int

    # Submodules
    elec_init: MoonElecInit
    elec_elec_emb: MoonElecEmb
    Gamma_ne: MoonEdgeFeatures
    nuc_mlp: MoonNucMLP
    Gamma_en: MoonEdgeFeatures
    elec_out: MoonElecOut

    # Low rank updates
    low_rank_buffer: int = 2

    @classmethod
    def create(
        cls,
        R: Nuclei,
        Z: Charges,
        n_electrons: int,
        n_up: int,
        cutoff: float,
        cutoff_1el: float,
        feature_dim: int,
        nuc_mlp_depth: int,
        pair_mlp_widths: tuple[int, int],
        pair_n_envelopes: int,
        low_rank_buffer: int,
        **_,
    ):
        return cls(
            R=R,
            Z=Z,
            n_electrons=n_electrons,
            n_up=n_up,
            cutoff=cutoff,
            cutoff_1el=cutoff_1el,
            feature_dim=feature_dim,
            pair_mlp_widths=pair_mlp_widths,
            pair_n_envelopes=pair_n_envelopes,
            nuc_mlp_depth=nuc_mlp_depth,
            elec_elec_emb=MoonElecEmb(cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes),
            Gamma_ne=MoonEdgeFeatures(cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes),
            Gamma_en=MoonEdgeFeatures(cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes, False),
            elec_init=MoonElecInit(cutoff_1el, pair_mlp_widths, feature_dim, pair_n_envelopes),
            nuc_mlp=MoonNucMLP(nuc_mlp_depth),
            elec_out=MoonElecOut(cutoff * 3, pair_mlp_widths, pair_n_envelopes),
            low_rank_buffer=low_rank_buffer,
        )

    @property
    def n_nuclei(self):
        return len(self.R)

    @property
    def spins(self):
        return jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_electrons - self.n_up)]).astype(jnp.float32)

    def get_static_input(
        self, r: Electrons, r_new: Optional[Electrons] = None, idx_changed: Optional[ElectronIdx] = None
    ) -> StaticInputMoon:
        n_el = r.shape[-2]
        dist_ee, dist_ne = get_full_distance_matrices(r, self.R)
        dist_en = jnp.swapaxes(dist_ne, -1, -2)
        dist_ee = dist_ee.at[..., np.arange(n_el), np.arange(n_el)].set(np.inf)

        # Neighbours
        n_neighbours = NrOfNeighbours(
            ee=jnp.max(jnp.sum(dist_ee < self.cutoff, axis=-1, dtype=jnp.int32)),
            ee_out=jnp.max(jnp.sum(dist_ee < 3 * self.cutoff, axis=-1, dtype=jnp.int32)),
            en=jnp.max(jnp.sum(dist_en < self.cutoff, axis=-1, dtype=jnp.int32)),
            ne=jnp.max(jnp.sum(dist_ne < self.cutoff, axis=-1, dtype=jnp.int32)),
            en_1el=jnp.max(jnp.sum(dist_en < self.cutoff_1el, axis=-1, dtype=jnp.int32)),
        )

        # Dependencies
        n_deps = NrOfDependencies(
            h_el_initial=n_neighbours.ee + 1,
            H_nuc=jnp.max(jnp.sum(dist_ne < self.cutoff * 2, axis=-1, dtype=jnp.int32)),
            h_el_out=jnp.max(jnp.sum(dist_ee < self.cutoff * 3, axis=-1, dtype=jnp.int32)) + 1,
        )
        if (r_new is None) or (idx_changed is None):
            return StaticInputMoon(n_deps, n_neighbours, NrOfChanges(1, 1, 1))

        # Changes for low-rank updates
        dist_ee_new, dist_ne_new = get_full_distance_matrices(r_new, self.R)
        dist_ee = jnp.minimum(dist_ee, dist_ee_new)
        dist_ne = jnp.minimum(dist_ne, dist_ne_new)
        dist_en = jnp.swapaxes(dist_ne, -1, -2)

        is_affected_r = jnp.zeros(n_el, dtype=jnp.bool_).at[idx_changed].set(True)
        is_affected_h0 = is_affected_r | jnp.any((dist_ee < self.cutoff) & is_affected_r[None, :], axis=-1)
        is_affected_H = jnp.any((dist_ne < self.cutoff) & is_affected_h0[None, :], axis=-1)
        is_affected_out = (
            is_affected_h0
            | jnp.any((dist_en < self.cutoff) & is_affected_H[None, :], axis=-1)
            | jnp.any((dist_ee < 3 * self.cutoff) & is_affected_r[None, :], axis=-1)
        )
        n_changes = NrOfChanges(
            jnp.sum(is_affected_h0, dtype=jnp.int32),
            jnp.sum(is_affected_H, dtype=jnp.int32),
            jnp.sum(is_affected_out, dtype=jnp.int32),
        )
        return StaticInputMoon(n_deps, n_neighbours, n_changes)

    def get_neighbour_indices(
        self,
        r: Electrons,
        max_n_neighbours: NrOfNeighbours[int],
    ) -> NeighbourIndices:
        n_el = r.shape[-2]
        dist_ee, dist_ne = get_full_distance_matrices(r, self.R)
        dist_en = dist_ne.T
        dist_ee = dist_ee.at[..., np.arange(n_el), np.arange(n_el)].set(np.inf)

        @functools.partial(jax.vmap, in_axes=(0, None, None))  # vmap over centers
        def _get_neighbours(dist, cutoff, max_size: int):
            # TODO: use this commented code but somehow let FOLX fwd_lap over it without crashing due to gradient of top_k operation
            # neg_dists, indices = jax.lax.top_k(-dist, max_size)
            # indices = jnp.where(neg_dists > -cutoff, indices, NO_NEIGHBOUR)
            # return indices
            return jnp.where(dist < cutoff, size=max_size, fill_value=NO_NEIGHBOUR)[0]

        # Neighbours
        return NeighbourIndices(
            ee=_get_neighbours(dist_ee, self.cutoff, max_n_neighbours.ee),
            en=_get_neighbours(dist_en, self.cutoff, max_n_neighbours.en),
            ne=_get_neighbours(dist_ne, self.cutoff, max_n_neighbours.ne),
            ee_out=_get_neighbours(dist_ee, 3 * self.cutoff, max_n_neighbours.ee_out),
            en_1el=_get_neighbours(dist_en, self.cutoff_1el, max_n_neighbours.en_1el),
        )

    def init(self, rng: Array, electrons: Array, static: StaticInputMoon):
        dtype = electrons.dtype
        rngs = jax.random.split(rng, 10)
        r_dummy = jnp.zeros([3], dtype)
        r_nb_dummy = jnp.zeros([1, 3], dtype)
        spin_dummy = jnp.zeros([], dtype)
        spin_nb_dummy = jnp.zeros([1], dtype)
        features_dummy = jnp.zeros([self.feature_dim], dtype)
        dummy_dyn_param = self._init_nuc_dependant_params(rngs[0], n_nuclei=1)

        params = MoonEmbeddingParams(
            dynamic_params_en=self._init_nuc_dependant_params(rngs[1], nuc_embedding=False),
            dynamic_params_ne=self._init_nuc_dependant_params(rngs[2]),
            dynamic_params_elec_init=self._init_nuc_dependant_params(rngs[3]),
            elec_init=self.elec_init.init(rngs[4], r_dummy, r_nb_dummy, dummy_dyn_param),
            elec_elec_emb=self.elec_elec_emb.init(
                rngs[5], r_dummy, r_nb_dummy, features_dummy, features_dummy[None], spin_dummy, spin_nb_dummy
            ),
            Gamma_ne=self.Gamma_ne.init(rngs[6], r_dummy, r_dummy, dummy_dyn_param),
            Gamma_en=self.Gamma_en.init(rngs[7], r_dummy, r_dummy, dummy_dyn_param),
            nuc_mlp=self.nuc_mlp.init(rngs[8], features_dummy, features_dummy),
            elec_out=self.elec_out.init(
                rngs[9],
                features_dummy,
                features_dummy,
                r_dummy,
                r_nb_dummy,
                spin_dummy,
                spin_nb_dummy,
                features_dummy[None],
            ),
            scales=MoonScales(h0=None, H1_up=None, H1_dn=None, msg=None),
        )
        _, scales = self.apply(params, electrons, static, return_scales=True)
        params = params.replace(scales=scales)
        return params

    def _init_nuc_dependant_params(self, rng, n_nuclei=None, nuc_embedding: bool = True):
        n_nuclei = n_nuclei or self.n_nuclei
        rngs = jax.random.split(rng, 4)
        if nuc_embedding:
            h_nuc = jax.random.normal(rngs[3], (n_nuclei, self.feature_dim), jnp.float32)
        else:
            h_nuc = None

        return NucleusDependentParams(
            filter=DynamicFilterParams(
                scales=scale_initializer(rngs[0], self.cutoff, (n_nuclei, self.pair_n_envelopes)),
                kernel=lecun_normal(rngs[1], (n_nuclei, 4, self.pair_mlp_widths[0])),
                bias=jax.random.normal(rngs[2], (n_nuclei, self.pair_mlp_widths[0]), jnp.float32) * 2.0,
            ),
            nuc_embedding=h_nuc,
        )

    def _get_Gamma_ne(self, params, R, r_nb_ne, dynamic_params, with_fwd_lap=False):
        get_gamma = functools.partial(self.Gamma_ne.apply, params)
        if with_fwd_lap:
            get_gamma = fwd_lap(get_gamma, argnums=1)
        get_gamma = jax.vmap(get_gamma, in_axes=(None, 0, None), out_axes=-2)  # vmap over neighbours (electrons)
        get_gamma = jax.vmap(get_gamma, in_axes=0, out_axes=-3)  # vmap over centers (nuclei)
        return get_gamma(R, r_nb_ne, dynamic_params)

    def _get_Gamma_en(self, params, r, R_nb_en, dynamic_params, with_fwd_lap=False):
        get_gamma = functools.partial(self.Gamma_en.apply, params)
        if with_fwd_lap:
            get_gamma = fwd_lap(get_gamma, argnums=0)
        get_gamma = jax.vmap(get_gamma, in_axes=(None, 0, 0), out_axes=-2)  # vmap over neighbours (nuclei)
        get_gamma = jax.vmap(get_gamma, in_axes=0, out_axes=-3)  # vmap over centers (electrons)
        return get_gamma(r, R_nb_en, dynamic_params)

    def apply(
        self,
        params: MoonEmbeddingParams,
        electrons: Electrons,
        static: StaticInputMoon,
        return_scales: bool = False,
        return_state: bool = False,
    ):
        idx_nb = self.get_neighbour_indices(electrons, static.n_neighbours)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en, R_nb_en_1el, r_nb_ee_out, spin_nb_ee_out = (
            get_neighbour_coordinates(electrons, self.R, idx_nb, self.spins)
        )

        @jax.vmap  # vmap over center electrons
        def get_hinit(r, R_nb, dyn_params):
            result = self.elec_init.apply(params.elec_init, r, R_nb, dyn_params)
            return tuple(cast(jax.Array, x) for x in result)

        dyn_params = tree_idx(params.dynamic_params_elec_init, idx_nb.en_1el)
        h_init, h_init_same, h_init_diff = get_hinit(electrons, R_nb_en_1el, dyn_params)
        h_init_nb = jnp.where(
            (self.spins[:, None] == spin_nb_ee)[..., None],
            get_with_fill(h_init_same, idx_nb.ee, 0),
            get_with_fill(h_init_diff, idx_nb.ee, 0),
        )

        @jax.vmap  # vmap over center electrons
        def get_h0(r, r_nb, h, h_nb, s, s_nb):
            return cast(jax.Array, self.elec_elec_emb.apply(params.elec_elec_emb, r, r_nb, h, h_nb, s, s_nb))

        # initial electron embedding
        h0 = get_h0(electrons, r_nb_ee, h_init, h_init_nb, self.spins, spin_nb_ee)
        h0, params.scales["h0"] = normalize(h0, params.scales["h0"], True)

        # construct nuclei embeddings
        Gamma_ne, edge_ne_emb = self._get_Gamma_ne(params.Gamma_ne, self.R, r_nb_ne, params.dynamic_params_ne)
        h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0)
        edge_ne_emb = nn.silu(h0_nb_ne + edge_ne_emb)
        edge_ne_up = jnp.where(spin_nb_ne[..., None] > 0, edge_ne_emb, 0)
        edge_ne_dn = jnp.where(spin_nb_ne[..., None] < 0, edge_ne_emb, 0)
        H1_up = contract(edge_ne_up, Gamma_ne)  # type: ignore
        H1_dn = contract(edge_ne_dn, Gamma_ne)  # type: ignore
        H1_up, params.scales["H1_up"] = normalize(H1_up, params.scales["H1_up"], True)
        H1_dn, params.scales["H1_dn"] = normalize(H1_dn, params.scales["H1_dn"], True)

        # construct electron embedding
        dyn_params = tree_idx(params.dynamic_params_en, idx_nb.en)
        gamma_en_out = self._get_Gamma_en(params.Gamma_en, electrons, R_nb_en, dyn_params)  # type: ignore

        # update electron embedding
        HL_up, HL_dn = self.nuc_mlp.apply(params.nuc_mlp, H1_up, H1_dn)
        HL_up, HL_dn = cast(tuple[jax.Array, jax.Array], (HL_up, HL_dn))
        HL_up_nb_en = get_with_fill(HL_up, idx_nb.en, 0)
        HL_dn_nb_en = get_with_fill(HL_dn, idx_nb.en, 0)
        HL_nb_en = jnp.where(self.spins[..., None, None] > 0, HL_up_nb_en, HL_dn_nb_en)
        msg = contract(HL_nb_en, gamma_en_out)
        msg, params.scales["msg"] = normalize(msg, params.scales["msg"], True)

        # Msg from hinit
        h_init_nb = jnp.where(
            (self.spins[:, None] == spin_nb_ee_out)[..., None],
            get_with_fill(h_init_same, idx_nb.ee_out, 0),
            get_with_fill(h_init_diff, idx_nb.ee_out, 0),
        )

        # readout
        @jax.vmap
        def compute_hout(h0, msg, electrons, r_nb_ee_out, spins, spin_nb_ee_out, h_init_nb):
            return self.elec_out.apply(
                params.elec_out,
                h0,
                msg,
                electrons,
                r_nb_ee_out,
                spins,
                spin_nb_ee_out,
                h_init_nb,
            )

        h_out = cast(
            jax.Array,
            compute_hout(
                h0,
                msg,
                electrons,
                r_nb_ee_out,
                self.spins,
                spin_nb_ee_out,
                h_init_nb,
            ),
        )

        if return_scales:
            return h_out, params.scales  # type: ignore

        if return_state:
            return h_out, MoonState(
                electrons=electrons,
                h_init=h_init,
                h_init_same=h_init_same,
                h_init_diff=h_init_diff,
                h0=h0,
                HL_up=HL_up,
                HL_dn=HL_dn,
                msg=msg,
                h_out=h_out,
            )
        return h_out

    def low_rank_update(
        self,
        params: MoonEmbeddingParams,
        electrons: Electrons,
        changed_electrons: ElectronIdx,
        static: StaticInputMoon,
        state: MoonState,
    ):
        idx_nb = self.get_neighbour_indices(electrons, static.n_neighbours)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en, R_nb_en_1el, r_nb_ee_out, spin_nb_ee_out = (
            get_neighbour_coordinates(electrons, self.R, idx_nb, self.spins)
        )
        idx_changed = get_changed_embeddings(electrons, state.electrons, changed_electrons, self.R, static, self.cutoff)

        # Compute hinit
        # Here every electron is updated invidivudally, so we only need to compute the hinit for the changed electrons.
        @jax.vmap  # vmap over center electrons
        def get_hinit(r, R_nb, dyn_params):
            result = self.elec_init.apply(params.elec_init, r, R_nb, dyn_params)
            return tuple(cast(jax.Array, x) for x in result)

        dyn_params = tree_idx(params.dynamic_params_elec_init, idx_nb.en_1el[changed_electrons])
        h_init, h_init_same, h_init_diff = get_hinit(
            electrons[changed_electrons], R_nb_en_1el[changed_electrons], dyn_params
        )
        h_init = state.h_init.at[changed_electrons].set(h_init)
        h_init_same = state.h_init_same.at[changed_electrons].set(h_init_same)
        h_init_diff = state.h_init_diff.at[changed_electrons].set(h_init_diff)

        # Compute h0
        # Here it already becomes more tricky since the set which depends on the changed electrons increases.
        # all h0 in range of the changed electrons need to be recomputed. For this we need all electrons
        # neighbhouring the neighbours of the changed electrons.
        # However, one can make this more efficient by subtracting the old state before the summation and adding the new state.
        @jax.vmap  # vmap over center electrons
        def get_h0(r, r_nb, h, h_nb, s, s_nb):
            return cast(jax.Array, self.elec_elec_emb.apply(params.elec_elec_emb, r, r_nb, h, h_nb, s, s_nb))

        h_init_nb = jnp.where(
            (self.spins[idx_changed.h0][:, None] == spin_nb_ee[idx_changed.h0])[..., None],
            get_with_fill(h_init_same, idx_nb.ee[idx_changed.h0], 0),
            get_with_fill(h_init_diff, idx_nb.ee[idx_changed.h0], 0),
        )
        h0_new = get_h0(
            electrons[idx_changed.h0],
            r_nb_ee[idx_changed.h0],
            h_init[idx_changed.h0],
            h_init_nb,
            self.spins[idx_changed.h0],
            spin_nb_ee[idx_changed.h0],
        )
        h0_new = normalize(h0_new, params.scales["h0"])
        h0 = state.h0.at[idx_changed.h0].set(h0_new)

        # construct nuclei embeddings
        Gamma_ne, edge_ne_emb = self._get_Gamma_ne(
            params.Gamma_ne,
            *jtu.tree_map(lambda x: jnp.asarray(x)[idx_changed.nuclei], (self.R, r_nb_ne, params.dynamic_params_ne)),
        )
        h0_nb_ne = get_with_fill(h0, idx_nb.ne[idx_changed.nuclei], 0)
        edge_ne_emb = nn.silu(h0_nb_ne + edge_ne_emb)
        edge_ne_up = jnp.where(spin_nb_ne[idx_changed.nuclei][..., None] > 0, edge_ne_emb, 0)
        edge_ne_dn = jnp.where(spin_nb_ne[idx_changed.nuclei][..., None] < 0, edge_ne_emb, 0)
        H1_up = contract(edge_ne_up, Gamma_ne)  # type: ignore
        H1_dn = contract(edge_ne_dn, Gamma_ne)  # type: ignore
        H1_up = normalize(H1_up, params.scales["H1_up"])
        H1_dn = normalize(H1_dn, params.scales["H1_dn"])

        # Update nuclei embeddings
        HL_up, HL_dn = self.nuc_mlp.apply(params.nuc_mlp, H1_up, H1_dn)
        HL_up, HL_dn = cast(tuple[jax.Array, jax.Array], (HL_up, HL_dn))
        HL_up = state.HL_up.at[idx_changed.nuclei].set(HL_up)
        HL_dn = state.HL_dn.at[idx_changed.nuclei].set(HL_dn)
        HL_up_nb_en = get_with_fill(HL_up, idx_nb.en[idx_changed.msg], 0)
        HL_dn_nb_en = get_with_fill(HL_dn, idx_nb.en[idx_changed.msg], 0)
        HL_nb_en = jnp.where(self.spins[idx_changed.msg][..., None, None] > 0, HL_up_nb_en, HL_dn_nb_en)

        # Compute gamma_en again, but with a different set of electrons
        dyn_params = tree_idx(params.dynamic_params_en, idx_nb.en[idx_changed.msg])
        gamma_en_out = self._get_Gamma_en(
            params.Gamma_en, electrons[idx_changed.msg], R_nb_en[idx_changed.msg], dyn_params
        )  # type: ignore
        msg = contract(HL_nb_en, gamma_en_out)
        msg = normalize(msg, params.scales["msg"])
        msg = state.msg.at[idx_changed.msg].set(msg)

        # Msg from hinit
        h_init_nb = jnp.where(
            (self.spins[idx_changed.out][:, None] == spin_nb_ee_out[idx_changed.out])[..., None],
            get_with_fill(h_init_same, idx_nb.ee_out[idx_changed.out], 0),
            get_with_fill(h_init_diff, idx_nb.ee_out[idx_changed.out], 0),
        )

        # readout
        @jax.vmap
        def compute_hout(h0, msg, electrons, r_nb_ee_out, spins, spin_nb_ee_out, h_init_nb):
            return self.elec_out.apply(
                params.elec_out,
                h0,
                msg,
                electrons,
                r_nb_ee_out,
                spins,
                spin_nb_ee_out,
                h_init_nb,
            )

        # readout
        h_out = cast(
            jax.Array,
            compute_hout(
                h0[idx_changed.out],
                msg[idx_changed.out],
                electrons[idx_changed.out],
                r_nb_ee_out[idx_changed.out],
                self.spins[idx_changed.out],
                spin_nb_ee_out[idx_changed.out],
                h_init_nb,
            ),
        )
        h_out = cast(jax.Array, h_out)
        h_out = state.h_out.at[idx_changed.out].set(h_out)

        return (
            h_out,
            idx_changed.out,
            MoonState(
                electrons=electrons,
                h_init=h_init,
                h_init_same=h_init_same,
                h_init_diff=h_init_diff,
                h0=h0,
                HL_up=HL_up,
                HL_dn=HL_dn,
                msg=msg,
                h_out=h_out,
            ),
        )

    def apply_with_fwd_lap(
        self, params: MoonEmbeddingParams, electrons: Electrons, static: StaticInputMoon
    ) -> tuple[FwdLaplArray, Dependency]:
        idx_nb = self.get_neighbour_indices(electrons, static.n_neighbours)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en, R_nb_en_1el, r_nb_ee_out, spin_nb_ee_out = (
            get_neighbour_coordinates(electrons, self.R, idx_nb, self.spins)
        )
        deps, dep_maps = get_all_dependencies(idx_nb, static.n_deps)
        n_deps_hout = deps.h_el_out.shape[-1]

        @functools.partial(jax.vmap, in_axes=(-3, -3, None), out_axes=-2)
        @functools.partial(fwd_lap, argnums=(0, 1))
        def contract_and_normalize(h, gamma, scale):
            h = contract(h, gamma)
            return normalize(h, scale)

        # Step -1: initial embedding
        @functools.partial(jax.vmap, in_axes=0, out_axes=-2)
        @functools.partial(fwd_lap, argnums=0)
        def get_hinit(r, R_nb, dyn_params):
            result = self.elec_init.apply(params.elec_init, r, R_nb, dyn_params)
            return tuple(cast(jax.Array, x) for x in result)

        dyn_params = tree_idx(params.dynamic_params_elec_init, idx_nb.en_1el)
        h_init, h_init_same, h_init_diff = get_hinit(electrons, R_nb_en_1el, dyn_params)
        h_init_nb_same = get_neighbour_features(h_init_same, idx_nb.ee)
        h_init_nb_diff = get_neighbour_features(h_init_diff, idx_nb.ee)
        h_init_nb_same = pad_pairwise_jacobian(h_init_nb_same, dep_maps.hinit_to_h0, static.n_deps.h_el_initial)
        h_init_nb_diff = pad_pairwise_jacobian(h_init_nb_diff, dep_maps.hinit_to_h0, static.n_deps.h_el_initial)
        h_init = zeropad_jacobian(h_init, static.n_deps.h_el_initial * 3)

        # Step 0: initialize electron jacobians
        @functools.partial(jax.vmap, in_axes=0, out_axes=(-2, -2))
        @fwd_lap
        def init(r, r_nb):
            return r, r_nb

        # Step 1: initial electron embedding
        @functools.partial(jax.vmap, in_axes=(-2, -2, -2, -3, -3, 0, 0), out_axes=-2)  # vmap over center electrons
        @functools.partial(fwd_lap, argnums=(0, 1, 2, 3, 4))
        def get_h0(r, r_nb, h, h_nb_s, h_nb_d, s, s_nb):
            h_nb = jnp.where((s == s_nb)[:, None], h_nb_s, h_nb_d)
            h0 = cast(jax.Array, self.elec_elec_emb.apply(params.elec_elec_emb, r, r_nb, h, h_nb, s, s_nb))
            return normalize(h0, params.scales["h0"])

        h0 = get_h0(*init(electrons, r_nb_ee), h_init, h_init_nb_same, h_init_nb_diff, self.spins, spin_nb_ee)

        # Step 2: Get features from electron -> nuclei
        @functools.partial(jax.vmap, in_axes=-3, out_axes=-3)
        @functools.partial(jax.vmap, in_axes=-2, out_axes=-2)
        @functools.partial(fwd_lap, argnums=(0, 1))
        def get_edge_ne_emb(h0_nb_ne, edge_ne_emb, spin_nb_ne):
            edge_ne_emb = nn.silu(h0_nb_ne + edge_ne_emb)
            edge_ne_up = jnp.where(spin_nb_ne > 0, edge_ne_emb, 0)
            edge_ne_dn = jnp.where(spin_nb_ne < 0, edge_ne_emb, 0)
            return edge_ne_up, edge_ne_dn

        Gamma_ne, edge_ne_emb = self._get_Gamma_ne(params.Gamma_ne, self.R, r_nb_ne, params.dynamic_params_ne, True)
        Gamma_ne = pad_pairwise_jacobian(Gamma_ne, dep_maps.Gamma_ne_to_Hnuc, static.n_deps.H_nuc)
        edge_ne_emb = pad_pairwise_jacobian(edge_ne_emb, dep_maps.Gamma_ne_to_Hnuc, static.n_deps.H_nuc)
        h0_nb_ne = get_neighbour_features(h0, idx_nb.ne)
        h0_nb_ne = pad_pairwise_jacobian(h0_nb_ne, dep_maps.h0_to_Hnuc, static.n_deps.H_nuc)

        # Step 3: Contract from electrons to nuclei and apply MLP
        @functools.partial(jax.vmap, in_axes=-3, out_axes=-2)  # vmap over nuclei
        @fwd_lap
        def contract_and_mlp(h_up, h_dn, gamma):
            H1_up = normalize(contract(h_up, gamma), params.scales["H1_up"])
            H1_dn = normalize(contract(h_dn, gamma), params.scales["H1_dn"])
            return self.nuc_mlp.apply(params.nuc_mlp, H1_up, H1_dn)

        edge_ne_up, edge_ne_dn = get_edge_ne_emb(h0_nb_ne, edge_ne_emb, spin_nb_ne[..., None])
        HL_up, HL_dn = contract_and_mlp(edge_ne_up, edge_ne_dn, Gamma_ne)

        # Step4: Get nucleus -> electron filters
        dyn_params = tree_idx(params.dynamic_params_en, idx_nb.en)
        gamma_en_out = self._get_Gamma_en(params.Gamma_en, electrons, R_nb_en, dyn_params, True)
        gamma_en_out = zeropad_jacobian(gamma_en_out, 3 * n_deps_hout)

        h0 = pad_jacobian(h0, dep_maps.h0_to_hout, static.n_deps.h_el_out)

        # Step 6: Contract deep nuclear embeddigns to output electron embeddings
        HL_up_nb_en = get_neighbour_features(HL_up, idx_nb.en[: self.n_up])
        HL_dn_nb_en = get_neighbour_features(HL_dn, idx_nb.en[self.n_up :])
        HL_nb_en = jtu.tree_map(lambda u, d: jnp.concatenate([u, d], axis=-3), HL_up_nb_en, HL_dn_nb_en)
        HL_nb_en = pad_pairwise_jacobian(HL_nb_en, dep_maps.Hnuc_to_hout, static.n_deps.h_el_out)
        msg = contract_and_normalize(HL_nb_en, gamma_en_out, params.scales["msg"])

        # readout
        @functools.partial(jax.vmap, in_axes=(-2, -2, -2, -3, 0, 0, -3, -3), out_axes=-2)
        @functools.partial(fwd_lap, argnums=(0, 1, 2, 3, 6, 7))
        def apply_elec_out(h0, msg, electrons, r_nb_ee_out, spins, spin_nb_ee_out, h_init_nb_same, h_init_nb_diff):
            h_init_nb = jnp.where(
                (spins == spin_nb_ee_out)[..., None],
                h_init_nb_same,
                h_init_nb_diff,
            )
            return self.elec_out.apply(
                params.elec_out,
                h0,
                msg,
                electrons,
                r_nb_ee_out,
                spins,
                spin_nb_ee_out,
                h_init_nb,
            )

        h_init_nb_same = get_neighbour_features(h_init_same, idx_nb.ee_out)
        h_init_nb_diff = get_neighbour_features(h_init_diff, idx_nb.ee_out)
        h_init_nb_same = pad_pairwise_jacobian(h_init_nb_same, dep_maps.hinit_to_hout, static.n_deps.h_el_out)
        h_init_nb_diff = pad_pairwise_jacobian(h_init_nb_diff, dep_maps.hinit_to_hout, static.n_deps.h_el_out)

        @functools.partial(jax.vmap, in_axes=0, out_axes=-2)
        @fwd_lap
        def init_r(r):
            return r

        r_out_in = init_r(electrons)
        r_out_nb_in = get_neighbour_features(r_out_in, idx_nb.ee_out, NO_NEIGHBOUR)
        r_out_nb_in = pad_pairwise_jacobian(r_out_nb_in, dep_maps.hinit_to_hout, static.n_deps.h_el_out)
        r_out_in = zeropad_jacobian(r_out_in, static.n_deps.h_el_out * 3)

        h_out = apply_elec_out(
            h0,
            msg,
            r_out_in,
            r_out_nb_in,
            self.spins,
            spin_nb_ee_out,
            h_init_nb_same,
            h_init_nb_diff,
        )
        return h_out, deps.h_el_out
