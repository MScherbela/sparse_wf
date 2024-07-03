import functools
from typing import Callable, Literal, NamedTuple, Optional, TypedDict, cast, overload

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.struct import PyTreeNode
from jaxtyping import Array, Float, Integer

from folx.api import FwdLaplArray
from sparse_wf.api import Charges, ElectronEmb, ElectronIdx, Electrons, Int, Nuclei, NucleiIdx, Parameters
from sparse_wf.jax_utils import fwd_lap, jit, nn_vmap, pmap, pmax_if_pmap
from sparse_wf.model.graph_utils import (
    NO_NEIGHBOUR,
    Dependency,
    DependencyMap,
    DistanceMatrix,
    NeighbourIndices,
    NrOfNeighbours,
    get_dependency_map,
    get_full_distance_matrices,
    get_neighbour_coordinates,
    get_neighbour_features,
    get_neighbour_indices,
    get_nr_of_neighbours,
    get_with_fill,
    merge_dependencies,
    pad_jacobian,
    pad_pairwise_jacobian,
    round_to_next_step,
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
from sparse_wf.tree_utils import tree_idx, tree_add


class NucleusDependentParams(NamedTuple):
    filter: DynamicFilterParams
    nuc_embedding: jnp.ndarray


class NrOfDependencies(NamedTuple):
    h_el_initial: int
    H_nuc: int
    h_el_out: int


class NrOfChanges(NamedTuple):
    # Maximum number of changes per moved electrons
    h0: int
    nuclei: int
    out: int


class StaticInputMoon(NamedTuple):
    n_deps: NrOfDependencies
    n_neighbours: NrOfNeighbours
    n_changes: NrOfChanges

    def to_log_data(self):
        return {
            "static/n_nb_en": self.n_neighbours.en,
            "static/n_nb_ee": self.n_neighbours.ee,
            "static/n_nb_ne": self.n_neighbours.ne,
            "static/n_nb_en_1el": self.n_neighbours.en_1el,
            "static/n_deps_h0": self.n_deps.h_el_initial,
            "static/n_deps_H": self.n_deps.H_nuc,
            "static/n_deps_hout": self.n_deps.h_el_out,
        }


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


def get_max_nr_of_dependencies(dist_ee: DistanceMatrix, dist_ne: DistanceMatrix, cutoff: float):
    @functools.partial(jnp.vectorize, signature="(elec,elec),(nuc,elec),()->(),(),()")
    def max_deps_vectorized(dist_ee: DistanceMatrix, dist_ne: DistanceMatrix, cutoff: float):
        def get_max_deps(deps):
            return pmax_if_pmap(jnp.max(jnp.sum(deps, axis=1)))

        h0_deps = dist_ee<cutoff
        n_deps_max_h0 = get_max_deps(h0_deps)

        H0 = dist_ne<cutoff
        def get_H_one_nuc(nearby_elec):
            return (h0_deps & nearby_elec).any(axis=1)
        H_deps = jax.vmap(get_H_one_nuc, 0, 0)(H0)
        n_deps_max_H = get_max_deps(H_deps)

        def get_h_out_1_elec(nearby_elec, nearby_nuc):
            elec_connected_to_nearby_nuclei = (nearby_nuc & H_deps.T).any(axis=1)
            return nearby_elec | elec_connected_to_nearby_nuclei
        h_out_deps = jax.vmap(get_h_out_1_elec, (1, 1), 0)(h0_deps, H0)
        n_deps_max_h_out = get_max_deps(h_out_deps)

        return n_deps_max_h0, n_deps_max_H, n_deps_max_h_out

    n_deps = max_deps_vectorized(dist_ee, dist_ne, cutoff)
    n_deps = jtu.tree_map(lambda x: pmax_if_pmap(jnp.max(x)), n_deps)
    # max_func = lambda l: pmax_if_pmap(jnp.max(jnp.asarray(l), axis=1))
    # n_deps_max_h0, n_deps_max_H, n_deps_max_h_out = max_func(n_deps_max_h0), max_func(n_deps_max_H), max_func(n_deps_max_h_out)
    return n_deps



def _get_static(electrons: Array, R: Nuclei, cutoff: float, cutoff_1el: float):
    n_el = electrons.shape[-2]
    n_nuc = len(R)
    dist_ee, dist_ne = get_full_distance_matrices(electrons, R)
    n_ee, n_en, n_ne, n_en_1el = get_nr_of_neighbours(dist_ee, dist_ne, cutoff, cutoff_1el)
    n_neighbours = NrOfNeighbours(
        ee=round_to_next_step(n_ee, 1.1, 1, n_el - 1),  # type: ignore
        en=round_to_next_step(n_en, 1.1, 1, n_nuc),  # type: ignore
        ne=round_to_next_step(n_ne, 1.1, 1, n_el),  # type: ignore
        en_1el=round_to_next_step(n_en_1el, 1.1, 1, n_nuc),  # type: ignore
    )
    n_deps_h0, n_deps_H, n_deps_hout = get_max_nr_of_dependencies(dist_ee, dist_ne, cutoff)  # noqa: F821
    n_deps = NrOfDependencies(
        h_el_initial=n_deps_h0,
        H_nuc=round_to_next_step(n_deps_H, 1.1, 1, n_el),  # type: ignore
        h_el_out=round_to_next_step(n_deps_hout, 1.1, 1, n_el),  # type: ignore
    )
    return n_neighbours, n_deps


get_static_pmapped = pmap(_get_static, in_axes=(0, None, None, None))
get_static_jitted = jit(_get_static)


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
        features_ee = jax.vmap(get_diff_features, in_axes=(None, 0))(r, r_nb)
        spin_mask = s == s_nb
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
        return result


class MoonEdgeFeatures(nn.Module):
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    n_gamma: int = 1

    @nn.compact
    def __call__(
        self,
        r_center: Float[Array, "dim=3"],
        r_neighbour: Float[Array, "dim=3"],
        dynamic_params: NucleusDependentParams,
    ):
        features = get_diff_features(r_center, r_neighbour)
        beta = PairwiseFilter(self.cutoff, self.filter_dims[1])(features, dynamic_params.filter)
        gamma = nn.Dense(self.n_gamma * self.feature_dim, use_bias=False)(beta)
        gamma = jnp.split(gamma, self.n_gamma, axis=-1)
        scaled_features = features / features[..., :1] * jnp.log1p(features[..., :1])
        edge_embedding = nn.Dense(self.feature_dim, use_bias=False)(scaled_features) + dynamic_params.nuc_embedding
        return (*gamma, edge_embedding)


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
        h_init = self.glu(result)
        h_init_same = self.dense_same(self.activation(h_init))
        h_init_diff = self.dense_diff(self.activation(h_init))
        return h_init, h_init_same, h_init_diff


class MoonElecOut(nn.Module):
    @nn.compact
    def __call__(self, elec, msg):
        dim = elec.shape[-1]
        elec = nn.silu(nn.Dense(dim)(elec))
        out = nn.silu(nn.Dense(dim)(elec) + msg)
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
    deps_neighbours = get_deps_nb(deps_H, idx_nb.en)
    deps_hout = merge_dependencies(deps_neighbours, deps_h0, jnp.arange(n_el)[:, None], n_deps_max.h_el_out)
    dep_map_H_to_hout = get_dep_map_for_all_centers(deps_neighbours, deps_hout)
    dep_map_h0_to_hout = jax.vmap(get_dependency_map)(deps_h0, deps_hout)

    # Assert that dependencies are consistent with static dims
    assert deps_h0.shape[-1] == n_deps_max.h_el_initial
    assert deps_H.shape[-1] == n_deps_max.H_nuc
    assert deps_hout.shape[-1] == n_deps_max.h_el_out
    assert dep_map_Gamma_ne_to_H.shape[-1] == 1
    assert dep_map_hinit_to_h0.shape[-1] == 1
    assert dep_map_h0_to_H.shape[-1] == n_deps_max.h_el_initial
    assert dep_map_H_to_hout.shape[-1] == n_deps_max.H_nuc
    assert dep_map_h0_to_hout.shape[-1] == n_deps_max.h_el_initial

    return DependenciesMoon(deps_h0, deps_H, deps_hout), DependencyMaps(
        dep_map_hinit_to_h0, dep_map_h0_to_H, dep_map_Gamma_ne_to_H, dep_map_H_to_hout, dep_map_h0_to_hout
    )


class EmbeddingChanges(NamedTuple):
    h0: ElectronIdx
    nuclei: NucleiIdx
    out: ElectronIdx


@jit(static_argnames="static")
def get_changed_embeddings(
    electrons: Electrons,
    previous_electrons: Electrons,
    changed_electrons: ElectronIdx,
    nuclei: Nuclei,
    static: StaticInputMoon,
    cutoff: float,
):
    num_changed = changed_electrons.shape[-1]
    n_electrons = electrons.shape[0]
    n_nuclei = nuclei.shape[0]
    num_changed_h0 = min(static.n_changes.h0 * num_changed, n_electrons)
    num_changed_nuclei = min(static.n_changes.nuclei * num_changed, n_nuclei)
    num_changed_out = min(
        static.n_changes.out * num_changed_nuclei,
        static.n_changes.out * num_changed,
        n_electrons,
    )

    # Finding affected electrons
    def affected_particles(old_x, old_y, new_x, new_y, num_changes, include=None):
        dist_old = jnp.linalg.norm(old_x[:, None] - old_y[None], axis=-1)
        dist_new = jnp.linalg.norm(new_x[:, None] - new_y[None], axis=-1)
        # we only care whether they were close or after the move, not which of these.
        dist_shortest = jnp.minimum(dist_old, dist_new)
        dist_shortest = jnp.min(dist_shortest, axis=0)  # shortest path to any particle
        # top k returns the k largest values and indices from an array, since we want the smallest distances we negate them
        neg_dists, order = jax.lax.top_k(-dist_shortest, num_changes)
        affected = jnp.where(neg_dists > (-cutoff), order, NO_NEIGHBOUR)
        if include is None:
            return affected
        return jnp.unique(jnp.concatenate([affected, include]), size=num_changes, fill_value=NO_NEIGHBOUR)

    changed_h0 = affected_particles(
        previous_electrons[changed_electrons],
        previous_electrons,
        electrons[changed_electrons],
        electrons,
        num_changed_h0,
    )
    changed_nuclei = affected_particles(
        previous_electrons[changed_h0],
        nuclei,
        electrons[changed_h0],
        nuclei,
        num_changed_nuclei,
    )
    changed_out = affected_particles(
        nuclei[changed_nuclei],
        previous_electrons,
        nuclei[changed_nuclei],
        electrons,
        num_changed_out,
        changed_h0,
    )
    return EmbeddingChanges(changed_h0, changed_nuclei, changed_out)


class MoonScales(TypedDict):
    h0: Optional[ScalingParam]
    H1_up: Optional[ScalingParam]
    H1_dn: Optional[ScalingParam]
    h1: Optional[ScalingParam]
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
    h1: Array
    HL_up: Array
    HL_dn: Array
    h_out: Array


class MoonEmbedding(PyTreeNode):
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
            Gamma_ne=MoonEdgeFeatures(cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes, n_gamma=1),
            Gamma_en=MoonEdgeFeatures(cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes, n_gamma=2),
            elec_init=MoonElecInit(cutoff_1el, pair_mlp_widths, feature_dim, pair_n_envelopes),
            nuc_mlp=MoonNucMLP(nuc_mlp_depth),
            elec_out=MoonElecOut(),
            low_rank_buffer=low_rank_buffer,
        )

    @property
    def n_nuclei(self):
        return len(self.R)

    @property
    def spins(self):
        return jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_electrons - self.n_up)]).astype(jnp.float32)

    def get_neighbour_indices(self, electrons: Electrons, n_neighbours: NrOfNeighbours):
        return get_neighbour_indices(electrons, self.R, n_neighbours, self.cutoff, self.cutoff_1el)

    def init(self, rng: Array, electrons: Array, static: StaticInputMoon) -> Parameters:
        dtype = electrons.dtype
        rngs = jax.random.split(rng, 10)
        r_dummy = jnp.zeros([3], dtype)
        r_nb_dummy = jnp.zeros([1, 3], dtype)
        spin_dummy = jnp.zeros([], dtype)
        spin_nb_dummy = jnp.zeros([1], dtype)
        features_dummy = jnp.zeros([self.feature_dim], dtype)
        dummy_dyn_param = self._init_nuc_dependant_params(rngs[0], n_nuclei=1)

        params = MoonEmbeddingParams(
            dynamic_params_en=self._init_nuc_dependant_params(rngs[1]),
            dynamic_params_ne=self._init_nuc_dependant_params(rngs[2]),
            dynamic_params_elec_init=self._init_nuc_dependant_params(rngs[3]),
            elec_init=self.elec_init.init(rngs[4], r_dummy, r_nb_dummy, dummy_dyn_param),
            elec_elec_emb=self.elec_elec_emb.init(
                rngs[5], r_dummy, r_nb_dummy, features_dummy, features_dummy[None], spin_dummy, spin_nb_dummy
            ),
            Gamma_ne=self.Gamma_ne.init(rngs[6], r_dummy, r_dummy, dummy_dyn_param),
            Gamma_en=self.Gamma_en.init(rngs[7], r_dummy, r_dummy, dummy_dyn_param),
            nuc_mlp=self.nuc_mlp.init(rngs[8], features_dummy, features_dummy),
            elec_out=self.elec_out.init(rngs[9], features_dummy, features_dummy),
            scales=MoonScales(h0=None, H1_up=None, H1_dn=None, h1=None, msg=None),
        )
        _, scales = self.apply(params, electrons, static, return_scales=True)
        params = params.replace(scales=scales)
        return params

    def _init_nuc_dependant_params(self, rng, n_nuclei=None):
        n_nuclei = n_nuclei or self.n_nuclei
        rngs = jax.random.split(rng, 4)
        return NucleusDependentParams(
            filter=DynamicFilterParams(
                scales=scale_initializer(rngs[0], self.cutoff, (n_nuclei, self.pair_n_envelopes)),
                kernel=lecun_normal(rngs[1], (n_nuclei, 4, self.pair_mlp_widths[0])),
                bias=jax.random.normal(rngs[2], (n_nuclei, self.pair_mlp_widths[0]), jnp.float32) * 2.0,
            ),
            nuc_embedding=jax.random.normal(rngs[3], (n_nuclei, self.feature_dim), jnp.float32),
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

    @overload
    def apply(
        self,
        params: MoonEmbeddingParams,
        electrons: Electrons,
        static: StaticInputMoon,
        return_scales: Literal[False] = False,
        return_state: Literal[False] = False,
    ) -> ElectronEmb: ...

    @overload
    def apply(
        self,
        params: MoonEmbeddingParams,
        electrons: Electrons,
        static: StaticInputMoon,
        return_scales: Literal[True],
        return_state: Literal[False] = False,
    ) -> tuple[ElectronEmb, MoonScales]: ...

    @overload
    def apply(
        self,
        params: MoonEmbeddingParams,
        electrons: Electrons,
        static: StaticInputMoon,
        return_scales: Literal[False],
        return_state: Literal[True],
    ) -> tuple[ElectronEmb, MoonState]: ...

    def apply(
        self,
        params: MoonEmbeddingParams,
        electrons: Electrons,
        static: StaticInputMoon,
        return_scales: bool = False,
        return_state: bool = False,
    ) -> ElectronEmb | tuple[ElectronEmb, MoonScales] | tuple[ElectronEmb, MoonState]:
        idx_nb = self.get_neighbour_indices(electrons, static.n_neighbours)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en, R_nb_en_1el = get_neighbour_coordinates(
            electrons, self.R, idx_nb, self.spins
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
        gamma_en_init, gamma_en_out, edge_en_emb = self._get_Gamma_en(params.Gamma_en, electrons, R_nb_en, dyn_params)  # type: ignore
        edge_en_emb = nn.silu(h0[:, None] + edge_en_emb)
        h1 = contract(edge_en_emb, gamma_en_init)
        h1, params.scales["h1"] = normalize(h1, params.scales["h1"], True)
        h1 += h0  # residual connection

        # update electron embedding
        HL_up, HL_dn = self.nuc_mlp.apply(params.nuc_mlp, H1_up, H1_dn)
        HL_up, HL_dn = cast(tuple[jax.Array, jax.Array], (HL_up, HL_dn))
        HL_up_nb_en = get_with_fill(HL_up, idx_nb.en, 0)
        HL_dn_nb_en = get_with_fill(HL_dn, idx_nb.en, 0)
        HL_nb_en = jnp.where(self.spins[..., None, None] > 0, HL_up_nb_en, HL_dn_nb_en)
        msg = contract(HL_nb_en, gamma_en_out)
        msg, params.scales["msg"] = normalize(msg, params.scales["msg"], True)

        # readout
        h_out = self.elec_out.apply(params.elec_out, h1, msg)
        h_out = cast(jax.Array, h_out)

        if return_scales:
            return h_out, params.scales  # type: ignore

        if return_state:
            return h_out, MoonState(
                electrons=electrons,
                h_init=h_init,
                h_init_same=h_init_same,
                h_init_diff=h_init_diff,
                h0=h0,
                h1=h1,
                HL_up=HL_up,
                HL_dn=HL_dn,
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
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en, R_nb_en_1el = get_neighbour_coordinates(
            electrons, self.R, idx_nb, self.spins
        )
        changed = get_changed_embeddings(electrons, state.electrons, changed_electrons, self.R, static, self.cutoff)

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
            (self.spins[changed.h0][:, None] == spin_nb_ee[changed.h0])[..., None],
            get_with_fill(h_init_same, idx_nb.ee[changed.h0], 0),
            get_with_fill(h_init_diff, idx_nb.ee[changed.h0], 0),
        )
        h0_new = get_h0(
            electrons[changed.h0],
            r_nb_ee[changed.h0],
            h_init[changed.h0],
            h_init_nb,
            self.spins[changed.h0],
            spin_nb_ee[changed.h0],
        )
        h0_new = normalize(h0_new, params.scales["h0"])
        h0 = state.h0.at[changed.h0].set(h0_new)

        # construct nuclei embeddings
        Gamma_ne, edge_ne_emb = self._get_Gamma_ne(
            params.Gamma_ne,
            *jtu.tree_map(lambda x: jnp.asarray(x)[changed.nuclei], (self.R, r_nb_ne, params.dynamic_params_ne)),
        )
        h0_nb_ne = get_with_fill(h0, idx_nb.ne[changed.nuclei], 0)
        edge_ne_emb = nn.silu(h0_nb_ne + edge_ne_emb)
        edge_ne_up = jnp.where(spin_nb_ne[changed.nuclei][..., None] > 0, edge_ne_emb, 0)
        edge_ne_dn = jnp.where(spin_nb_ne[changed.nuclei][..., None] < 0, edge_ne_emb, 0)
        H1_up = contract(edge_ne_up, Gamma_ne)  # type: ignore
        H1_dn = contract(edge_ne_dn, Gamma_ne)  # type: ignore
        H1_up = normalize(H1_up, params.scales["H1_up"])
        H1_dn = normalize(H1_dn, params.scales["H1_dn"])

        # construct electron embedding
        # We need to update all electrons where h0 changed.
        dyn_params = tree_idx(params.dynamic_params_en, idx_nb.en[changed.h0])
        gamma_en_init, _, edge_en_emb = self._get_Gamma_en(
            params.Gamma_en, electrons[changed.h0], R_nb_en[changed.h0], dyn_params
        )  # type: ignore
        edge_en_emb = nn.silu(h0[changed.h0][:, None] + edge_en_emb)
        h1 = contract(edge_en_emb, gamma_en_init)
        h1 = normalize(h1, params.scales["h1"])
        h1 += h0[changed.h0]  # residual connection
        h1 = state.h1.at[changed.h0].set(h1)

        # Update nuclei embeddings
        # Compute gamma_en again, but with a different set of electrons
        dyn_params = tree_idx(params.dynamic_params_en, idx_nb.en[changed.out])
        _, gamma_en_out, _ = self._get_Gamma_en(
            params.Gamma_en, electrons[changed.out], R_nb_en[changed.out], dyn_params
        )  # type: ignore
        HL_up, HL_dn = self.nuc_mlp.apply(params.nuc_mlp, H1_up, H1_dn)
        HL_up, HL_dn = cast(tuple[jax.Array, jax.Array], (HL_up, HL_dn))
        HL_up = state.HL_up.at[changed.nuclei].set(HL_up)
        HL_dn = state.HL_dn.at[changed.nuclei].set(HL_dn)
        HL_up_nb_en = get_with_fill(HL_up, idx_nb.en[changed.out], 0)
        HL_dn_nb_en = get_with_fill(HL_dn, idx_nb.en[changed.out], 0)
        HL_nb_en = jnp.where(self.spins[changed.out][..., None, None] > 0, HL_up_nb_en, HL_dn_nb_en)
        msg = contract(HL_nb_en, gamma_en_out)
        msg = normalize(msg, params.scales["msg"])

        # readout
        h_out = self.elec_out.apply(params.elec_out, h1[changed.out], msg)
        h_out = cast(jax.Array, h_out)
        h_out = state.h_out.at[changed.out].set(h_out)

        return (
            h_out,
            changed.out,
            MoonState(
                electrons=electrons,
                h_init=h_init,
                h_init_same=h_init_same,
                h_init_diff=h_init_diff,
                h0=h0,
                h1=h1,
                HL_up=HL_up,
                HL_dn=HL_dn,
                h_out=h_out,
            ),
        )

    def apply_with_fwd_lap(
        self, params: MoonEmbeddingParams, electrons: Electrons, static: StaticInputMoon
    ) -> tuple[FwdLaplArray, Dependency]:
        idx_nb = self.get_neighbour_indices(electrons, static.n_neighbours)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en, R_nb_en_1el = get_neighbour_coordinates(
            electrons, self.R, idx_nb, self.spins
        )
        deps, dep_maps = get_all_dependencies(idx_nb, static.n_deps)
        n_deps_h0 = deps.h0.shape[-1]
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
        @functools.partial(jax.vmap, in_axes=0, out_axes=(-2, -3))
        @fwd_lap
        def init(r, r_nb):
            return r, r_nb

        # Step 1: initial electron embedding
        @functools.partial(jax.vmap, in_axes=(-2, -3, -2, -3, -3, 0, 0), out_axes=-2)  # vmap over center electrons
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
        gamma_en_init, gamma_en_out, edge_en_emb = self._get_Gamma_en(
            params.Gamma_en, electrons, R_nb_en, dyn_params, True
        )
        gamma_en_out = zeropad_jacobian(gamma_en_out, 3 * n_deps_hout)
        gamma_en_init = zeropad_jacobian(gamma_en_init, 3 * n_deps_h0)
        edge_en_emb = zeropad_jacobian(edge_en_emb, 3 * n_deps_h0)
        edge_en_emb = jax.vmap(fwd_lap(lambda h, e: jax.nn.silu(h[None, :] + e)), in_axes=(-2, -3), out_axes=-3)(
            h0, edge_en_emb
        )

        # Step 5: Contract initial electron-nucleus features
        h1 = contract_and_normalize(edge_en_emb, gamma_en_init, params.scales["h1"])
        h1 = tree_add(h1, h0)
        h1 = pad_jacobian(h1, dep_maps.h0_to_hout, static.n_deps.h_el_out)

        # Step 6: Contract deep nuclear embeddigns to output electron embeddings
        HL_up_nb_en = get_neighbour_features(HL_up, idx_nb.en[: self.n_up])
        HL_dn_nb_en = get_neighbour_features(HL_dn, idx_nb.en[self.n_up :])
        HL_nb_en = jtu.tree_map(lambda u, d: jnp.concatenate([u, d], axis=-3), HL_up_nb_en, HL_dn_nb_en)
        HL_nb_en = pad_pairwise_jacobian(HL_nb_en, dep_maps.Hnuc_to_hout, static.n_deps.h_el_out)
        msg = contract_and_normalize(HL_nb_en, gamma_en_out, params.scales["msg"])

        # readout
        apply_elec_out = fwd_lap(lambda h, m: self.elec_out.apply(params.elec_out, h, m))
        apply_elec_out = jax.vmap(apply_elec_out, in_axes=-2, out_axes=-2)
        h_out = apply_elec_out(h1, msg)
        return h_out, deps.h_el_out

    def get_static_input(self, electrons: Array) -> StaticInputMoon:
        if electrons.ndim == 4:
            # [device x local_batch x el x 3] => electrons are split across gpus;
            n_neighbours, n_dependencies = get_static_pmapped(electrons, self.R, self.cutoff, self.cutoff_1el)
            # Data is synchronized across all devices, so we can just take the 0-th element
            n_dependencies = [int(x[0]) for x in n_dependencies]
            n_neighbours = [int(x[0]) for x in n_neighbours]
        else:
            n_neighbours, n_dependencies = get_static_jitted(electrons, self.R, self.cutoff, self.cutoff_1el)
            n_dependencies = [int(x) for x in n_dependencies]
            n_neighbours = [int(x) for x in n_neighbours]

        n_neighbours = NrOfNeighbours(*n_neighbours)
        n_dependencies = NrOfDependencies(*n_dependencies)
        n_changes = NrOfChanges(
            h0=n_dependencies.h_el_initial + self.low_rank_buffer,
            nuclei=n_dependencies.H_nuc + self.low_rank_buffer,
            out=n_dependencies.h_el_out + self.low_rank_buffer,
        )
        return StaticInputMoon(
            n_neighbours=n_neighbours,
            n_deps=n_dependencies,
            n_changes=n_changes,
        )
