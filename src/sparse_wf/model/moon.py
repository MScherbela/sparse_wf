from typing import Callable, Optional, NamedTuple, cast, TypedDict
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from sparse_wf.api import Electrons, Int, Nuclei, Parameters, Charges
from sparse_wf.jax_utils import jit
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
    NrOfNeighbours,
    pad_jacobian,
    pad_pairwise_jacobian,
    zeropad_jacobian,
    get_neighbour_features,
)
from sparse_wf.model.utils import (
    DynamicFilterParams,
    FixedScalingFactor,
    PairwiseFilter,
    contract,
    get_diff_features,
    get_diff_features_vmapped,
    lecun_normal,
    normalize,
    scale_initializer,
    ScalingParam,
)
from sparse_wf.tree_utils import tree_idx
import functools
import jax.tree_util as jtu
from flax.struct import PyTreeNode
from folx.api import FwdLaplArray
from sparse_wf.jax_utils import fwd_lap, pmax_if_pmap, pmap


class NucleusDependentParams(NamedTuple):
    filter: DynamicFilterParams
    nuc_embedding: jnp.ndarray


class NrOfDependencies(NamedTuple):
    h_el_initial: int
    H_nuc: int
    h_el_out: int


class StaticInputMoon(NamedTuple):
    n_deps: NrOfDependencies
    n_neighbours: NrOfNeighbours


class DependenciesMoon(NamedTuple):
    h0: Dependency
    H_nuc: Dependency
    h_el_out: Dependency


class DependencyMaps(NamedTuple):
    h0_to_Hnuc: DependencyMap
    Gamma_ne_to_Hnuc: DependencyMap
    Hnuc_to_hout: DependencyMap
    h0_to_hout: DependencyMap


def get_max_nr_of_dependencies(dist_ee: DistanceMatrix, dist_ne: DistanceMatrix, cutoff: float):
    # Thest first electron message passing step can depend at most on electrons within 1 * cutoff
    n_deps_max_h0 = pmax_if_pmap(jnp.max(jnp.sum(dist_ee < cutoff, axis=-1)))

    # The nuclear embeddings are computed with 2 message passing steps and can therefore depend at most on electrons within 2 * cutoff
    n_deps_max_H = pmax_if_pmap(jnp.max(jnp.sum(dist_ne < cutoff * 2, axis=-1)))

    # The output electron embeddings are computed with 3 message passing step and can therefore depend at most on electrons within 3 * cutoff
    n_deps_max_h_out = pmax_if_pmap(jnp.max(jnp.sum(dist_ee < cutoff * 3, axis=-1)))
    return n_deps_max_h0, n_deps_max_H, n_deps_max_h_out


def _get_static(electrons: Array, R: Nuclei, cutoff: float):
    print("Compiling _get_static in moon-embedding")
    n_el = electrons.shape[-2]
    dist_ee, dist_ne = get_full_distance_matrices(electrons, R)
    n_neighbours = get_nr_of_neighbours(dist_ee, dist_ne, cutoff)
    n_deps = get_max_nr_of_dependencies(dist_ee, dist_ne, cutoff)  # noqa: F821
    return jtu.tree_map(lambda x: round_to_next_step(x, 1.2, 1, n_el), (n_neighbours, n_deps))


get_static_pmapped = pmap(_get_static, in_axes=(0, None, None))
get_static_jitted = jit(_get_static)


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


class MoonEdgeFeatures(nn.Module):
    R: Nuclei
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


class MoonScales(TypedDict):
    h0: Optional[ScalingParam]
    H1_up: Optional[ScalingParam]
    H1_dn: Optional[ScalingParam]
    h1: Optional[ScalingParam]
    msg: Optional[ScalingParam]


class MoonEmbeddingParams(PyTreeNode):
    elec_elec_emb: Parameters
    Gamma_ne: Parameters
    Gamma_en: Parameters
    nuc_mlp: Parameters
    elec_out: Parameters
    dynamic_params_en: NucleusDependentParams
    dynamic_params_ne: NucleusDependentParams
    scales: MoonScales


class MoonEmbedding(PyTreeNode):
    # Molecule
    R: Nuclei
    Z: Charges
    n_electrons: int
    n_up: int

    # Hyperparams
    cutoff: float
    feature_dim: int
    pair_mlp_widths: tuple[int, int]
    pair_n_envelopes: int
    nuc_mlp_depth: int

    # Submodules
    elec_elec_emb: MoonElecEmb
    Gamma_ne: MoonEdgeFeatures
    Gamma_en: MoonEdgeFeatures
    nuc_mlp: MoonNucMLP
    elec_out: MoonElecOut

    @classmethod
    def create(
        cls,
        R: Nuclei,
        Z: Charges,
        n_electrons: int,
        n_up: int,
        cutoff: float,
        feature_dim: int,
        nuc_mlp_depth: int,
        pair_mlp_widths: tuple[int, int],
        pair_n_envelopes: int,
    ):
        return cls(
            R=R,
            Z=Z,
            n_electrons=n_electrons,
            n_up=n_up,
            cutoff=cutoff,
            feature_dim=feature_dim,
            pair_mlp_widths=pair_mlp_widths,
            pair_n_envelopes=pair_n_envelopes,
            nuc_mlp_depth=nuc_mlp_depth,
            elec_elec_emb=MoonElecEmb(R, cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes),
            Gamma_ne=MoonEdgeFeatures(R, cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes, n_gamma=1),
            Gamma_en=MoonEdgeFeatures(R, cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes, n_gamma=2),
            nuc_mlp=MoonNucMLP(nuc_mlp_depth),
            elec_out=MoonElecOut(),
        )

    @property
    def n_nuclei(self):
        return len(self.R)

    @property
    def spins(self):
        return jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_electrons - self.n_up)]).astype(jnp.float32)

    def init(self, rng: Array, electrons: Array, static: StaticInputMoon) -> Parameters:
        rngs = jax.random.split(rng, 7)
        r_dummy = jnp.zeros([3])
        r_nb_dummy = jnp.zeros([1, 3])
        spin_dummy = jnp.zeros([])
        spin_nb_dummy = jnp.zeros([1])
        features_dummy = jnp.zeros([self.feature_dim])

        dynamic_params_en = self._init_nuc_dependant_params(rngs[0])
        dynamic_params_ne = self._init_nuc_dependant_params(rngs[1])
        dummy_dyn_param = jtu.tree_map(lambda x: x[1:], dynamic_params_en)

        params = MoonEmbeddingParams(
            dynamic_params_en=dynamic_params_en,
            dynamic_params_ne=dynamic_params_ne,
            elec_elec_emb=self.elec_elec_emb.init(rngs[2], r_dummy, r_nb_dummy, spin_dummy, spin_nb_dummy),
            Gamma_ne=self.Gamma_ne.init(rngs[3], r_dummy, r_dummy, dummy_dyn_param),
            Gamma_en=self.Gamma_en.init(rngs[4], r_dummy, r_dummy, dummy_dyn_param),
            nuc_mlp=self.nuc_mlp.init(rngs[5], features_dummy, features_dummy),
            elec_out=self.elec_out.init(rngs[6], features_dummy, features_dummy),
            scales=MoonScales(h0=None, H1_up=None, H1_dn=None, h1=None, msg=None),
        )
        _, scales = self.apply(params, electrons, static, return_scales=True)
        params = params.replace(scales=scales)
        return params

    def _init_nuc_dependant_params(self, rng):
        rngs = jax.random.split(rng, 4)
        return NucleusDependentParams(
            filter=DynamicFilterParams(
                scales=scale_initializer(rngs[0], self.cutoff, (self.n_nuclei, self.pair_n_envelopes)),
                kernel=lecun_normal(rngs[1], (self.n_nuclei, 4, self.pair_mlp_widths[0])),
                bias=jax.random.normal(rngs[2], (self.n_nuclei, self.pair_mlp_widths[0]), jnp.float32) * 2.0,
            ),
            nuc_embedding=jax.random.normal(rngs[3], (self.n_nuclei, self.feature_dim), jnp.float32),
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
        self, params: MoonEmbeddingParams, electrons: Electrons, static: StaticInputMoon, return_scales=False
    ) -> Electrons:
        idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en = get_neighbour_coordinates(
            electrons, self.R, idx_nb, self.spins
        )

        @jax.vmap  # vmap over center electrons
        def get_h0(r, r_nb, s, s_nb):
            return cast(jax.Array, self.elec_elec_emb.apply(params.elec_elec_emb, r, r_nb, s, s_nb))

        # initial electron embedding
        h0 = get_h0(electrons, r_nb_ee, self.spins, spin_nb_ee)
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
        return h_out

    def apply_with_fwd_lap(
        self, params: MoonEmbeddingParams, electrons: Electrons, static: StaticInputMoon
    ) -> tuple[FwdLaplArray, Dependency]:
        idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en = get_neighbour_coordinates(
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

        # Step 1: initial electron embedding
        @functools.partial(jax.vmap, in_axes=0, out_axes=-2)  # vmap over center electrons
        @functools.partial(fwd_lap, argnums=(0, 1))
        def get_h0(r, r_nb, s, s_nb):
            h0 = cast(jax.Array, self.elec_elec_emb.apply(params.elec_elec_emb, r, r_nb, s, s_nb))
            return normalize(h0, params.scales["h0"])

        h0 = get_h0(electrons, r_nb_ee, self.spins, spin_nb_ee)

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
            n_neighbours, n_dependencies = get_static_pmapped(electrons, self.R, self.cutoff)
            # Data is synchronized across all devices, so we can just take the 0-th element
            n_dependencies = [int(x[0]) for x in n_dependencies]
            n_neighbours = [int(x[0]) for x in n_neighbours]
        else:
            n_neighbours, n_dependencies = get_static_jitted(electrons, self.R, self.cutoff)
            n_dependencies = [int(x) for x in n_dependencies]
            n_neighbours = [int(x) for x in n_neighbours]

        return StaticInputMoon(n_neighbours=NrOfNeighbours(*n_neighbours), n_deps=NrOfDependencies(*n_dependencies))
