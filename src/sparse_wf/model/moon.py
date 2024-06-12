from typing import Callable, Optional, NamedTuple, cast, TypedDict
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
import pyscf.gto
from sparse_wf.api import JastrowArgs

from sparse_wf.api import Electrons, Int, Nuclei, Parameters
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
)
from sparse_wf.model.utils import (
    DynamicFilterParams,
    FixedScalingFactor,
    PairwiseFilter,
    contract,
    get_diff_features,
    get_diff_features_vmapped,
    scale_initializer,
    IsotropicEnvelope,
    ElElCusp,
)
from sparse_wf.model.wave_function import MoonLikeWaveFunction, NrOfDependencies, StaticInput
from sparse_wf.tree_utils import tree_idx
import functools
import jax.tree_util as jtu
from flax.struct import PyTreeNode


def lecun_normal(rng, shape):
    fan_in = shape[0]
    scale = 1 / jnp.sqrt(fan_in)
    return jax.random.truncated_normal(rng, -1, 1, shape, jnp.float32) * scale


class NucleusDependentParams(NamedTuple):
    filter: DynamicFilterParams
    nuc_embedding: jnp.ndarray


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


class MoonEdgeFeatures(nn.Module):
    R: Nuclei
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    pair_feature_dim: int = 4
    gamma_dim: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        r_center: Float[Array, "dim=3"],
        r_neighbour: Float[Array, "dim=3"],
        dynamic_params: NucleusDependentParams,
    ):
        features = get_diff_features(r_center, r_neighbour)
        beta = PairwiseFilter(self.cutoff, self.filter_dims[1])(features, dynamic_params.filter)
        gamma = nn.Dense(self.gamma_dim if self.gamma_dim else self.feature_dim, use_bias=False)(beta)
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


ScalingParam = Float[Array, ""]


class MoonScales(TypedDict):
    h0: Optional[ScalingParam]
    H1_up: Optional[ScalingParam]
    H1_dn: Optional[ScalingParam]
    h1: Optional[ScalingParam]
    msg: Optional[ScalingParam]
    nuc: Optional[ScalingParam]


class MoonParams(PyTreeNode):
    elec_elec_emb: Parameters
    Gamma_ne: Parameters
    Gamma_en: Parameters
    nuc_mlp: Parameters
    elec_out: Parameters
    dynamic_params_en: NucleusDependentParams
    dynamic_params_ne: NucleusDependentParams
    scales: MoonScales


def normalize(x, scale: ScalingParam | None):
    if scale is None:
        scale = 1.0 / jnp.std(x)
        scale = jnp.where(jnp.isfinite(scale), scale, 1.0)
    return x * scale, scale


class Moon(MoonLikeWaveFunction):
    elec_elec_emb: MoonElecEmb
    Gamma_ne: MoonEdgeFeatures
    Gamma_en: MoonEdgeFeatures
    nuc_mlp: MoonNucMLP
    elec_out: MoonElecOut

    @classmethod
    def create(
        cls,
        mol: pyscf.gto.Mole,
        cutoff: float,
        feature_dim: int,
        nuc_mlp_depth: int,
        pair_mlp_widths: tuple[int, int],
        pair_n_envelopes: int,
        n_determinants: int,
        n_envelopes: int,
        use_e_e_cusp: bool,
        mlp_jastrow: JastrowArgs,
        log_jastrow: JastrowArgs,
        use_yukawa_jastrow: bool,
    ):
        R = np.asarray(mol.atom_coords(), dtype=jnp.float32)
        Z = np.asarray(mol.atom_charges())
        if use_e_e_cusp and use_yukawa_jastrow:
            raise KeyError("Use either electron-electron cusp or Yukawa")
        return cls(
            R=R,
            Z=Z,
            n_electrons=mol.nelectron,
            n_up=mol.nelec[0],
            n_determinants=n_determinants,
            n_envelopes=n_envelopes,
            cutoff=cutoff,
            feature_dim=feature_dim,
            pair_mlp_widths=pair_mlp_widths,
            pair_n_envelopes=pair_n_envelopes,
            nuc_mlp_depth=nuc_mlp_depth,
            use_e_e_cusp=use_e_e_cusp,
            mlp_jastrow_args=mlp_jastrow,
            log_jastrow_args=log_jastrow,
            use_yukawa_jastrow=use_yukawa_jastrow,
            to_orbitals=nn.Dense(n_determinants * mol.nelectron, name="lin_orbitals"),
            envelope=IsotropicEnvelope(n_determinants, mol.nelectron, n_envelopes),
            e_e_cusp=ElElCusp(mol.nelec[0]) if use_e_e_cusp else None,
            elec_elec_emb=MoonElecEmb(R, cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes),
            Gamma_ne=MoonEdgeFeatures(R, cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes),
            Gamma_en=MoonEdgeFeatures(
                R, cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes, gamma_dim=2 * feature_dim
            ),
            nuc_mlp=MoonNucMLP(nuc_mlp_depth),
            elec_out=MoonElecOut(),
        )

    def init_embedding(self, rng: Array, electrons: Array, static: StaticInput) -> Parameters:
        rngs = jax.random.split(rng, 5)
        r_dummy = jnp.zeros([3])
        r_nb_dummy = jnp.zeros([1, 3])
        spin_dummy = jnp.zeros([])
        spin_nb_dummy = jnp.zeros([1])
        features_dummy = jnp.zeros([self.feature_dim])

        dynamic_params_en = self._init_nuc_dependant_params(rngs[0])
        dynamic_params_ne = self._init_nuc_dependant_params(rngs[1])
        dummy_dyn_param = jtu.tree_map(lambda x: x[1:], dynamic_params_en)

        params = MoonParams(
            elec_elec_emb=self.elec_elec_emb.init(rngs[2], r_dummy, r_nb_dummy, spin_dummy, spin_nb_dummy),
            Gamma_ne=self.Gamma_ne.init(rngs[3], r_dummy, r_dummy, dummy_dyn_param),
            Gamma_en=self.Gamma_en.init(rngs[4], r_dummy, r_dummy, dummy_dyn_param),
            nuc_mlp=self.nuc_mlp.init(rngs[5], features_dummy, features_dummy),
            elec_out=self.elec_out.init(rngs[6], features_dummy, features_dummy),
            dynamic_params_en=dynamic_params_en,
            dynamic_params_ne=dynamic_params_ne,
            scales=MoonScales(h0=None, H1_up=None, H1_dn=None, h1=None, msg=None, nuc=None),
        )
        _, scales = self.embedding(params, electrons, static, return_scales=True)
        params = params.replace(scales=scales)
        return params

    def _init_nuc_dependant_params(self, rng):
        rngs = jax.random.split(rng, 4)
        return NucleusDependentParams(
            filter=DynamicFilterParams(
                scales=scale_initializer(rngs[0], self.cutoff, (self.n_nuclei, self.pair_n_envelopes)),
                kernel=lecun_normal(rngs[1], (self.n_nuclei, 4, self.pair_mlp_widths[0])),
                bias=jax.random.normal(rngs[2], (self.n_nuclei, self.pair_mlp_widths[0])) * 2.0,
            ),
            nuc_embedding=jax.random.normal(rngs[3], (self.n_nuclei, self.feature_dim)),
        )

    def embedding(
        self, params: MoonParams, electrons: Electrons, static: StaticInput, return_scales=False
    ) -> Electrons:
        idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)
        spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en = get_neighbour_coordinates(
            electrons, self.R, idx_nb, self.spins
        )

        @jax.vmap  # vmap over center electrons
        def get_h0(r, r_nb, s, s_nb):
            return cast(jax.Array, self.elec_elec_emb.apply(params.elec_elec_emb, r, r_nb, s, s_nb))

        @jax.vmap  # vmap over centers (nuclei)
        @functools.partial(jax.vmap, in_axes=(None, 0, None))  # vmap over neighbours (electrons)
        def get_Gamma_ne(R, r_ne, dyn_params):
            Gamma, edge_features = self.Gamma_ne.apply(params.Gamma_ne, R, r_ne, dyn_params)
            return cast(tuple[jax.Array, jax.Array], (Gamma, edge_features))

        @jax.vmap
        @functools.partial(jax.vmap, in_axes=(None, 0, 0))
        def get_Gamma_en(r, R_nb_en, dyn_params):
            Gamma, edge_features = self.Gamma_en.apply(params.Gamma_en, r, R_nb_en, dyn_params)
            Gamma, edge_features = cast(tuple[jax.Array, jax.Array], (Gamma, edge_features))
            return Gamma[: self.feature_dim], Gamma[self.feature_dim :], edge_features

        # initial electron embedding
        h0 = get_h0(electrons, r_nb_ee, self.spins, spin_nb_ee)
        h0, params.scales["h0"] = normalize(h0, params.scales["h0"])

        # construct nuclei embeddings
        Gamma_ne, edge_ne_emb = get_Gamma_ne(self.R, r_nb_ne, params.dynamic_params_ne)
        h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0)
        edge_ne_emb = nn.silu(h0_nb_ne + edge_ne_emb)
        edge_ne_up = jnp.where(spin_nb_ne[..., None] > 0, edge_ne_emb, 0)
        edge_ne_dn = jnp.where(spin_nb_ne[..., None] < 0, edge_ne_emb, 0)
        H1_up = contract(edge_ne_up, Gamma_ne)
        H1_dn = contract(edge_ne_dn, Gamma_ne)
        H1_up, params.scales["H1_up"] = normalize(h0, params.scales["H1_up"])
        H1_dn, params.scales["H1_dn"] = normalize(h0, params.scales["H1_dn"])

        # construct electron embedding
        dyn_params = tree_idx(params.dynamic_params_en, idx_nb.en)
        gamma_en_init, gamma_en_out, edge_en_emb = get_Gamma_en(electrons, R_nb_en, dyn_params)
        edge_en_emb = nn.silu(h0[:, None] + edge_en_emb)
        h1 = contract(edge_en_emb, gamma_en_init)
        h1, params.scales["h1"] = normalize(h1, params.scales["h1"])

        # update electron embedding
        HL_up, HL_down = self.nuc_mlp.apply(params.nuc_mlp, H1_up, H1_dn)
        HL_up, HL_down = cast(tuple[jax.Array, jax.Array], (HL_up, HL_down))
        HL_up_nb_en = get_with_fill(HL_up, idx_nb.en, 0)
        HL_down_nb_en = get_with_fill(HL_down, idx_nb.en, 0)
        HL_nb_en = jnp.where(self.spins[..., None, None] > 0, HL_up_nb_en, HL_down_nb_en)
        msg = contract(HL_nb_en, gamma_en_out)
        msg, params.scales["msg"] = normalize(msg, params.scales["msg"])

        # readout
        hL = self.elec_out.apply(params.elec_out, h1, msg)
        hL = cast(jax.Array, hL)
        if return_scales:
            return hL, params.scales  # type: ignore
        return hL

    # def _embedding_with_fwd_lap(
    #     self, params, electrons: Electrons, static: StaticInput
    # ) -> tuple[FwdLaplArray, Dependency]:
    #     spins = self.get_spins()
    #     idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)
    #     spin_nb_ee, r_nb_ee, spin_nb_ne, r_nb_ne, R_nb_en = get_neighbour_coordinates(electrons, self.R, idx_nb, spins)
    #     deps, dep_maps = get_all_dependencies(idx_nb, static.n_deps)

    #     @functools.partial(jax.vmap, in_axes=0, out_axes=-2)  # vmap over center electrons
    #     @functools.partial(fwd_lap, argnums=(0, 1))
    #     def h0_apply(r, r_nb, s, s_nb):
    #         h0 = self.apply(params, r, r_nb, s, s_nb, method=self._apply_elec_elec_emb)
    #         h0 = self.apply(params, "h0", h0, method=self._apply_scale)
    #         return h0

    #     h0 = h0_apply(electrons, r_nb_ee, spins, spin_nb_ee)
    #     h0 += self.test

    #     @functools.partial(jax.vmap, in_axes=0, out_axes=-3)  # vmap over center nuclei
    #     @functools.partial(jax.vmap, in_axes=0, out_axes=-2)  # vmap over neighbouring electrons
    #     @fwd_lap
    #     def get_Gamma_ne(r_ne):
    #         return self.apply(params, r_ne, method=self._apply_Gamma_ne)

    #     Gamma_ne, edge_ne_emb = get_Gamma_ne(r_nb_ne)

    #     h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0)
    #     edge_ne_emb = nn.silu(h0_nb_ne + edge_ne_emb)
    #     # edge_ne_up = jnp.where(spin_nb_ne[..., None] > 0, edge_ne_emb, 0)
    #     # edge_ne_down = jnp.where(spin_nb_ne[..., None] < 0, edge_ne_emb, 0)

    #     return h0, deps.h0

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
