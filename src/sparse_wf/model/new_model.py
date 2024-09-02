import functools
from typing import Callable, NamedTuple, Optional, Generic, TypeVar
from sparse_wf.static_args import round_with_padding

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
from folx.api import FwdJacobian, FwdLaplArray
from jaxtyping import Array, Float, Integer

from sparse_wf.api import Charges, ElectronIdx, Electrons, Embedding, Int, Nuclei, Parameters
from sparse_wf.jax_utils import fwd_lap, jit, nn_vmap, rng_sequence
from sparse_wf.model.graph_utils import (
    NO_NEIGHBOUR,
    Dependency,
    _pad_jacobian_to_output_deps,
    affected_particles,
    get_dependency_map,
    get_full_distance_matrices,
)
from sparse_wf.model.moon import GatedLinearUnit
from sparse_wf.model.utils import (
    DynamicFilterParams,
    PairwiseFilter,
    ScalingParam,
    get_diff_features,
    lecun_normal,
    normalize,
    scale_initializer,
    iter_list_with_pad,
    AppendingList,
)


class NucleusDependentParams(NamedTuple):
    filter: DynamicFilterParams
    nuc_embedding: jax.Array | None


class EdgeFeatures(nn.Module):
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
        scaled_features = features / features[..., :1] * jnp.log1p(features[..., :1])
        edge_embedding = nn.Dense(self.feature_dim, use_bias=False)(scaled_features) + dynamic_params.nuc_embedding
        return gamma, edge_embedding


class ElecInit(nn.Module):
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    n_updates: int
    activation: Callable = nn.silu

    @nn.compact
    def __call__(
        self,
        r: Float[Array, " dim=3"],
        R_nb: Float[Array, "n_neighbours dim=3"],
        dynamic_params: NucleusDependentParams,
    ):
        n_out = self.n_updates * 2 + 2
        edge_feat = EdgeFeatures(self.cutoff, self.filter_dims, self.feature_dim, self.n_envelopes)
        features, Gamma = nn_vmap(edge_feat, in_axes=(None, 0, 0))(r, R_nb, dynamic_params)  # vmap over nuclei
        h_init = jnp.einsum("...Jd,...Jd->...d", features, Gamma)
        h_init = nn.LayerNorm()(h_init)
        h_init = GatedLinearUnit(self.feature_dim)(h_init)
        h_init = self.activation(nn.Dense(self.feature_dim)(h_init))
        # let's parallize this will by having all operations in one go
        h_out = jnp.split(nn.Dense(self.feature_dim * n_out)(h_init), n_out, -1)
        # split the output into the different parts
        h_init, to_params, h_nb_same, h_nb_diff = (
            h_out[0],
            h_out[1],
            h_out[2 : 2 + self.n_updates],
            h_out[2 + self.n_updates :],
        )

        # Params for EE edges
        def scale_init(rng, shape, dtype=jnp.float32):
            return scale_initializer(rng, self.cutoff, shape, dtype)

        to_params = nn.silu(to_params)
        scale = nn.Dense(self.n_envelopes, bias_init=scale_init)(to_params)
        kernel = nn.Dense(4 * self.filter_dims[0])(to_params).reshape(4, self.filter_dims[0]) / jnp.sqrt(4)
        bias = nn.Dense(self.filter_dims[0], bias_init=jax.nn.initializers.normal(2, dtype=jnp.float32))(to_params)
        dynamic_params_ee = DynamicFilterParams(scales=scale, kernel=kernel, bias=bias)

        return h_init, h_nb_same, h_nb_diff, dynamic_params_ee


class ElecElecEdges(nn.Module):
    cutoff: float
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    n_updates: int

    @nn.compact
    def __call__(
        self,
        r: Float[Array, " dim=3"],
        r_nb: Float[Array, " dim=3"],
        s: Int,
        s_nb: Int,
        dynamic_params_ee: DynamicFilterParams,
    ):
        spin_mask = s == s_nb
        features_ee = get_diff_features(r, r_nb)
        beta = PairwiseFilter(self.cutoff, self.filter_dims, name="beta_ee")
        beta_ee = beta(features_ee, dynamic_params_ee)
        dense = nn.Dense(self.feature_dim * self.n_updates * 2, use_bias=False)
        gamma_ee_same, gamma_ee_diff = jnp.split(dense(beta_ee), 2, -1)
        gamma = jnp.where(spin_mask, gamma_ee_same, gamma_ee_diff)

        # logarithmic rescaling
        inp_ee = features_ee / features_ee[..., :1] * jnp.log1p(features_ee[..., :1])
        feat_ee_same, feat_ee_diff = jnp.split(nn.Dense(self.feature_dim * 2 * self.n_updates)(inp_ee), 2, -1)
        feat_ee = jnp.where(spin_mask, feat_ee_same, feat_ee_diff)

        return jnp.split(gamma, self.n_updates, -1), jnp.split(feat_ee, self.n_updates, -1)


class ElecUpdate(nn.Module):
    @nn.compact
    def __call__(
        self,
        h: Float[Array, " feature_dim"],
        gamma: Float[Array, "n_neighbours feature_dim"],
        ee_feat: Float[Array, "n_neighbours feature_dim"],
        nb_same: Float[Array, "n_neighbours feature_dim"],
        nb_diff: Float[Array, "n_neighbours feature_dim"],
        spin: Int,
        spin_nb: Integer[Array, " n_neighbours"],
    ):
        feat_dim = h.shape[-1]
        spin_mask = (spin == spin_nb)[..., None]

        # message passing
        nb = jnp.where(spin_mask, nb_same, nb_diff)
        msg = jnp.einsum("...Jd,...Jd->...d", gamma, nn.silu(ee_feat + nb + h))

        # combination
        out = nn.silu((h + msg) / jnp.sqrt(2.0))
        out = nn.silu(nn.Dense(feat_dim)(out))
        out = nn.silu(nn.Dense(feat_dim)(out))

        # Skip connection
        return out + h


class EmbeddingParams(NamedTuple):
    dynamic_params_en: NucleusDependentParams
    init_params: Parameters
    edge_params: Parameters
    update_params: tuple[Parameters, ...]
    scales: tuple[ScalingParam, ...]


class EmbeddingState(NamedTuple):
    electrons: jax.Array
    h_init: jax.Array
    h_init_same: tuple[jax.Array, ...]
    h_init_diff: tuple[jax.Array, ...]
    h_out: jax.Array
    edg_params: DynamicFilterParams


T = TypeVar("T", bound=int | Int)


class NrOfNeighbours(NamedTuple, Generic[T]):
    ee: T
    en: T


@struct.dataclass
class StaticInputNewModel(Generic[T]):
    n_neighbours: NrOfNeighbours[T]

    # @override
    def round_with_padding(self, padding_factor, n_el, n_up, n_nuc):
        return StaticInputNewModel(
            NrOfNeighbours(
                ee=round_with_padding(self.n_neighbours.ee, padding_factor, n_el - 1),
                en=round_with_padding(self.n_neighbours.ee, padding_factor, n_nuc),
            )
        )

    def to_static(self):
        return jtu.tree_map(lambda x: int(jnp.max(x)), self)


class NewEmbedding(struct.PyTreeNode, Embedding[EmbeddingParams, StaticInputNewModel, EmbeddingState]):
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
    n_updates: int

    # Low rank
    low_rank_buffer: int

    # Submodules
    elec_init: ElecInit
    edge: ElecElecEdges
    updates: tuple[ElecUpdate, ...]

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
        pair_mlp_widths: tuple[int, int],
        pair_n_envelopes: int,
        low_rank_buffer: int,
        n_updates: int,
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
            n_updates=n_updates,
            low_rank_buffer=low_rank_buffer,
            elec_init=ElecInit(cutoff_1el, pair_mlp_widths, feature_dim, pair_n_envelopes, n_updates),
            edge=ElecElecEdges(cutoff, pair_mlp_widths, feature_dim, pair_n_envelopes, n_updates),
            updates=tuple(ElecUpdate() for _ in range(n_updates)),
        )

    @property
    def n_nuclei(self):
        return len(self.R)

    @property
    def spins(self):
        return jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_electrons - self.n_up)]).astype(jnp.float32)

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

    def init(self, rng: Array, electrons: Array, static: StaticInputNewModel):
        dtype = electrons.dtype
        rng_seq = iter(rng_sequence(rng))
        r_dummy = jnp.zeros([3], dtype)
        r_nb_dummy = jnp.zeros([1, 3], dtype)
        spin_dummy = jnp.zeros([], dtype)
        spin_nb_dummy = jnp.zeros([1], dtype)
        features_dummy = jnp.zeros([self.feature_dim], dtype)
        features_nb_dummy = jnp.zeros([1, self.feature_dim], dtype)
        dummy_dyn_param = self._init_nuc_dependant_params(next(rng_seq), n_nuclei=1)

        params = EmbeddingParams(
            dynamic_params_en=self._init_nuc_dependant_params(next(rng_seq)),
            init_params=self.elec_init.init(next(rng_seq), r_dummy, r_nb_dummy, dummy_dyn_param),
            edge_params=self.edge.init(next(rng_seq), r_dummy, r_dummy, spin_dummy, spin_dummy, dummy_dyn_param.filter),
            update_params=tuple(
                upd.init(
                    next(rng_seq),
                    features_dummy,
                    features_nb_dummy,
                    features_nb_dummy,
                    features_nb_dummy,
                    features_nb_dummy,
                    spin_dummy,
                    spin_nb_dummy,
                )
                for upd in self.updates
            ),
            scales=(),
        )
        _, scales = self.apply(params, electrons, static, return_scales=True)
        params = params._replace(scales=scales)
        return params

    def apply(
        self,
        params: EmbeddingParams,
        electrons: Electrons,
        static: StaticInputNewModel,
        return_scales: bool = False,
        return_state: bool = False,
    ):
        scale_seq = iter_list_with_pad(params.scales)
        new_scales = AppendingList()

        # Indexing
        en_idx, ee_idx = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff, self.cutoff_1el)
        R_nb = jnp.array(self.R).at[en_idx].get(mode="fill", fill_value=NO_NEIGHBOUR)
        s_nb = self.spins.at[ee_idx].get(mode="fill", fill_value=0)
        r_nb = electrons.at[ee_idx].get(mode="fill", fill_value=NO_NEIGHBOUR)

        # hinit
        init_fn = jax.vmap(self.elec_init.apply, in_axes=(None, 0, 0, 0))  # type: ignore
        dyn_params = jtu.tree_map(lambda x: x[en_idx], params.dynamic_params_en)
        h_init, h_init_same, h_init_diff, edg_params = init_fn(params.init_params, electrons, R_nb, dyn_params)  # type: ignore
        h_init, new_scales.h_init = normalize(h_init, next(scale_seq), return_scale=True)

        # edges
        edge_fn = jax.vmap(
            jax.vmap(
                self.edge.apply,
                in_axes=(None, None, 0, None, 0, None),
            ),
            in_axes=(None, 0, 0, 0, 0, 0),
        )
        ee_gamma, ee_feat = edge_fn(params.edge_params, electrons, r_nb, self.spins, s_nb, edg_params)

        # update layers
        h = h_init
        for mod, p, gamma, feat, same, diff in zip(
            self.updates, params.update_params, ee_gamma, ee_feat, h_init_same, h_init_diff
        ):
            fwd_fn = jax.vmap(mod.apply, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))  # type: ignore
            same, new_scales.same = normalize(same, next(scale_seq), return_scale=True)
            diff, new_scales.diff = normalize(diff, next(scale_seq), return_scale=True)
            h = fwd_fn(p, h, gamma, feat, same[ee_idx], diff[ee_idx], self.spins, s_nb)
            h, new_scales.h = normalize(h, next(scale_seq), return_scale=True)

        # return
        if return_state:
            return h, EmbeddingState(
                electrons=electrons,
                h_init=h_init,
                h_init_same=h_init_same,
                h_init_diff=h_init_diff,
                h_out=h,
                edg_params=edg_params,
            )
        if return_scales:
            return h, tuple(new_scales)
        return h

    def low_rank_update(
        self,
        params: EmbeddingParams,
        electrons: Electrons,
        changed_electrons: ElectronIdx,
        static: StaticInputNewModel,
        state: EmbeddingState,
    ):
        scale_seq = iter_list_with_pad(params.scales)

        # Changed out
        # TODO: Better calculation of nr of affected
        max_affected = (static.n_neighbours.ee + 1 + self.low_rank_buffer) * len(changed_electrons)
        max_affected = min(max_affected, self.n_electrons)
        changed_out = affected_particles(
            state.electrons[changed_electrons],
            state.electrons,
            electrons[changed_electrons],
            electrons,
            max_affected,
            cutoff=self.cutoff,
            include_idx=changed_electrons,
        )

        # Indexing
        en_idx, ee_idx = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff, self.cutoff_1el)
        en_idx = en_idx[changed_electrons]
        ee_idx = ee_idx[changed_out]
        R_nb = jnp.array(self.R).at[en_idx].get(mode="fill", fill_value=NO_NEIGHBOUR)
        s_nb = self.spins.at[ee_idx].get(mode="fill", fill_value=0)
        r_nb = electrons.at[ee_idx].get(mode="fill", fill_value=NO_NEIGHBOUR)

        # hinit
        init_fn = jax.vmap(self.elec_init.apply, in_axes=(None, 0, 0, 0))  # type: ignore
        dyn_params = jtu.tree_map(lambda x: x[en_idx], params.dynamic_params_en)
        h_init, h_init_same, h_init_diff, edg_params = init_fn(
            params.init_params, electrons[changed_electrons], R_nb, dyn_params
        )  # type: ignore
        h_init = normalize(h_init, next(scale_seq))

        # Update state
        h_init = state.h_init.at[changed_electrons].set(h_init)
        h_init_same = [h.at[changed_electrons].set(h_new) for h, h_new in zip(state.h_init_same, h_init_same)]
        h_init_diff = [h.at[changed_electrons].set(h_new) for h, h_new in zip(state.h_init_diff, h_init_diff)]
        edg_params = jtu.tree_map(lambda x, y: x.at[changed_electrons].set(y), state.edg_params, edg_params)

        # edges
        edge_fn = jax.vmap(
            jax.vmap(
                self.edge.apply,
                in_axes=(None, None, 0, None, 0, None),
            ),
            in_axes=(None, 0, 0, 0, 0, 0),
        )
        ee_gamma, ee_feat = edge_fn(
            params.edge_params,
            electrons[changed_out],
            r_nb,
            self.spins[changed_out],
            s_nb,
            jtu.tree_map(lambda x: x[changed_out], edg_params),
        )

        # update layers
        h = h_init[changed_out]
        for module, p, gamma, feat, same, diff in zip(
            self.updates, params.update_params, ee_gamma, ee_feat, h_init_same, h_init_diff
        ):
            fwd_fn = jax.vmap(module.apply, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))  # type: ignore
            same = normalize(same, next(scale_seq))
            diff = normalize(diff, next(scale_seq))
            h = fwd_fn(p, h, gamma, feat, same[ee_idx], diff[ee_idx], self.spins[changed_out], s_nb)
            h = normalize(h, next(scale_seq))

        # prepare output
        h = state.h_out.at[changed_out].set(h)
        out_state = EmbeddingState(
            electrons=electrons,
            h_init=h_init,
            h_init_same=h_init_same,
            h_init_diff=h_init_diff,
            h_out=h,
            edg_params=edg_params,
        )
        return h, changed_out, out_state

    def apply_with_fwd_lap(
        self,
        params: EmbeddingParams,
        electrons: Array,
        static: StaticInputNewModel,
    ):
        scale_seq = iter_list_with_pad(params.scales)
        # Indexing
        en_idx, ee_idx = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff, self.cutoff_1el)
        dep_out, dep_map, dep_map_gamma = get_all_dependencies(ee_idx)
        n_dep_out = static.n_neighbours.ee + 1
        R_nb = jnp.array(self.R).at[en_idx].get(mode="fill", fill_value=NO_NEIGHBOUR)
        s_nb = self.spins.at[ee_idx].get(mode="fill", fill_value=0)
        r_nb = electrons.at[ee_idx].get(mode="fill", fill_value=NO_NEIGHBOUR)

        pad_jacobian = jax.vmap(_pad_jacobian_to_output_deps, in_axes=(0, 0, None), out_axes=0)
        pad_pairwise_jacobian = jax.vmap(pad_jacobian, in_axes=(1, 1, None), out_axes=1)

        @functools.partial(jax.vmap, in_axes=(0, None))
        @functools.partial(fwd_lap, argnums=0)
        def _normalize(arr, scale):
            return normalize(arr, scale)

        # hinit
        @jax.vmap
        @functools.partial(fwd_lap, argnums=0)
        def get_hinit(r, R_nb, dyn_params):
            return self.elec_init.apply(params.init_params, r, R_nb, dyn_params)

        dyn_params = jtu.tree_map(lambda x: x[en_idx], params.dynamic_params_en)
        h_init, h_init_same, h_init_diff, edg_params = get_hinit(electrons, R_nb, dyn_params)
        h_init = _normalize(h_init, next(scale_seq))

        # edges
        @jax.vmap
        @functools.partial(jax.vmap, in_axes=(None, 0, None, 0, None))
        @functools.partial(fwd_lap, argnums=(0, 1, 4))
        def get_edges(r, r_nb, s, s_nb, edg_params):
            return self.edge.apply(params.edge_params, r, r_nb, s, s_nb, edg_params)

        @jax.vmap
        @functools.partial(jax.vmap, in_axes=(None, 0), out_axes=(None, 0))
        @fwd_lap
        def init_lap(r, r_nb):
            return r, r_nb

        ee_gamma, ee_feat = get_edges(*init_lap(electrons, r_nb), self.spins, s_nb, edg_params)

        # update layers
        h = h_init
        # h = jax.vmap(zeropad_jacobian, in_axes=(0, None))(h, 3 * n_dep_out
        for module, p, gamma, feat, same, diff in zip(
            self.updates, params.update_params, ee_gamma, ee_feat, h_init_same, h_init_diff
        ):

            @functools.partial(jax.vmap, in_axes=0)
            @functools.partial(fwd_lap, argnums=(0, 1, 2, 3, 4))
            def fwd_fn(h, edg, feat, same, diff, s, s_nb):
                return module.apply(p, h, edg, feat, same, diff, s, s_nb)

            # Padding
            def to_pairwise(x):
                x = _normalize(x, next(scale_seq))
                x = jtu.tree_map(lambda x: x.at[ee_idx].get(mode="fill", fill_value=0), x)
                return pad_pairwise_jacobian(x, dep_map, n_dep_out)

            same, diff = to_pairwise(same), to_pairwise(diff)
            gamma = pad_pairwise_jacobian(gamma, dep_map_gamma, n_dep_out)
            feat = pad_pairwise_jacobian(feat, dep_map_gamma, n_dep_out)

            def move_nb_axis(x: FwdLaplArray):
                return FwdLaplArray(
                    x=x.x,
                    jacobian=FwdJacobian(jnp.swapaxes(x.jacobian.data, 1, 2)),
                    laplacian=x.laplacian,
                )

            gamma, feat, same, diff = map(move_nb_axis, (gamma, feat, same, diff))
            # Fwd pass
            h = fwd_fn(h, gamma, feat, same, diff, self.spins, s_nb)
            h = _normalize(h, next(scale_seq))

        @functools.partial(jax.vmap, in_axes=0, out_axes=-2)
        def move_axis(x):
            return x

        return move_axis(h), dep_out

    def get_static_input(
        self, electrons: Electrons, electrons_new: Optional[Electrons] = None, idx_changed: Optional[ElectronIdx] = None
    ) -> StaticInputNewModel[Int]:
        dist_ee, dist_ne = get_full_distance_matrices(electrons, self.R)
        n_ee = jnp.max(jnp.sum(dist_ee < self.cutoff, axis=-1, dtype=jnp.int32))
        n_en = jnp.max(jnp.sum(dist_ne.T < self.cutoff_1el, axis=-1, dtype=jnp.int32))
        return StaticInputNewModel(NrOfNeighbours(ee=n_ee, en=n_en))


@jit(static_argnames=("n_neighbours", "cutoff", "cutoff_1el"))
def get_neighbour_indices(
    r: Electrons,
    R: Nuclei,
    n_neighbours: NrOfNeighbours,
    cutoff: float,
    cutoff_1el: float,
):
    dist_ee, dist_ne = get_full_distance_matrices(r, R)

    def _get_ind_neighbour(dist, max_n_neighbours: int, cutoff, exclude_diagonal=False):
        if exclude_diagonal:
            n_particles = dist.shape[-1]
            dist += jnp.diag(jnp.inf * jnp.ones(n_particles, dist.dtype))
        in_cutoff = dist < cutoff

        # TODO: dynamically assert that n_neighbours <= max_n_neighbours
        n_neighbours = jnp.max(jnp.sum(in_cutoff, axis=-1))  # noqa: F841

        @jax.vmap
        def _get_ind(in_cutoff_):
            indices = jnp.nonzero(in_cutoff_, size=max_n_neighbours, fill_value=NO_NEIGHBOUR)[0]
            return jnp.unique(indices, size=max_n_neighbours, fill_value=NO_NEIGHBOUR)

        return _get_ind(in_cutoff)

    en_idx = _get_ind_neighbour(dist_ne.T, n_neighbours.en, cutoff_1el)
    ee_idx = _get_ind_neighbour(dist_ee, n_neighbours.ee, cutoff, exclude_diagonal=True)
    return en_idx, ee_idx


@jit
def get_all_dependencies(ee_idx: jax.Array):
    """Get the indices of electrons on which each embedding will depend on."""
    n_el = ee_idx.shape[-2]
    batch_dims = ee_idx.shape[:-2]
    self_dependency = jnp.arange(n_el)[:, None]
    self_dependency = jnp.tile(self_dependency, batch_dims + (1, 1))

    get_dep_map_for_all_centers = jax.vmap(jax.vmap(get_dependency_map, in_axes=(0, None)))

    # hinit to h
    deps: Dependency = jnp.concatenate([self_dependency, ee_idx], axis=-1)
    deps_neighbours = ee_idx[..., None]
    dep_map = get_dep_map_for_all_centers(deps_neighbours, deps)
    # Gamma to h
    deps_gamma = jnp.concatenate(
        [
            jnp.tile(self_dependency[..., None, :], (1,) * len(batch_dims) + (1, deps_neighbours.shape[-2], 1)),
            deps_neighbours,
        ],
        axis=-1,
    )
    gamma_dep_map = get_dep_map_for_all_centers(deps_gamma, deps)
    return deps, dep_map, gamma_dep_map
