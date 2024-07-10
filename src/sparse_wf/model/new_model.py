from collections.abc import Iterator
from typing import Callable, Literal, NamedTuple, cast, overload
import flax.linen as nn
from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from jaxtyping import Array, Float, Integer
from sparse_wf.api import ElectronEmb, Electrons, Int, Nuclei, Charges, Parameters
from sparse_wf.jax_utils import jit, nn_vmap, pmap, rng_sequence
from sparse_wf.model.graph_utils import NO_NEIGHBOUR, get_full_distance_matrices, round_to_next_step
from sparse_wf.model.moon import GatedLinearUnit
from sparse_wf.model.utils import (
    DynamicFilterParams,
    PairwiseFilter,
    ScalingParam,
    get_diff_features,
    lecun_normal,
    normalize,
    scale_initializer,
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
        beta = PairwiseFilter(self.cutoff, self.filter_dims[1])(features, dynamic_params.filter)
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

    def setup(self):
        self.edge = EdgeFeatures(self.cutoff, self.filter_dims, self.feature_dim, self.n_envelopes)
        self.glu = GatedLinearUnit(self.feature_dim)
        self.dense_same = nn.Dense(self.feature_dim * self.n_updates, use_bias=False)
        self.dense_diff = nn.Dense(self.feature_dim * self.n_updates, use_bias=False)

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
        return h_init, jnp.split(h_init_same, self.n_updates, -1), jnp.split(h_init_diff, self.n_updates, -1)


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
        r_nb: Float[Array, "n_neighbours dim=3"],
        s: Int,
        s_nb: Integer[Array, " n_neighbours"],
    ):
        features_ee = jax.vmap(get_diff_features, in_axes=(None, 0))(r, r_nb)
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
        gamma_ee_same = nn.Dense(self.feature_dim * self.n_updates, use_bias=False)(beta_ee)
        gamma_ee_diff = nn.Dense(self.feature_dim * self.n_updates, use_bias=False)(beta_ee)
        gamma = jnp.where((s == s_nb)[..., None], gamma_ee_same, gamma_ee_diff)
        return jnp.split(gamma, self.n_updates, -1)


class ElecUpdate(nn.Module):
    @nn.compact
    def __call__(
        self,
        h: Float[Array, " feature_dim"],
        gamma: Float[Array, "n_neighbours feature_dim"],
        nb_same: Float[Array, "n_neighbours feature_dim"],
        nb_diff: Float[Array, "n_neighbours feature_dim"],
        spin: Int,
        spin_nb: Integer[Array, " n_neighbours"],
    ):
        feat_dim = h.shape[-1]
        spin_mask = (spin == spin_nb)[..., None]

        # message passing
        h_nb = jnp.where(spin_mask, nb_same, nb_diff)
        msg = jnp.einsum("...Jd,...Jd->...d", gamma, nn.silu(h_nb + h))

        # update
        h = nn.silu(nn.Dense(feat_dim)(h))

        # combination
        out = nn.silu(nn.Dense(feat_dim)(h + msg))
        out = nn.silu(nn.Dense(feat_dim)(out))

        # Skip connection
        return out + h


class EmbeddingParams(NamedTuple):
    dynamic_params_en: NucleusDependentParams
    init_params: Parameters
    edge_params: Parameters
    update_params: tuple[Parameters, ...]
    scales: tuple[ScalingParam, ...]


class EmbeddingState(NamedTuple): ...


class NrOfNeighbours(NamedTuple):
    ee: int
    en: int


class StaticInput(NamedTuple):
    n_neighbours: NrOfNeighbours

    def to_log_data(self):
        return {
            "static/n_nb_ee": self.n_neighbours.ee,
            "static/n_nb_en": self.n_neighbours.en,
        }


class DefaultList(list):
    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        yield from iterator
        while True:
            yield None


class AppendingList(list):
    def __setattr__(self, name, value):
        self.append(value)


def _get_static(electrons: Array, R: Nuclei, cutoff: float, cutoff_1el: float):
    n_el = electrons.shape[-2]
    n_nuc = len(R)
    dist_ee, dist_ne = get_full_distance_matrices(electrons, R)
    n_ee = jnp.max(jnp.sum(dist_ee < cutoff, axis=-1))
    n_en = jnp.max(jnp.sum(dist_ne.T < cutoff_1el, axis=-1))
    n_neighbours = NrOfNeighbours(
        ee=round_to_next_step(n_ee, 1.1, 1, n_el - 1),  # type: ignore
        en=round_to_next_step(n_en, 1.1, 1, n_nuc),  # type: ignore
    )
    return n_neighbours


get_static_pmapped = pmap(_get_static, in_axes=(0, None, None, None))
get_static_jitted = jit(_get_static)


class Embedding(PyTreeNode):
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
    n_updates: int

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
        nuc_mlp_depth: int,
        pair_mlp_widths: tuple[int, int],
        pair_n_envelopes: int,
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
            nuc_mlp_depth=nuc_mlp_depth,
            n_updates=n_updates,
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

    def init(self, rng: Array, electrons: Array, static: StaticInput):
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
            edge_params=self.edge.init(next(rng_seq), r_dummy, r_nb_dummy, spin_dummy, spin_nb_dummy),
            update_params=tuple(
                upd.init(
                    next(rng_seq),
                    features_dummy,
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

    @overload
    def apply(
        self,
        params: EmbeddingParams,
        electrons: Electrons,
        static: StaticInput,
        return_scales: Literal[False] = False,
        return_state: Literal[False] = False,
    ) -> ElectronEmb: ...

    @overload
    def apply(
        self,
        params: EmbeddingParams,
        electrons: Electrons,
        static: StaticInput,
        return_scales: Literal[True],
        return_state: Literal[False] = False,
    ) -> tuple[ElectronEmb, tuple[ScalingParam, ...]]: ...

    @overload
    def apply(
        self,
        params: EmbeddingParams,
        electrons: Electrons,
        static: StaticInput,
        return_scales: Literal[False],
        return_state: Literal[True],
    ) -> tuple[ElectronEmb, EmbeddingState]: ...

    def apply(
        self,
        params: EmbeddingParams,
        electrons: Array,
        static: StaticInput,
        return_scales: bool = False,
        return_state: bool = False,
    ):
        if return_state:
            raise NotImplementedError("return_state not implemented")

        scale_seq = iter(DefaultList(params.scales))
        new_scales = AppendingList()

        en_idx, ee_idx = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff, self.cutoff_1el)
        R_nb = jnp.array(self.R).at[en_idx].get(mode="fill", fill_value=NO_NEIGHBOUR)
        s_nb = self.spins.at[ee_idx].get(mode="fill", fill_value=0)
        r_nb = electrons.at[ee_idx].get(mode="fill", fill_value=NO_NEIGHBOUR)

        init_fn = jax.vmap(self.elec_init.apply, in_axes=(None, 0, 0, 0))  # type: ignore
        dyn_params = jtu.tree_map(lambda x: x[en_idx], params.dynamic_params_en)
        h_init, h_init_same, h_init_diff = init_fn(params.init_params, electrons, R_nb, dyn_params)  # type: ignore
        h_init, new_scales.h_init = normalize(h_init, next(scale_seq), return_scale=True)
        edge_fn = jax.vmap(self.edge.apply, in_axes=(None, 0, 0, 0, 0))
        edges = cast(tuple[jax.Array, ...], edge_fn(params.edge_params, electrons, r_nb, self.spins, s_nb))

        # update layers
        h = h_init
        for module, p, edg, same, diff in zip(self.updates, params.update_params, edges, h_init_same, h_init_diff):
            fwd_fn = jax.vmap(module.apply, in_axes=(None, 0, 0, 0, 0, 0, 0))  # type: ignore
            same, new_scales.same = normalize(same, next(scale_seq), return_scale=True)
            diff, new_scales.diff = normalize(diff, next(scale_seq), return_scale=True)
            h = fwd_fn(p, h, edg, same[ee_idx], diff[ee_idx], self.spins, s_nb)
            h, new_scales.h = normalize(h, next(scale_seq), return_scale=True)
        if return_scales:
            return h, tuple(new_scales)
        return h

    def get_static_input(self, electrons: Electrons) -> StaticInput:
        if electrons.ndim == 4:
            # [device x local_batch x el x 3] => electrons are split across gpus;
            n_neighbours = get_static_pmapped(electrons, self.R, self.cutoff, self.cutoff_1el)
            # Data is synchronized across all devices, so we can just take the 0-th element
            n_neighbours = [int(x[0]) for x in n_neighbours]
        else:
            n_neighbours = get_static_jitted(electrons, self.R, self.cutoff, self.cutoff_1el)
            n_neighbours = [int(x) for x in n_neighbours]
        n_neighbours = NrOfNeighbours(*n_neighbours)
        return StaticInput(n_neighbours)


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
