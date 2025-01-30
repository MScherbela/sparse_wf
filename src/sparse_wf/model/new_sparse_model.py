# %%
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import PyTreeNode
from sparse_wf.api import (
    ElectronIdx,
    Parameters,
    Int,
    Nuclei,
    Electrons,
    Charges,
    ScalingParam,
    StaticInput,
)
from sparse_wf.static_args import round_with_padding
from sparse_wf.model.utils import (
    GatedLinearUnit,
    get_diff_features,
    PairwiseFilter,
    DynamicFilterParams,
    lecun_normal,
    normalize,
    scale_initializer,
    iter_list_with_pad,
    AppendingList,
)
from sparse_wf.model.graph_utils import NO_NEIGHBOUR, is_neighbour
from sparse_wf.model.sparse_fwd_lap import NodeWithFwdLap, get, SparseMLP
import jax
import numpy as np
from typing import NamedTuple, Generic, TypeVar, Callable, cast
from sparse_wf.jax_utils import fwd_lap, nn_vmap, rng_sequence
import functools
from folx.api import FwdLaplArray, FwdJacobian
from jaxtyping import Float, Array
import jax.tree_util as jtu
from flax import struct

T = TypeVar("T", bound=int | Int)


@struct.dataclass
class StaticArgs(StaticInput, Generic[T]):
    n_pairs_same: T
    n_pairs_diff: T
    n_triplets: T
    n_neighbours_en: T
    n_changed_hout: T
    n_changed_pair_same: T
    n_changed_pair_diff: T
    n_pp_elecs: T

    def get_max(self, n_el, n_up, n_nuc):
        n_dn = n_el - n_up
        max_pairs_same = n_up * (n_up - 1) + n_dn * (n_dn - 1)
        max_pairs_diff = 2 * n_up * n_dn
        max_triplets = n_el * (n_el - 1) ** 2
        return StaticArgs(
            max_pairs_same,
            max_pairs_diff,
            max_triplets,
            n_nuc,
            n_el,
            max_pairs_same,
            max_pairs_diff,
            n_el * n_nuc * 12,  # TODO: This 12 is the maximum number of integration points for the pseudopotential
        )

    # @override
    def round_with_padding(self, padding_factor, n_el, n_up, n_nuc):
        max_values = self.get_max(n_el, n_up, n_nuc)
        return jtu.tree_map(
            lambda x, max_x: round_with_padding(x, padding_factor, max_x),
            self,
            max_values,
        )


def contract(h_residual, h_ct, h_nb, Gamma, edges, idx_ctr, idx_nb):
    pair_msg = Gamma * nn.silu(edges + get(h_ct, idx_ctr) + get(h_nb, idx_nb))
    h_out = h_residual.at[idx_ctr].add(pair_msg, mode="drop")
    return h_out


def contract_with_fwd_lap(
    h_residual: NodeWithFwdLap,
    h_ct: FwdLaplArray,
    h_nb: FwdLaplArray,
    Gamma: FwdLaplArray,
    edges: FwdLaplArray,
    idx_ctr,
    idx_nb,
    idx_jac,
) -> NodeWithFwdLap:
    n_el, feature_dim = h_ct.x.shape
    padding = jnp.zeros([n_el, 3, feature_dim], dtype=h_ct.x.dtype)
    h_center = FwdLaplArray(
        h_ct.x,
        FwdJacobian(jnp.concatenate([h_ct.jacobian.data, padding], axis=1)),
        h_ct.laplacian,
    )
    h_neighb = FwdLaplArray(
        h_nb.x,
        FwdJacobian(jnp.concatenate([padding, h_nb.jacobian.data], axis=1)),
        h_nb.laplacian,
    )

    # Compute the message for each pair of nodes using folx
    def get_msg_pair(gamma, edge, i, j):
        return fwd_lap(lambda G, e, h_i, h_j: G * nn.silu(e + h_i + h_j))(
            gamma, edge, get(h_center, i), get(h_neighb, j)
        )

    msg_pair = jax.vmap(get_msg_pair)(Gamma, edges, idx_ctr, idx_nb)  # vmap over pairs

    # Aggregate the messages to each center node
    h_out = h_residual.x.at[idx_ctr].add(msg_pair.x)
    lap_out = h_residual.lap.at[idx_ctr].add(msg_pair.laplacian)
    J_out = h_residual.jac.at[idx_ctr].add(msg_pair.jacobian.data[:, :3])  # dmsg/dx_ctr
    J_out = J_out.at[idx_jac].add(msg_pair.jacobian.data[:, 3:])  # dmsg/dx_nb
    return NodeWithFwdLap(h_out, J_out, lap_out, h_residual.idx_ctr, h_residual.idx_dep)


class NucleusDependentParams(NamedTuple):
    filter: DynamicFilterParams
    nuc_embedding: jax.Array | None


class ElecNucEdge(nn.Module):
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
        beta = PairwiseFilter(self.filter_dims)(self.cutoff, features, dynamic_params.filter)
        gamma = nn.Dense(self.feature_dim, use_bias=False)(beta)
        scaled_features = features / features[..., :1] * jnp.log1p(features[..., :1])
        edge_embedding = nn.Dense(self.feature_dim, use_bias=False)(scaled_features) + dynamic_params.nuc_embedding
        return gamma, edge_embedding


class ElecElecEdges(nn.Module):
    filter_dims: tuple[int, int]
    feature_dim: int
    n_envelopes: int
    n_updates: int

    @nn.compact
    def __call__(
        self,
        cutoff,
        r: Float[Array, " dim=3"],
        r_nb: Float[Array, " dim=3"],
    ):
        features_ee = get_diff_features(r, r_nb)
        beta = PairwiseFilter(self.filter_dims, name="beta_ee")(cutoff, features_ee)
        gamma = nn.Dense(self.feature_dim * self.n_updates, use_bias=False)(beta)

        # logarithmic rescaling
        inp_ee = features_ee / features_ee[..., :1] * jnp.log1p(features_ee[..., :1])
        feat_ee = nn.Dense(self.feature_dim * self.n_updates)(inp_ee)

        return jnp.split(gamma, self.n_updates, -1), jnp.split(feat_ee, self.n_updates, -1)


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
        n_out = self.n_updates * 4 + 1
        edge_feat = ElecNucEdge(self.cutoff, self.filter_dims, self.feature_dim, self.n_envelopes)
        Gamma, features = nn_vmap(edge_feat, in_axes=(None, 0, 0))(r, R_nb, dynamic_params)  # vmap over nuclei
        h_init = jnp.einsum("...Jd,...Jd->...d", Gamma, features)
        h_init = nn.LayerNorm()(h_init)
        h_init = GatedLinearUnit(self.feature_dim)(h_init)
        h_init = self.activation(nn.Dense(self.feature_dim)(h_init))
        # let's parallize this will by having all operations in one go
        h_out = jnp.split(nn.Dense(self.feature_dim * n_out)(h_init), n_out, axis=-1)
        h_msg = [h_out[offset : offset + self.n_updates] for offset in range(1, n_out, self.n_updates)]
        return h_out[0], h_msg


class EmbeddingParams(NamedTuple):
    dynamic_params_en: NucleusDependentParams
    elec_init: Parameters
    edge_same: Parameters
    edge_diff: Parameters
    updates: Parameters
    scales: tuple[ScalingParam, ...]
    cutoff_param: jax.Array


class EmbeddingState(NamedTuple):
    electrons: jax.Array
    h0: jax.Array
    msg_ct_same: tuple[jax.Array, ...]
    msg_ct_diff: tuple[jax.Array, ...]
    msg_nb_same: tuple[jax.Array, ...]
    msg_nb_diff: tuple[jax.Array, ...]
    h_out: jax.Array


def get_pair_indices(r, n_up, cutoff, n_pairs_same: int, n_pairs_diff: int):
    assert r.ndim == 2
    n_el = r.shape[0]
    n_dn = n_el - n_up
    spin = np.concatenate([np.zeros(n_up, bool), np.ones(n_dn, bool)])
    dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
    in_cutoff = dist < cutoff
    is_same_spin = spin[:, None] == spin[None, :]

    pair_indices_same = jnp.where(in_cutoff & is_same_spin, size=n_pairs_same, fill_value=NO_NEIGHBOUR)
    pair_indices_diff = jnp.where(in_cutoff & ~is_same_spin, size=n_pairs_diff, fill_value=NO_NEIGHBOUR)
    idx_ct, idx_nb = jnp.where(in_cutoff, size=n_pairs_same + n_pairs_diff, fill_value=NO_NEIGHBOUR)
    jac_indices = (
        jnp.concatenate([np.arange(n_el), idx_ct]),
        jnp.concatenate([np.arange(n_el), idx_nb]),
    )

    search_key = idx_ct * n_el + idx_nb
    jac_idx_same = jnp.searchsorted(search_key, pair_indices_same[0] * n_el + pair_indices_same[1]) + n_el
    jac_idx_diff = jnp.searchsorted(search_key, pair_indices_diff[0] * n_el + pair_indices_diff[1]) + n_el
    return (
        (*pair_indices_same, jac_idx_same),
        (*pair_indices_diff, jac_idx_diff),
        jac_indices,
    )


def get_distinct_triplet_indices(r, cutoff, n_triplets_max: int):
    n_el = r.shape[0]
    dist = jnp.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist = dist.at[jnp.arange(n_el), jnp.arange(n_el)].set(jnp.inf)
    in_cutoff = dist < cutoff
    is_triplet = in_cutoff[:, None, :] & in_cutoff[None, :, :]
    return jnp.where(is_triplet, size=n_triplets_max, fill_value=NO_NEIGHBOUR)


def get_pair_indices_from_triplets(pair_indices, triplet_indices, n_el):
    pair_search_keys = pair_indices[0] * n_el + pair_indices[1]
    pair_in = jnp.searchsorted(pair_search_keys, triplet_indices[0] * n_el + triplet_indices[2])
    pair_kn = jnp.searchsorted(pair_search_keys, triplet_indices[1] * n_el + triplet_indices[2])
    return pair_in, pair_kn


class NewSparseEmbedding(PyTreeNode):
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
    edge_same: ElecElecEdges
    edge_diff: ElecElecEdges
    updates: tuple[SparseMLP, ...]

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
            edge_same=ElecElecEdges(pair_mlp_widths, feature_dim, pair_n_envelopes, n_updates),
            edge_diff=ElecElecEdges(pair_mlp_widths, feature_dim, pair_n_envelopes, n_updates),
            updates=tuple([SparseMLP([feature_dim] * 2) for _ in range(n_updates)]),
        )

    @property
    def n_nuclei(self):
        return len(self.R)

    @property
    def spins(self):
        return jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_electrons - self.n_up)]).astype(jnp.float32)

    def init(self, rng: Array, electrons: Array, static: StaticArgs):
        dtype = electrons.dtype
        rng_seq = iter(rng_sequence(rng))
        r_dummy = jnp.zeros([3], dtype)
        r_nb_dummy = jnp.ones([3], dtype)
        R_nb_dummy = jnp.ones([1, 3], dtype)
        features_dummy = jnp.zeros([self.feature_dim], dtype)
        dummy_dyn_param = self._init_nuc_dependant_params(next(rng_seq), n_nuclei=1)

        params = EmbeddingParams(
            dynamic_params_en=self._init_nuc_dependant_params(next(rng_seq)),
            elec_init=self.elec_init.init(next(rng_seq), r_dummy, R_nb_dummy, dummy_dyn_param),
            edge_same=self.edge_same.init(next(rng_seq), self.cutoff, r_dummy, r_nb_dummy),
            edge_diff=self.edge_diff.init(next(rng_seq), self.cutoff, r_dummy, r_nb_dummy),
            updates=tuple(
                upd.init(
                    next(rng_seq),
                    features_dummy,
                )
                for upd in self.updates
            ),
            scales=(),
            cutoff_param=jnp.ones((), jnp.float32) * 10.0,
        )
        _, scales = self.apply(params, electrons, static, return_scales=True)
        params = params._replace(scales=scales)
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

    def get_static_input(self, electrons_old, electrons_new=None, idx_changed=None) -> StaticArgs[Int]:
        n_elec = electrons_old.shape[0]
        electrons = electrons_old if (electrons_new is None) else electrons_new
        assert electrons.ndim == 2

        dist_en = jnp.linalg.norm(electrons[:, None, :] - self.R, axis=-1)
        n_neighbours_en = jnp.max(jnp.sum(dist_en < self.cutoff_1el, axis=-1, dtype=jnp.int32), axis=0)

        is_same_spin = self.spins[:, None] == self.spins[None, :]
        in_cutoff = is_neighbour(electrons, self.cutoff)
        n_pairs_same = jnp.sum(in_cutoff & is_same_spin, dtype=jnp.int32)
        n_pairs_diff = jnp.sum(in_cutoff & ~is_same_spin, dtype=jnp.int32)
        n_triplets = jnp.sum(in_cutoff[:, None, :] & in_cutoff[None, :, :], dtype=jnp.int32)

        if electrons_new is not None:
            in_cutoff_old = is_neighbour(electrons_old, self.cutoff)
            is_affected_hout = jnp.any(in_cutoff[:, idx_changed] | in_cutoff_old[:, idx_changed], axis=-1)
            is_affected_hout = is_affected_hout.at[idx_changed].set(True)
            is_required_pair = in_cutoff & is_affected_hout[:, None]
            n_changed_hout = jnp.sum(is_affected_hout, dtype=jnp.int32)
            n_changed_pair_same = jnp.sum(is_required_pair & is_same_spin, dtype=jnp.int32)
            n_changed_pair_diff = jnp.sum(is_required_pair & ~is_same_spin, dtype=jnp.int32)
        else:
            n_changed_hout, n_changed_pair_same, n_changed_pair_diff = 0, 0, 0  # type: ignore
        return StaticArgs(
            n_pairs_same,
            n_pairs_diff,
            n_triplets,
            n_neighbours_en,
            n_changed_hout,
            n_changed_pair_same,
            n_changed_pair_diff,
            n_elec,
        )

    def get_low_rank_pair_indices(
        self,
        electrons: Electrons,
        electrons_new: Electrons,
        idx_changed: ElectronIdx,
        n_changed_hout: int,
        n_changed_pair_same: int,
        n_changed_pair_diff: int,
    ):
        # Get the indices of all electrons, for which h_out changes, and which must thus be recomputed
        in_cutoff_old = is_neighbour(electrons, self.cutoff)
        in_cutoff_new = is_neighbour(electrons_new, self.cutoff)
        is_affected_hout = jnp.any(in_cutoff_old[:, idx_changed] | in_cutoff_new[:, idx_changed], axis=-1)
        is_affected_hout = is_affected_hout.at[idx_changed].set(True)
        idx_changed_hout = jnp.where(is_affected_hout, size=n_changed_hout, fill_value=NO_NEIGHBOUR)[0]

        # Get all pairs of electrons, where the center electron needs to be recomputed
        is_required_pair = in_cutoff_new[idx_changed_hout, :]  # => [n_changsed_hout x n_el]
        n_changed_pairs_total = n_changed_pair_same + n_changed_pair_diff
        idx_pair_ct, idx_pair_nb = jnp.where(is_required_pair, size=n_changed_pairs_total, fill_value=NO_NEIGHBOUR)

        # Divide the pairs into pairs with same and different spin
        is_same_spin = (idx_changed_hout[idx_pair_ct] < self.n_up) == (idx_pair_nb < self.n_up)
        pair_idx_same = jnp.where(is_same_spin, size=n_changed_pair_same, fill_value=NO_NEIGHBOUR)
        pair_idx_diff = jnp.where(~is_same_spin, size=n_changed_pair_diff, fill_value=NO_NEIGHBOUR)

        # Get the center and neighbour indices for the pairs
        get_padded = functools.partial(get, fill_value=NO_NEIGHBOUR)
        # Getting the centers requires 1 more lookup, because we need to convert from the idx of changed_hout to the idx of all electrons
        # This is a result of is_required_pair having a non-square shape
        pair_idx_same_ct = get_padded(idx_changed_hout, get_padded(idx_pair_ct, pair_idx_same))
        pair_idx_diff_ct = get_padded(idx_changed_hout, get_padded(idx_pair_ct, pair_idx_diff))
        pair_idx_same_nb = get_padded(idx_pair_nb, pair_idx_same)
        pair_idx_diff_nb = get_padded(idx_pair_nb, pair_idx_diff)

        return idx_changed_hout, (pair_idx_same_ct, pair_idx_same_nb), (pair_idx_diff_ct, pair_idx_diff_nb)

    def get_neighbouring_nuclei(self, r, dynamic_params_en, n_neighbours_en: int):
        assert r.shape == (3,)
        dist_en = jnp.linalg.norm(r - self.R, axis=-1)
        in_cutoff = dist_en < self.cutoff_1el
        idx_en = jnp.where(in_cutoff, size=n_neighbours_en, fill_value=NO_NEIGHBOUR)[0]
        return get(jnp.array(self.R, r.dtype), idx_en, 1e6), get(dynamic_params_en, idx_en)

    def apply(
        self,
        params: EmbeddingParams,
        electrons: Electrons,
        static: StaticArgs[int],
        return_scales: bool = False,
        return_state: bool = False,
    ):
        assert electrons.ndim == 2
        scale_seq = iter_list_with_pad(params.scales)
        new_scales = AppendingList()

        (idx_ct_same, idx_nb_same, _), (idx_ct_diff, idx_nb_diff, _), _ = get_pair_indices(
            electrons,
            self.n_up,
            self.cutoff,
            static.n_pairs_same,
            static.n_pairs_diff,
        )
        R_nb_en, nuc_params_en = jax.vmap(
            lambda r: self.get_neighbouring_nuclei(r, params.dynamic_params_en, static.n_neighbours_en)
        )(electrons)

        cutoff = self.cutoff * nn.sigmoid(params.cutoff_param)

        h0_fn = functools.partial(self.elec_init.apply, params.elec_init)
        edge_fn_same = functools.partial(self.edge_same.apply, params.edge_same, cutoff)
        edge_fn_diff = functools.partial(self.edge_diff.apply, params.edge_diff, cutoff)

        h0, (msg_ct_same, msg_ct_diff, msg_nb_same, msg_nb_diff) = cast(
            tuple[
                jax.Array,
                tuple[list[jax.Array], list[jax.Array], list[jax.Array], list[jax.Array]],
            ],
            jax.vmap(h0_fn, out_axes=-2)(electrons, R_nb_en, nuc_params_en),
        )
        Gamma_same, edge_same = cast(
            tuple[list[jax.Array], list[jax.Array]],
            jax.vmap(edge_fn_same)(
                get(electrons, idx_ct_same, 0.0),
                get(electrons, idx_nb_same, self.cutoff),
            ),
        )
        Gamma_diff, edge_diff = cast(
            tuple[list[jax.Array], list[jax.Array]],
            jax.vmap(edge_fn_diff)(
                get(electrons, idx_ct_diff, 0.0),
                get(electrons, idx_nb_diff, self.cutoff),
            ),
        )

        h = h0
        for n in range(self.n_updates):
            h = contract(
                h,
                msg_ct_same[n],
                msg_nb_same[n],
                Gamma_same[n],
                edge_same[n],
                idx_ct_same,
                idx_nb_same,
            )
            h = contract(
                h,
                msg_ct_diff[n],
                msg_nb_diff[n],
                Gamma_diff[n],
                edge_diff[n],
                idx_ct_diff,
                idx_nb_diff,
            )
            h, new_scales.add = normalize(h, next(scale_seq), return_scale=True)
            h = cast(jax.Array, self.updates[n].apply(params.updates[n], h))
            h, new_scales.add = normalize(h, next(scale_seq), return_scale=True)

        if return_scales:
            return h, tuple(new_scales)
        if return_state:
            return h, EmbeddingState(
                electrons,
                h0,
                tuple(msg_ct_same),
                tuple(msg_ct_diff),
                tuple(msg_nb_same),
                tuple(msg_nb_diff),
                h,
            )
        return h

    def low_rank_update(
        self,
        params: EmbeddingParams,
        electrons: Electrons,
        idx_changed_el: ElectronIdx,
        static: StaticArgs,
        state: EmbeddingState,
    ):
        assert electrons.ndim == 2
        scale_seq = iter(params.scales)

        (
            idx_changed_hout,
            (idx_ct_same, idx_nb_same),
            (idx_ct_diff, idx_nb_diff),
        ) = self.get_low_rank_pair_indices(
            state.electrons,
            electrons,
            idx_changed_el,
            static.n_changed_hout,
            static.n_changed_pair_same,
            static.n_changed_pair_diff,
        )

        # Get h0, which only depends on a single electron
        R_nb_en, nuc_params_en = jax.vmap(
            lambda r: self.get_neighbouring_nuclei(r, params.dynamic_params_en, static.n_neighbours_en)
        )(electrons[idx_changed_el])
        h0_fn = functools.partial(self.elec_init.apply, params.elec_init)
        h0, (msg_ct_same, msg_ct_diff, msg_nb_same, msg_nb_diff) = jax.vmap(h0_fn)(
            electrons[idx_changed_el], R_nb_en, nuc_params_en
        )  # type: ignore
        h0 = state.h0.at[idx_changed_el].set(h0)
        msg_ct_same = [msg_state.at[idx_changed_el].set(msg) for msg_state, msg in zip(state.msg_ct_same, msg_ct_same)]
        msg_ct_diff = [msg_state.at[idx_changed_el].set(msg) for msg_state, msg in zip(state.msg_ct_diff, msg_ct_diff)]
        msg_nb_same = [msg_state.at[idx_changed_el].set(msg) for msg_state, msg in zip(state.msg_nb_same, msg_nb_same)]
        msg_nb_diff = [msg_state.at[idx_changed_el].set(msg) for msg_state, msg in zip(state.msg_nb_diff, msg_nb_diff)]

        # Get all pairs, which are required to update h_out
        cutoff = self.cutoff * nn.sigmoid(params.cutoff_param)
        edge_fn_same = functools.partial(self.edge_same.apply, params.edge_same, cutoff)
        edge_fn_diff = functools.partial(self.edge_diff.apply, params.edge_diff, cutoff)
        Gamma_same, edge_same = cast(
            tuple[list[jax.Array], list[jax.Array]],
            jax.vmap(edge_fn_same)(
                get(electrons, idx_ct_same, 0.0),
                get(electrons, idx_nb_same, self.cutoff),
            ),
        )
        Gamma_diff, edge_diff = cast(
            tuple[list[jax.Array], list[jax.Array]],
            jax.vmap(edge_fn_diff)(
                get(electrons, idx_ct_diff, 0.0),
                get(electrons, idx_nb_diff, self.cutoff),
            ),
        )

        # Loop over layers;
        h = state.h_out.at[idx_changed_hout].set(get(h0, idx_changed_hout))
        for n in range(self.n_updates):
            h = contract(
                h,
                msg_ct_same[n],
                msg_nb_same[n],
                Gamma_same[n],
                edge_same[n],
                idx_ct_same,
                idx_nb_same,
            )
            h = contract(
                h,
                msg_ct_diff[n],
                msg_nb_diff[n],
                Gamma_diff[n],
                edge_diff[n],
                idx_ct_diff,
                idx_nb_diff,
            )
            h_update = get(h, idx_changed_hout) * next(scale_seq)
            h_update = self.updates[n].apply(params.updates[n], h_update)
            h_update = h_update * next(scale_seq)
            h = h.at[idx_changed_hout].set(h_update)

        return (
            h,
            idx_changed_hout,
            EmbeddingState(
                electrons,
                h0,
                tuple(msg_ct_same),
                tuple(msg_ct_diff),
                tuple(msg_nb_same),
                tuple(msg_nb_diff),
                h,
            ),
        )

    def apply_with_fwd_lap(self, params: EmbeddingParams, electrons: Electrons, static: StaticArgs[int]):
        scale_seq = iter(params.scales)
        (
            (idx_ct_same, idx_nb_same, idx_jac_same),
            (idx_ct_diff, idx_nb_diff, idx_jac_diff),
            (idx_jac_ctr, idx_jac_dep),
        ) = get_pair_indices(electrons, self.n_up, self.cutoff, static.n_pairs_same, static.n_pairs_diff)
        n_pairs = static.n_pairs_same + static.n_pairs_diff
        R_nb_en, nuc_params_en = jax.vmap(
            lambda r: self.get_neighbouring_nuclei(r, params.dynamic_params_en, static.n_neighbours_en)
        )(electrons)

        h0_fn = functools.partial(self.elec_init.apply, params.elec_init)
        cutoff = self.cutoff * nn.sigmoid(params.cutoff_param)
        edge_fn_same = functools.partial(self.edge_same.apply, params.edge_same, cutoff)
        edge_fn_diff = functools.partial(self.edge_diff.apply, params.edge_diff, cutoff)

        h0, (h_ct_same, h_ct_diff, h_nb_same, h_nb_diff) = jax.vmap(fwd_lap(h0_fn, argnums=0))(
            electrons, R_nb_en, nuc_params_en
        )
        Gamma_same, edge_same = jax.vmap(fwd_lap(edge_fn_same))(
            get(electrons, idx_ct_same, 0.0), get(electrons, idx_nb_same, 1.0)
        )
        Gamma_diff, edge_diff = jax.vmap(fwd_lap(edge_fn_diff))(
            get(electrons, idx_ct_diff, 0.0), get(electrons, idx_nb_diff, 1.0)
        )
        jac_h0_padded = jnp.concatenate(
            [
                h0.jacobian.data.reshape([self.n_electrons, 3, self.feature_dim]),
                jnp.zeros([n_pairs, 3, self.feature_dim], dtype=h0.jacobian.data.dtype),
            ]
        )
        h = NodeWithFwdLap(
            h0.x,
            jac_h0_padded,
            h0.laplacian,
            idx_jac_ctr,
            idx_jac_dep,
        )
        for n in range(self.n_updates):
            h = contract_with_fwd_lap(
                h,
                h_ct_same[n],
                h_nb_same[n],
                Gamma_same[n],
                edge_same[n],
                idx_ct_same,
                idx_nb_same,
                idx_jac_same,
            )
            h = contract_with_fwd_lap(
                h,
                h_ct_diff[n],
                h_nb_diff[n],
                Gamma_diff[n],
                edge_diff[n],
                idx_ct_diff,
                idx_nb_diff,
                idx_jac_diff,
            )
            h = h * next(scale_seq)  # type: ignore
            h = self.updates[n].apply(params.updates[n], h)  # type: ignore
            h = h * next(scale_seq)  # type: ignore

        return h
