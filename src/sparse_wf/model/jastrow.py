import functools
from typing import Literal, Sequence, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from folx.api import FwdJacobian, FwdLaplArray
from jaxtyping import Array, Float

from sparse_wf.api import ElectronIdx, Electrons, SignedLogAmplitude
from sparse_wf.jax_utils import fwd_lap
from sparse_wf.model.graph_utils import NO_NEIGHBOUR, sum_fwd_lap
from sparse_wf.model.utils import MLP, get_dist_same_diff, get_logscaled_diff_features
from sparse_wf.model.sparse_fwd_lap import ElementWise, Linear, SparseMLP
from sparse_wf.tree_utils import tree_add
from sparse_wf.model.sparse_fwd_lap import NodeWithFwdLap
from flax.struct import PyTreeNode
import numpy as np


class JastrowState(PyTreeNode):
    one_el: jax.Array
    two_el_same: jax.Array
    two_el_diff: jax.Array
    attention: jax.Array
    values: jax.Array


class YukawaJastrow(nn.Module):
    n_up: int

    @nn.compact
    def __call__(self, electrons: Electrons) -> Float[Array, " *batch_dims"]:
        A_same = jax.nn.softplus(self.param("A_same", nn.initializers.ones, (), jnp.float32))
        F_same = jnp.sqrt(2 * A_same)
        A_diff = jax.nn.softplus(self.param("A_diff", nn.initializers.ones, (), jnp.float32))
        F_diff = jnp.sqrt(2 * A_diff)

        dist_same, dist_diff = get_dist_same_diff(electrons, self.n_up)
        # Supposed to be a minus in front of the whole jastrow, but I use expm1 instead of 1-exp, so it should work out
        u_same = A_same * jnp.sum(1 / dist_same * jnp.expm1(-dist_same / F_same), axis=-1)
        u_diff = A_diff * jnp.sum(1 / dist_diff * jnp.expm1(-dist_diff / F_diff), axis=-1)

        return u_same + u_diff


class ElElCusp(nn.Module):
    n_up: int

    @nn.compact
    def __call__(self, electrons: Electrons) -> Float[Array, " *batch_dims"]:
        dist_same, dist_diff = get_dist_same_diff(electrons, self.n_up)

        alpha_same = self.param("alpha_same", nn.initializers.ones, (), jnp.float32)
        alpha_diff = self.param("alpha_diff", nn.initializers.ones, (), jnp.float32)
        factor_same = self.param("factor_same", nn.initializers.zeros, (), jnp.float32)
        factor_diff = self.param("factor_diff", nn.initializers.zeros, (), jnp.float32)
        # factor_same, factor_diff = -0.25, -0.5

        cusp_same = jnp.sum(alpha_same**2 / (alpha_same + dist_same), axis=-1)
        cusp_diff = jnp.sum(alpha_diff**2 / (alpha_diff + dist_diff), axis=-1)

        return factor_same * cusp_same + factor_diff * cusp_diff


class GlobalAttentionJastrow(nn.Module):
    n_up: int
    n_register: int
    register_dim: int
    out_dims: Sequence[int]

    def setup(self):
        self.register_keys = self.param(
            "register_keys",
            nn.initializers.normal(1),
            (self.n_register, self.register_dim),
        )
        self.Q_W = Linear(self.n_register * self.register_dim)
        self.V_W = Linear(self.n_register * self.register_dim)
        self.out = MLP([*self.out_dims, 2])
        self.scale = self.param("scale", nn.initializers.zeros, (2,), jnp.float32)
        self.bias = self.param("bias", nn.initializers.ones, (1,), jnp.float32)

    def attention_and_values(self, h: Float[Array, "... feature_dim"]):
        queries = self.Q_W(h).reshape(*h.shape[:-1], self.n_register, self.register_dim)
        values = self.V_W(h).reshape(*h.shape[:-1], self.n_register, self.register_dim)

        attention = (queries * self.register_keys).sum(-1)
        attention = ElementWise(lambda x: jnp.exp(-x / jnp.sqrt(self.register_dim)))(attention)
        return attention, attention[..., None] * values  # attention is ... x reg, values is ... x reg x dim

    def readout(
        self,
        normalizer: Float[Array, "n_nodes n_reg"],
        values: Float[Array, "n_nodes n_reg feature_dim"],
    ):
        values = (values / normalizer[..., None]).reshape(-1)
        jastrow = self.out(values)
        return jastrow * self.scale + jnp.concatenate([jnp.zeros(1, dtype=self.bias.dtype), self.bias])

    @staticmethod
    def contract(
        attention: Float[Array, "n_nodes n_reg"],
        values: Float[Array, "n_nodes n_reg feature_dim"],
        n_up: int,
        dependencies=None,
    ):
        n_el = attention.shape[-2]
        if isinstance(attention, FwdLaplArray):
            # sparse folx fwd
            up_norm = sum_fwd_lap(
                jtu.tree_map(lambda x: x[..., :n_up, :], attention),
                dependencies[:n_up],
                n_el,
            )
            down_norm = sum_fwd_lap(
                jtu.tree_map(lambda x: x[..., n_up:, :], attention),
                dependencies[n_up:],
                n_el,
            )
            up_values = sum_fwd_lap(
                jtu.tree_map(lambda x: x[..., :n_up, :, :], values),
                dependencies[:n_up],
                n_el,
            )
            down_values = sum_fwd_lap(
                jtu.tree_map(lambda x: x[..., n_up:, :, :], values),
                dependencies[n_up:],
                n_el,
            )
        elif isinstance(attention, NodeWithFwdLap):
            # flattened custom fwd lapl
            up_norm = attention.sum_from_to(0, n_up)
            down_norm = attention.sum_from_to(n_up, n_el)
            up_values = values.sum_from_to(0, n_up)
            down_values = values.sum_from_to(n_up, n_el)
        else:
            # regular fwd
            up_norm, down_norm = attention[:n_up].sum(0), attention[n_up:].sum(0)
            up_values, down_values = values[:n_up].sum(0), values[n_up:].sum(0)

        def stack(x, y):
            return jnp.stack([x, y], axis=0)

        if isinstance(up_norm, FwdLaplArray):
            stack = fwd_lap(stack, argnums=(0, 1))

        norm = stack(up_norm, down_norm)
        values = stack(up_values, down_values)
        return norm, values

    def __call__(self, h: Float[Array, "n_nodes feature_dim"]):
        attention, values = self.attention_and_values(h)
        return self.readout(*self.contract(attention, values, self.n_up))


def get_all_pair_indices(n_el: int, n_up: int):
    idx_grids = np.meshgrid(np.arange(n_el), np.arange(n_el), indexing="ij")
    spin = np.concatenate([np.zeros(n_up, dtype=int), np.ones(n_el - n_up, dtype=int)])
    indices = np.stack([idx.flatten() for idx in idx_grids], axis=0)
    indices = indices[:, indices[0] != indices[1]]
    is_same = spin[indices[0]] == spin[indices[1]]
    return indices[:, is_same], indices[:, ~is_same]


def get_changed_pair_indices(n_el: int, n_up: int, idx_changed: ElectronIdx):
    n_changes = len(idx_changed)
    n_dn = n_el - n_up
    (idx_ct_same, idx_nb_same), (idx_ct_diff, idx_nb_diff) = get_all_pair_indices(n_el, n_up)
    is_changed_same = jnp.any(idx_ct_same[:, None] == idx_changed, axis=-1) | jnp.any(
        idx_nb_same[:, None] == idx_changed, axis=-1
    )
    is_changed_diff = jnp.any(idx_ct_diff[:, None] == idx_changed, axis=-1) | jnp.any(
        idx_nb_diff[:, None] == idx_changed, axis=-1
    )
    # TODO: this is not tight if n_up != n_dn, but it's not a big deal.
    # To make it tight, one would need to know the number of up and down electrons that are changed, which requires static args
    # n_pairs_same = n_changes * (max(n_up, n_dn) - 1)
    n_pairs_same = n_changes * (2 * max(n_up, n_dn) - 1)
    n_pairs_diff = n_changes * 2 * max(n_up, n_dn)
    n_pairs_same = min(n_pairs_same, len(idx_ct_same))
    n_pairs_diff = min(n_pairs_diff, len(idx_ct_diff))
    idx_pair_same = jnp.nonzero(is_changed_same, size=n_pairs_same, fill_value=NO_NEIGHBOUR)
    idx_pair_diff = jnp.nonzero(is_changed_diff, size=n_pairs_diff, fill_value=NO_NEIGHBOUR)
    return idx_pair_same, idx_pair_diff


class Jastrow(nn.Module):
    n_up: int
    e_e_cusps: Literal["none", "psiformer", "yukawa"]
    use_e_e_mlp: bool
    use_log_jastrow: bool
    use_mlp_jastrow: bool
    mlp_depth: int
    mlp_width: int
    sparse_embedding: bool = False
    use_attention: bool = False
    attention_heads: int = 16
    attention_dim: int = 8

    def setup(self):
        if self.e_e_cusps == "none":
            self.pairwise_cusps = None
        elif self.e_e_cusps == "psiformer":
            self.pairwise_cusps = ElElCusp(self.n_up)
        elif self.e_e_cusps == "yukawa":
            self.pairwise_cusps = YukawaJastrow(self.n_up)
        else:
            raise ValueError(f"Unknown e_e_cusps: {self.e_e_cusps}")

        if self.use_mlp_jastrow or self.use_log_jastrow:
            if self.sparse_embedding:
                self.mlp = MLP([self.mlp_width] * self.mlp_depth + [2], activate_final=False)
            else:
                self.mlp = SparseMLP([self.mlp_width] * self.mlp_depth + [2])
            self.mlp_scale = self.param("mlp_scale", nn.initializers.zeros, (2,), jnp.float32)
            if self.use_log_jastrow:
                self.log_bias = self.param("log_bias", nn.initializers.ones, (), jnp.float32)
        else:
            self.mlp = None

        if self.use_attention:
            self.att = GlobalAttentionJastrow(
                self.n_up,
                self.attention_heads,
                self.attention_dim,
                (self.mlp_width,) * self.mlp_depth,
            )
        else:
            self.att = None
        if self.use_e_e_mlp:
            self.e_e_mlp_same = MLP(
                [self.mlp_width] * self.mlp_depth + [1],
                activate_final=False,
                output_bias=False,
            )
            self.e_e_mlp_diff = MLP(
                [self.mlp_width] * self.mlp_depth + [1],
                activate_final=False,
                output_bias=False,
            )
            self.e_e_mlp_scale_same = self.param("e_e_mlp_scale_same", nn.initializers.zeros, (), jnp.float32)
            self.e_e_mlp_scale_diff = self.param("e_e_mlp_scale_diff", nn.initializers.zeros, (), jnp.float32)
        else:
            self.e_e_mlp_same, self.e_e_mlp_diff = None, None

    def _get_all_electron_pairs(self, electrons):
        n_el = electrons.shape[-2]
        pair_indices = get_all_pair_indices(n_el, self.n_up)
        return jtu.tree_map(lambda idx: electrons[idx], pair_indices)

    def _apply_pairwise_mlp_same(self, r1: Float[Array, " dim=3"], r2: Float[Array, " dim=3"]) -> jax.Array:
        features = get_logscaled_diff_features(r1 - r2)
        return self.e_e_mlp_same(features) * self.e_e_mlp_scale_same

    def _apply_pairwise_mlp_diff(self, r1: Float[Array, " dim=3"], r2: Float[Array, " dim=3"]) -> jax.Array:
        features = get_logscaled_diff_features(r1 - r2)
        return self.e_e_mlp_diff(features) * self.e_e_mlp_scale_diff

    def __call__(
        self,
        electrons: Electrons,
        embeddings: jax.Array,
        return_state: bool = False,
    ) -> SignedLogAmplitude | tuple[SignedLogAmplitude, JastrowState]:
        sign = jnp.ones((), electrons.dtype)
        logpsi = jnp.zeros([], electrons.dtype)
        if self.pairwise_cusps:
            logpsi += self.pairwise_cusps(electrons)
        if self.use_e_e_mlp:
            (r1_same, r2_same), (r1_diff, r2_diff) = self._get_all_electron_pairs(electrons)
            jastrow_same = self._apply_pairwise_mlp_same(r1_same, r2_same)
            jastrow_diff = self._apply_pairwise_mlp_diff(r1_diff, r2_diff)
            logpsi += jnp.sum(jastrow_same) + jnp.sum(jastrow_diff)
        else:
            jastrow_same, jastrow_diff = (
                jnp.zeros([], electrons.dtype),
                jnp.zeros([], electrons.dtype),
            )

        jastrows_after_sum = jnp.zeros((2,), electrons.dtype)
        # Attention Jastrow
        if self.use_attention:
            attention, values = self._apply_attention_and_values(embeddings)
            jastrows_after_sum += self._apply_attention_readout(*self.att.contract(attention, values, self.n_up))
        else:
            attention, values = (
                jnp.zeros((), electrons.dtype),
                jnp.zeros((), electrons.dtype),
            )

        # MLP
        if self.mlp:
            jastrows_before_sum = self._apply_mlp(embeddings)
            jastrows_after_sum += jastrows_before_sum.sum(axis=0)
        else:
            jastrows_before_sum = jnp.zeros((), electrons.dtype)

        # Apply jastrows
        J_sign, J_logpsi = self._mlp_to_logpsi(jastrows_after_sum)
        sign *= J_sign
        logpsi += J_logpsi

        if return_state:
            return (sign, logpsi), JastrowState(jastrows_before_sum, jastrow_same, jastrow_diff, attention, values)
        return (sign, logpsi)

    def low_rank_update(
        self,
        electrons: Electrons,
        embeddings: jax.Array,
        changed_electrons: ElectronIdx,
        changed_embeddings: ElectronIdx,
        state: JastrowState,
    ) -> tuple[SignedLogAmplitude, JastrowState]:
        sign = jnp.ones((), electrons.dtype)
        logpsi = jnp.zeros([], electrons.dtype)
        n_el = electrons.shape[-2]
        if self.pairwise_cusps:
            # TODO: one could do low-rank updates on the cusps, though they should be cheap anyway.
            # NG: I benchmarked this on 200-electrons and it accounts for <5% of the runtime.
            # If we want to implement this, we can use the changed_electrons variable.
            logpsi += self.pairwise_cusps(electrons)
        if self.use_e_e_mlp:
            (r1_same, r2_same), (r1_diff, r2_diff) = self._get_all_electron_pairs(electrons)
            idx_same, idx_diff = get_changed_pair_indices(n_el, self.n_up, changed_electrons)
            jastrow_same = state.two_el_same.at[idx_same].set(
                self._apply_pairwise_mlp_same(r1_same[idx_same], r2_same[idx_same])
            )
            jastrow_diff = state.two_el_diff.at[idx_diff].set(
                self._apply_pairwise_mlp_diff(r1_diff[idx_diff], r2_diff[idx_diff])
            )
            logpsi += jnp.sum(jastrow_same) + jnp.sum(jastrow_diff)
        else:
            jastrow_same, jastrow_diff = (
                jnp.zeros([], electrons.dtype),
                jnp.zeros([], electrons.dtype),
            )

        jastrows_after_sum = jnp.zeros((2,), electrons.dtype)
        # Attention jastrow
        if self.use_attention:
            attention, values = self._apply_attention_and_values(embeddings[changed_embeddings])
            attention = state.attention.at[changed_embeddings].set(attention)
            values = state.values.at[changed_embeddings].set(values)
            jastrows_after_sum += self._apply_attention_readout(*self.att.contract(attention, values, self.n_up))
        else:
            attention, values = jnp.zeros(()), jnp.zeros(())

        # Elementwise MLP jastrow
        if self.mlp:
            jastrows_before_sum = self._apply_mlp(embeddings[changed_embeddings])
            jastrows_before_sum = state.one_el.at[changed_embeddings].set(jastrows_before_sum)
            jastrows_after_sum += jastrows_before_sum.sum(axis=0)
        else:
            jastrows_before_sum = jnp.zeros(())

        # Apply jastrows
        J_sign, J_logpsi = self._mlp_to_logpsi(jastrows_after_sum)
        sign *= J_sign
        logpsi += J_logpsi
        return (sign, logpsi), JastrowState(jastrows_before_sum, jastrow_same, jastrow_diff, attention, values)

    def _apply_mlp(self, embeddings):
        return self.mlp(embeddings) * self.mlp_scale + jnp.array([0, self.log_bias])

    def _apply_attention_and_values(self, embeddings):
        return self.att.attention_and_values(embeddings)

    def _apply_attention_readout(self, norm, values):
        return self.att.readout(norm, values)

    def _mlp_to_logpsi(self, jastrows):
        assert jastrows.shape == (2,)
        logpsi = jnp.zeros((), dtype=jastrows.dtype)
        sign = jnp.ones((), dtype=jastrows.dtype)
        if self.use_mlp_jastrow:
            logpsi += jastrows[0]
        if self.use_log_jastrow:
            log_J = jastrows[1]
            sign *= jnp.sign(log_J).prod()
            logpsi += jnp.log(jnp.abs(log_J))
        return sign, logpsi

    def _apply_pairwise_cusps(self, electrons):
        return self.pairwise_cusps(electrons)

    def apply_with_fwd_lap(
        self,
        params,
        electrons: Electrons,
        embeddings: FwdLaplArray | NodeWithFwdLap,
        dependencies,
    ) -> jax.Array:
        n_el = electrons.shape[-2]
        zeros = jnp.zeros([], electrons.dtype)
        logpsi = FwdLaplArray(zeros, FwdJacobian(data=zeros), zeros)
        if self.e_e_cusps != "none":

            @functools.partial(fwd_lap, sparsity_threshold=0.6)
            def get_pairwise_jastrow(r):
                return self.apply(params, r, method=self._apply_pairwise_cusps)

            logpsi = tree_add(logpsi, get_pairwise_jastrow(electrons))
        if self.use_e_e_mlp:
            (idx1_same, idx2_same), (idx1_diff, idx2_diff) = get_all_pair_indices(n_el, self.n_up)

            @fwd_lap
            def get_jastrow(r):
                r1_same, r2_same = r[idx1_same], r[idx2_same]
                r1_diff, r2_diff = r[idx1_diff], r[idx2_diff]
                J_same = self.apply(params, r1_same, r2_same, method=self._apply_pairwise_mlp_same)
                J_diff = self.apply(params, r1_diff, r2_diff, method=self._apply_pairwise_mlp_diff)
                return jnp.sum(J_same) + jnp.sum(J_diff)

            logpsi = tree_add(logpsi, get_jastrow(electrons))
        if self.use_mlp_jastrow or self.use_log_jastrow or self.use_attention:
            jastrows = FwdLaplArray(
                jnp.zeros((2,), dtype=electrons.dtype),
                FwdJacobian(data=jnp.zeros((1, 2), dtype=electrons.dtype)),
                jnp.zeros((2,), dtype=electrons.dtype),
            )
            # Attention
            if self.use_attention:
                if isinstance(embeddings, FwdLaplArray):
                    _get_att_inp = functools.partial(self.apply, params, method=self._apply_attention_and_values)
                    _get_att_inp = jax.vmap(fwd_lap(_get_att_inp, argnums=0), in_axes=-2, out_axes=(-2, -3))
                    attention, values = _get_att_inp(embeddings)
                    norm, values = GlobalAttentionJastrow.contract(attention, values, self.n_up, dependencies)  # type: ignore
                    _get_att_jastrow = functools.partial(self.apply, params, method=self._apply_attention_readout)
                    jastrows = tree_add(
                        jastrows,
                        fwd_lap(_get_att_jastrow, argnums=(0, 1))(norm, values),
                    )
                else:
                    _get_att_inp = functools.partial(self.apply, params, method=self._apply_attention_and_values)
                    attention, values = _get_att_inp(embeddings)
                    norm, values = GlobalAttentionJastrow.contract(attention, values, self.n_up, dependencies)  # type: ignore
                    _get_att_out = functools.partial(self.apply, params, method=self._apply_attention_readout)
                    _get_att_out = fwd_lap(_get_att_out, argnums=(0, 1))
                    jastrows = tree_add(jastrows, _get_att_out(norm, values))

            # Elementwise MLP
            if self.use_mlp_jastrow or self.use_log_jastrow:
                _get_jastrows = functools.partial(self.apply, params, method=self._apply_mlp)
                if isinstance(embeddings, FwdLaplArray):
                    _get_jastrows = fwd_lap(_get_jastrows, argnums=0)
                    _get_jastrows = jax.vmap(_get_jastrows, in_axes=-2, out_axes=-2)  # vmap over eletrons
                    mlp_jastrows = _get_jastrows(embeddings)
                    n_el = electrons.shape[-2]
                    jastrows = tree_add(jastrows, sum_fwd_lap(mlp_jastrows, dependencies, n_el))
                elif isinstance(embeddings, NodeWithFwdLap):
                    jastrows = tree_add(
                        jastrows,
                        cast(NodeWithFwdLap, _get_jastrows(embeddings)).sum_over_nodes(),
                    )

            logpsi_jastrow = fwd_lap(lambda J: self._mlp_to_logpsi(J)[1])(jastrows)
            logpsi = tree_add(logpsi, logpsi_jastrow)
        return logpsi
