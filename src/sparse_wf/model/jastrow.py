import functools
from typing import Literal, NamedTuple, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from folx.api import FwdJacobian, FwdLaplArray
from jaxtyping import Array, Float

from sparse_wf.api import ElectronIdx, Electrons, SignedLogAmplitude
from sparse_wf.jax_utils import fwd_lap
from sparse_wf.model.utils import MLP, get_dist_same_diff
from sparse_wf.model.sparse_fwd_lap import SparseMLP
from sparse_wf.tree_utils import tree_add
from sparse_wf.model.sparse_fwd_lap import NodeWithFwdLap


class JastrowState(NamedTuple):
    elementwise: jax.Array
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


class GlobalAttentionJastrowAttentionValues(nn.Module):
    n_register: int
    register_dim: int
    n_up: int

    @nn.compact
    def __call__(self, h: Float[Array, "n_nodes feature_dim"]):
        register_keys = self.param("register_keys", nn.initializers.normal(1), (self.n_register, self.register_dim))
        queries = nn.Dense(self.n_register * self.register_dim)(h).reshape(
            *h.shape[:-1], self.n_register, self.register_dim
        )
        values = nn.Dense(self.n_register * self.register_dim)(h).reshape(
            *h.shape[:-1], self.n_register, self.register_dim
        )

        # folx-friendly version of the inner product
        attention = (register_keys * queries).sum(-1)
        attention = jnp.exp(-attention / jnp.sqrt(self.register_dim))
        return attention, attention[..., None] * values  # attention is el x reg, values is el x reg x dim


def sum_fwd_lap(x: FwdLaplArray, dependencies, n_el: int) -> FwdLaplArray:
    out_jac = jnp.zeros((n_el, 3, *x.shape[1:]), x.x.dtype)
    jac = x.jacobian.data
    jac = jac.reshape((n_el, 3, *jac.shape[1:]))
    jac = jnp.swapaxes(jac, 1, 2)
    out_jac = out_jac.at[dependencies.T].add(jac, mode="drop")
    out_jac = out_jac.reshape(n_el * 3, *x.shape[1:])
    y = x.x.sum(0)
    y_lapl = x.laplacian.sum(0)
    return FwdLaplArray(x=y, jacobian=FwdJacobian(out_jac), laplacian=y_lapl)


def att_to_mlp_inp(attention, values, n_up, dependencies=None):
    n_el = attention.shape[-2]
    if isinstance(attention, FwdLaplArray):
        up_norm = sum_fwd_lap(jax.tree.map(lambda x: x[..., :n_up, :], attention), dependencies[:n_up], n_el)
        down_norm = sum_fwd_lap(jax.tree.map(lambda x: x[..., n_up:, :], attention), dependencies[n_up:], n_el)
        up_values = sum_fwd_lap(jax.tree.map(lambda x: x[..., :n_up, :, :], values), dependencies[:n_up], n_el)
        down_values = sum_fwd_lap(jax.tree.map(lambda x: x[..., n_up:, :, :], values), dependencies[n_up:], n_el)
    else:
        up_norm, down_norm = attention[:n_up].sum(0), attention[n_up:].sum(0)
        up_values, down_values = values[:n_up].sum(0), values[n_up:].sum(0)

    def stack(x, y):
        return jnp.stack([x, y], axis=0)

    if isinstance(up_norm, FwdLaplArray):
        stack = fwd_lap(stack, argnums=(0, 1))

    norm = stack(up_norm, down_norm)
    values = stack(up_values, down_values)
    return norm, values


class AttentionOut(nn.Module):
    @nn.compact
    def __call__(self, norm: Float[Array, " n_register"], values: Float[Array, "n_register feature_dim"]):
        values = (values / norm[..., None]).reshape(-1)
        jastrow = MLP([256, 256, 2])(values)
        scale = self.param("scale", nn.initializers.zeros, (2,), jnp.float32)
        bias = self.param("bias", nn.initializers.ones, (1,), jnp.float32)
        return jastrow * scale + jnp.concatenate([jnp.zeros(1), bias])


class Jastrow(nn.Module):
    n_up: int
    e_e_cusps: Literal["none", "psiformer", "yukawa"]
    use_log_jastrow: bool
    use_mlp_jastrow: bool
    mlp_depth: int
    mlp_width: int
    sparse_embedding: bool = False
    use_attention: bool = False

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
            self.att_inp = GlobalAttentionJastrowAttentionValues(4, 32, self.n_up)
            self.att_out = AttentionOut()
        else:
            self.att_inp = None
            self.att_out = None

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
        if self.mlp:
            jastrows_before_sum = self._apply_mlp(embeddings)
            jastrows_after_sum = jastrows_before_sum.sum(axis=0)
            J_attention, J_values = self._apply_att_jastrow_inp(embeddings)
            jastrows_after_sum += self._apply_att_jastrow(*att_to_mlp_inp(J_attention, J_values, self.n_up))
            J_sign, J_logpsi = self._mlp_to_logpsi(jastrows_before_sum.sum(axis=0))
            sign *= J_sign
            logpsi += J_logpsi
        else:
            jastrows_before_sum = jnp.zeros(())
            J_attention, J_values = jnp.zeros(()), jnp.zeros(())

        if return_state:
            return (sign, logpsi), JastrowState(jastrows_before_sum, J_attention, J_values)
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
        if self.pairwise_cusps:
            # TODO: one could do low-rank updates on the cusps, though they should be cheap anyway.
            # NG: I benchmarked this on 200-electrons and it accounts for <5% of the runtime.
            # If we want to implement this, we can use the changed_electrons variable.
            logpsi += self.pairwise_cusps(electrons)
        if self.mlp:
            jastrows_before_sum = self._apply_mlp(embeddings[changed_embeddings])
            jastrows_before_sum = state.elementwise.at[changed_embeddings].set(jastrows_before_sum)
            jastrows_after_sum = jastrows_before_sum.sum(axis=0)
            # TODO: low rank update for attention jastrow
            if self.use_attention:
                J_attention, J_values = self._apply_att_jastrow_inp(embeddings[changed_embeddings])
                J_attention = state.attention.at[changed_embeddings].set(J_attention)
                J_values = state.values.at[changed_embeddings].set(J_values)
                jastrows_after_sum += self._apply_att_jastrow(*att_to_mlp_inp(J_attention, J_values, self.n_up))
            else:
                J_attention, J_values = jnp.zeros(()), jnp.zeros(())
            J_sign, J_logpsi = self._mlp_to_logpsi(jastrows_after_sum)
            sign *= J_sign
            logpsi += J_logpsi
        else:
            jastrows_before_sum = jnp.zeros(())
        return (sign, logpsi), JastrowState(jastrows_before_sum, J_attention, J_values)

    def _apply_mlp(self, embeddings):
        return self.mlp(embeddings) * self.mlp_scale + jnp.array([0, self.log_bias])

    def _apply_att_jastrow_inp(self, embeddings):
        if self.att_inp:
            return self.att_inp(embeddings)
        return jnp.zeros(()), jnp.zeros(())

    def _apply_att_jastrow(self, attention, values):
        if self.att_out:
            return self.att_out(attention, values)
        return jnp.array([0, 1])

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
        self, params, electrons: Electrons, embeddings: FwdLaplArray | NodeWithFwdLap, dependencies
    ) -> jax.Array:
        zeros = jnp.zeros([], electrons.dtype)
        logpsi = FwdLaplArray(zeros, FwdJacobian(data=zeros), zeros)
        if self.e_e_cusps != "none":

            @functools.partial(fwd_lap, sparsity_threshold=0.6)
            def get_pairwise_jastrow(r):
                return self.apply(params, r, method=self._apply_pairwise_cusps)

            logpsi = tree_add(logpsi, get_pairwise_jastrow(electrons))
        if self.use_mlp_jastrow or self.use_log_jastrow:
            _get_jastrows = functools.partial(self.apply, params, method=self._apply_mlp)
            if isinstance(embeddings, FwdLaplArray):
                _get_jastrows = fwd_lap(_get_jastrows, argnums=0)
                _get_jastrows = jax.vmap(_get_jastrows, in_axes=-2, out_axes=-2)  # vmap over eletrons
                jastrows = _get_jastrows(embeddings)
                n_el = electrons.shape[-2]
                jastrows = sum_fwd_lap(jastrows, dependencies, n_el)

                if self.use_attention:
                    _get_att_inp = functools.partial(self.apply, params, method=self._apply_att_jastrow_inp)
                    _get_att_inp = jax.vmap(fwd_lap(_get_att_inp, argnums=0), in_axes=-2, out_axes=(-2, -3))
                    J_attention, J_values = _get_att_inp(embeddings)
                    J_norm, J_vals = att_to_mlp_inp(J_attention, J_values, self.n_up, dependencies)
                    _get_att_jastrow = functools.partial(self.apply, params, method=self._apply_att_jastrow)
                    jastrows = tree_add(jastrows, fwd_lap(_get_att_jastrow, argnums=(0, 1))(J_norm, J_vals))
            elif isinstance(embeddings, NodeWithFwdLap):
                jastrows = cast(NodeWithFwdLap, _get_jastrows(embeddings)).sum_over_nodes()
                # TODO: Implement attention for new sparse format
            logpsi_jastrow = fwd_lap(lambda J: self._mlp_to_logpsi(J)[1])(jastrows)
            logpsi = tree_add(logpsi, logpsi_jastrow)
        return logpsi
