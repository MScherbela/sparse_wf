import functools
from typing import Literal, TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp
from folx.api import FwdJacobian, FwdLaplArray
from jaxtyping import Array, Float

from sparse_wf.api import ElectronIdx, Electrons, SignedLogAmplitude
from sparse_wf.jax_utils import fwd_lap
from sparse_wf.model.graph_utils import pad_jacobian_to_dense
from sparse_wf.model.utils import MLP, get_dist_same_diff
from sparse_wf.tree_utils import tree_add

JastrowState: TypeAlias = jax.Array


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
        factor_same, factor_diff = -0.25, -0.5

        cusp_same = jnp.sum(alpha_same**2 / (alpha_same + dist_same), axis=-1)
        cusp_diff = jnp.sum(alpha_diff**2 / (alpha_diff + dist_diff), axis=-1)

        return factor_same * cusp_same + factor_diff * cusp_diff


class Jastrow(nn.Module):
    n_up: int
    e_e_cusps: Literal["none", "psiformer", "yukawa"]
    use_log_jastrow: bool
    use_mlp_jastrow: bool
    mlp_depth: int
    mlp_width: int

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
            self.mlp = MLP([self.mlp_width] * self.mlp_depth + [2], activate_final=False)
            self.mlp_scale = self.param("mlp_scale", nn.initializers.zeros, (2,), jnp.float32)
            if self.use_log_jastrow:
                self.log_bias = self.param("log_bias", nn.initializers.ones, (), jnp.float32)
        else:
            self.mlp = None

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
            J_sign, J_logpsi = self._mlp_to_logpsi(jastrows_before_sum)
            sign *= J_sign
            logpsi += J_logpsi
        else:
            jastrows_before_sum = jnp.zeros(())
        if return_state:
            return (sign, logpsi), jastrows_before_sum
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
            jastrows_before_sum = state.at[changed_embeddings].set(jastrows_before_sum)
            J_sign, J_logpsi = self._mlp_to_logpsi(jastrows_before_sum)
            sign *= J_sign
            logpsi += J_logpsi
        else:
            jastrows_before_sum = jnp.zeros(())
        return (sign, logpsi), jastrows_before_sum

    def _apply_mlp(self, embeddings):
        return self.mlp(embeddings) * self.mlp_scale + jnp.array([0, self.log_bias])

    def _mlp_to_logpsi(self, jastrows):
        logpsi = jnp.zeros(())
        sign = jnp.ones(())
        if self.use_mlp_jastrow:
            logpsi += jastrows[..., 0].sum()
        if self.use_log_jastrow:
            log_J = jastrows[..., 1].sum()
            sign *= jnp.sign(log_J).prod()
            logpsi += jnp.log(jnp.abs(log_J)).sum()
        return sign, logpsi

    def _apply_pairwise_cusps(self, electrons):
        return self.pairwise_cusps(electrons)

    def apply_with_fwd_lap(self, params, electrons: Electrons, embeddings: FwdLaplArray, dependencies) -> jax.Array:
        zeros = jnp.zeros([], electrons.dtype)
        logpsi = FwdLaplArray(zeros, FwdJacobian(data=zeros), zeros)
        if self.e_e_cusps != "none":

            @functools.partial(fwd_lap, sparsity_threshold=0.6)
            def get_pairwise_jastrow(r):
                return self.apply(params, r, method=self._apply_pairwise_cusps)

            logpsi = tree_add(logpsi, get_pairwise_jastrow(electrons))
        if self.use_mlp_jastrow or self.use_log_jastrow:
            _get_jastrows = functools.partial(self.apply, params, method=self._apply_mlp)
            _get_jastrows = fwd_lap(_get_jastrows, argnums=0)
            _get_jastrows = jax.vmap(_get_jastrows, in_axes=-2, out_axes=-2)
            jastrows = _get_jastrows(embeddings)
            # TODO: this generates an n_el x n_el x 2 tensor, which is very sparse, and which is immediately summed over in the the next step.
            # Find a way to avoid this. Probably using reverse dependency mapping.
            n_el = electrons.shape[-2]
            jastrows = jax.vmap(pad_jacobian_to_dense, in_axes=(-2, 0, None), out_axes=-2)(jastrows, dependencies, n_el)

            @fwd_lap
            def _get_logpsi(jastrows):
                return self._mlp_to_logpsi(jastrows)[1]

            logpsi = tree_add(logpsi, _get_logpsi(jastrows))
        return logpsi
