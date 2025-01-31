from typing import NamedTuple, TypeVar

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax.struct import PyTreeNode

from sparse_wf.api import Electrons, Int, ParameterizedWaveFunction, SpinOperator, SpinOperatorArgs, StaticInput
from sparse_wf.jax_utils import psum, pmean, pgather
from sparse_wf.tree_utils import tree_mul, tree_add

P = TypeVar("P")
L = TypeVar("L")


def mask_mean(x, mask):
    return psum(jnp.vdot(x, mask)) / psum(mask.sum())


def outlier_mask(ratio, threshold: float):
    if threshold > 0.0:
        full = pgather(ratio, axis=0, tiled=True)
        clip_center = jnp.median(full)
        mad = pmean(jnp.mean(jnp.abs(full - clip_center), keepdims=True))
        lower = clip_center - threshold * mad
        upper = clip_center + threshold * mad
        return (ratio > lower) & (ratio < upper)
    return jnp.ones_like(ratio, dtype=jnp.bool_)


def clip_ratios(ratio, clip_threshold: float, mask):
    if clip_threshold > 0.0:
        clip_center = mask_mean(ratio, mask)
        mad = mask_mean(jnp.abs(ratio - clip_center), mask)
        max_dev = clip_threshold * mad
        ratio = jnp.clip(ratio, clip_center - max_dev, clip_center + max_dev)
    return ratio


class SplusState(NamedTuple):
    beta: Int


class SplusOperator(SpinOperator[P, SplusState], PyTreeNode):
    wf: ParameterizedWaveFunction[P, L]
    grad_scale: float
    clip_threshold: float
    mask_threshold: float

    def init_state(self):
        return SplusState(beta=jnp.array(self.wf.n_up, dtype=jnp.int32))

    def __call__(self, params: P, electrons: Electrons, static: StaticInput, state: SplusState):
        # https://www.nature.com/articles/s43588-024-00730-4
        n_electrons = electrons.shape[-2]
        n_up = self.wf.n_up
        n_down = n_electrons - n_up
        batch_size = np.prod(electrons.shape[:-2]) * jax.device_count()

        # compute the base state and prepare its vjp function
        vmapped_logpsi = jax.vmap(self.wf.log_psi_with_state, in_axes=(None, 0, None))
        ((base_sign, base_logpsi), logpsi_state), vjp_fn = jax.vjp(
            lambda p: vmapped_logpsi(p, electrons, static), params
        )

        # Here we compute the gradient of the operator in two steps, first we compute it for every swap
        # and then we compute it for the initial state in the outer loop.
        def ratio_alpha_beta(params: P, base_logpsi, logpsi_state, alpha: Int):
            idx = jnp.array([alpha, state.beta], dtype=jnp.int32)
            new_electrons = electrons.at[:, idx].set(electrons[:, idx[::-1]])
            new_sign, new_logpsi = jax.vmap(
                self.wf.log_psi_low_rank_update,
                in_axes=(None, 0, None, None, 0),
            )(params, new_electrons, idx, static, logpsi_state)[0]
            swap_ratio = -new_sign * base_sign * jnp.exp(new_logpsi - base_logpsi)
            return swap_ratio.sum(), swap_ratio

        # The loop aggregates the gradient towards the parameters and the gradients w.r.t. the base case.
        def loop_fn(gradient, beta):
            (_, swap_ratio), ratio_grad = jax.value_and_grad(ratio_alpha_beta, argnums=(0, 1, 2), has_aux=True)(
                params, base_logpsi, logpsi_state, beta
            )
            return tree_add(gradient, ratio_grad), swap_ratio

        grad_components, swap_ratio = jax.lax.scan(
            loop_fn,
            jax.tree_map(jnp.zeros_like, (params, base_logpsi, logpsi_state)),
            jnp.arange(n_up),
        )
        gradient, dR_dlogpsi, dR_dlogpsi_state = grad_components
        # summation over swaps
        R_beta = 1 + swap_ratio.sum(0)
        mask = outlier_mask(R_beta, self.mask_threshold)
        R_beta = clip_ratios(R_beta, self.clip_threshold, mask)
        # summation over batch
        P_plus = mask_mean(R_beta, mask)
        # Compute the full gradient, this adds remaining gradient from the loop with the local operator deviation
        gradient = tree_add(
            gradient,
            vjp_fn(((jnp.zeros_like(base_sign), dR_dlogpsi + 2 * mask * (R_beta - P_plus)), dR_dlogpsi_state))[0],
        )
        gradient = psum(gradient)

        # Rescale
        # 2 * P_plus as in equation
        # 1 / batch_size for the mean in the vjp above
        # self.grad_scale for the gap scaling
        gradient = tree_mul(gradient, 2 * P_plus / batch_size * self.grad_scale)
        # Catch NaNs
        is_nan = jnp.isnan(jfu.ravel_pytree(gradient)[0]).any()
        gradient = jtu.tree_map(lambda x: jnp.where(is_nan, jnp.zeros_like(x), x), gradient)
        # Round robin on the beta electron
        new_spin_state = SplusState(beta=(state.beta - n_up + 1) % n_down + n_up)

        sz = jnp.abs(n_up - n_down) * 0.5
        aux_data = {
            "spin/P_plus": P_plus,
            "spin/estimator": sz * (sz + 1) + P_plus,
            "spin/var": mask_mean((R_beta - P_plus) ** 2, mask),
            "spin/num_outlier": psum((~mask).sum()),
        }
        return P_plus, gradient, new_spin_state, aux_data


class NoSpinOperator(SpinOperator[P, jax.Array], PyTreeNode):
    wf: ParameterizedWaveFunction[P, L]

    def init_state(self):
        return jnp.zeros(())

    def __call__(self, params: P, electrons: Electrons, static: StaticInput, state: jax.Array):
        return jnp.zeros(()), jtu.tree_map(jnp.zeros_like, params), state, {}


def make_spin_operator(wf: ParameterizedWaveFunction[P, L], args: SpinOperatorArgs):
    match args["operator"].lower():
        case "splus":
            return SplusOperator(
                wf=wf,
                grad_scale=args["grad_scale"],
                clip_threshold=args["clip_threshold"],
                mask_threshold=args["mask_threshold"],
            )
        case "none":
            return NoSpinOperator(wf=wf)
        case _:
            raise ValueError(f"Unknown spin operator {args['operator']}")
