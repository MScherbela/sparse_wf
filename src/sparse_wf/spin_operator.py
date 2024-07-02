from typing import NamedTuple, TypeVar

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax.struct import PyTreeNode

from sparse_wf.api import Electrons, Int, ParameterizedWaveFunction, SpinOperator, SpinOperatorArgs
from sparse_wf.jax_utils import psum
from sparse_wf.tree_utils import tree_mul, tree_add

P = TypeVar("P")
S = TypeVar("S")
L = TypeVar("L")


class SplusState(NamedTuple):
    alpha: Int


class SplusOperator(SpinOperator[P, S, SplusState], PyTreeNode):
    wf: ParameterizedWaveFunction[P, S, L]
    grad_scale: float

    def init_state(self):
        return SplusState(alpha=jnp.zeros((), dtype=jnp.int32))

    def __call__(self, params: P, electrons: Electrons, static: S, state: SplusState):
        n_electrons = electrons.shape[-2]
        n_up = self.wf.n_up
        batch_size = np.prod(electrons.shape[:-2]) * jax.device_count()

        # compute the base state and prepare its vjp function
        vmapped_logpsi = jax.vmap(self.wf.log_psi_with_state, in_axes=(None, 0, None))
        ((base_sign, base_logpsi), logpsi_state), vjp_fn = jax.vjp(
            lambda p: vmapped_logpsi(p, electrons, static), params
        )

        # Here we compute the gradient of the operator in two steps, first we compute it for every swap
        # and then we compute it for the initial state in the outer loop.
        def R_alpha_element(params: P, base_logpsi, logpsi_state, beta: Int):
            idx = jnp.array([beta, state.alpha])
            new_electrons = electrons.at[:, idx].set(electrons[:, idx[::-1]])
            new_sign, new_logpsi = jax.vmap(
                self.wf.log_psi_low_rank_update,
                in_axes=(None, 0, None, None, 0),
            )(params, new_electrons, idx, static, logpsi_state)[0]
            summation_elements = -new_sign * base_sign * jnp.exp(new_logpsi - base_logpsi)
            return summation_elements.sum(), summation_elements

        # The loop aggregates the gradient towards the parameters and the gradients w.r.t. the base case.
        def loop_fn(gradient, beta):
            (_, R_alpha_element_val), R_alpha_element_grad = jax.value_and_grad(
                R_alpha_element, argnums=(0, 1, 2), has_aux=True
            )(params, base_logpsi, logpsi_state, beta)
            return tree_add(gradient, R_alpha_element_grad), R_alpha_element_val

        gradient, R_alpha = jax.lax.scan(
            loop_fn,
            jax.tree_map(jnp.zeros_like, (params, base_logpsi, logpsi_state)),
            jnp.arange(n_up, n_electrons),
        )
        gradient, dlogpsi, dlogpsi_state = gradient
        # summation over swaps
        R_alpha = 1 + R_alpha.sum(0)
        # summation over batch
        P_plus = R_alpha.sum()
        P_plus = psum(P_plus) / batch_size
        # Compute the full gradient, this adds remaining gradient from the loop with the local operator deviation
        gradient = tree_add(
            gradient, vjp_fn(((jnp.zeros_like(base_sign), dlogpsi + 2 * (R_alpha - P_plus)), dlogpsi_state))[0]
        )
        gradient = psum(gradient)

        # Rescale
        gradient = tree_mul(gradient, 2 * P_plus / batch_size * self.grad_scale)
        # Catch NaNs
        is_nan = jnp.isnan(jfu.ravel_pytree(gradient)[0]).any()
        gradient = jtu.tree_map(lambda x: jnp.where(is_nan, jnp.zeros_like(x), x), gradient)
        # Round robin on the alpha electron
        new_spin_state = SplusState(alpha=(state.alpha + 1) % n_up)
        return P_plus, gradient, new_spin_state


class NoSpinOperator(SpinOperator[P, S, jax.Array], PyTreeNode):
    wf: ParameterizedWaveFunction[P, S, L]

    def init_state(self):
        return jnp.zeros(())

    def __call__(self, params: P, electrons: Electrons, static: S, state: jax.Array):
        return jnp.zeros(()), jtu.tree_map(jnp.zeros_like, params), state


def make_spin_operator(wf: ParameterizedWaveFunction[P, S, L], args: SpinOperatorArgs):
    match args["operator"].lower():
        case "splus":
            return SplusOperator(wf=wf, grad_scale=args["grad_scale"])
        case "none":
            return NoSpinOperator(wf=wf)
        case _:
            raise ValueError(f"Unknown spin operator {args['operator']}")
