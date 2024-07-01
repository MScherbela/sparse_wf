from typing import NamedTuple, TypeVar

import jax
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

        def compute_R_alpha(params: P):
            (base_sign, base_logpsi), logpsi_state = jax.vmap(self.wf.log_psi_with_state, in_axes=(None, 0, None))(
                params, electrons, static
            )

            @jax.vmap
            def swapped_logpsi(beta: Int):
                idx = jnp.array([beta, state.alpha])
                new_electrons = electrons.at[:, idx].set(electrons[:, idx[::-1]])
                new_sign, new_logpsi = jax.vmap(
                    self.wf.log_psi_low_rank_update,
                    in_axes=(None, 0, None, None, 0),
                )(params, new_electrons, idx, static, logpsi_state)[0]
                return new_sign, new_logpsi

            swapped_sign, swapped_logpsis = swapped_logpsi(jnp.arange(n_up, n_electrons))
            # summation over swaps
            R_alpha = 1 - (swapped_sign * base_sign * jnp.exp(swapped_logpsis - base_logpsi)).sum(axis=0)
            P_plus = R_alpha.sum()  # summation over batch
            return P_plus, R_alpha

        (P_plus, R_alpha), R_alpha_grad = jax.value_and_grad(compute_R_alpha, has_aux=True)(params)
        P_plus = psum(P_plus) / batch_size
        grad = psum(
            tree_add(
                R_alpha_grad,
                jax.vjp(lambda p: jax.vmap(self.wf, in_axes=(None, 0, None))(p, electrons, static), params)[1](
                    2 * (R_alpha - P_plus)
                )[0],
            )
        )
        grad = tree_mul(grad, 2 * P_plus / batch_size * self.grad_scale)
        new_spin_state = SplusState(alpha=(state.alpha + 1) % n_up)
        return P_plus, grad, new_spin_state


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
