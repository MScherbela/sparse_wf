import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.sparse.linalg import cg

from sparse_wf.api import (
    Electrons,
    EnergyCotangent,
    NaturalGradient,
    NaturalGradientState,
    Parameters,
    ParametrizedWaveFunction,
    StaticInput,
)
from sparse_wf.jax_utils import pall_to_all, pgather, pidx, psum
from sparse_wf.tree_utils import tree_add, tree_mul, tree_sub


def make_cg_preconditioner(
    wave_function: ParametrizedWaveFunction,
    damping: float = 1e-3,
    maxiter: int = 100,
) -> NaturalGradient:
    def init(params):
        return NaturalGradientState(last_grad=jax.tree_map(lambda x: jnp.zeros_like(x), params))

    def precondition(
        params: Parameters,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        natgrad_state: NaturalGradientState,
    ):
        N = dE_dlogpsi.size * jax.device_count()

        def log_p_closure(p: Parameters):
            return jax.vmap(wave_function, in_axes=(None, 0, 0))(p, electrons, static)

        _, vjp = jax.vjp(log_p_closure, params)
        _, jvp = jax.linearize(vjp, params)

        grad = psum(vjp(dE_dlogpsi)[0])

        def Fisher_matmul(v):
            w = jvp(v) / N
            undamped = vjp(w)[0]
            result = tree_add(undamped, tree_mul(v, damping))
            return psum(result)

        natgrad = cg(A=Fisher_matmul, b=grad, x0=natgrad_state.last_grad, tol=0, atol=0, maxiter=maxiter)
        return natgrad, NaturalGradientState(last_grad=natgrad), {}

    return NaturalGradient(init, precondition)


def make_spring_preconditioner(
    wave_function: ParametrizedWaveFunction,
    damping: float = 1e-3,
    decay_factor: float = 0.99,
) -> NaturalGradient:
    def init(params):
        return NaturalGradientState(last_grad=jax.tree_map(lambda x: jnp.zeros_like(x), params))

    def precondition(
        params: Parameters,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        natgrad_state: NaturalGradientState,
    ):
        n_dev = jax.device_count()
        N = dE_dlogpsi.size * n_dev
        normalization = 1 / jnp.sqrt(N)

        def log_p(params: Parameters, electrons: Electrons, static: StaticInput):
            return wave_function(params, electrons, static) * normalization

        # Gather individual jacobians
        jac_fn = jax.vmap(jax.grad(log_p), in_axes=(None, 0, 0))
        jacobians = jtu.tree_leaves(jac_fn(params, electrons, static))
        jacobians = jtu.tree_map(lambda x: x.reshape(N, -1), jacobians)

        # Compute T
        T = jnp.zeros((N, N))
        for jac in jacobians:
            if jac.shape[-1] % n_dev != 0:
                jac = jnp.concatenate(
                    [
                        jac,
                        jnp.zeros((jac.shape[0], n_dev - jac.shape[-1] % n_dev)),
                    ],
                    axis=-1,
                )
            jac = pall_to_all(jac, split_axis=1, concat_axis=0, tiled=True)
            jac = jac - jac.mean(0)
            T += jac @ jac.T
        T = psum(T)

        def log_p_closed(params: Parameters):
            result = jax.vmap(wave_function, in_axes=(None, 0, 0))(params, electrons, static)
            return result * normalization

        prim_out, vjp = jax.vjp(log_p_closed, params)

        def centered_jvp(x):
            uncentered = jax.jvp(log_p_closed, (params,), (x,))[1]
            return uncentered - psum(uncentered.sum()) / N

        avg_grad = psum(vjp(jnp.ones_like(prim_out) / N)[0])

        def centered_vjp(x):
            uncentered = psum(vjp(x.reshape(prim_out.shape).astype(prim_out.dtype)))[0]
            return tree_sub(uncentered, tree_mul(avg_grad, psum(x.sum())))

        last_grad = natgrad_state.last_grad
        decayed_last_grad = tree_mul(last_grad, decay_factor)
        cotangent = dE_dlogpsi.reshape(-1) * normalization
        cotangent -= centered_jvp(decayed_last_grad).reshape(-1)
        cotangent = pgather(cotangent, axis=0, tiled=True)

        T = T + damping * jnp.eye(T.shape[-1]) + 1 / N

        natgrad = centered_vjp(jnp.linalg.solve(T, cotangent).reshape(n_dev, -1)[pidx()])
        natgrad = tree_add(natgrad, decayed_last_grad)
        return natgrad, NaturalGradientState(last_grad=natgrad), {}

    return NaturalGradient(init, precondition)
