import functools
from typing import NamedTuple, Optional, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from folx import batched_vmap
from jax.scipy.sparse.linalg import cg

from sparse_wf.api import (
    Electrons,
    EnergyCotangent,
    ParameterizedWaveFunction,
    Preconditioner,
    PreconditionerArgs,
    PreconditionerState,
    StaticInput,
)
from sparse_wf.jax_utils import pall_to_all, pgather, pidx, pmean, psum, vector_to_tree_like
from sparse_wf.tree_utils import tree_add, tree_mul, tree_sub, ravel_with_padding

P, MS = TypeVar("P"), TypeVar("MS")


def symmetric_inv_with_damping(H, damping, max_cond_nr=1e10):
    s, U = jnp.linalg.eigh(H)
    damping = jnp.maximum(damping, damping - s[0])  # Use larger damping in case of negative eigenvalues
    if max_cond_nr is not None:
        required_damping = (s[-1] - max_cond_nr * s[0]) / (max_cond_nr - 1)
        damping = jnp.maximum(damping, required_damping)
    cond_nr = (s[-1] + damping) / (s[0] + damping)
    U = U / jnp.sqrt(s + damping)
    return U @ U.T, damping.astype(jnp.float32), cond_nr.astype(jnp.float32)


def make_identity_preconditioner(
    wave_function: ParameterizedWaveFunction[P, MS],
):
    def init(params: P) -> PreconditionerState[P]:
        return PreconditionerState(last_grad=jax.tree_map(jnp.zeros_like, params), damping=jnp.zeros([]))

    def precondition(
        params: P,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        aux_grad: P,
        natgrad_state: PreconditionerState[P],
    ):
        N = dE_dlogpsi.size * jax.device_count()

        def log_p_closure(p: P):
            return jax.vmap(lambda r: wave_function(p, r, static))(electrons) / N  # type: ignore

        _, vjp = jax.vjp(log_p_closure, params)
        grad = psum(vjp(dE_dlogpsi)[0])
        grad = tree_add(grad, aux_grad)
        precond_grad_norm = jnp.sqrt(pmean(sum([jnp.sum(g**2) for g in jtu.tree_leaves(grad)])))
        return grad, natgrad_state, {"opt/precond_grad_norm": precond_grad_norm}

    return Preconditioner(init, precondition)


def make_cg_preconditioner(
    wave_function: ParameterizedWaveFunction[P, MS],
    damping: float = 1e-3,
    maxiter: int = 100,
):
    def init(params: P) -> PreconditionerState[P]:
        return PreconditionerState(
            last_grad=jax.tree_map(jnp.zeros_like, params),
            damping=jnp.array(damping, jnp.float32),
        )

    def precondition(
        params: P,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        aux_grad: P,
        natgrad_state: PreconditionerState[P],
    ):
        N = dE_dlogpsi.size * jax.device_count()

        def log_p_closure(p: P):
            return jax.vmap(lambda r: wave_function(p, r, static))(electrons) / jnp.sqrt(N)  # type: ignore

        _, vjp = jax.vjp(log_p_closure, params)
        _, jvp = jax.linearize(log_p_closure, params)

        grad = psum(vjp(dE_dlogpsi / jnp.sqrt(N).astype(jnp.float32))[0])
        grad = tree_add(grad, aux_grad)

        def Fisher_matmul(v: P):
            w = jvp(v)
            undamped = vjp(w)[0]
            result = tree_add(undamped, tree_mul(v, natgrad_state.damping))
            return psum(result)

        natgrad, _ = cg(
            A=Fisher_matmul,
            b=grad,
            x0=natgrad_state.last_grad,
            tol=0,
            atol=0,
            maxiter=maxiter,
        )
        return (
            natgrad,
            PreconditionerState(last_grad=natgrad, damping=natgrad_state.damping),
            {},
        )

    return Preconditioner(init, precondition)


def get_jacjacT(local_jacT, global_block_size, use_float64: bool = True):
    """Compute jac @ jac.T from local/sharded transposed jacobians.

    This function transposes the local jaobians across devices, such that all devices have all samples,
    but only a subset of parameters.
    From these sharded transposed jacobians the full jac @ jac.T is computed.
    To do this memory efficiently we split the parameters into blocks of size param_block_size, and transpose block-by-block.
    We loop over and aggregate all blocks using scan, and finally add the (non-full) block of leftover params.

    Args:
        local_jacT: Jacobian.T, shape (n_params, n_samples_per_device)
    Returns:
        T: Jacobian times Jacobian transposed, shape (n_samples_total, n_samples_total)
    """
    n_devices = jax.device_count()
    n_params, n_samples_local = local_jacT.shape
    assert n_params % global_block_size == 0, "n_params must be divisible by global_block_size"
    n_blocks = n_params // global_block_size
    local_jac_T = local_jacT.reshape([n_blocks, global_block_size, n_samples_local])

    def scan_func(T_carry, loc_jac_T):
        # pall_to_all: [block_size_global x n_samples_local] => [block_size_local x n_samples_global]
        jac_T = pall_to_all(loc_jac_T, split_axis=0, concat_axis=1, tiled=True)
        if use_float64:
            jac_T = jac_T.astype(jnp.float64)
        jac_T -= jac_T.mean(axis=1, keepdims=True)
        T_carry += jac_T.T @ jac_T
        return T_carry, None

    T = jnp.zeros((n_samples_local * n_devices, n_samples_local * n_devices), dtype=local_jac_T)
    if use_float64:
        T = T.astype(jnp.float64)

    T, _ = jax.lax.scan(scan_func, T, local_jac_T)
    T = psum(T)
    T = (T + T.T) / 2  # Not required, but potentially better numerics
    return T


def make_dense_spring_preconditioner(
    wave_function: ParameterizedWaveFunction[P, MS],
    damping: float,
    decay_factor: float,
    max_batch_size: int,
    use_float64: bool,
    param_block_size: int = 65536,
):
    n_dev = jax.device_count()
    global_param_block_size = param_block_size * n_dev
    ravel_params = functools.partial(ravel_with_padding, block_size=global_param_block_size)

    def init(params: P) -> PreconditionerState[P]:
        return PreconditionerState(
            last_grad=0 * ravel_params(params)[0],
            damping=jnp.array(damping, jnp.float32),
        )

    def precondition(
        params: P,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        aux_grad: P,
        natgrad_state: PreconditionerState[P],
    ):
        local_batch_size = dE_dlogpsi.size
        N = local_batch_size * n_dev
        normalization = 1 / jnp.sqrt(N)

        # We could cast the params first to float64, or at the jacobian, or at solving? Or not at all?
        flat_params, unravel = ravel_params(params)

        def log_p(params: jax.Array, electrons: Electrons, static: StaticInput):
            return wave_function(unravel(params), electrons, static) * normalization  # type: ignore

        jac_fn = batched_vmap(jax.grad(log_p), in_axes=(None, 0, None), out_axes=1, max_batch_size=max_batch_size)
        jacT = jac_fn(flat_params, electrons, static)
        T = get_jacjacT(jacT, global_param_block_size, use_float64)
        T += 1 / N
        T_inv, actual_damping, cond_nr = symmetric_inv_with_damping(T, damping)
        local_T_inv = T_inv.reshape([n_dev, local_batch_size, N])[pidx()]

        # The remainder needs a centered jacobian - this can be done in float32
        jacT -= pmean(jacT.mean(axis=1, keepdims=True))

        @jax.vmap
        def split_grad_by_J(grad):
            coeffs = psum((grad @ jacT) @ local_T_inv)
            grad_in_J = psum(jacT @ coeffs.reshape(n_dev, local_batch_size)[pidx()])
            return coeffs, grad_in_J, grad - grad_in_J

        last_grad = natgrad_state.last_grad
        decayed_last_grad = decay_factor * last_grad
        flat_aux_grad = ravel_params(aux_grad)[0]
        cotangent = dE_dlogpsi.reshape(-1) * normalization
        cotangent = pgather(cotangent, axis=0, tiled=True)

        # Here we decompose the aux_grad into a part that is in the jacobian and a part that is not
        # The part that is in the span of J is added as linear combination to the cotangent
        # The remainder is added to the update as is
        ((coeff_aux, coeff_last_grad), (aux_in_J, last_grad_in_J), (aux_not_J, last_grad_not_J)) = split_grad_by_J(
            jnp.stack([flat_aux_grad, decayed_last_grad], axis=0)
        )
        cotangent += coeff_aux + coeff_last_grad * actual_damping
        local_precond_cotangents = local_T_inv @ cotangent  # T^(-1)@contangent for local samples
        local_precond_cotangents = local_precond_cotangents.astype(jnp.float32)
        local_natgrad = jacT @ local_precond_cotangents
        natgrad = psum(local_natgrad)
        # Add momentum term
        natgrad += last_grad_not_J
        # Add aux_not_in_J part without rescaling with damping
        natgrad += aux_not_J

        # Diagnose and adjust stability
        aux_data = {}
        aux_data["opt/log10_S_cond_nr"] = jnp.log10(cond_nr)
        aux_data["opt/damping"] = actual_damping
        aux_data["opt/spring/aux_grad_norm"] = jnp.sqrt(jnp.sum(flat_aux_grad**2))
        aux_data["opt/spring/aux_in_J_norm"] = jnp.sqrt(jnp.sum(aux_in_J**2))
        aux_data["opt/spring/aux_not_in_J_norm"] = jnp.sqrt(jnp.sum(aux_not_J**2))
        aux_data["opt/spring/last_grad_norm"] = jnp.sqrt(jnp.sum(last_grad**2))
        aux_data["opt/spring/last_grad_in_J_norm"] = jnp.sqrt(jnp.sum(last_grad_in_J**2))
        aux_data["opt/spring/last_grad_not_in_J_norm"] = jnp.sqrt(jnp.sum(last_grad_not_J**2))
        is_nan = ~jnp.all(jnp.isfinite(natgrad))
        natgrad = jnp.where(is_nan, jnp.zeros_like(natgrad), natgrad)

        # Add the aux_not_in_J part to the update
        update = unravel(natgrad.astype(jnp.float32))
        return update, PreconditionerState(last_grad=natgrad, damping=actual_damping), aux_data

    return Preconditioner(init, precondition)


def make_spring_preconditioner(
    wave_function: ParameterizedWaveFunction[P, MS],
    damping: float = 1e-3,
    decay_factor: float = 0.99,
):
    def init(params: P) -> PreconditionerState[P]:
        return PreconditionerState(
            last_grad=jax.tree_map(jnp.zeros_like, params),
            damping=jnp.array(damping, jnp.float32),
        )

    def precondition(
        params: P,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        aux_grad: P,
        natgrad_state: PreconditionerState[P],
    ):
        # dtype = dE_dlogpsi.dtype
        n_dev = jax.device_count()
        local_batch_size = dE_dlogpsi.size
        N = local_batch_size * n_dev
        normalization = 1 / jnp.sqrt(N)

        def log_p(params: P, electrons: Electrons, static: StaticInput):
            return wave_function(params, electrons, static) * normalization  # type: ignore

        # Gather individual jacobians
        jac_fn = jax.vmap(jax.grad(log_p), in_axes=(None, 0, None))
        jacobians = jtu.tree_leaves(jac_fn(params, electrons, static))
        jacobians = jtu.tree_map(lambda x: x.reshape(local_batch_size, -1), jacobians)

        # Compute T
        t_dtype = jnp.float64
        T = jnp.zeros((N, N), t_dtype)
        for jac in jacobians:
            if jac.shape[-1] % n_dev != 0:
                jac = jnp.concatenate(
                    [
                        jac,
                        jnp.zeros((jac.shape[0], n_dev - jac.shape[-1] % n_dev), t_dtype),
                    ],
                    axis=-1,
                )
            jac = pall_to_all(jac, split_axis=1, concat_axis=0, tiled=True)
            jac = jac - jac.mean(0)
            T += jac @ jac.T
        T = psum(T)

        def log_p_closed(params: P):
            return jax.vmap(lambda r: wave_function(params, r, static))(electrons) * normalization  # type: ignore

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
        decayed_last_grad = tree_add(decayed_last_grad, tree_mul(aux_grad, 1 / natgrad_state.damping))
        cotangent = dE_dlogpsi.reshape(-1) * normalization
        cotangent -= centered_jvp(decayed_last_grad).reshape(-1)
        cotangent = pgather(cotangent, axis=0, tiled=True)

        T = T + natgrad_state.damping * jnp.eye(T.shape[-1], dtype=T.dtype) + 1 / N

        natgrad = centered_vjp(jnp.linalg.solve(T, cotangent).reshape(n_dev, -1)[pidx()])
        natgrad = tree_add(natgrad, decayed_last_grad)
        natgrad = jax.tree_util.tree_map(lambda x: jnp.astype(x, jnp.float32), natgrad)
        return (
            natgrad,
            PreconditionerState(last_grad=natgrad, damping=natgrad_state.damping),
            {},
        )

    return Preconditioner(init, precondition)


class SVDPreconditionerState(NamedTuple):
    last_grad: jax.Array
    X_history: Optional[jax.Array]


def make_svd_preconditioner(
    wave_function: ParameterizedWaveFunction[P, MS],
    damping: float,
    ema_natgrad: float,
    ema_S: float,
    history_length: int,
):
    def init(params: P):
        n_params = sum([p.size for p in jtu.tree_leaves(params)])
        return SVDPreconditionerState(
            last_grad=jnp.zeros(n_params),
            X_history=jnp.zeros([n_params, history_length]),
        )

    def precondition(
        params: P,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        aux_grad: P,  # TODO: this doesn't work currently
        natgrad_state,
    ):
        assert jax.device_count() == 1
        total_batch_size = dE_dlogpsi.size * jax.device_count()
        N_total = total_batch_size + history_length

        def get_dlogpsi_dparam(r: Electrons):
            g = jax.grad(wave_function)(params, r, static)
            g = jtu.tree_flatten(g)[0]
            g = jnp.concatenate([x.flatten() for x in g])
            return g

        # Compute jacobian X = dlogpsi/dparam and concatenate with jacobian history
        X = jax.vmap(get_dlogpsi_dparam, out_axes=-1)(electrons)  # [n_params x n_samples]
        X = X - jnp.mean(X, axis=1, keepdims=True)  # center grads
        X_full = jnp.concatenate([X, natgrad_state.X_history], axis=1)

        # Compute SVD of merged jacobian
        U, s, Vt = jnp.linalg.svd(X_full, full_matrices=False, compute_uv=True)
        D1 = (N_total / total_batch_size) * s / (N_total * damping + s**2)
        D2 = ema_natgrad * s**2 / (N_total * damping + s**2)

        # Compute natural gradient
        E_padded = jnp.concatenate([dE_dlogpsi, jnp.zeros(history_length)])
        grad_term = jnp.einsum("i,ij,j->i", D1, Vt, E_padded)  # D1 @ V.T @ dE_dlogpsi
        momentum_term = jnp.einsum("i,ji,j->i", D2, U, natgrad_state.last_grad)  # D2 @ U.T @ last_grad
        natgrad = ema_natgrad * natgrad_state.last_grad + U @ (grad_term - momentum_term)

        # New history = U @ S corresponding to the largest singular values
        s_history = s[:history_length]
        X_history = U[:, :history_length] * (s_history * ema_S)

        # Compute auxiliary data and convert to output format
        s2_residual = jnp.sum(s[history_length:] ** 2) / jnp.sum(s**2)
        aux_data = {"opt/svd_s2_residual": s2_residual}
        precond_state = SVDPreconditionerState(last_grad=natgrad, X_history=X_history)
        natgrad = vector_to_tree_like(natgrad, params)
        return natgrad, precond_state, aux_data

    return Preconditioner(init, precondition)  # type: ignore


def make_preconditioner(wf: ParameterizedWaveFunction[P, MS], args: PreconditionerArgs):
    preconditioner = args["preconditioner"].lower()
    if preconditioner == "identity":
        return make_identity_preconditioner(wf)
    elif preconditioner == "cg":
        return make_cg_preconditioner(wf, **args["cg_args"])
    elif preconditioner == "spring":
        return make_spring_preconditioner(wf, **args["spring_args"])
    elif preconditioner == "spring_dense":
        return make_dense_spring_preconditioner(wf, **args["spring_dense_args"])
    elif preconditioner == "svd":
        return make_svd_preconditioner(wf, **args["svd_args"])
    else:
        raise ValueError(f"Unknown preconditioner: {args['preconditioner']}")
