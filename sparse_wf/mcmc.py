import jax
import jax.numpy as jnp
from jax import lax

from sparse_wf.api import (
    Int,
    ClosedLogLikelihood,
    LogAmplitude,
    Electrons,
    MCStep,
    Parameters,
    PMove,
    ParameterizedLogPsi,
    StaticInput,
    Width,
    PRNGKeyArray,
    WidthScheduler,
    WidthSchedulerState,
)
from .jax_utils import jit


def mh_update(
    log_prob_fn: ClosedLogLikelihood,
    key: PRNGKeyArray,
    electrons: Electrons,
    log_prob: LogAmplitude,
    num_accepts: Int,
    width: Width,
) -> tuple[PRNGKeyArray, Electrons, LogAmplitude, Int]:
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, electrons.shape, dtype=electrons.dtype) * width
    new_electrons = electrons + eps
    new_log_prob = log_prob_fn(new_electrons)
    log_ratio = new_log_prob - log_prob

    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, log_ratio.shape)
    accept = log_ratio > jnp.log(u)
    new_electrons = jnp.where(accept[..., None, None], new_electrons, electrons)
    new_log_prob = jnp.where(accept, new_log_prob, log_prob)
    num_accepts += jnp.sum(accept).astype(jnp.int32)

    return key, new_electrons, new_log_prob, num_accepts


def make_mcmc(
    network: ParameterizedLogPsi,
    steps: int = 10,
) -> MCStep:
    batch_network = jax.vmap(network, in_axes=(None, 0))

    def mcmc_step(
        key: PRNGKeyArray, params: Parameters, electrons: Electrons, static: StaticInput, width: Width
    ) -> tuple[Electrons, PMove]:
        def log_prob_fn(electrons: Electrons) -> LogAmplitude:
            return 2 * batch_network(params, electrons, static)

        def step_fn(_, x):
            return mh_update(log_prob_fn, *x, width)  # type: ignore

        logprob = log_prob_fn(electrons)
        num_accepts = jnp.zeros((), dtype=jnp.int32)

        key, electrons, logprob, num_accepts = lax.fori_loop(0, steps, step_fn, (key, electrons, logprob, num_accepts))

        pmove = num_accepts / (steps * electrons.shape[0])
        return electrons, pmove

    return jit(mcmc_step, static_argnames="static")


def make_width_scheduler(
    window_size: int = 20,
    target_pmove: float = 0.525,
    error: float = 0.025,
    width_multiplier: float = 1.1,
) -> WidthScheduler:
    @jit
    def init(init_width: Width) -> WidthSchedulerState:
        return WidthSchedulerState(
            width=jnp.array(init_width, dtype=jnp.float32),
            pmoves=jnp.zeros((window_size, *init_width.shape), dtype=jnp.float32),
            i=jnp.zeros((), dtype=jnp.int32),
        )

    @jit
    def update(state: WidthSchedulerState, pmove: PMove) -> WidthSchedulerState:
        pmoves = state.pmoves.at[jnp.mod(state.i, window_size)].set(pmove)
        pm_mean = state.pmoves.mean()
        i = state.i + 1

        upd_width = jnp.where(pm_mean < target_pmove - error, state.width / width_multiplier, state.width)
        upd_width = jnp.where(pm_mean > target_pmove + error, upd_width * width_multiplier, upd_width)
        width = jnp.where(
            jnp.mod(state.i, window_size) == 0,
            upd_width,
            state.width,
        )
        return WidthSchedulerState(width=width, pmoves=pmoves, i=i)

    return WidthScheduler(init, update)
