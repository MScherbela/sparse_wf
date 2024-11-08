from enum import Enum
from typing import Literal, TypeVar, Callable, Sequence

import jax
import jax.numpy as jnp
import optax

from sparse_wf.api import (
    ClippingArgs,
    Electrons,
    LocalEnergy,
    MCStep,
    OptState,
    ParameterizedWaveFunction,
    Preconditioner,
    PRNGKeyArray,
    SpinOperator,
    Trainer,
    TrainingState,
    Width,
    WidthScheduler,
)
from sparse_wf.hamiltonian import make_local_energy
from sparse_wf.jax_utils import (
    pgather,
    pmap,
    pmean,
    replicate,
    plogsumexp,
    vmap_reduction,
    pmax_if_pmap,
    PMAP_AXIS_NAME,
    copy_from_main,
)
from sparse_wf.tree_utils import tree_dot


class ClipStatistic(Enum):
    MEDIAN = "median"
    MEAN = "mean"


def clip_local_energies(e_loc: LocalEnergy, clip_local_energy: float, stat: str | ClipStatistic) -> LocalEnergy:
    stat = ClipStatistic(stat)

    def stat_fn(x):
        if stat == ClipStatistic.MEAN:
            return pmean(jnp.mean(x, keepdims=True))
        elif stat == ClipStatistic.MEDIAN:
            return pmean(jnp.median(x, keepdims=True))
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    if clip_local_energy > 0.0:
        full_e = pgather(e_loc, axis=0, tiled=True)
        clip_center = stat_fn(full_e)
        mad = pmean(jnp.mean(jnp.abs(full_e - clip_center), keepdims=True))
        max_dev = clip_local_energy * mad
        e_loc = jnp.clip(e_loc, clip_center - max_dev, clip_center + max_dev)
    return e_loc


def local_energy_diff(e_loc: LocalEnergy, clip_local_energy: float, stat: str | ClipStatistic) -> LocalEnergy:
    e_loc = clip_local_energies(e_loc, clip_local_energy, stat)
    center = pmean(jnp.mean(e_loc, keepdims=True))
    e_loc -= center
    return e_loc


P = TypeVar("P")
S = TypeVar("S")
MS = TypeVar("MS")
SS = TypeVar("SS")


def make_trainer(
    wave_function: ParameterizedWaveFunction[P, S, MS],
    mcmc_step: MCStep[P, S],
    width_scheduler: WidthScheduler,
    optimizer: optax.GradientTransformation,
    preconditioner: Preconditioner[P, S],
    clipping_args: ClippingArgs,
    max_batch_size: int,
    spin_operator: SpinOperator[P, S, SS],
    energy_operator: Literal["sparse", "dense"],
    pseudopotentials: Sequence[str],
    pp_grid_points: int,
):
    energy_fn = make_local_energy(wave_function, energy_operator, pseudopotentials, pp_grid_points)
    batched_energy_fn = vmap_reduction(
        energy_fn,
        (lambda x: x, lambda x: pmax_if_pmap(jnp.max(x))),
        max_batch_size=max_batch_size,
        in_axes=(0, None, 0, None),
    )

    def init(key: PRNGKeyArray, params: P, electrons: Electrons, init_width: Width):
        params = copy_from_main(replicate(params))
        return TrainingState(
            key=key,
            params=params,
            opt_state=OptState(
                opt=pmap(optimizer.init)(params),
                natgrad=pmap(preconditioner.init)(params),
            ),
            electrons=electrons.reshape(jax.local_device_count(), -1, *electrons.shape[1:]),
            width_state=replicate(width_scheduler.init(init_width)),
            spin_state=replicate(spin_operator.init_state()),
            step=replicate(jnp.zeros([], jnp.int32)),
        )

    # @pmap(static_broadcasted_argnums=(1, 2, 3))
    def sampling_step(
        state: TrainingState[P, SS], static: S, pp_static: S, compute_energy: bool, overlap_fn: Callable | None = None
    ):
        batch_size = state.electrons.shape[0]
        key, subkey = jax.random.split(state.key)
        electrons, stats = mcmc_step(subkey, state.params, state.electrons, static, state.width_state.width)
        width_state = width_scheduler.update(state.width_state, stats.pmove)
        state = state.replace(key=key, electrons=electrons, width_state=width_state)
        aux_data = {}
        if compute_energy:
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch_size)
            E_loc, pp_static_max = batched_energy_fn(keys, state.params, electrons, pp_static)
            E_mean = pmean(E_loc.mean())
            E_std = pmean(((E_loc - E_mean) ** 2).mean()) ** 0.5
            aux_data["eval/E"] = E_mean
            aux_data["eval/E_std"] = E_std
        else:
            pp_static_max = pp_static
        if overlap_fn is not None:
            signpsi, logpsi = jax.vmap(wave_function.signed, in_axes=(None, 0, None))(state.params, electrons, static)
            signphi, logphi = jax.vmap(overlap_fn)(electrons)
            overlap_signs = signphi * signpsi[:, None]
            log_overlap_ratios = logphi - logpsi[:, None]
            log_overlap, sign_overlap = plogsumexp(log_overlap_ratios, overlap_signs)
            for i, (o, s) in enumerate(zip(log_overlap, sign_overlap)):
                aux_data[f"eval/log_overlap_{i}"] = o
                aux_data[f"eval/sign_overlap_{i}"] = s

        return state, aux_data, stats, pp_static_max

    # @pmap(static_broadcasted_argnums=(1, 2))
    def step(state: TrainingState[P, SS], static: S, pp_static: S):
        batch_size = state.electrons.shape[0]
        key, subkey = jax.random.split(state.key)
        electrons, stats = mcmc_step(subkey, state.params, state.electrons, static, state.width_state.width)
        width_state = width_scheduler.update(state.width_state, stats.pmove)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        energy, pp_static_max = batched_energy_fn(keys, state.params, electrons, pp_static)
        energy_diff = local_energy_diff(energy, **clipping_args)

        E_mean = pmean(energy.mean())
        E_std = pmean(((energy - E_mean) ** 2).mean()) ** 0.5
        aux_data = {"opt/E": E_mean, "opt/E_std": E_std}

        spin_op_value, spin_grad, spin_state = spin_operator(state.params, electrons, static, state.spin_state)
        aux_data["opt/S"] = spin_op_value

        natgrad, precond_state, preconditioner_aux = preconditioner.precondition(
            state.params,
            electrons,
            static,
            energy_diff,
            spin_grad,
            state.opt_state.natgrad,
        )
        aux_data.update(preconditioner_aux)
        aux_data["opt/update_norm"] = tree_dot(natgrad, natgrad) ** 0.5
        aux_data["opt/elec_max_extend"] = jnp.abs(electrons).max()

        updates, opt = optimizer.update(natgrad, state.opt_state.opt, state.params)
        params = optax.apply_updates(state.params, updates)

        return (
            state.replace(
                key=key,
                params=params,
                electrons=electrons,
                opt_state=OptState(opt, precond_state),
                width_state=width_state,
                spin_state=spin_state,
                step=state.step + 1,
            ),
            energy,
            aux_data,
            stats,
            pp_static_max,
        )

    return Trainer(
        init=init,
        step=jax.pmap(step, PMAP_AXIS_NAME, static_broadcasted_argnums=(1, 2)),
        sampling_step=jax.pmap(sampling_step, PMAP_AXIS_NAME, static_broadcasted_argnums=(1, 2, 3, 4)),
        wave_function=wave_function,
        mcmc=mcmc_step,
        width_scheduler=width_scheduler,
        optimizer=optimizer,
        preconditioner=preconditioner,
        spin_operator=spin_operator,
    )
