import functools
from typing import Callable, TypeVar, cast, ParamSpec, overload
import folx

import jax
import jax.tree_util as jtu

from sparse_wf.api import PRNGKeyArray, PyTree

T = TypeVar("T")

R = TypeVar("R")
P = ParamSpec("P")

broadcast = jax.pmap(lambda x: x)
instance = functools.partial(jtu.tree_map, lambda x: x[0])

_p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def p_split(key: PRNGKeyArray) -> tuple[PRNGKeyArray, ...]:
    return _p_split(key)


def replicate(pytree: PyTree) -> PyTree:
    n = jax.local_device_count()
    stacked_pytree = jtu.tree_map(lambda x: jax.lax.broadcast(x, (n,)), pytree)
    return broadcast(stacked_pytree)


# Axis name we pmap over.
PMAP_AXIS_NAME = "qmc_pmap_axis"


# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
@functools.wraps(jax.pmap)
def pmap(fn: Callable[P, R], *p_args, **p_kwargs) -> Callable[P, R]:
    pmapped = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)(fn, *p_args, **p_kwargs)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return pmapped(*args, **kwargs)

    return wrapper


pmean = functools.partial(jax.lax.pmean, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(jax.lax.psum, axis_name=PMAP_AXIS_NAME)
pmax = functools.partial(jax.lax.pmax, axis_name=PMAP_AXIS_NAME)
pgather = functools.partial(jax.lax.all_gather, axis_name=PMAP_AXIS_NAME)
pall_to_all = functools.partial(jax.lax.all_to_all, axis_name=PMAP_AXIS_NAME)
pidx = functools.partial(jax.lax.axis_index, axis_name=PMAP_AXIS_NAME)


@overload
def jit(fun: None = None, *jit_args, **jit_kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def jit(fun: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]: ...


def jit(
    fun: Callable[P, R] | None = None, *jit_args, **jit_kwargs
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    def inner_jit(fun: Callable[P, R]) -> Callable[P, R]:
        jitted = jax.jit(fun, *jit_args, **jit_kwargs)

        @functools.wraps(fun)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cast(R, jitted(*args, **kwargs))

        return wrapper

    if fun is None:
        return inner_jit

    return inner_jit(fun)

@functools.wraps(folx.forward_laplacian)
def fwd_lap(f, argnums=None, sparsity_threshold=0.6):
    """Applies forward laplacian transform using the folx package, but adds the option to specifiy which args are being differentiated,
    and chooses sparse jacobians by default."""

    if argnums is None:
        # Take laplacian wrt to all arguments. This is the default of folx anyway so lets just use that
        return folx.forward_laplacian(f, sparsity_threshold=sparsity_threshold)

    if isinstance(argnums, int):
        argnums = (argnums,)
    argnums = sorted(argnums)
    assert len(set(argnums)) == len(argnums), "argnums must be unique"

    @functools.wraps(f)
    def transformed(*args):
        should_take_lap = [i in argnums for i in range(len(args))]

        # Create a new function that only depends on the argments that should be differentiated (specified by argnums)
        # and apply the forward laplacian transform to this function
        @functools.partial(folx.forward_laplacian, sparsity_threshold=sparsity_threshold)
        def func_with_only_args_to_diff(*args_to_diff_):
            # Combine the differentiable and non-differentiable arguments in their original order and pass them to the original function
            idx_arg_diff = 0
            combined_args = []
            for i, do_lap in enumerate(should_take_lap):
                if do_lap:
                    combined_args.append(args_to_diff_[idx_arg_diff])
                    idx_arg_diff += 1
                else:
                    combined_args.append(args[i])
            return f(*combined_args)

        args_to_diff = [arg for arg, do_lap in zip(args, should_take_lap) if do_lap]
        lap_array = func_with_only_args_to_diff(*args_to_diff)
        return lap_array

    return transformed

