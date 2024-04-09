import functools
from typing import Callable, TypeVar, cast, ParamSpec, overload

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
