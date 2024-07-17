import functools
import os
import sys
from typing import Callable, ParamSpec, TypeVar, cast, overload, Sequence

import numpy as np
import folx
import jax
import jax.numpy as jnp
import jax.core
import jax.tree_util as jtu

from sparse_wf.api import PRNGKeyArray
from sparse_wf.tree_utils import tree_squared_norm
import flax.linen as nn

T = TypeVar("T")

R = TypeVar("R")
P = ParamSpec("P")

broadcast = jax.pmap(lambda x: x)
instance = functools.partial(jtu.tree_map, lambda x: x[0])

_p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def p_split(key: PRNGKeyArray) -> tuple[PRNGKeyArray, ...]:
    return _p_split(key)


def replicate(pytree: T) -> T:
    n = jax.local_device_count()
    stacked_pytree = jtu.tree_map(lambda x: jax.lax.broadcast(x, (n,)), pytree)
    return broadcast(stacked_pytree)


def vector_to_tree_like(v: jax.Array, tree):
    tree_flat, tree_def = jtu.tree_flatten(tree)
    tree_shapes = [x.shape for x in tree_flat]
    sizes = [int(np.prod(x)) for x in tree_shapes]
    assert sum(sizes) == len(v), f"Size of vector {len(v)} does not match size of tree {sum(sizes)}"

    output_list = []
    start = 0
    for shape, size in zip(tree_shapes, sizes):
        output_list.append(v[start : start + size].reshape(shape))
        start += size
    return jtu.tree_unflatten(tree_def, output_list)


# Axis name we pmap over.
PMAP_AXIS_NAME = "qmc_pmap_axis"
pmean = functools.partial(jax.lax.pmean, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(jax.lax.psum, axis_name=PMAP_AXIS_NAME)
pmax = functools.partial(jax.lax.pmax, axis_name=PMAP_AXIS_NAME)
pmin = functools.partial(jax.lax.pmin, axis_name=PMAP_AXIS_NAME)
pgather = functools.partial(jax.lax.all_gather, axis_name=PMAP_AXIS_NAME)
pall_to_all = functools.partial(jax.lax.all_to_all, axis_name=PMAP_AXIS_NAME)
pidx = functools.partial(jax.lax.axis_index, axis_name=PMAP_AXIS_NAME)


def only_in_pmap(fun):
    @functools.wraps(fun)
    def wrapper(x):
        try:
            jax.core.axis_frame(PMAP_AXIS_NAME)
            return fun(x)
        except NameError:
            return x

    return wrapper


pmax_if_pmap = only_in_pmap(pmax)
psum_if_pmap = only_in_pmap(psum)


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


M = TypeVar("M", bound=nn.Module)


def nn_multi_vmap(
    module: M, in_axes: Sequence[int | None | Sequence[int | None]], out_axes: int | Sequence[int] = 0
) -> M:
    if isinstance(out_axes, int):
        out_axes = [out_axes] * len(in_axes)
    assert len(in_axes) == len(out_axes)

    def call_module(m, *args):
        return m(*args)

    for in_ax, out_ax in zip(in_axes, out_axes):
        call_module = nn.vmap(
            call_module,
            variable_axes={"params": None, "scaling_factors": None},
            split_rngs={"params": False, "scaling_factors": False},
            in_axes=in_ax,  # type: ignore # too restrictive typing by flax
            out_axes=out_ax,
        )
    return cast(M, functools.partial(call_module, module))


def nn_vmap(module: M, in_axes: int | None | Sequence[int | None] = 0, out_axes: int = 0) -> M:
    return nn_multi_vmap(module, [in_axes], [out_axes])


@overload
def vectorize(fun: None = None, *vec_args, **vec_kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def vectorize(fun: Callable[P, R], *vec_args, **vec_kwargs) -> Callable[P, R]: ...


def vectorize(
    fun: Callable[P, R] | None = None, *vec_args, **vec_kwargs
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    def inner_jit(fun: Callable[P, R]) -> Callable[P, R]:
        vectorized = jnp.vectorize(fun, *vec_args, **vec_kwargs)

        @functools.wraps(fun)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cast(R, vectorized(*args, **kwargs))

        return wrapper

    if fun is None:
        return inner_jit

    return inner_jit(fun)


@overload
def pmap(fun: None = None, *jit_args, **jit_kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def pmap(fun: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]: ...


# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
def pmap(
    fun: Callable[P, R] | None = None, *pmap_args, **pmap_kwargs
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    def inner_pmap(fun: Callable[P, R]) -> Callable[P, R]:
        pmapped = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)(fun, *pmap_args, **pmap_kwargs)

        @functools.wraps(fun)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cast(R, pmapped(*args, **kwargs))

        return wrapper

    if fun is None:
        return inner_pmap

    return inner_pmap(fun)


@functools.wraps(folx.forward_laplacian)
def fwd_lap(f, argnums=None, sparsity_threshold: float | int = 0):
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


def copy_from_main(x: T) -> T:
    def _copy_from_main(x):
        return jax.tree_util.tree_map(lambda x: pgather(x, axis=0)[0], x)

    return pmap(_copy_from_main)(x)


def is_main_process():
    return int(os.environ.get("SLURM_PROCID", 0)) == 0


class ChildProcessSkip(Exception): ...


class MainProcessExecuteContext:
    def __enter__(self):
        if not is_main_process():
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        raise ChildProcessSkip()

    def __exit__(self, type, value, traceback):
        if type is None:
            return  # No exception
        if issubclass(type, ChildProcessSkip):
            return True  # Suppress special SkipWithBlock exception


@overload
def only_on_main_process(func: Callable[P, R]) -> Callable[P, R | None]: ...


@overload
def only_on_main_process(func: None = None) -> MainProcessExecuteContext: ...


def only_on_main_process(
    func: Callable[P, R] | None = None,
) -> Callable[P, R | None] | MainProcessExecuteContext:
    if callable(func):

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            if is_main_process():
                return func(*args, **kwargs)
            return None

        return wrapper
    else:
        return MainProcessExecuteContext()


def assert_identical_copies(x, threshold=1e-8):
    @pmap
    def check_tree_identity(x):
        main = jax.tree_util.tree_map(lambda x: pgather(x, axis=0)[0], x)
        diff = jax.tree_util.tree_map(jnp.subtract, x, main)
        delta = tree_squared_norm(diff) ** 0.5
        is_okay = delta <= threshold
        is_okay = pmin(is_okay)
        return is_okay, delta

    is_okay, delta = check_tree_identity(x)
    assert is_okay.any().item(), f"Tensors are not identical! Delta is {delta.ravel()[0].item()}"


def get_from_main_process(data, has_device_axis=False):
    if not has_device_axis:
        data = jax.tree_util.tree_map(lambda x: x.reshape((1, *x.shape)), data)
        data = jax.tree_util.tree_map(lambda x: jnp.tile(x, [jax.local_device_count()] + [1] * (x.ndim - 1)), data)

    def get_from_main(x):
        return pgather(x, axis=0)[0]

    data_on_main = pmap(lambda x: jax.tree_util.tree_map(get_from_main, x))(data)

    if not has_device_axis:
        data_on_main = jax.tree_util.tree_map(lambda x: x[0], data_on_main)
    return data_on_main


def rng_sequence(key):
    while True:
        key, subkey = jax.random.split(key)
        yield subkey
