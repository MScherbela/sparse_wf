from typing import Sequence, TypeVar
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util as jtu

from jaxtyping import PyTree, ArrayLike, Array

T = TypeVar("T", bound=PyTree[ArrayLike])


def tree_scale(tree: T, x: ArrayLike) -> T:
    return jtu.tree_map(lambda a: a * x, tree)


def tree_mul(tree: T, x: T | ArrayLike) -> T:
    if isinstance(x, ArrayLike):  # type: ignore
        return tree_scale(tree, x)
    return jtu.tree_map(lambda a, b: a * b, tree, x)


def tree_shift(tree1: T, x: ArrayLike) -> T:
    return jtu.tree_map(lambda a: a + x, tree1)


def tree_add(tree1: T, tree2: T | ArrayLike) -> T:
    if isinstance(tree2, ArrayLike):  # type: ignore
        return tree_shift(tree1, tree2)
    return jtu.tree_map(lambda a, b: a + b, tree1, tree2)


def tree_sub(tree1: T, tree2: T) -> T:
    return jtu.tree_map(lambda a, b: a - b, tree1, tree2)


def tree_dot(a: T, b: T) -> Array:
    return jtu.tree_reduce(jnp.add, jtu.tree_map(jnp.sum, jax.tree_map(jax.lax.mul, a, b)))


def tree_sum(tree: PyTree[ArrayLike]) -> Array:
    return jtu.tree_reduce(jnp.add, jtu.tree_map(jnp.sum, tree))


def tree_squared_norm(tree: PyTree[ArrayLike]) -> Array:
    return jtu.tree_reduce(jnp.add, jtu.tree_map(lambda x: jnp.einsum("...,...->", x, x), tree))


def tree_concat(trees: Sequence[T], axis: int = 0) -> T:
    return jtu.tree_map(lambda *args: jnp.concatenate(args, axis=axis), *trees)


def tree_split(tree: PyTree[Array], sizes: tuple[int]) -> tuple[PyTree[Array], ...]:
    idx = 0
    result: list[PyTree[Array]] = []
    for s in sizes:
        result.append(jtu.tree_map(lambda x: x[idx : idx + s], tree))
        idx += s
    result.append(jtu.tree_map(lambda x: x[idx:], tree))
    return tuple(result)


def tree_idx(tree: T, idx) -> T:
    return jtu.tree_map(lambda x: x[idx], tree)


def tree_expand(tree: T, axis) -> T:
    return jtu.tree_map(lambda x: jnp.expand_dims(x, axis), tree)


def tree_take(tree: T, idx, axis) -> T:
    def take(x):
        indices = idx
        if isinstance(indices, slice):
            slices = [slice(None)] * x.ndim
            slices[axis] = idx
            return x[tuple(slices)]
        return jnp.take(x, indices, axis)

    return jtu.tree_map(take, tree)


def tree_max(tree: T, axis=None) -> T:
    return jtu.tree_map(lambda x: jnp.max(x, axis), tree)


def tree_maximum(tree1: T, tree2: T) -> T:
    return jtu.tree_map(jnp.maximum, tree1, tree2)


def tree_zeros_like(tree, dtype=None, shape=None):
    return jtu.tree_map(lambda x: jnp.zeros_like(x, dtype, shape), tree)


def tree_to_flat_dict(tree: PyTree, prefix: str = "") -> dict:
    def to_str(k):
        if hasattr(k, "key"):
            return k.key
        elif hasattr(k, "idx"):
            return str(k.idx)
        elif hasattr(k, "name"):
            return k.name
        return str(k)

    out_dict = {}
    for key, v in jtu.tree_leaves_with_path(tree):
        key_string = "/".join([to_str(k) for k in key])
        out_dict[prefix + key_string] = v
    return out_dict


def ravel_with_padding(data, block_size: int):
    # This ravel function pads the dense tensor to a multiple of the block size

    n_elements = sum([p.size for p in jax.tree_util.tree_leaves(data)])
    dtype = jax.tree_util.tree_leaves(data)[0].dtype
    padding = jnp.zeros((-n_elements % block_size,), dtype=dtype)
    flat_data, unravel = jax.flatten_util.ravel_pytree((data, padding))

    def unravel_data(flat):
        return unravel(flat)[0]

    return flat_data, unravel_data
