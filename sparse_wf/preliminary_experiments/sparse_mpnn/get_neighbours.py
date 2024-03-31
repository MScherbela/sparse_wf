# %%
import numpy as np
import jax.numpy as jnp
import jax
import functools

NO_NEIGHBOUR = 1_000_000


@functools.partial(jax.jit, static_argnums=(2,))
def _get_cutoff_matrix(r, cutoff, include_self=False):
    dist = jnp.linalg.norm(r[..., :, None, :] - r[..., None, :, :], axis=-1)
    if not include_self:
        dist += np.inf * jnp.eye(r.shape[-2])
    in_cutoff = dist < cutoff
    max_n_neighbours_1 = jnp.max(jnp.sum(in_cutoff, axis=-1))
    return in_cutoff, max_n_neighbours_1


@functools.partial(jax.jit, static_argnums=(1,))
@functools.partial(jax.vmap, in_axes=(0, None))
def _get_ind_neighbours(in_cutoff, n_max):
    indices = jnp.nonzero(in_cutoff, size=n_max, fill_value=NO_NEIGHBOUR)[0]
    indices = jnp.unique(indices, size=n_max, fill_value=NO_NEIGHBOUR)
    return indices


def get_ind_neighbours(r, cutoff, include_self=False):
    in_cutoff, n_neighbours = _get_cutoff_matrix(r, cutoff, include_self)
    n_neighbours = int(n_neighbours)
    ind_neighbours = _get_ind_neighbours(in_cutoff, n_neighbours)
    return ind_neighbours


def get_max_nr_of_dependencies(r, cutoff, n_steps_max):
    cutoffs = cutoff * jnp.arange(1, n_steps_max + 1)
    dist = jnp.linalg.norm(r[..., :, None, :] - r[..., None, :, :], axis=-1, keepdims=True)
    in_cutoff = dist < cutoffs
    n_dependencies = jnp.sum(in_cutoff, axis=-2)
    return jnp.max(n_dependencies, axis=tuple(np.arange(n_dependencies.ndim - 1)))


def get_with_fill(arr, ind, fill=NO_NEIGHBOUR):
    return arr.at[ind].get(mode="fill", fill_value=fill)


def multi_vmap(f, n):
    for _ in range(n):
        f = jax.vmap(f)
    return f


@functools.partial(jax.jit, static_argnums=(1,))
def merge_dependencies(ind_dep_in, n_dep_out_max=None):
    """
    Get the set of common dependencies and a translation map between the input indices and the common dependencies.

    Args:
    -----
    ind_dep_in: jax.Array of shape [..., n_elements, n_dep_in] containing the indices of the electrons, that each element depends on.
    n_dep_out_max: int, optional. The maximum number of dependencies after the merge. If None, it is assumed there are no overlapping input dependcies
        and therefore n_dep_out_max = n_elements * n_dep_in.
    """
    n_elements, n_dep_in = ind_dep_in.shape[-2:]
    n_batch_dims = ind_dep_in.ndim - 2

    n_dep_in_total = n_elements * n_dep_in
    ind_dep_in = ind_dep_in.reshape(ind_dep_in.shape[:-2] + (n_dep_in_total,))
    n_dep_in_total = ind_dep_in.shape[-1]
    n_dep_out_max = n_dep_out_max if n_dep_out_max is not None else n_dep_in_total

    def get_ind_dep_out(i):
        return jnp.unique(i, return_inverse=True, size=n_dep_out_max, fill_value=NO_NEIGHBOUR)

    ind_dep_out, dep_map = multi_vmap(get_ind_dep_out, n_batch_dims)(ind_dep_in)

    n_dep_out = jnp.max(jnp.sum(ind_dep_out != NO_NEIGHBOUR, axis=-1))
    dep_map = dep_map.reshape(ind_dep_in.shape[:-1] + (n_elements, n_dep_in))
    return ind_dep_out, dep_map, n_dep_out


if __name__ == "__main__":
    n_el = 50

    rng_r = jax.random.PRNGKey(0)
    R = jnp.arange(-n_el // 2, n_el // 2)[:, None] * jnp.array([1, 0, 0])
    r = jax.random.normal(rng_r, (n_el, 3)) + R

    n_dependencies = get_max_nr_of_dependencies(r, cutoff=3.0, n_steps_max=2)
    n_dependencies = [int(n) for n in n_dependencies]
    ind_neighbours = get_ind_neighbours(r, cutoff=4.0, include_self=False)
    ind_dep = jnp.concatenate([jnp.arange(n_el)[:, None], ind_neighbours], axis=-1)

    i = 1
    ind_dep_merged = merge_dependencies(get_with_fill(ind_dep, ind_neighbours[i]), n_dep_out_max=n_dependencies[1])
