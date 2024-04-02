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


@functools.partial(jax.jit, static_argnums=(2,))
def merge_dependencies(deps, fixed_deps, n_deps_max):
    new_deps = jnp.where(jnp.isin(deps, fixed_deps), NO_NEIGHBOUR, deps)

    n_new_deps_max = n_deps_max - len(fixed_deps)
    new_unique_deps = jnp.unique(new_deps, size=n_new_deps_max, fill_value=NO_NEIGHBOUR)
    deps_out = jnp.concatenate([fixed_deps, new_unique_deps], axis=-1)

    dep_map = multi_vmap(lambda d: jnp.argwhere(d == deps_out, size=1, fill_value=NO_NEIGHBOUR)[0][0], 2)(deps)
    return deps_out, dep_map


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
    fixed_deps = ind_dep[i]
    deps = get_with_fill(ind_dep, ind_neighbours[i])
    deps_out, dep_map = merge_dependencies(deps, fixed_deps, n_dependencies[1])

    print("Fixed deps: ")
    print(fixed_deps)
    print("")
    print("Dependecies: ")
    print(deps)
    print("")
    print("Merged deps: ")
    print(deps_out)
    print("")
    print("Dep map: ")
    print(dep_map)


