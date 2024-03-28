#%%
import numpy as np
import jax.numpy as jnp
import jax
import functools

NO_NEIGHBOUR = 1_000_000

n_el = 50

rng_r = jax.random.PRNGKey(0)
R = jnp.arange(-n_el // 2, n_el // 2)[:, None] * jnp.array([1, 0, 0])
r = jax.random.normal(rng_r, (n_el, 3)) + R

def get_connectivity(r: jax.Array, 
                     cutoff: float, 
                     n_steps_max: int):
    """
    Compute the n-step connectivity matrices for given coordinates and cutoff.

    Returns:
    -------
    neighbour_indices: List[jax.Array] of shape (n_el, n_neighbours_max[l]). 
        For each level l, the matrix neighbour_indices[n, m] contains the unique indices of electrons that 
        can interact with electron n via at most l steps.
    maps_reverse: List[jax.Array] of shape (n_el, n_neighbours_max[1], n_neighbours_max[l-1]) with values in [0...n_neighbours_max[l]]
        For each level l contains the mapping from the indices of the neighbours at level l-1 to the indices of the neighbours at level l.    
    """
    @jax.jit
    def get_cutoff_matrix(r, cutoff):
        dist = jnp.linalg.norm(r[..., :, None, :] - r[..., None, :, :], axis=-1)
        in_cutoff = dist < cutoff
        max_n_neighbours_1 = jnp.max(jnp.sum(in_cutoff, axis=-1))
        return in_cutoff, max_n_neighbours_1
    
    @functools.partial(jax.jit, static_argnums=(1,))
    @functools.partial(jax.vmap, in_axes=(0, None))
    def get_ind_neighbours_1step(in_cutoff, n_max):
        indices = jnp.nonzero(in_cutoff, size=n_max, fill_value=NO_NEIGHBOUR)[0]
        indices = jnp.unique(indices, size=n_max, fill_value=NO_NEIGHBOUR)
        return indices

    @jax.jit
    def get_ind_neighbours_next_step(ind_neighbours_1, ind_neighbours):
        max_n_neighbours_1 = ind_neighbours_1.shape[-1]
        max_n_neighbours_current = ind_neighbours.shape[-1]
        max_n_neighbours_next = max_n_neighbours_1 * max_n_neighbours_current

        ind_neighbours = ind_neighbours_1.at[ind_neighbours].get(mode="fill", fill_value=NO_NEIGHBOUR)
        ind_neighbours = ind_neighbours.reshape(ind_neighbours.shape[:-2] + (-1,))
        ind_neighbours, map_reverse = jax.vmap(lambda i: jnp.unique(i, size=max_n_neighbours_next, fill_value=NO_NEIGHBOUR, return_inverse=True))(ind_neighbours)
        max_n_neigbours = jnp.max(jnp.sum(ind_neighbours != NO_NEIGHBOUR, axis=-1))
        return ind_neighbours, map_reverse, max_n_neigbours
    

    in_cutoff, max_n_neighbours_1 = get_cutoff_matrix(r, cutoff)
    max_n_neighbours_1 = int(max_n_neighbours_1)
    ind_neighbours_1 = get_ind_neighbours_1step(in_cutoff, max_n_neighbours_1)
    ind_neighbours = ind_neighbours_1

    neighbour_indices = [ind_neighbours_1]
    maps_reverse = [None]
    for n in range(1, n_steps_max):
        ind_neighbours, map_reverse, max_n_neighbours = get_ind_neighbours_next_step(ind_neighbours_1, ind_neighbours)
        ind_neighbours = ind_neighbours[..., :max_n_neighbours]
        map_reverse = map_reverse.reshape(map_reverse.shape[:-1] + (max_n_neighbours_1, -1))
        neighbour_indices.append(ind_neighbours)
        maps_reverse.append(map_reverse)
    return neighbour_indices, maps_reverse

if __name__ == '__main__':
    ind_neighbours, maps_reverse = get_connectivity(r, cutoff=5.0, n_steps_max = 4)
    for n, (ind, maps) in enumerate(zip(ind_neighbours, maps_reverse)):
        print(f"Step {n}: {ind.shape=};" + (f" {maps.shape=}: [{maps.min()}...{maps.max()}]" if maps is not None else ""))
