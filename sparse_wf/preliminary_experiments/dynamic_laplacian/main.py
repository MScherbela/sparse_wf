#%%
import jax.numpy as jnp
import folx
from sparse_wf.preliminary_experiments.dynamic_laplacian.model import build_layers
import numpy as np
import functools
import jax


def get_neighbours(r, cutoff):
    """Get indices of neighbours within cutoff"""
    n_el = r.shape[-2]
    distance_matrix = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    in_cutoff = distance_matrix + cutoff * np.eye(n_el) < cutoff
    n_neighbours = jnp.max(jnp.sum(in_cutoff, axis=-1))
    ind_neighbours = jax.vmap(lambda x: jnp.nonzero(x, size=n_neighbours)[0])(in_cutoff)
    return ind_neighbours

@jax.vmap
@functools.partial(folx.forward_laplacian, sparsity_threshold=0)
def build_initial_input(r, r_neighbours):
    """Build input features from coordinates and convert them to folx tuples"""
    h = r
    h_neighbour = r_neighbours
    diff = r - r_neighbours
    return h, h_neighbour, diff

def get_neighbor_embeddings(h, ind_neighbours):
    """Gather embeddings of the neighbours specified by ind_neighbours"""
    h_neighbour = folx.api.FwdLaplArray(
        h.x[ind_neighbours], 
        folx.api.FwdJacobian(data=h.jacobian.data[ind_neighbours].swapaxes(1, 2)),
        h.laplacian[ind_neighbours])
    return h_neighbour

if __name__ == '__main__':
    width = 64
    n_layers = 3
    n_el = 20
    cutoff = 5.0

    layers = build_layers(width, n_layers)
    layers = [folx.forward_laplacian(l) for l in layers]

    rng = jax.random.PRNGKey(0)
    r = jax.random.normal(rng, (n_el, 3)) + np.arange(n_el)[:, None] * np.array([1, 0, 0]) 
    ind_neighbours = get_neighbours(r, cutoff)
    n_neighbours = ind_neighbours.shape[-1]
    print(f"{n_el} electrons, each with {n_neighbours} neighbours: Dense input dim = (1 + {n_neighbours}) * 3 = {3 * (n_neighbours + 1)}")

    h, h_neighbour, diff_neighbour = build_initial_input(r, r[ind_neighbours])
    for ind_layer, layer in enumerate(layers):
        print(f"Layer {ind_layer}")
        print("Jac input shapes: h={h.jacobian.data.shape}, h_neighbour={h_neighbour.jacobian.data.shape}, diff={diff_neighbour.jacobian.data.shape}")
        h = jax.vmap(layer)(h, h_neighbour, diff_neighbour)
        print(f"Jac output shape: h={h.jacobian.data.shape}")
        print("-"*20)
        h_neighbour = get_neighbor_embeddings(h, ind_neighbours)

