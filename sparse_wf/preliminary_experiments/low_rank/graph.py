# %%
import functools

import chex
import jax
import jax.numpy as jnp

# TODO: choose something nicer
SENTINEL_INDEX = 100_000


@chex.dataclass
class Edges:
    j: jnp.array
    weight: jnp.array
    diff: jnp.array
    dist: jnp.array

    @jax.vmap
    def get_subset(self, electron_indices):
        return Edges(
            j=self.j[electron_indices, :],
            weight=self.weight[electron_indices, :],
            diff=self.diff[electron_indices, :],
            dist=self.dist[electron_indices],
        )


@functools.partial(jax.jit, static_argnums=(3,))
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def _build_edges(diff, dist, in_cutoff, n_edges_max):
    ind_neighbor = jnp.nonzero(in_cutoff, size=n_edges_max, fill_value=SENTINEL_INDEX)[0]
    return Edges(
        j=ind_neighbor,
        weight=jnp.where(ind_neighbor != SENTINEL_INDEX, 1.0, 0.0),
        diff=diff[ind_neighbor],
        dist=dist[ind_neighbor],
    )


@jax.jit
def get_diff_dist(r):
    diff = r[..., :, None, :] - r[..., None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist


@jax.jit
@jax.vmap
def _get_affected_electrons(i_moved, edges_old, edges_new):
    js_old = edges_old.j[..., i_moved, :]
    js_new = edges_new.j[..., i_moved, :]
    js_merged = jnp.concatenate([js_old, js_new], axis=-1)
    affected_electrons = jnp.unique(js_merged, size=js_merged.shape[-1], fill_value=SENTINEL_INDEX)
    return affected_electrons


def get_affected_electrons(i_moved, edges_old, edges_new):
    # jited
    indices_affected = _get_affected_electrons(i_moved, edges_old, edges_new)

    # not jited
    n_affected_max = int(jnp.max(jnp.sum(indices_affected != SENTINEL_INDEX, axis=-1)))
    return indices_affected[:, :n_affected_max]


def build_edges(r, cutoff):
    # jited
    diff, dist = get_diff_dist(r)

    # not jited
    in_cutoff = dist <= cutoff
    n_edges_max = int(jnp.max(in_cutoff.sum(axis=-1)))

    # jited
    edges = _build_edges(diff, dist, in_cutoff, n_edges_max)
    return edges


if __name__ == "__main__":
    batch_size = 50
    n_el = 20
    rng = jax.random.PRNGKey(0)
    r = jnp.arange(n_el)[:, None] + jax.random.normal(rng, (batch_size, n_el, 3)) * 1
    cutoff = 5.0
    edges = build_edges(r, cutoff)
