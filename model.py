#%%
import flax.linen as nn
from graph import Edges, build_edges, get_affected_electrons
import jax
import jax.numpy as jnp
import numpy as np

def cutoff_envelope(d, p=4):
    a = -(p+1)*(p+2) * 0.5
    b = p*(p+2)
    c = -p*(p+1) * 0.5
    cutoff = 1 + a * d**p + b * d**(p+1) + c * d**(p+2)
    return jnp.where(d < 1, cutoff, 0.0)



class MLP(nn.Module):
    width: int
    depth: int
    activate_final: bool

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = nn.Dense(self.width)(x)
            if (i < self.depth - 1) or self.activate_final:
                x = jax.nn.silu(x)
        return x

class SparseEmbedding(nn.Module):
    width_1el: int
    width_2el: int
    depth: int
    cutoff: float

    @nn.compact
    def __call__(self, r, edges:Edges):
        distance_scale = 10.0
        h1 = MLP(self.width_1el, self.depth, activate_final=False)(r / distance_scale)

        h2 = MLP(self.width_2el, self.depth, activate_final=True)(edges.diff / distance_scale)
        kernel = cutoff_envelope(edges.dist / self.cutoff) * edges.weight
        h2 = jnp.einsum("...ij,...ijf->...if", kernel, h2) # i,j = electrons; f = features

        h = h1 + nn.Dense(self.width_1el, use_bias=False)(h2)
        return h
    
class Wavefunction(nn.Module):
    width_1el: int
    width_2el: int
    depth: int
    n_orbitals: int
    cutoff: float
    R: jnp.array

    @nn.compact
    def __call__(self, r, edges:Edges):
        n_el, n_neighbors = edges.j.shape[-2:]
        print(f"Compiling model for n_el={n_el}, n_neighbors={n_neighbors}")
        h = SparseEmbedding(self.width_1el, self.width_2el, self.depth, self.cutoff)(r, edges)
        phi = nn.Dense(self.n_orbitals)(h) # project to orbitals

        # el_nuc_dist = jnp.linalg.norm(r[..., : , None, :] - self.R, axis=-1)
        R_orb = self.R
        envelope_dist = jnp.linalg.norm(r[..., :, None, :] - R_orb, axis=-1)
        envelope = jnp.exp(-envelope_dist / 10.0)
        return phi * envelope

    





