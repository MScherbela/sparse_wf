# %%
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def cutoff_function(d, p=4):
    a = -(p + 1) * (p + 2) * 0.5
    b = p * (p + 2)
    c = -p * (p + 1) * 0.5
    cutoff = 1 + a * d**p + b * d ** (p + 1) + c * d ** (p + 2)
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

class MessagePassingLayer(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, h_center, h_neighbors, diff_features):
        h_center = nn.Dense(self.out_dim, name="proj_i")(h_center)
        filter_kernel = nn.Dense(self.out_dim, use_bias=False, name="Gamma")(diff_features)
        
        h_out = jax.nn.silu(h_center + jnp.sum(filter_kernel * h_neighbors, axis=-2))
        return h_out
    
class InitialEmbeddings(nn.Module):
    out_dim: int
    depth: int = 2

    @nn.compact
    def __call__(self, diff, diff_features):
        dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        g = jnp.concatenate([diff, dist], axis=-1)
        g = MLP(self.out_dim, self.depth, activate_final=True)(g)

        filter_kernel = nn.Dense(self.out_dim, use_bias=False, name="Gamma")(diff_features)
        return jnp.sum(g * filter_kernel, axis=-2)
    

class PairwiseFeatures(nn.Module):
    width_hidden: int
    width_out: int
    n_envelopes: int
    cutoff: float

    def scale_initializer(self, rng, shape):
        noise = 1 + 0.25 * jax.random.normal(rng, shape)
        return self.cutoff * noise * 0.5

    @nn.compact
    def __call__(self, diff):
        dist = jnp.linalg.norm(diff, axis=-1)
        scales = self.param("scales", self.scale_initializer, (self.n_envelopes,))
        scales = jax.nn.softplus(scales)
        envelopes = jnp.exp(-dist[:, :, None]**2 / scales)
        envelopes = nn.Dense(self.width_out)(envelopes)

        diff = nn.Dense(self.width_hidden)(diff)
        diff = jax.nn.silu(diff)
        diff = nn.Dense(self.width_out)(diff)

        return envelopes * diff * cutoff_function(dist / self.cutoff)[:, :, None]
    
class OrbitalLayer(nn.Module):
    R_orb: np.ndarray

    @nn.compact
    def __call__(self, r, h_out):
        n_orb = len(self.R_orb)
        dist_el_orb = jnp.linalg.norm(r[:, None, :] - self.R_orb[None, :, :], axis=-1)
        orbital_envelope = jnp.exp(-dist_el_orb * 0.2)
        phi = nn.Dense(n_orb)(h_out) * orbital_envelope
        return phi

def get_neighbours(x, ind_neighbour=None):
    assert x.ndim == 2
    assert (ind_neighbour is None) or (ind_neighbour.ndim == 2)
    if ind_neighbour is None:
        return x[None, :, :]
    else:
        return x[ind_neighbour]
   
class SparseWavefunction(nn.Module):
    R_orb: np.ndarray
    cutoff: float
    width: int = 64
    depth: int = 3
    beta_width_hidden: int = 16
    beta_width_out: int = 8
    beta_n_envelpoes: int = 16

    @nn.compact
    def __call__(self, r, ind_neighbour=None, weight_neighbour=None):
        r_neighbour = get_neighbours(r, ind_neighbour)
        diff = r[:, None, :] - r_neighbour
        beta = PairwiseFeatures(self.beta_width_hidden, self.beta_width_out, self.beta_n_envelpoes, self.cutoff)(diff)
        if weight_neighbour is not None:
            beta *= weight_neighbour[:, :, None]

        h0 = InitialEmbeddings(self.width)(diff, beta)
        h = MLP(self.width, self.depth, activate_final=False)(h0)
        h_neighbour = get_neighbours(h, ind_neighbour)
        h_out = MessagePassingLayer(self.width)(h0, h_neighbour, beta)
        phi = OrbitalLayer(self.R_orb)(r, h_out)
        return phi
        # return jnp.slogdet(phi)[1]

