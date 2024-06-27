from sparse_wf.model.utils import (
    ElecInp,
    ElecNucDifferences,
    ElecNucDistances,
    cutoff_function,
    init_glu_feedforward,
    apply_glu_feedforward,
    swap_bottom_blocks,
)
from sparse_wf.api import Charges, SlaterMatrices, Parameters
from sparse_wf.tree_utils import tree_idx
from typing import Optional
import numpy as np
import einops
from typing import Protocol

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp


class Envelope(Protocol):
    def apply(self, params: Parameters, diffs: ElecNucDifferences) -> jax.Array: ...
    def init(self, key, diffs: ElecNucDifferences) -> Parameters: ...


class IsotropicEnvelope(nn.Module):
    n_determinants: int
    n_orbitals: int
    cutoff: Optional[float] = None

    def _sigma_initializer(self, key, shape, dtype=jnp.float32):
        assert shape[-1] == self.envelope_size
        scale = jnp.geomspace(0.2, 10.0, self.envelope_size)
        scale *= jax.random.truncated_normal(key, 0.5, 1.5, shape, dtype)
        return scale.astype(jnp.float32)

    @nn.compact
    def __call__(self, diffs: ElecNucDifferences) -> jax.Array:
        dists = jnp.linalg.norm(diffs, axis=-1)
        n_nuc = dists.shape[-1]
        sigma = self.param(
            "sigma", self._sigma_initializer, (n_nuc, self.n_determinants * self.n_orbitals), jnp.float32
        )
        sigma = nn.softplus(sigma)
        pi = self.param("pi", jnn.initializers.ones, (n_nuc, self.n_determinants * self.n_orbitals), jnp.float32)
        scaled_dists = dists[..., None] * sigma
        env = jnp.exp(-scaled_dists)
        if self.cutoff is not None:
            env *= cutoff_function(dists / self.cutoff)
        out = jnp.einsum("...nd,nd->...d", env, pi)
        return out


class EfficientIsotropicEnvelopes(nn.Module):
    n_determinants: int
    n_orbitals: int
    n_envelopes: int
    cutoff: Optional[float] = None

    @nn.compact
    def __call__(self, diffs: ElecNucDifferences) -> jax.Array:
        dists = jnp.linalg.norm(diffs, axis=-1)
        n_nuc = dists.shape[-1]
        sigma = self.param("sigma", jnn.initializers.ones, (n_nuc, self.n_determinants, self.n_envelopes), jnp.float32)
        sigma = nn.softplus(sigma)
        pi = self.param(
            "pi",
            jnn.initializers.normal(1 / jnp.sqrt(self.n_envelopes)),
            (n_nuc, self.n_determinants, self.n_envelopes, self.n_orbitals),
            jnp.float32,
        )
        scaled_dists = dists[..., None, None] * sigma
        env = jnp.exp(-scaled_dists)
        if self.cutoff is not None:
            env *= cutoff_function(dists / self.cutoff)
        out = jnp.einsum("...nde,ndeo->...do", env, pi)
        return out.reshape(*out.shape[:-2], -1)


class GLUEnvelopes(Envelope):
    def __init__(
        self,
        Z: Charges,
        n_determinants: int,
        n_orbitals: int,
        n_envelopes: int,
        width: int,
        depth: int,
        cutoff: Optional[float] = None,
    ):
        assert cutoff is None, "Cutoff not implemented for GLU envelopes"
        self.Z = Z
        self.n_determinants = n_determinants
        self.n_orbitals = n_orbitals
        self.n_envelopes = n_envelopes
        self.width = width
        self.depth = depth
        self.cutoff = cutoff
        unique_Z = {z: i for i, z in enumerate(np.unique(self.Z))}
        self.n_unique_Z = len(unique_Z)
        self.atom_types = np.array([unique_Z[z] for z in self.Z], int)

    def init(self, key, diffs):
        input_dim = 3
        rngs = jax.random.split(key, 3)
        rngs_glu = jax.random.split(rngs[0], self.n_unique_Z)
        glu_params = jax.vmap(lambda k: init_glu_feedforward(k, self.width, self.depth, input_dim, self.n_envelopes))(
            rngs_glu
        )
        sigma = 1.0 + jax.random.normal(rngs[1], (self.n_unique_Z, self.n_envelopes), jnp.float32) * 0.1
        pi = jax.random.normal(
            rngs[2], (len(self.Z), self.n_envelopes, self.n_determinants * self.n_orbitals), jnp.float32
        ) / jnp.sqrt(self.n_envelopes)
        return dict(glu=glu_params, sigma=sigma, pi=pi)

    def apply(self, params, diffs):
        assert diffs.ndim == 2, "GLU envelopes expects diffs shape (n_nuclei, 3); use vmap for multiple electrons"
        dists = jnp.linalg.norm(diffs, axis=-1)
        glu_params = tree_idx(params["glu"], self.atom_types)
        sigma = jax.nn.softplus(params["sigma"][self.atom_types])
        pi = params["pi"]
        glu_outputs = jax.vmap(apply_glu_feedforward, in_axes=0)(glu_params, diffs)  # vmap over nuclei
        exponents = jnp.exp(-sigma * dists[..., None])
        basis_functions = glu_outputs * exponents  # (..., n_nuclei, n_envelopes)
        envelopes = jnp.einsum("Iek,...Ie->...k", pi, basis_functions)
        return envelopes


class SlaterOrbitals(nn.Module):
    n_determinants: int
    envelope_size: int
    spins: tuple[int, int]

    @nn.compact
    def __call__(self, h_one: ElecInp, dists: ElecNucDistances) -> SlaterMatrices:
        n_el = h_one.shape[-2]
        spins = np.array(self.spins)
        orbitals = nn.Dense(self.n_determinants * n_el)(h_one)
        orbitals *= EfficientIsotropicEnvelopes(self.n_determinants, n_el, self.envelope_size)(dists)
        orbitals = einops.rearrange(
            orbitals, "... el (det orb) -> ... det el orb", el=n_el, orb=n_el, det=self.n_determinants
        )
        orbitals = swap_bottom_blocks(orbitals, spins[0])  # reverse bottom two blocks
        return (orbitals,)
