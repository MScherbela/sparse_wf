import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax
import jax.numpy as jnp
from sparse_wf.model.utils import MLP
from sparse_wf.jax_utils import nn_vmap, nn_multi_vmap
import flax.linen as nn
import time
import functools
from folx import ForwardLaplacianOperator, LoopLaplacianOperator
import numpy as np
print(jax.devices())

def contract(x, y, contraction_method):
    if contraction_method == "einsum":
        if x.ndim == 2:
            return jnp.einsum("jf,jf->f", x, y)
        else:
            return jnp.einsum("ijf,ijf->if", x, y)
    elif contraction_method == "product":
        return (x * y).sum(axis=-2)
    else:
        raise NotImplementedError


class MyModel(nn.Module):
    feature_dim: int = 256

    def setup(self):
        self.mlp1 = MLP([self.feature_dim], activate_final=True)
        self.mlp2 = MLP([self.feature_dim], activate_final=True)

    def _get_edges(self, r):
        return r[:, None, :] - r[None, :, :]


    def _embeddings_vectorized(self, r, contraction_method):
        edges = self._get_edges(r)
        x = self.mlp1(edges)
        y = self.mlp2(edges)
        return contract(x, y, contraction_method)

    def _embeddings_vmapped_edges(self, r, contraction_method):
        edges = self._get_edges(r)
        x = nn_multi_vmap(self.mlp1, [0, 0])(edges)
        y = nn_multi_vmap(self.mlp2, [0, 0])(edges)
        return contract(x, y, contraction_method)

    def _embeddings_vmapped_centers(self, r, contraction_method):
        def _get_emb(r_):
            edges = r_[None, :] - r
            x = self.mlp1(edges)
            y = self.mlp2(edges)
            return contract(x, y, contraction_method)
        return jax.vmap(_get_emb)(r)

    def __call__(self, r, method, contraction_method):
        if method == "vectorized":
            h = self._embeddings_vectorized(r, contraction_method)
        elif method == "vmapped_edges":
            h = self._embeddings_vmapped_edges(r, contraction_method)
        elif method == "vmapped_centers":
            h = self._embeddings_vmapped_centers(r, contraction_method)
        else:
            raise NotImplementedError
        return jax.nn.silu(h).sum()

n_el = 20
batch_size = 512
rng = jax.random.PRNGKey(0)
key_r, key_params = jax.random.split(rng)
r = jax.random.normal(key_r, (batch_size, n_el, 3))

model = MyModel(256)
params = model.init(key_params, r[0], "vectorized", "product")
model_jitted = jax.jit(jax.vmap(model.apply, in_axes=(None, 0, None, None)), static_argnums=(2,3))

@functools.partial(jax.jit, static_argnums=(2,3))
@functools.partial(jax.vmap, in_axes=(None, 0, None, None))
def laplacian(params, electrons, method, contraction_method):
    lap, jac = ForwardLaplacianOperator(0.6)(lambda r: model.apply(params, r, method, contraction_method))(electrons)
    # lap, jac = LoopLaplacianOperator()(lambda r: model.apply(params, r, method))(electrons)
    return lap


METHODS = ["vectorized", "vmapped_edges", "vmapped_centers"]
CONTRACTION_METHODS = ["einsum", "product"]
for method in METHODS:
    for contraction_method in CONTRACTION_METHODS:
        timings = []
        for i in range(5):
            t0 = time.perf_counter()
            h = model_jitted(params, r, method, contraction_method).block_until_ready()
            t1 = time.perf_counter()
            lap = laplacian(params, r, method, contraction_method).block_until_ready()
            t2 = time.perf_counter()
            timings.append((t1-t0, t2-t1))
        # print(h.sum(), lap.sum())
        timings = np.array(timings)
        timings = np.median(timings[1:], axis=0) * 1000
        print(f"{method:<17}+ {contraction_method:<8}, t_fwd = {timings[0]:3.1f} ms, t_lap = {timings[1]:5.1f} ms")


