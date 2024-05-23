import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax
import jax.numpy as jnp
import time
import functools
from folx import ForwardLaplacianOperator, LoopLaplacianOperator
import numpy as np
print(jax.devices())

def mlp(params, x):
    for p in params:
        x = jnp.dot(x, p["w"]) + p["b"]
        x = jax.nn.silu(x)
    return x

class MyModel:
    def _get_edges(self, r):
        return r[:, None, :] - r[None, :, :]

    def init(self):
        feature_dim = 256
        n_layers = 1
        params = []
        for _ in range(2):
            params.append([])
            input_dim = 3
            for _ in range(n_layers):
                params[-1].append(dict(w=jnp.ones([input_dim, feature_dim]), b=jnp.zeros(feature_dim)))
                input_dim = feature_dim
        return params

    def __call__(self, params, r, contraction_method):
        edges = self._get_edges(r)
        x = mlp(params[0], edges)
        y = mlp(params[1], edges)
        if contraction_method == "einsum":
            h = jnp.einsum("...jf,...jf->...f", x, y)
        elif contraction_method == "product":
            h = (x * y).sum(axis=-2)
        else:
            raise NotImplementedError
        return jax.nn.silu(h).sum()

n_el = 20
batch_size = 512
r = jnp.ones([batch_size, n_el, 3])

model = MyModel()
params = model.init()
model_jitted = jax.jit(jax.vmap(model, in_axes=(None, 0, None)), static_argnums=(2,))

@functools.partial(jax.jit, static_argnums=(2,))
@functools.partial(jax.vmap, in_axes=(None, 0, None))
def fwd_lap_jitted(params, electrons, contraction_method):
    lap, jac = ForwardLaplacianOperator(0.6)(lambda r: model(params, r, contraction_method))(electrons)
    return lap

@functools.partial(jax.jit, static_argnums=(2,))
@functools.partial(jax.vmap, in_axes=(None, 0, None))
def loop_lap_jitted(params, electrons, contraction_method):
    lap, jac = LoopLaplacianOperator()(lambda r: model(params, r, contraction_method))(electrons)
    return lap


CONTRACTION_METHODS = ["einsum", "product"]
for contraction_method in CONTRACTION_METHODS:
    timings = []
    for i in range(5):
        t0 = time.perf_counter()
        h = model_jitted(params, r, contraction_method).block_until_ready()
        t1 = time.perf_counter()
        lap = fwd_lap_jitted(params, r, contraction_method).block_until_ready()
        t2 = time.perf_counter()
        lap = loop_lap_jitted(params, r, contraction_method).block_until_ready()
        t3 = time.perf_counter()
        timings.append((t1-t0, t2-t1, t3-t2))
    timings = np.array(timings)
    timings = np.median(timings[1:], axis=0) * 1000 # drop the first run (compilation)
    print(f"{contraction_method:<8}, t_fwd_pass = {timings[0]:3.1f} ms, t_fwd_lap = {timings[1]:5.1f} ms, t_loop_lap = {timings[2]:5.1f} ms")

# Output when run on a single A100
# einsum  , t_fwd_pass = 1.2 ms, t_fwd_lap = 288.2 ms, t_loop_lap = 299.4 ms
# product , t_fwd_pass = 1.2 ms, t_fwd_lap =  54.7 ms, t_loop_lap = 308.8 ms


