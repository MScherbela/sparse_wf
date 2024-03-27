# %%
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import functools

class MLP(nn.Module):
    width: int
    depth: int
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = nn.Dense(self.width)(x)
            if (i < self.depth - 1) or self.activate_final:
                x = jax.nn.silu(x)
        return x

class MessagePassingLayer(nn.Module):
    width: int

    @nn.compact
    def __call__(self, h_center, h_neighbors, differences):
        filter_kernel = MLP(width=self.width, depth=2)(differences)
        h_neighbors = nn.Dense(self.width)(h_neighbors)
        msg = filter_kernel * h_neighbors
        msg = jnp.sum(msg, axis=-2)
        
        h_out = nn.Dense(self.width)(h_center) + msg
        h_out = jax.nn.silu(h_out)
        return h_out


def build_layers(width, n_layers):
    rng = jax.random.PRNGKey(0)
    x_dummy = np.zeros([2, 3])
    diff_dummy = np.zeros([2, 2, 3])
    layer_functions = []
    for n in range(n_layers):
        rng, key = jax.random.split(rng)
        layer = MessagePassingLayer(width)
        x_dummy, params = layer.init_with_output(key, x_dummy[0], x_dummy, diff_dummy)
        layer_func = functools.partial(layer.apply, params)
        layer_functions.append(layer_func)
    return layer_functions