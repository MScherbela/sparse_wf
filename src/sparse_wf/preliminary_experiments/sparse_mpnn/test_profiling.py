import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
import jax.numpy as jnp


@jax.jit
def my_func(x):
    x = jnp.sin(x)
    y = jnp.sum(x**2)
    return y

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000,))

print(jax.devices())

y = my_func(x)
y.block_until_ready()

with jax.profiler.trace("/tmp/tensorboard"):
    y = my_func(x).block_until_ready()