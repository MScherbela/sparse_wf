#%%
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import folx
import jax.numpy as jnp
import jax

@jax.checkpoint
def f(x):
    return jnp.sum(jnp.sin(x))

get_lap = folx.ForwardLaplacianOperator(0.6)(f)

x = jnp.ones((3,))
get_lap(x)





