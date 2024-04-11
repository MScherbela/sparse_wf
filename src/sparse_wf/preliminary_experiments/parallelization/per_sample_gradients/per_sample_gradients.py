import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "" # Uncomment to run on CPU, in which case the code works
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax.numpy as jnp
import jax.distributed


def model(params, x):
    return x @ params


if __name__ == "__main__":
    jax.print_environment_info()
    print("Devices: ", jax.devices())
    n_devices = jax.device_count()  # 2
    devices = mesh_utils.create_device_mesh((n_devices,))
    sharding = PositionalSharding(devices)

    feature_dim = 3
    batch_size_total = 8

    # Get example data
    x = jnp.ones((batch_size_total, feature_dim))
    params = jnp.ones(feature_dim)

    # Shard data, replicate params
    x = jax.device_put(x, sharding.reshape(n_devices, 1))
    params = jax.device_put(params, sharding.replicate(axis=0))

    y = model(params, x)
    print("Forward pass (with vectorizable function) works")

    y = jax.vmap(model, in_axes=(None, 0))(params, x)
    print("Forward pass with explicity vmap also works")

    grad = jax.grad(lambda p: model(p, x).sum())(params)
    print("Gradient of global function works")

    per_sample_grads = jax.vmap(jax.grad(model), in_axes=(None, 0))(params, x)
    print("This is never reached: per-sample gradients combining vmap and grad fails when run on GPUs")
