import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax.numpy as jnp


def model(params, x):
    return x @ params


def get_jacobian(params, x):
    return jax.vmap(jax.grad(model), in_axes=(None, 0))(params, x)


if __name__ == "__main__":
    # Get some dummy data
    dim = 3
    total_batch_size = 8
    x = jnp.ones((total_batch_size, dim))
    params = jnp.ones(dim)

    # Set up sharding across batch size
    print("Devices: ", jax.devices())
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    sharding = PositionalSharding(devices)

    # Commit data to devices
    x = jax.device_put(x, sharding.reshape(-1, 1))
    params = jax.device_put(params, sharding.replicate(axis=0))

    # Evaluate model and jacobian
    y = model(params, x)
    jac = get_jacobian(params, x)  # This line fails on GPUs, but works on CPUs
    print("Success")
