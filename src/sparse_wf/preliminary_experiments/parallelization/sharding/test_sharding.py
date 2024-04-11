import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax.numpy as jnp
import jax.distributed
from jax.experimental.multihost_utils import broadcast_one_to_all
import numpy as np
import functools

@jax.jit
# @functools.partial(jax.vmap, in_axes=(None, 0))
def model(params, x):
    y = x @ params
    # y = y - jnp.mean(y, axis=0)
    return y

@jax.jit
def get_jacobian(params, x):
    return jax.vmap(jax.grad(model), in_axes=(None, 0))(params, x)

def print_with_process_id(msg):
    print(f"Process {jax.process_index()}: {msg}", flush=True)


if __name__ == "__main__":
    # Set up sharding across batch size
    jax.distributed.initialize()
    n_devices = jax.device_count()
    print_with_process_id(f"Found {n_devices} devices: {jax.devices()}")
    devices = mesh_utils.create_device_mesh((n_devices,))
    sharding = PositionalSharding(devices)

    # Get some dummy data
    dim = 3
    global_batch_size = 8
    local_batch_size = global_batch_size // n_devices

    print_with_process_id("Generating local data and putting them on local device")
    print_with_process_id(f"{sharding.addressable_devices=}")
    x_local = jnp.ones((local_batch_size, dim)) * jax.process_index()
    x_local = jax.device_put(x_local, jax.local_devices()[0])
    print_with_process_id(f"{x_local.devices()=}")

    global_shape = (global_batch_size, dim)
    print_with_process_id(f"Merging into global data array: {global_shape=}, {sharding=}")
    x_global = jax.make_array_from_single_device_arrays(global_shape, sharding.reshape(-1, 1), [x_local])
    print_with_process_id("Merge successful")



    params = jnp.ones(dim)
    print_with_process_id(f"Broadcasting params to all")
    params = broadcast_one_to_all(params)
    print_with_process_id("Broadcast successful")


    # # Commit data to devices
    # # x = jax.device_put(x, sharding.reshape(-1, 1))
    # params = jax.device_put(params, sharding.replicate(axis=0))

    # Evaluate model and jacobian
    print_with_process_id("Computing forward pass...")
    y = model(params, x_global)
    print_with_process_id(f"Forward pass successful: {np.array(y.addressable_data(0))=}")

    print_with_process_id("Computing jacobian...")
    jac = get_jacobian(params, x_global)  # This line fails on GPUs, but works on CPUs
    print_with_process_id("Jacobian successful")
