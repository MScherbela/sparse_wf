import jax.distributed
import jax

if __name__ == "__main__":
    jax.distributed.initialize()
    print(
        f"Process {jax.process_index()}: Global devices: {jax.devices()}, Local devices: {jax.local_devices()}",
        flush=True,
    )
