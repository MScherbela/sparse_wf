import jax.distributed
import jax

if __name__ == "__main__":
    jax.distributed.initialize()
    print(f"Process {jax.process_index()}: Devices: {jax.devices()}")
