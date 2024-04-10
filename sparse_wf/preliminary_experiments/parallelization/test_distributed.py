import jax
import flax.linen as nn
import jax.distributed

class Model(nn.Module):
    @nn.compact
    def apply(self, x):
        return jax.nn.silu(nn.Dense(features=1)(x))

if __name__ == '__main__':
    jax.distributed.initialize()

    print(f"Initialized {jax.process_index()} / {jax.device_count()}")

    seed = 42
    model = Model()
    rng = jax.random.PRNGKey(seed)
    rng_params, rng_samples = jax.random.split(rng)
    # params = model.init(rng)