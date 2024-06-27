#%%
import jax.numpy as jnp
import jax
from sparse_wf.model.utils import init_glu_feedforward, apply_glu_feedforward
import numpy as np

rng = jax.random.PRNGKey(0)
params = init_glu_feedforward(rng, 32, 2, 3, 8)

diffs = np.random.normal(size=(100,3)) * 1
dists = np.linalg.norm(diffs, axis=-1)
glu_output = apply_glu_feedforward(params, diffs)
envelopes = glu_output * jnp.exp(-dists[..., None])

print(np.std(glu_output))
print(np.std(envelopes))


