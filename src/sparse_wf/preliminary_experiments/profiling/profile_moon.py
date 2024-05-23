import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from sparse_wf.model.moon import Moon
from sparse_wf.model.two_step_moon import TwoStepMoon
from sparse_wf.api import JastrowArgs
import pyscf
import jax.numpy as jnp
import jax
print(jax.devices())

mol = pyscf.gto.M(atom="N 0 0 0; N 0 0 2.0", unit="bohr", basis="6-31g")

model = TwoStepMoon.create(
    mol,
    cutoff=20.0,
    feature_dim=256,
    nuc_mlp_depth=3,
    pair_mlp_widths=(16,8),
    pair_n_envelopes=32,
    n_determinants=16,
    n_envelopes=16,
    use_e_e_cusp=True,
    mlp_jastrow=JastrowArgs(use=False, embedding_n_hidden=None, soe_n_hidden=None)
)
batch_size = 1024

rng = jax.random.PRNGKey(0)
rng_r, rng_params = jax.random.split(rng)
electrons = jax.random.normal(rng_r, (batch_size, 14, 3))
params = model.init(rng_params, electrons[0])

static = model.get_static_input(electrons)
# model_batched = jax.vmap(lambda p, r: model(p, r, static), in_axes=(None, 0))
# model_jitted = jax.jit(model_batched)
model_jitted = jax.jit(lambda p, r: model.local_energy(p, r, static))

print("Warmup")
logpsi = model_jitted(params, electrons)

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
with jax.profiler.trace("/tmp/tensorboard"):
    for i in range(3):
        print(i)
        logpsi = model_jitted(params, electrons)
print("finished")

