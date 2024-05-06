# %%
import os
import flax.serialization
import jax.tree_util as jtu
import jax.numpy as jnp

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def to_flat_dict(nested_dict, parent_key=""):
    flat_dict = {}
    for key, value in nested_dict.items():
        if parent_key:
            key = f"{parent_key}.{key}"
        if isinstance(value, dict):
            flat_dict.update(to_flat_dict(value, key))
        else:
            flat_dict[key] = value
    return flat_dict


fname = "/home/scherbelam20/develop/sparse_wf/runs/svd_precond/dump_sv/dump_sv_10_scale_linspace/grad_002000.msgpk"
with open(fname, "rb") as f:
    data = f.read()
data = flax.serialization.msgpack_restore(data)
s = data["s"]
Vt_all = to_flat_dict(data["Vt"])
params = to_flat_dict(data["params"])
grad = data["gradients"][0]

# %%
ind_sv = 20
print(f"{s[ind_sv]:.1f}")
Vt = jtu.tree_map(lambda x: x[ind_sv], Vt_all)
norms = jtu.tree_map(lambda x: jnp.linalg.norm(x) ** 2, Vt)
norms = to_flat_dict(norms)

start = 0
for k, v in norms.items():
    param_norm = jnp.linalg.norm(params[k])
    size = params[k].size
    g = jnp.sum(grad[:, start : start + size] ** 2, axis=-1)
    line = f"{k:<40}: {v:.2f} p={param_norm:6.2f} g={g.mean():6.0f}"
    if v > 0.05:
        line += " <=="
    # if k == "ee_filter.params.scales":
    #     print(params[k])
    print(line)

    start += size


# %%
