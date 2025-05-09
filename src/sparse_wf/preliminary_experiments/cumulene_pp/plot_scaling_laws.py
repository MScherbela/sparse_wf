# %%
import jax.scipy.optimize
import pandas as pd
import numpy as np
from sparse_wf.plot_utils import get_outlier_mask, savefig, MILLIHARTREE
import matplotlib.pyplot as plt
import scienceplots
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import optax
plt.style.use(['science', 'grid'])

def fractional_smoothing(x, indices, frac):
    y = np.zeros_like(indices, dtype=float)
    for i, idx in enumerate(indices):
        y[i] = np.mean(x[int(idx * (1 - frac)) : idx])
    return y

def preprocess_data(df, window_outlier=100, n_steps_min=500, n_samples=200, frac_smoothing=0.1):
    window_outlier = 100
    n_steps_min = 500
    n_samples = 200
    frac_smoothing = 0.1

    smooth_data = []
    for molecule in df_all.molecule.unique():
        df = df_all[df_all.molecule == molecule].copy()
        mask = get_outlier_mask(df["opt/E"], window_size=window_outlier)
        print(f"Found {mask.sum()} outliers for {molecule}")
        df = df[~mask]
        indices = np.geomspace(n_steps_min, len(df), n_samples).astype(int)
        indices = np.unique(indices)

        smooth_data.append(
            pd.DataFrame(
                {
                    "step": indices,
                    "E": fractional_smoothing(df["opt/E"], indices, frac_smoothing),
                    "molecule": molecule,
                    "n_el": df["n_el"].iloc[0],
                }
            )
        )
    df_smooth = pd.concat(smooth_data)
    return df_smooth


def fit_powerlaw(df, opt_steps=20_000):
    df = df.copy()
    pivot = df.groupby(["n_el", "molecule"]).agg(E_min=("E", "min")).reset_index()
    pivot["idx_mol"] = np.arange(len(pivot))
    df = df.merge(pivot[["molecule", "idx_mol"]], on="molecule")
    E, n, idx_mol, step = df.E.values, df.n_el.values, df.idx_mol.values, df.step.values
    E_min_per_mol = pivot.E_min.values

    X = np.stack([-np.log(step), np.log(n), np.ones_like(n)], axis=1)
    M = np.linalg.inv(X.T @ X) @ X.T

    optimizer = optax.adam(lambda t: 1e-3 / (1 + t/1000))

    def loss_fn(E_inf):
        delta_E = E - E_inf[idx_mol]
        y = jnp.log(delta_E)
        alpha = M @ y
        return jnp.sum((y - X @ alpha)**2), alpha

    @jax.jit
    def opt_step(state, E_inf):
        (loss, alpha), grads = jax.value_and_grad(loss_fn, has_aux=True)(E_inf)
        updates, state = optimizer.update(grads, state)
        E_inf = optax.apply_updates(E_inf, updates)
        E_inf = jnp.minimum(E_inf, E_min_per_mol - 1e-4)
        return state, loss, E_inf, alpha

    E_inf = E_min_per_mol - 5e-3
    state = optimizer.init(E_inf)
    loss_values = []
    for step in range(opt_steps):
        state, loss, E_inf, alpha = opt_step(state, E_inf)
        loss_values.append(loss)
    loss_values = jnp.array(loss_values)
    pivot["E_inf"] = E_inf
    return pivot, alpha, loss_values

df_smoothed = []
df_all = pd.read_csv("cumulene_pp_energies.csv")
df_all = df_all[df_all["cutoff"] == 3.0]
df_all["molecule"] = "C" + df_all["n_carbon"].astype(str) + "H4_" + df_all["angle"].astype(str)
df_all["n_el"] = df_all["n_carbon"] * 4 + 4
df_smoothed.append(preprocess_data(df_all))

df_all = pd.read_csv("../acene/acene.csv", header=[0, 1, 2], index_col=0)
df_all = df_all["E"]
df_all = df_all.melt(value_name="opt/E").reset_index()
df_all = df_all[df_all["opt/E"].notnull()]
df_all["molecule"] = df_all.Molecule + "_" + df_all.State
df_all["n_rings"] = df_all.Molecule.apply(lambda x: ['naphthalene', 'anthracene', 'tetracene', 'pentacene', 'hexacene', 'heptacene'].index(x) + 2)
df_all = df_all[df_all["n_rings"] <= 6]
df_all["n_el"] = df_all.n_rings * 18 + 10
df_smoothed.append(preprocess_data(df_all))


#%%

# fig, axes = plt.subplots(1, 2, figsize=(5.5, 3))
fig, axes = plt.subplots(3, 1, figsize=(4, 5), height_ratios=[1, 1, 0.07])
cax = axes[-1]
axes = axes[:-1]

ls_singlet, ls_triplet, ls_fit = ":", "--", "-"

sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"), norm=plt.Normalize(0, 130, clip=True))
for df_smooth, molecule_class, ax_label, ax in zip(df_smoothed, ["cumulenes", "acenes"], "ab", axes):
    fit_pivot, fit_params, loss_values = fit_powerlaw(df_smooth, opt_steps=1000)
    alpha, beta, const = fit_params
    def powerlaw(t, n):
        return np.exp(const) * t**(-alpha) * n**beta
    fitted_equation = f"$(E - E_\infty) \sim t^{{-{alpha:.1f}}}\\; n_\\mathrm{{el}}^{{{beta:.1f}}}$"
    # fitted_equation = f"$\Delta E \sim t^{{-{alpha:.1f}}}\\; n_\\mathrm{{el}}^{{{beta:.1f}}}$"

    for _, r in fit_pivot.iterrows():
        color = sm.cmap(sm.norm(r.n_el))
        df = df_smooth[df_smooth.molecule == r.molecule]
        ls = ls_triplet if ("90" in r.molecule or "triplet" in r.molecule) else ls_singlet
        ax.plot(df.step, powerlaw(df.step, df.n_el), color=color, ls=ls_fit, alpha=1)
        ax.plot(df.step, df.E - r.E_inf, color=color, alpha=1, ls=ls)
    ax.plot([], [], color="black", ls=ls_singlet, label="Singlet")
    ax.plot([], [], color="black", ls=ls_triplet, label="Triplet")
    ax.plot([], [], color="black", ls=ls_fit, label="power-law fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(molecule_class)
    ax.set_xlabel("optimization step $t$", labelpad=0)
    # if molecule_class == "cumulenes":
    ax.set_ylabel(f"$(E - E_\infty)$ " + MILLIHARTREE)
    # legend with less spacing between lines
    ax.legend(loc="upper right", handlelength=1, labelspacing=0.3)
    t=ax.text(0.02, 0.07, fitted_equation, transform=ax.transAxes, fontsize=11)
    t.set_bbox(dict(facecolor="white", alpha=1, edgecolor="white", linewidth=0, boxstyle="square,pad=0"))
    ax.text(0, 1.02, f"\\textbf{{{ax_label})}}", transform=ax.transAxes, va="bottom", ha="left")
    print(f"Opt-step induced order beta/alpha: {beta/alpha:.2f}")
axes[0].set_ylim([1e-4, 2e0])

fig.tight_layout()
plt.colorbar(sm, cax=cax, orientation="horizontal")
cax.set_xlabel("$n_\\mathrm{el}$", labelpad=-2)

# cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal")
# cbar.set_label("$n_\\mathrm{el}$", labelpad=-4)
fig.subplots_adjust(hspace=0.5)
savefig(fig, "scaling_laws")
