#%%
import jax.config
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax
from model import Wavefunction
from graph import build_edges, get_affected_electrons
from mcmc import single_electron_proposal, MCMCState
import numpy as np
import time
import jax.tree_util as jtu

@jax.jit
@jax.vmap
def update_log_psi(phi_inv, affected_indices, delta_phi):
    n_changes, n_el = delta_phi.shape[-2:]
    eye_nel = jnp.eye(n_el)
    eye_changes = jnp.eye(n_changes) 
    U = eye_nel.at[:, affected_indices].get(mode="drop", fill_value=0)
    M = eye_changes + delta_phi @ phi_inv @ U

    # TODO: there should probably a faster way to get the determinant and inverse together
    delta_log_det = jnp.linalg.slogdet(M)[1]
    M_inv = jnp.linalg.inv(M)

    U_proj = phi_inv @ U
    V_proj = delta_phi @ phi_inv
    delta_phi_inv = - U_proj @ M_inv @ V_proj
    return delta_log_det, delta_phi_inv

@jax.jit
@jax.vmap
def update_slater_matrix(phi_old, phi_update, indices_update):
    delta_phi = phi_update - phi_old.at[indices_update].get(mode="fill", fill_value=0)
    return phi_old.at[indices_update].set(phi_update), delta_phi
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    rng = jax.random.PRNGKey(0)
    rng_mcmc, rng_model = jax.random.split(rng)

    batch_size = 20
    n_el = 200
    cutoff = 10.0
    R = jnp.arange(-n_el // 2, n_el//2)[:, None] * jnp.array([1, 0, 0])

    r = R + jax.random.normal(rng_mcmc, (batch_size, n_el, 3)) * 1
    edges = build_edges(r, cutoff)

    model = Wavefunction(width_1el=512, 
                         width_2el=32, 
                         depth=4,
                         n_orbitals=n_el, 
                         cutoff=cutoff, 
                         R=R)
    params = model.init(rng_model, r, edges)
    print(f"Nr of params: {sum([p.size for p in jtu.tree_leaves(params)]):,}")
    phi_func = jax.jit(model.apply)

    # Full computation of WF once at the beginning
    edges = build_edges(r, cutoff)
    phi = phi_func(params, r, edges)

    mcmc_state = MCMCState(r=r, 
                           log_psi=jnp.linalg.slogdet(phi)[1], 
                           phi=phi, 
                           phi_inv=jnp.linalg.inv(phi), 
                           rng = jax.random.split(rng_mcmc, batch_size))

    n_steps = 500
    log_psi_values_full = np.zeros([n_steps, batch_size])
    log_psi_values_update = np.zeros([n_steps, batch_size])
    log_psi_values_full[0] = mcmc_state.log_psi
    log_psi_values_update[0] = mcmc_state.log_psi
    t_full = np.zeros(n_steps)
    t_update = np.zeros(n_steps)

    for ind_step in range(1, n_steps):
        if ind_step % 10 == 0:
            print(f"Step {ind_step}/{n_steps}")
        # Single-electron move
        rng, ind_move, r = single_electron_proposal(mcmc_state.rng, mcmc_state.r)
        edges_new = build_edges(r, cutoff)

        # Partial update: re-compute only the changed embeddings and update the determinant using low-rank update
        t0 = time.perf_counter()
        affected_indices = get_affected_electrons(ind_move, edges, edges_new)
        edges_affected = edges_new.get_subset(affected_indices)
        r_affected = jax.vmap(lambda r_, i_: r_[i_])(r, affected_indices)
        phi_affected = phi_func(params, r_affected, edges_affected)
        phi, delta_phi = update_slater_matrix(mcmc_state.phi, phi_affected, affected_indices)
        delta_log_psi, delta_phi_inv = update_log_psi(mcmc_state.phi_inv, affected_indices, delta_phi)
        mcmc_state = MCMCState(r=r, 
                               log_psi=mcmc_state.log_psi + delta_log_psi,
                               phi=phi,
                               phi_inv=mcmc_state.phi_inv + delta_phi_inv,
                               rng=rng)
        mcmc_state = jax.block_until_ready(mcmc_state)
        edges = edges_new
        t1 = time.perf_counter()

        # Full computation of WF after MCMC step (for verification only)
        phi_full = phi_func(params, r, edges_new)
        log_psi_full = jnp.linalg.slogdet(phi_full)[1]
        log_psi_full = jax.block_until_ready(log_psi_full)
        t2 = time.perf_counter()

        # Track log_psi for debugging
        log_psi_values_update[ind_step] = mcmc_state.log_psi
        log_psi_values_full[ind_step] = log_psi_full
        t_update[ind_step] = t1 - t0
        t_full[ind_step] = t2 - t1

    print(f"t update: {np.median(t_update):4f} sec")
    print(f"t full:   {np.median(t_full):4f} sec")

    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ind_batch in range(1):
        axes[0].plot(log_psi_values_update[:, ind_batch], label="update")
        axes[0].plot(log_psi_values_full[:, ind_batch], label="full")
        delta_log_psi = log_psi_values_update[:, ind_batch] - log_psi_values_full[:, ind_batch]
        axes[1].plot(np.exp(2 * delta_log_psi), label="p(update) / p(full)")
    axes[0].set_title("Log|Psi|")
    axes[0].legend()
    axes[0].set_xlabel("MCMC step")
    axes[1].set_title("Probability ratio")
    axes[1].legend()
    axes[1].set_xlabel("MCMC step")
    axes[1].set_ylim([0.99, 1.01])
    fig.tight_layout()
    fig.savefig("low_rank_update.png", bbox_inches="tight")


    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.imshow(mcmc_state.phi[0], cmap="bwr", clim=[-5,5])


