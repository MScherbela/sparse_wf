# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np


# For system in n=..., pp=...
# Build model
# For each model, run: psi, psi_low_rank, E_kin, E_pot, embedding, jastrow, determinant

from sparse_wf.system import from_str
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.api import NewEmbeddingArgs, JastrowArgs
from sparse_wf.mcmc import init_electrons
import jax
import jax.numpy as jnp
import time
import timeit
from sparse_wf.pseudopotentials import make_pseudopotential
from sparse_wf.hamiltonian import potential_energy
from sparse_wf.loggers import to_log_data
import argparse
import functools


def get_cumulene(n_carbon: int, use_ecp):
    bond_length = 2.53222
    R = np.zeros([n_carbon + 4, 3])
    R[:-4, :] = np.arange(n_carbon)[:, None] * np.array([bond_length, 0, 0])[None, :]
    R_H = np.array([[-1.02612, 1.77728, 0.0], [-1.02612, -1.77728, 0.0]])
    R[-4:-2] = R_H
    R[-2:] = R[n_carbon - 1] - R_H
    Z = np.array([6] * n_carbon + [1, 1, 1, 1])

    mol = from_str(atom=[(z, r) for z, r in zip(Z, R)])  # type: ignore
    if use_ecp:
        mol.ecp = {6: "ccecp"}
    mol.basis = "sto-6g"
    mol.build()
    return mol


def get_model(mol, cutoff):
    embedding_config = dict(
        embedding="new_sparse",
        new=dict(
            cutoff=cutoff,
            cutoff_1el=20.0,
            feature_dim=256,
            pair_mlp_widths=[16, 8],
            pair_n_envelopes=32,
            low_rank_buffer=2,
            n_updates=1,
        ),
    )
    jastrow_config = dict(
        e_e_cusps="psiformer",
        use_e_e_mlp=False,
        use_mlp_jastrow=True,
        use_log_jastrow=True,
        mlp_width=256,
        mlp_depth=2,
        use_attention=True,
        attention_heads=16,
        attention_dim=16,
    )
    envelope_args = dict(envelope="isotropic", isotropic_args=dict(n_envelopes=8))

    wf = MoonLikeWaveFunction.create(
        mol,
        embedding=embedding_config,
        jastrow=jastrow_config,
        envelopes=envelope_args,
        n_determinants=4,
        spin_restricted=False,
    )
    return wf


def make_potential_energy(wf, use_ecp):
    if use_ecp:
        pseudopotentials = ["C", "N", "O"]
    eff_charges, pp_local, pp_nonlocal = make_pseudopotential(wf.Z, pseudopotentials)

    def get_potential_energy(key: jax.Array, params, electrons, static):
        potential = potential_energy(electrons, wf.R, eff_charges)
        potential += pp_local(electrons, wf.R)
        nl_pp, new_static = pp_nonlocal(key, wf, params, electrons, static)
        potential += nl_pp
        return potential, new_static

    return get_potential_energy


def single_electron_move(rng, electrons, stepsize):
    batch_size, n_el, _ = electrons.shape
    rng = jax.random.split(rng, (batch_size, 2))

    def update(r, k):
        idx = jax.random.randint(k[0], [], 0, n_el)
        dr = jax.random.normal(k[1], (3,)) * stepsize
        return r.at[idx].add(dr), idx[None]

    return jax.vmap(update)(electrons, rng)


def jit_and_await(f, static_argnums=None):
    f = jax.jit(f, static_argnums=static_argnums)
    return lambda *args, **kwargs: jax.tree_map(lambda x: x.block_until_ready(), f(*args, **kwargs))


def get_max_static(static, wf):
    static = jax.tree_map(lambda x: int(jnp.max(x)), static)
    return static.round_with_padding(1.0, wf.n_electrons, wf.n_up, wf.n_nuclei)


def build_looped_logpsi_full(wf, n_iterations):
    def looped_logpsi(params, electrons_new, static):
        _, state = wf.log_psi_with_state(params, electrons_new, static)

        def loop_body(i, offset):
            r_new = electrons_new + offset
            logpsi = wf(params, r_new, static)
            return jnp.mean(logpsi) * 1e-12

        return jax.lax.fori_loop(0, n_iterations - 1, loop_body, jnp.zeros([])), state

    looped_logpsi = jax.vmap(looped_logpsi, in_axes=(None, 0, None))
    return jit_and_await(looped_logpsi, static_argnums=(2,))


def build_looped_logpsi_lowrank(wf, n_iterations):
    def looped_logpsi(params, electrons_new, idx_changed_el, static, state):
        def loop_body(i, carry):
            _state, offset = carry
            r_new = electrons_new + offset
            idx_el = idx_changed_el + jnp.minimum(i, 0)
            (sign, logpsi), _state = wf.log_psi_low_rank_update(params, r_new, idx_el, static, _state)
            return _state, jnp.mean(sign * logpsi) * 1e-12

        return jax.lax.fori_loop(0, n_iterations, loop_body, (state, jnp.zeros([])))[0]

    looped_logpsi = jax.vmap(looped_logpsi, in_axes=(None, 0, 0, None, 0))
    return jit_and_await(looped_logpsi, static_argnums=(3,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_ecp", type=int, default=1)
    parser.add_argument("--system_size_min", type=int, default=4)
    parser.add_argument("--system_size_max", type=int, default=256)
    parser.add_argument("--system_size_steps", type=int, default=13)
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--system", type=str, default="cumulene")
    parser.add_argument("--move_stepsize", type=float, default=0.5)
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("-o", "--output", type=str, default="timings.txt")

    args = parser.parse_args()

    system_sizes = np.geomspace(args.system_size_min, args.system_size_max, args.system_size_steps).astype(int)
    system_sizes = np.unique(np.round(system_sizes).astype(int))
    for system_size in system_sizes:
        batch_size = args.batch_size
        cutoff = args.cutoff
        system = args.system
        move_stepsize = args.move_stepsize
        n_iterations = args.n_iterations
        use_ecp = bool(args.use_ecp)

        if system_size >= 32:
            batch_size //= 2
        if system_size >= 64:
            batch_size //= 2
        if system_size >= 128:
            batch_size //= 2

        settings = dict(
            batch_size=batch_size,
            cutoff=cutoff,
            system=system,
            system_size=system_size,
            move_stepsize=move_stepsize,
            n_iterations=n_iterations,
            use_ecp=use_ecp,
        )

        print(f"system_size = {system_size}: Initializing model, electrons, params, and static input for ")
        rng = jax.random.PRNGKey(0)
        rng_electrons, rng_move, rng_params, rng_pp = jax.random.split(rng, 4)
        rng_pp = jax.random.split(rng_pp, batch_size)
        mol = get_cumulene(system_size, use_ecp)
        wf = get_model(mol, 3.0)
        electrons = init_electrons(rng_electrons, mol, batch_size)
        electrons_new, idx_changed_el = single_electron_move(rng_move, electrons, move_stepsize)
        params = wf.init(rng_params, electrons[0])

        get_static = jit_and_await(jax.vmap(wf.get_static_input))
        get_logpsi_full = build_looped_logpsi_full(wf, n_iterations)
        get_logpsi_lowrank = build_looped_logpsi_lowrank(wf, n_iterations)
        get_E_kin = jit_and_await(jax.vmap(wf.kinetic_energy, in_axes=(None, 0, None)), static_argnums=(2,))
        get_E_pot = jit_and_await(
            jax.vmap(make_potential_energy(wf, use_ecp), in_axes=(0, None, 0, None)), static_argnums=(3,)
        )

        static = get_static(electrons, electrons_new, idx_changed_el)
        static = get_max_static(static, wf)

        print("Running warmup/compilation", flush=True)
        state = get_logpsi_full(params, electrons, static)[1]
        get_logpsi_lowrank(params, electrons_new, idx_changed_el, static, state)
        get_E_kin(params, electrons, static)
        pp_static = get_E_pot(rng_pp, params, electrons, static)[1]
        pp_static = get_max_static(pp_static, wf)
        get_E_pot(rng_pp, params, electrons, pp_static)

        timing = functools.partial(timeit.timeit, globals=globals(), number=10)

        print("Running for timings", flush=True)
        t_wf_full = timing("get_logpsi_full(params, electrons, static)") / n_iterations
        t_wf_lr = timing("get_logpsi_lowrank(params, electrons_new, idx_changed_el, static, state)") / n_iterations
        t_E_kin = timing("get_E_kin(params, electrons, static)")
        t_E_pot = timing("get_E_pot(rng_pp, params, electrons, pp_static)")

        results = dict(
            **settings,
            n_el_core=sum([mol.atom_nelec_core(i) for i in range(mol.natm)]),
            n_el=mol.nelectron,
            t_wf_full=t_wf_full,
            t_wf_lr=t_wf_lr,
            t_E_kin=t_E_kin,
            t_E_pot=t_E_pot,
            **to_log_data(static, "static/"),
            **to_log_data(pp_static, "pp_static/"),
        )

        with open(args.output, "a") as f:
            f.write(str(results) + "\n")
        print(str(results))
