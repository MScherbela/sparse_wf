# %%
import numpy as np
from sparse_wf.system import from_str
from sparse_wf.model.wave_function import MoonLikeWaveFunction
from sparse_wf.mcmc import init_electrons
import jax
import jax.numpy as jnp
import timeit
from sparse_wf.pseudopotentials import make_pseudopotential
from sparse_wf.hamiltonian import potential_energy
import argparse
import pathlib
import yaml
import jax.tree_util as jtu
import jaxlib.xla_extension

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent / "../../../../config/default.yaml"


def get_timing(expression, n_repeat=10):
    timer = timeit.Timer(expression, globals=globals())
    n_timeit_reps = timer.autorange()[0]
    timings = timer.repeat(repeat=n_repeat, number=n_timeit_reps)
    return min(timings) / n_timeit_reps


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
    mol.basis = "ccecp-cc-pvdz"
    mol.build()
    return mol


def get_model(mol, cutoff):
    config = load_yaml(DEFAULT_CONFIG_PATH)
    model_config = config["model_args"]
    model_config["embedding"]["new"]["cutoff"] = cutoff
    wf = MoonLikeWaveFunction.create(mol, **model_config)
    return wf


# def make_potential_energy(wf, use_ecp):
#     pseudopotentials = ["C", "N", "O"] if use_ecp else []
#     eff_charges, pp_local, pp_nonlocal = make_pseudopotential(wf.Z, pseudopotentials, n_quad_points=dict(default=4))

#     def get_potential_energy(key: jax.Array, params, electrons, static):
#         potential = potential_energy(electrons, wf.R, eff_charges)
#         potential += pp_local(electrons, wf.R)
#         nl_pp, new_static = pp_nonlocal(key, wf, params, electrons, static)
#         potential += nl_pp
#         return potential, new_static

#     return get_potential_energy


def single_electron_move(rng, electrons, stepsize):
    batch_size, n_el, _ = electrons.shape
    rng = jax.random.split(rng, (batch_size, 2))

    def update(r, key):
        idx = jax.random.randint(key[0], [], 0, n_el)
        dr = jax.random.normal(key[1], (3,)) * stepsize
        return r.at[idx].add(dr), idx[None]

    return jax.vmap(update)(electrons, rng)


# def electron_swap(rng, electrons):
#     batch_size, n_el, _ = electrons.shape
#     n_up = n_el // 2
#     rng = jax.random.split(rng, (batch_size, 2))

#     def update(r, key):
#         idx_up = jax.random.randint(key[0], [], 0, n_up)
#         idx_dn = jax.random.randint(key[1], [], n_up, n_el)
#         r_up = r[idx_up]
#         r = r.at[idx_up].set(r[idx_dn])
#         r = r.at[idx_dn].set(r_up)
#         return r, jnp.stack([idx_up, idx_dn])

#     return jax.vmap(update)(electrons, rng)


def jit_and_await(f, static_argnums=None):
    f = jax.jit(f, static_argnums=static_argnums)
    return lambda *args, **kwargs: jtu.tree_map(lambda x: x.block_until_ready(), f(*args, **kwargs))


def get_max_static(wf, r, r_new, idx_changed):
    static = jax.vmap(wf.get_static_input)(r, r_new, idx_changed)
    static = jtu.tree_map(lambda x: int(jnp.max(x)), static)
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
    def looped_logpsi(params, electrons, static, state):
        def loop_body(i, carry):
            r, _state, shift = carry
            idx_changed_el = jnp.array([i], jnp.int32)
            r = r.at[i].add(1e-12 * shift)
            (sign, logpsi), _state = wf.log_psi_low_rank_update(params, r, idx_changed_el, static, _state)
            return r, _state, jnp.mean(sign * logpsi) * 1e-12

        return jax.lax.fori_loop(0, n_iterations, loop_body, (electrons, state, jnp.zeros([])))[0]

    looped_logpsi = jax.vmap(looped_logpsi, in_axes=(None, 0, None, 0))
    return jit_and_await(looped_logpsi, static_argnums=(2,))


def benchmark_runtime(operation, system_size, batch_size, use_ecp, move_stepsize, n_iterations):
    mol = get_cumulene(system_size, use_ecp)
    rng = jax.random.PRNGKey(0)
    rng_electrons, rng_move, rng_params = jax.random.split(rng, 3)

    # Initialize electrons and get statics
    print("Initializing electrons", flush=True)
    wf = get_model(mol, args.cutoff)

    r = init_electrons(rng_electrons, mol, batch_size)
    r_new_local, idx_changed_local = single_electron_move(rng_move, r, move_stepsize)
    static_local = get_max_static(wf, r, r_new_local, idx_changed_local)

    print("Initializing parameters", flush=True)
    params = wf.init(rng_params, r[0])

    # Build operations
    if operation in "wf_full":
        n_iterations = args.n_iterations
        get_logpsi_full = build_looped_logpsi_full(wf, n_iterations)

        def test_func():
            get_logpsi_full(params, r, static_local)
    elif operation == "wf_lowrank":
        n_iterations = args.n_iterations
        get_logpsi_lowrank = build_looped_logpsi_lowrank(wf, n_iterations)
        # Calculate state from a single sample and copy it, to avoid oom from full wf
        _, state = wf.log_psi_with_state(params, r[0], static_local)
        state = jtu.tree_map(lambda x: jnp.tile(x, [batch_size] + [1] * x.ndim), state)

        def test_func():
            get_logpsi_lowrank(params, r, static_local, state)
    elif operation in ["E_kin", "E_kin_dense"]:
        n_iterations = 1
        get_E_kin = wf.kinetic_energy_dense if operation == "E_kin_dense" else wf.kinetic_energy
        get_E_kin = jax.vmap(get_E_kin, in_axes=(None, 0, None))
        get_E_kin = jit_and_await(get_E_kin, static_argnums=(2,))

        def test_func():
            get_E_kin(params, r, static_local)

    print("Running warmup/compilation", flush=True)
    test_func()

    print("Running for timings", flush=True)
    t = get_timing(test_func) / n_iterations
    return t


if __name__ == "__main__":
    default_system_sizes = np.round(np.geomspace(16, 124, 25)).astype(int)
    default_system_sizes = np.concatenate([default_system_sizes, np.arange(24, 125, 25)])
    default_system_sizes = np.unique(default_system_sizes)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--use_ecp", action="store_true", default=False)
    parser.add_argument("--system_sizes", nargs="+", type=int, default=default_system_sizes)
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--system", type=str, default="cumulene")
    parser.add_argument("--move_stepsize", type=float, default=0.5)
    parser.add_argument("--n_iterations", type=int, default=20)
    parser.add_argument("-o", "--output", type=str, default="timings.txt")
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--operations", nargs="+", default=["wf_full", "wf_lowrank", "E_kin", "E_kin_dense"])
    args = parser.parse_args()

    settings = {k: v for k, v in vars(args).items() if k not in ["output", "profile", "system_sizes"]}
    for operation in args.operations:
        batch_size = args.batch_size
        for idx_system_size, system_size in enumerate(sorted(args.system_sizes)):
            while batch_size > 0:
                print(
                    f"op={operation}, system_size={system_size} ({idx_system_size + 1}/{len(args.system_sizes)}), batch_size={batch_size}",
                    flush=True,
                )
                try:
                    t = benchmark_runtime(
                        operation, system_size, batch_size, args.use_ecp, args.move_stepsize, args.n_iterations
                    )
                except jaxlib.xla_extension.XlaRuntimeError as e:
                    # out of memory
                    print(f"Out of memory with batch-size {batch_size}", flush=True)
                    batch_size = batch_size // 2
                    continue
                with open(args.output, "a") as f:
                    f.write(f"FiRE,{operation},{system_size},{batch_size},{t}\n")
                break
            else:
                # we land here if the loop completes without breaking
                print(
                    f"Could not complete {operation} for system_size={system_size}. Skipping larger system sizes.",
                    flush=True,
                )
                break
