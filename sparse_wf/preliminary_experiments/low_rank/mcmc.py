import chex
import jax


@chex.dataclass
class MCMCState:
    r: jax.Array
    log_psi: jax.Array
    phi: jax.Array
    phi_inv: jax.Array
    rng: jax.random.PRNGKey

    @property
    def n_el(self):
        return self.r.shape[-2]


@jax.vmap
def single_electron_proposal(rng, r):
    n_el = r.shape[-2]
    rng, rng_ind, rng_proposal = jax.random.split(rng, 3)
    ind_move = jax.random.randint(rng_ind, (), 0, n_el)
    dr = jax.random.normal(rng_proposal, (3,))
    r = r.at[ind_move].add(dr)
    return rng, ind_move, r
