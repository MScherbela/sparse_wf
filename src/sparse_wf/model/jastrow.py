from jaxtyping import Array, Float
from sparse_wf.api import Electrons, JastrowFactorArgs
from sparse_wf.model.utils import MLP, get_dist_same_diff
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, cast, Optional
import jax


class JastrowFactor(nn.Module):
    embedding_n_hidden: Optional[Sequence[int]]
    soe_n_hidden: Optional[Sequence[int]]
    use: bool = True

    @nn.compact
    def __call__(self, embeddings):
        """
        There are three options here (i is the electron index):
        (1) J_i = MLP_[embedding_n_hidden, 1](embeddings) , J=sum(J_i)
        (2) J_i = MLP_[embedding_n_hidden](embeddings), J=MLP_[soe_n_hidden, 1](sum(J_i))
        (3) J=MLP_[soe_n_hidden, 1](sum(embeddings_i))
        """
        if self.embedding_n_hidden is None and self.soe_n_hidden is None:
            raise KeyError("Either embedding_n_hidden or soe_n_hidden must be specified when using mlp jastrow.")

        if self.embedding_n_hidden is not None:
            if self.soe_n_hidden is None:  # Option (1)
                jastrow = jnp.squeeze(
                    MLP([*self.embedding_n_hidden, 1], activate_final=False, residual=False, output_bias=False)(
                        embeddings
                    ),
                    axis=-1,
                )
                jastrow = jnp.sum(jastrow, axis=-1)
            else:  # Option (2) part 1
                jastrow = MLP(self.embedding_n_hidden, activate_final=False, residual=False, output_bias=False)(
                    embeddings
                )
        else:  # Option (3) part 2
            jastrow = embeddings

        if self.soe_n_hidden is not None:  # Option (2 or 3)
            jastrow = jnp.sum(jastrow, axis=-2)  # Sum over electrons.
            jastrow = jnp.squeeze(
                MLP([*self.soe_n_hidden, 1], activate_final=False, residual=False, output_bias=False)(jastrow), axis=-1
            )

        return jastrow


class YukawaJastrow(nn.Module):
    n_up: int

    @nn.compact
    def __call__(self, electrons: Electrons) -> Float[Array, " *batch_dims"]:
        A_same = jax.nn.softplus(self.param("A_same", nn.initializers.ones, (), jnp.float32))
        F_same = jnp.sqrt(2 * A_same)
        A_diff = jax.nn.softplus(self.param("A_diff", nn.initializers.ones, (), jnp.float32))
        F_diff = jnp.sqrt(2 * A_diff)

        dist_same, dist_diff = get_dist_same_diff(electrons, self.n_up)
        # Supposed to be a minus in front of the whole jastrow, but I use expm1 instead of 1-exp, so it should work out
        u_same = A_same * jnp.sum(1 / dist_same * jnp.expm1(-dist_same / F_same), axis=-1)
        u_diff = A_diff * jnp.sum(1 / dist_diff * jnp.expm1(-dist_diff / F_diff), axis=-1)

        return u_same + u_diff


class ElElCusp(nn.Module):
    n_up: int

    @nn.compact
    def __call__(self, electrons: Electrons) -> Float[Array, " *batch_dims"]:
        dist_same, dist_diff = get_dist_same_diff(electrons, self.n_up)

        alpha_same = self.param("alpha_same", nn.initializers.ones, (), jnp.float32)
        alpha_diff = self.param("alpha_diff", nn.initializers.ones, (), jnp.float32)
        factor_same, factor_diff = -0.25, -0.5

        cusp_same = jnp.sum(alpha_same**2 / (alpha_same + dist_same), axis=-1)
        cusp_diff = jnp.sum(alpha_diff**2 / (alpha_diff + dist_diff), axis=-1)

        return factor_same * cusp_same + factor_diff * cusp_diff


class Jastrow(nn.Module):
    n_up: int
    mlp: JastrowFactorArgs
    log: JastrowFactorArgs
    use_yukawa_jastrow: bool
    use_e_e_cusp: bool

    @nn.compact
    def __call__(self, electrons: Electrons, embeddings: jax.Array) -> jax.Array:
        assert not (self.use_yukawa_jastrow and self.use_e_e_cusp), "Do not use both Yukawa and ElElCusp"
        logpsi = 0.0
        if self.mlp["use"]:
            logpsi += JastrowFactor(**self.mlp, name="mlp_jastrow")(embeddings)
        if self.log["use"]:
            j = JastrowFactor(**self.log, name="mlp_jastrow")(embeddings)
            logpsi += jnp.log(jnp.abs(j))
        if self.use_e_e_cusp:
            logpsi += ElElCusp(self.n_up)(electrons)
        if self.use_yukawa_jastrow:
            logpsi += YukawaJastrow(self.n_up)(electrons)
        return cast(jax.Array, logpsi)
