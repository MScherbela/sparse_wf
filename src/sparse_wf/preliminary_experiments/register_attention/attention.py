import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, Float, Integer
from sparse_wf.model.utils import MLP

class ElecToRegister(nn.Module):
    n_register: int
    register_dim: int

    @nn.compact
    def __call__(
        self,
        h: Float[Array, "n_nodes feature_dim"],
        spin: Integer[Array, " n_nodes"],
    ):
        # Inspired by https://openreview.net/forum?id=2dnO3LLiJ1
        register_keys = self.param("register_keys", nn.initializers.normal(1), (self.n_register, self.register_dim))
        queries = nn.Dense(self.n_register * self.register_dim)(h).reshape(-1, self.n_register, self.register_dim)
        values = nn.Dense(self.n_register * self.register_dim)(h).reshape(-1, self.n_register, self.register_dim)

        attention = jnp.einsum("rd,nrd->nr", register_keys, queries)
        attention = jax.nn.softmax(attention / jnp.sqrt(self.register_dim), axis=0)
        register_vals = jnp.einsum("nr,nrd->rd", attention, values)
        # These registers we would have to treat as new particles for our jacobian.
        register = register_vals.reshape(-1)
        register = nn.LayerNorm()(register)
        return register


class RegisterToElec(nn.Module):
    heads: int
    out_dim: int

    @nn.compact
    def __call__(
        self,
        h: Float[Array, "n_nodes feature_dim"],
        register: Float[Array, " reg_dim"],
        spin: Integer[Array, " n_nodes"],
    ):
        att_dim = self.out_dim // self.heads
        register = nn.LayerNorm()(register)
        register = MLP([h.shape[-1]] * 2 + [self.heads * att_dim * 2])(register)
        queries, values = jnp.split(register.reshape(2 * self.heads, att_dim), 2, 0)
        keys = nn.Dense(self.heads * att_dim)(h).reshape(-1, self.heads, att_dim)
        attention = jnp.einsum("nrd,rd->nr", keys, queries)
        attention = jax.nn.softmax(attention / jnp.sqrt(att_dim), axis=1)
        out = jnp.einsum("nr,rd->nrd", attention, values).reshape(-1, self.out_dim)

        # This is from psiformer
        h += out
        h = nn.LayerNorm()(h)

        mlp_out = nn.silu(nn.Dense(self.out_dim)(h))

        h += mlp_out
        h = nn.LayerNorm()(h)
        return h


class RegisterAttention(nn.Module):
    out_dim: int
    n_register: int = 4
    register_dim: int = 4
    heads: int = 4

    @nn.compact
    def __call__(self, h: Float[Array, "n_nodes feature_dim"], spin: Integer[Array, " n_nodes"]):
        register = ElecToRegister(self.n_register, self.register_dim)(h, spin)
        h = RegisterToElec(self.heads, self.out_dim)(h, register, spin)
        return h
