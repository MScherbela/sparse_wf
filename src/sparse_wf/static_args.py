from sparse_wf.api import Int, StaticInput
from sparse_wf.tree_utils import tree_zeros_like
from typing import Optional
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


def to_static(static: StaticInput[jax.Array]) -> StaticInput[int]:
    return jtu.tree_map(lambda x: int(jnp.max(x)), static)


def round_with_padding(n, padding_factor, max_val):
    if padding_factor <= 1.0:
        return min(n, max_val)
    power = np.log(n) / np.log(padding_factor)
    n_padded = padding_factor ** np.ceil(power)
    return int(np.minimum(n_padded, max_val))


class StaticScheduler:
    def __init__(self, n_electrons: int, n_nuclei: int, history_length: int = 5, padding_factor: float = 1.1):
        self.step = 0
        self.history_length = history_length
        self.history: Optional[StaticInput[np.array]] = None
        self.n_electrons = n_electrons
        self.n_nuclei = n_nuclei
        self.padding_factor = padding_factor

    def __call__(self, actual_static: StaticInput[Int]) -> StaticInput[int]:
        if self.history is None:
            self.history = tree_zeros_like(actual_static, jnp.int32, self.history_length)
        self.history = jtu.tree_map(
            lambda history, new: history.at[self.step].set(jnp.max(new)), self.history, actual_static
        )
        self.step = (self.step + 1) % self.history_length
        static = jtu.tree_map(lambda x: int(jnp.max(x)), self.history)
        static = StaticInput(
            mcmc=round_with_padding(static.mcmc, self.padding_factor, self.n_electrons),
            model=static.model.round_with_padding(self.padding_factor, self.n_electrons, self.n_nuclei),
        )

        return static
