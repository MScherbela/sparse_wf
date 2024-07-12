from sparse_wf.api import Int, StaticInput
from sparse_wf.tree_utils import tree_zeros_like
from typing import Optional
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


class StaticScheduler:
    def __init__(self, n_electrons: int, history_length: int = 5, padding_factor: float = 1.1):
        self.step = 0
        self.history_length = history_length
        self.history: Optional[StaticInput[np.array]] = None
        self.n_electrons = n_electrons
        self.padding_factor = padding_factor

    def round_with_padding(self, n, min_val=None):
        if self.padding_factor <= 1.0:
            return n
        min_val = min_val or self.n_electrons
        power = np.log(n) / np.log(self.padding_factor)
        n_padded = self.padding_factor ** np.ceil(power)
        return int(np.minimum(n_padded, min_val))

    def __call__(self, actual_static: StaticInput[Int]) -> StaticInput[int]:
        if self.history is None:
            self.history = tree_zeros_like(actual_static, jnp.int32, self.history_length)
        self.history = jtu.tree_map(
            lambda history, new: history.at[self.step].set(jnp.max(new)), self.history, actual_static
        )
        self.step = (self.step + 1) % self.history_length
        static = jtu.tree_map(lambda x: self.round_with_padding(jnp.max(x)), self.history)
        return static
