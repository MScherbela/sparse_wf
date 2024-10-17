from sparse_wf.api import StaticInput
from sparse_wf.tree_utils import tree_zeros_like
from typing import Optional
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


def round_with_padding(n, padding_factor, max_val):
    if padding_factor <= 1.0:
        return min(n, max_val)
    power = np.log(n) / np.log(padding_factor)
    n_padded = padding_factor ** np.ceil(power)
    return int(np.minimum(n_padded, max_val))


class StaticScheduler:
    def __init__(
        self,
        n_electrons: int,
        n_up: int,
        n_nuclei: int,
        history_length: int = 100,
        max_padding_factor: float = 1.2,
        min_padding_factor: float = 1.05,
    ):
        self.step = 0
        self.steps_since_recompile = 0
        self.history_length = history_length
        self.history: Optional[StaticInput] = None
        self.previous_static: Optional[StaticInput] = None
        self.n_electrons = n_electrons
        self.n_up = n_up
        self.n_nuclei = n_nuclei
        self.padding_factor = max_padding_factor
        self.max_padding_factor = max_padding_factor
        self.min_padding_factor = min_padding_factor

    def __call__(self, actual_static: StaticInput) -> StaticInput:
        if self.history is None:
            self.history = tree_zeros_like(actual_static, jnp.int32, self.history_length)
            self.previous_static = tree_zeros_like(actual_static, jnp.int32)
        self.history = jtu.tree_map(
            lambda history, new: history.at[self.step].set(jnp.max(new)), self.history, actual_static
        )
        self.step = (self.step + 1) % self.history_length
        static = jtu.tree_map(lambda x: int(jnp.max(x)), self.history)
        static = static.round_with_padding(self.padding_factor, self.n_electrons, self.n_up, self.n_nuclei)

        # Check if the static input has changed (ie. probably triggering a recompile)
        if any([x != y for x, y in zip(jtu.tree_leaves(static), jtu.tree_leaves(self.previous_static))]):
            self.steps_since_recompile = 0
            self.previous_static = static
            self.padding_factor = self.max_padding_factor

        # If the statics have not changed in a while, reduce the padding factor
        if self.steps_since_recompile > 100:
            self.padding_factor = self.min_padding_factor
        return static
