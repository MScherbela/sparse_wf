from sparse_wf.api import StaticInput, StaticInputs
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


class StaticSchedulers:
    def __init__(
        self,
        n_electrons: int,
        n_up: int,
        n_nuclei: int,
        history_length: int = 500,
        max_padding_factor: float = 1.2,
        min_padding_factor: float = 1.05,
        keys=("mcmc", "mcmc_jump", "pp"),
    ):
        self.schedulers = {
            key: StaticScheduler(n_electrons, n_up, n_nuclei, history_length, max_padding_factor, min_padding_factor)
            for key in keys
        }

    def __call__(
        self, actual_statics: dict[str, StaticInput], compilation_cache_size: int | None = None
    ) -> StaticInputs:
        statics_dict = {k: s(actual_statics.get(k), compilation_cache_size) for k, s in self.schedulers.items()}
        return StaticInputs(**statics_dict)


class StaticScheduler:
    def __init__(
        self,
        n_electrons: int,
        n_up: int,
        n_nuclei: int,
        history_length: int = 500,
        max_padding_factor: float = 1.2,
        min_padding_factor: float = 1.05,
    ):
        self.step = 0
        self.did_recompile = np.ones(history_length, bool)
        self.history_length = history_length
        self.history: Optional[StaticInput] = None
        self.n_electrons = n_electrons
        self.n_up = n_up
        self.n_nuclei = n_nuclei
        self.max_padding_factor = max_padding_factor
        self.min_padding_factor = min_padding_factor
        self.padding_factor = max_padding_factor
        self.compilation_cache_size = 0

    def __call__(self, actual_static: StaticInput | None, compilation_cache_size: int | None = None) -> StaticInput:
        if self.history is None:
            self.history = tree_zeros_like(actual_static, jnp.int32, self.history_length)

        if actual_static is not None:
            self.history = jtu.tree_map(
                lambda history, new: history.at[self.step].set(jnp.max(new)), self.history, actual_static
            )
            self.step = (self.step + 1) % self.history_length

        # Adjust the padding depending on how often we have recompiled
        if compilation_cache_size is not None:
            self.did_recompile[self.step] = self.compilation_cache_size != compilation_cache_size
            self.compilation_cache_size = compilation_cache_size
            n_recompiles = np.sum(self.did_recompile)
            if n_recompiles <= 2:
                self.padding_factor = self.min_padding_factor
            elif n_recompiles >= 5:
                self.padding_factor = self.max_padding_factor

        static = jtu.tree_map(lambda x: int(jnp.max(x)), self.history)
        static = static.round_with_padding(self.padding_factor, self.n_electrons, self.n_up, self.n_nuclei)
        return static
