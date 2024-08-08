#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax
import jax.tree_util as jtu
from typing import Generic, TypeVar, NamedTuple
from sparse_wf.api import StaticInput
from abc import ABC, abstractmethod
import flax.struct
import jax.numpy as jnp
import dataclasses
import functools
import time

S = TypeVar('S')


class NrOfNeighbours(NamedTuple, Generic[S]):
    ee: S
    en: S

@flax.struct.dataclass
class StaticInputDerived(StaticInput, Generic[S]):
    a: S
    b: S
    neighbours: NrOfNeighbours[S]


@functools.partial(jax.jit, static_argnums=0)
def my_func(s):
    return jnp.ones(s.a + s.b)


static: StaticInputDerived[int] = StaticInputDerived(1, 2, NrOfNeighbours(3, 4))
print(static.to_log_data())
y = my_func(static)
print(static)

#%%

class TimingContext:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

with TimingContext() as tc:
    time.sleep(0.1)
print(tc.interval)
