#%%
import jax
import flax.linen as nn
from flax.struct import PyTreeNode

class A(PyTreeNode):
    a: int

    @classmethod
    def create(cls, x):
        return cls(a=2*x)

class B(A):
    b: int

    @classmethod
    def create(cls, x):
        a = super(B, cls).create(x)
        return cls(**a, b=3*x)

var_a = A.create(2)
var_b = B.create(3)




