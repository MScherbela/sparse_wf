from typing import Sequence, cast

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pyscf
from flax.struct import PyTreeNode
from jaxtyping import Array, Float

from sparse_wf.api import (
    ElectronIdx,
    Electrons,
    HFOrbitals,
    ParameterizedWaveFunction,
    PRNGKeyArray,
    SlaterMatrices,
)
from sparse_wf.hamiltonian import make_local_energy
from sparse_wf.jax_utils import vectorize
from sparse_wf.model.utils import ElecInp, SlaterOrbitals, hf_orbitals_to_fulldet_orbitals, signed_logpsi_from_orbitals

PairInp = Float[Array, "*batch n_electrons n_electrns n_pair_in"]
ElecOut = Float[Array, "*batch n_electrons n_out"]
PairOut = Float[Array, "*batch n_electrons n_electrns n_pair_out"]
FermiLayerOut = tuple[ElecOut, PairOut]


def residual(x: Array, y: Array) -> Array:
    if x.shape == y.shape:
        return (x + y) / jnp.asarray(np.sqrt(2), dtype=x.dtype)
    return x


# IDEA: Isn't that exactly the same as nn.Dense?
class RepeatedDense(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel = self.param("kernel", jnn.initializers.lecun_normal(), (x.shape[-1], self.out_dim), jnp.float32)
        bias = self.param("bias", jnn.initializers.zeros, (self.out_dim,), jnp.float32)
        out = x @ kernel + bias
        return cast(jax.Array, out)


class GroupDense(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, *groups: tuple[jax.Array, ...]) -> tuple[tuple[jax.Array, ...], ...]:
        result: list[list[Array]] = []
        for group in groups:
            shapes = [x.shape[:-1] for x in group]
            sizes = [np.prod(s) for s in shapes]
            cum_sizes = np.cumsum(sizes)
            inp = jnp.concatenate([x.reshape(-1, x.shape[-1]) for x in group], axis=0)
            out = nn.Dense(self.out_dim)(inp)
            result.append([x.reshape(*s, self.out_dim) for s, x in zip(shapes, jnp.split(out, cum_sizes[:-1], axis=0))])
        return tuple(map(tuple, result))


class FermiLayer(nn.Module):
    spins: tuple[int, int]
    single_dim: int
    pair_dim: int
    activation: str

    @nn.compact
    def __call__(self, h_one: ElecInp, h_two: PairInp) -> FermiLayerOut:
        norm = np.asarray(1 / np.sqrt(3.0), dtype=jnp.float32)
        out = nn.Dense(self.single_dim, use_bias=False)(h_one) * norm
        spins = np.array(self.spins)

        uu, ud, du, dd = [
            s for split in jnp.split(h_two, spins[:1], axis=0) for s in jnp.split(split, spins[:1], axis=1)
        ]
        pair_inp = jnp.concatenate(
            [jnp.concatenate(pairs, axis=-1) for pairs in [[uu.mean(1), ud.mean(1)], [dd.mean(1), du.mean(1)]]], axis=-2
        )
        out += nn.Dense(self.single_dim, use_bias=False)(pair_inp).reshape(spins.sum(), -1) * norm

        up, down = jnp.split(h_one, spins[:1], axis=0)
        up, down = jnp.mean(up, axis=0), jnp.mean(down, axis=0)
        global_inp = jnp.stack([jnp.concatenate([up, down], axis=-1), jnp.concatenate([down, up], axis=-1)], axis=0)
        out += RepeatedDense(self.single_dim)(global_inp).repeat(spins, axis=0) * norm

        act = getattr(nn, self.activation)
        out = act(out)

        (uu, dd), (ud, du) = GroupDense(self.pair_dim)((uu, dd), (ud, du))
        pair_out = jnp.concatenate(
            [
                jnp.concatenate([uu, ud], axis=1),
                jnp.concatenate([du, dd], axis=1),
            ],
            axis=0,
        )
        pair_out = act(pair_out)
        return out, pair_out


class FermiNetOrbitals(nn.Module):
    mol: pyscf.gto.Mole
    n_determinants: int = 16
    n_envelopes: int = 16
    hidden_dims: Sequence[tuple[int, int]] = ((256, 32), (256, 32), (256, 32), (256, 32))
    activation: str = "tanh"

    @nn.compact
    def __call__(self, electrons: Electrons, static: None):
        electrons = electrons.reshape(-1, 3)
        n_ele = electrons.shape[0]
        assert n_ele == self.mol.nelectron
        spins = int(self.mol.nelec[0]), int(self.mol.nelec[1])
        nuclei = jnp.array(self.mol.atom_coords(), dtype=electrons.dtype)
        r_im = electrons[..., None, :] - nuclei
        r_im = jnp.concatenate([r_im, jnp.linalg.norm(r_im, axis=-1, keepdims=True)], axis=-1)

        r_ij = electrons[..., None, :] - electrons
        diag_mask = jnp.eye(n_ele, dtype=electrons.dtype)[..., None]
        r_ij = jnp.concatenate(
            [r_ij, jnp.linalg.norm(r_ij + diag_mask, axis=-1, keepdims=True) * (1 - diag_mask)], axis=-1
        )

        h_one, h_two = r_im.reshape(n_ele, -1), r_ij

        for dim in self.hidden_dims:
            h_one_new, h_two_new = FermiLayer(spins, dim[0], dim[1], self.activation)(h_one, h_two)
            h_one, h_two = residual(h_one_new, h_one), residual(h_two_new, h_two)
        dist_im = r_im[..., -1]

        return SlaterOrbitals(self.n_determinants, self.n_envelopes, spins)(h_one, dist_im)


FermiNetParams = dict[str, "FermiNetParams"] | Array


class DenseFermiNet(ParameterizedWaveFunction[FermiNetParams, None, None], PyTreeNode):
    mol: pyscf.gto.Mole
    ferminet: FermiNetOrbitals

    def get_static_input(self, electrons: Electrons):
        return None

    @classmethod
    def create(cls, mol: pyscf.gto.Mole):
        return cls(mol, FermiNetOrbitals(mol))

    def init(self, key: PRNGKeyArray, electrons: Electrons) -> FermiNetParams:
        return cast(FermiNetParams, self.ferminet.init(key, electrons, self.get_static_input(electrons)))

    def orbitals(self, params: FermiNetParams, electrons: Electrons, static) -> SlaterMatrices:
        return self.ferminet.apply(params, electrons, static)  # type: ignore

    def signed(self, params: FermiNetParams, electrons: Electrons, static):
        orbitals = self.orbitals(params, electrons, static)
        return signed_logpsi_from_orbitals(orbitals)

    def __call__(self, params: FermiNetParams, electrons: Electrons, static):
        return self.signed(params, electrons, static)[1]

    def hf_transformation(self, hf_orbitals: HFOrbitals) -> SlaterMatrices:
        return hf_orbitals_to_fulldet_orbitals(hf_orbitals)

    @vectorize(signature="(nel,dim)->()", excluded=(0, 1, 3))
    def local_energy(self, params: FermiNetParams, electrons: Electrons, static):
        return make_local_energy(self, self.mol.atom_coords(), self.mol.atom_charges())(params, electrons, static)

    @vectorize(signature="(nel,dim)->()", excluded=(0, 1, 3))
    def local_energy_dense(self, params: FermiNetParams, electrons: Electrons, static):
        return make_local_energy(self, self.mol.atom_coords(), self.mol.atom_charges())(params, electrons, static)

    def log_psi_with_state(self, params: FermiNetParams, electrons: Electrons, static):
        return self(params, electrons, static), None

    def log_psi_low_rank_update(
        self, params: FermiNetParams, electrons: Electrons, changed_electrons: ElectronIdx, static, state
    ):
        return self(params, electrons, static), state
