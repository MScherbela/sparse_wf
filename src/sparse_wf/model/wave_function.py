from typing import NamedTuple, cast

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pyscf

from sparse_wf.api import (
    Charges,
    Electrons,
    HFOrbitals,
    JastrowArgs,
    LocalEnergy,
    LogAmplitude,
    Nuclei,
    ParameterizedWaveFunction,
    Parameters,
    PRNGKeyArray,
    SignedLogAmplitude,
    SlaterMatrices,
)
from sparse_wf.hamiltonian import make_local_energy
from sparse_wf.jax_utils import nn_vmap, vectorize
from sparse_wf.model.graph_utils import NrOfNeighbours
from sparse_wf.model.utils import (
    ElElCusp,
    IsotropicEnvelope,
    JastrowFactor,
    hf_orbitals_to_fulldet_orbitals,
    signed_logpsi_from_orbitals,
    swap_bottom_blocks,
    YakuwaJastrow,
)


class NrOfDependencies(NamedTuple):
    h_el_initial: int
    H_nuc: int
    h_el_out: int


class StaticInput(NamedTuple):
    n_neighbours: NrOfNeighbours
    n_deps: NrOfDependencies


class MoonLikeWaveFunction(nn.Module, ParameterizedWaveFunction[Parameters, StaticInput]):
    # Configuration
    R: Nuclei
    Z: Charges
    n_electrons: int
    n_up: int
    # Model
    n_determinants: int
    n_envelopes: int
    cutoff: float
    use_e_e_cusp: bool
    feature_dim: int
    pair_mlp_widths: tuple[int, int]
    pair_n_envelopes: int
    nuc_mlp_depth: int
    mlp_jastrow_args: JastrowArgs
    log_jastrow_args: JastrowArgs
    use_yukawa_jastrow: bool

    def setup(self):
        self.to_orbitals = nn.Dense(self.n_determinants * self.n_electrons, name="lin_orbitals")
        self.envelope = IsotropicEnvelope(self.n_determinants, self.n_electrons, self.n_envelopes)
        if self.use_e_e_cusp:
            self.e_e_cusp = ElElCusp(self.n_up)
        else:
            self.e_e_cusp = None
        if self.use_yukawa_jastrow:
            self.yakuwa_jastrow = YakuwaJastrow(self.n_up)
        else:
            self.yakuwa_jastrow = None
        if self.mlp_jastrow_args["use"]:
            self.mlp_jastrow = JastrowFactor(
                self.mlp_jastrow_args["embedding_n_hidden"], self.mlp_jastrow_args["soe_n_hidden"]
            )
        else:
            self.mlp_jastrow = None
        if self.log_jastrow_args["use"]:
            self.log_jastrow = JastrowFactor(
                self.log_jastrow_args["embedding_n_hidden"], self.log_jastrow_args["soe_n_hidden"]
            )
        else:
            self.log_jastrow = None
        self.spins = self.get_spins()

    def get_spins(self):
        return jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_electrons - self.n_up)]).astype(jnp.float32)

    def init(self, rng: PRNGKeyArray, electrons: Electrons) -> Parameters:  # type: ignore
        return nn.Module.init(self, rng, electrons, self.get_static_input(electrons), method=self._signed)  # type: ignore

    @classmethod
    def create(
        cls,
        mol: pyscf.gto.Mole,
        cutoff: float,
        feature_dim: int,
        nuc_mlp_depth: int,
        pair_mlp_widths: tuple[int, int],
        pair_n_envelopes: int,
        n_determinants: int,
        n_envelopes: int,
        use_e_e_cusp: bool,
        mlp_jastrow: JastrowArgs,
        log_jastrow: JastrowArgs,
        use_yukawa_jastrow: bool,
    ):
        if use_e_e_cusp and use_yukawa_jastrow:
            raise KeyError("Use either electron-electron cusp or Yukawa")
        return cls(
            R=np.asarray(mol.atom_coords(), dtype=jnp.float32),
            Z=np.asarray(mol.atom_charges()),
            n_electrons=mol.nelectron,
            n_up=mol.nelec[0],
            n_determinants=n_determinants,
            n_envelopes=n_envelopes,
            cutoff=cutoff,
            feature_dim=feature_dim,
            pair_mlp_widths=pair_mlp_widths,
            pair_n_envelopes=pair_n_envelopes,
            nuc_mlp_depth=nuc_mlp_depth,
            use_e_e_cusp=use_e_e_cusp,
            mlp_jastrow_args=mlp_jastrow,
            log_jastrow_args=log_jastrow,
            use_yukawa_jastrow=use_yukawa_jastrow,
        )

    def _embedding(self, electrons: Electrons, static: StaticInput) -> jax.Array:
        raise NotImplementedError

    def _orbitals(self, electrons: Electrons, static: StaticInput, embeddings=None) -> SlaterMatrices:
        if embeddings is None:
            embeddings = self._embedding(electrons, static)
        dist_en_full = jnp.linalg.norm(electrons[:, None, :] - self.R, axis=-1)
        orbitals = self.to_orbitals(embeddings) * nn_vmap(self.envelope)(dist_en_full)
        orbitals = einops.rearrange(orbitals, "el (det orb) -> det el orb", det=self.n_determinants)
        return (swap_bottom_blocks(orbitals, self.n_up),)

    def _signed(self, electrons: Electrons, static: StaticInput) -> SignedLogAmplitude:
        if self.mlp_jastrow or self.log_jastrow:
            embeddings = self._embedding(electrons, static)
        else:
            embeddings = None
        orbitals = self._orbitals(electrons, static, embeddings)
        signpsi, logpsi = signed_logpsi_from_orbitals(orbitals)
        if self.e_e_cusp:
            logpsi += self.e_e_cusp(electrons)
        if self.yakuwa_jastrow:
            logpsi += self.yakuwa_jastrow(electrons)
        if self.mlp_jastrow:
            logpsi += self.mlp_jastrow(embeddings)
        if self.log_jastrow:
            logpsi += jnp.log(jnp.abs(self.log_jastrow(embeddings)))
        return signpsi, logpsi

    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput) -> LogAmplitude:
        return cast(LogAmplitude, self.apply(params, electrons, static, method=self._signed)[1])

    def get_static_input(self, electrons: Electrons) -> StaticInput:
        raise NotImplementedError

    def orbitals(self, params: Parameters, electrons: Electrons, static: StaticInput) -> SlaterMatrices:
        return cast(SlaterMatrices, self.apply(params, electrons, static, method=self._orbitals))

    def hf_transformation(self, hf_orbitals: HFOrbitals) -> SlaterMatrices:
        return hf_orbitals_to_fulldet_orbitals(hf_orbitals)

    @vectorize(signature="(nel,dim)->()", excluded=(0, 1, 3))
    def local_energy(self, params: Parameters, electrons: Electrons, static: StaticInput) -> LocalEnergy:
        if electrons.batch_dim > 0:
            from folx import batched_vmap

            return batched_vmap(
                self.local_energy_dense(params, electrons, static), max_batch_size=64, in_axes=(None, 0, None)
            )
        else:
            return self.local_energy_dense(params, electrons, static)

    @vectorize(signature="(nel,dim)->()", excluded=(0, 1, 3))
    def local_energy_dense(self, params: Parameters, electrons: Electrons, static: StaticInput) -> LocalEnergy:
        return make_local_energy(self, self.R, self.Z)(params, electrons, static)

    @vectorize(signature="(nel,dim)->()", excluded=(0, 1, 3))
    def signed(self, params: Parameters, electrons: Electrons, static: StaticInput) -> SignedLogAmplitude:
        return cast(SignedLogAmplitude, self.apply(params, electrons, static, method=self._signed))
