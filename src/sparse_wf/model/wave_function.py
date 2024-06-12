from typing import NamedTuple, cast, Optional

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
from sparse_wf.jax_utils import vectorize
from sparse_wf.model.graph_utils import NrOfNeighbours
from sparse_wf.model.utils import (
    ElElCusp,
    IsotropicEnvelope,
    hf_orbitals_to_fulldet_orbitals,
    signed_logpsi_from_orbitals,
    swap_bottom_blocks,
)
from flax.struct import PyTreeNode


class NrOfDependencies(NamedTuple):
    h_el_initial: int
    H_nuc: int
    h_el_out: int


class StaticInput(NamedTuple):
    n_neighbours: NrOfNeighbours
    n_deps: NrOfDependencies


class MoonLikeParams(NamedTuple):
    embedding: Parameters
    to_orbitals: Parameters
    envelope: Parameters
    e_e_cusp: Optional[Parameters]


class MoonLikeWaveFunction(ParameterizedWaveFunction[Parameters, StaticInput], PyTreeNode):
    # Configuration
    R: Nuclei
    Z: Charges
    n_electrons: int
    n_up: int
    # Model hyperparams
    n_determinants: int
    n_envelopes: int
    cutoff: float
    use_e_e_cusp: bool
    feature_dim: int
    pair_mlp_widths: tuple[int, int]
    pair_n_envelopes: int
    nuc_mlp_depth: int

    # TODO: refactor
    mlp_jastrow_args: JastrowArgs
    log_jastrow_args: JastrowArgs
    use_yukawa_jastrow: bool

    # Submodules
    to_orbitals: nn.Dense
    envelope: IsotropicEnvelope
    e_e_cusp: Optional[ElElCusp]
    # TODO: refactor that all jastrow variants are in one class
    # jastrow_factor: JastrowFactor
    # yakuwa_jastrow: YakuwaJastrow
    # mlp_jastrow: JastrowFactor
    # log_jastrow: JastrowFactor

    @property
    def n_nuclei(self):
        return len(self.R)

    @property
    def spins(self):
        return jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_electrons - self.n_up)]).astype(jnp.float32)

    def init(self, rng: PRNGKeyArray, electrons: Electrons) -> Parameters:  # type: ignore
        rngs = jax.random.split(rng, 4)
        params = MoonLikeParams(
            embedding=self.init_embedding(rngs[0], electrons, self.get_static_input(electrons)),
            to_orbitals=self.to_orbitals.init(rngs[1], jnp.zeros((self.feature_dim,))),
            envelope=self.envelope.init(rngs[2], jnp.zeros([self.n_nuclei])),
            e_e_cusp=self.e_e_cusp.init(rngs[3], electrons) if self.e_e_cusp else None,
        )
        return params

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
            to_orbitals=nn.Dense(n_determinants * mol.nelectron, name="lin_orbitals"),
            envelope=IsotropicEnvelope(n_determinants, mol.nelectron, n_envelopes),
            e_e_cusp=ElElCusp(mol.nelec[0]) if use_e_e_cusp else None,
        )

    def init_embedding(self, rng: PRNGKeyArray, electrons: Electrons, static: StaticInput) -> Parameters:
        return NotImplementedError

    def embedding(self, params: Parameters, electrons: Electrons, static: StaticInput) -> jax.Array:
        raise NotImplementedError

    def _orbitals(self, params: MoonLikeParams, electrons: Electrons, embeddings) -> SlaterMatrices:
        dist_en_full = jnp.linalg.norm(electrons[:, None, :] - self.R, axis=-1)
        orbitals = self.to_orbitals.apply(params.to_orbitals, embeddings)
        envelopes = jax.vmap(lambda d: self.envelope.apply(params.envelope, d))(dist_en_full)
        orbitals = einops.rearrange(orbitals * envelopes, "el (det orb) -> det el orb", det=self.n_determinants)
        return (swap_bottom_blocks(orbitals, self.n_up),)

    @vectorize(signature="(nel,dim)->(),()", excluded=(0, 1, 3))
    def signed(self, params: Parameters, electrons: Electrons, static: StaticInput) -> SignedLogAmplitude:
        embeddings = self.embedding(params.embedding, electrons, static)
        orbitals = self._orbitals(params, electrons, embeddings)
        signpsi, logpsi = signed_logpsi_from_orbitals(orbitals)
        # if self.e_e_cusp:
        #     logpsi += self.e_e_cusp.apply(params.e_e_cusp, electrons)
        # if self.yakuwa_jastrow:
        #     logpsi += self.yakuwa_jastrow(electrons)
        # if self.mlp_jastrow:
        #     logpsi += self.mlp_jastrow(embeddings)
        # if self.log_jastrow:
        #     logpsi += jnp.log(jnp.abs(self.log_jastrow(embeddings)))
        return signpsi, logpsi

    def orbitals(self, params: Parameters, electrons: Electrons, static: StaticInput) -> SlaterMatrices:
        embeddings = self.embedding(params.embedding, electrons, static)
        orbitals = self._orbitals(params, electrons, embeddings)
        return cast(SlaterMatrices, orbitals)

    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput) -> LogAmplitude:
        return self.signed(params, electrons, static)[1]

    def get_static_input(self, electrons: Electrons) -> StaticInput:
        raise NotImplementedError

    def hf_transformation(self, hf_orbitals: HFOrbitals) -> SlaterMatrices:
        return hf_orbitals_to_fulldet_orbitals(hf_orbitals)

    def local_energy(self, params: Parameters, electrons: Electrons, static: StaticInput) -> LocalEnergy:
        return self.local_energy_dense(params, electrons, static)

    def local_energy_dense(self, params: Parameters, electrons: Electrons, static: StaticInput) -> LocalEnergy:
        return make_local_energy(self, self.R, self.Z)(params, electrons, static)
