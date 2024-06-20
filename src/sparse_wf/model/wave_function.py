from typing import cast, NamedTuple
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import pyscf
import numpy as np
import jax.tree_util as jtu
import functools

from sparse_wf.api import (
    Charges,
    Electrons,
    HFOrbitals,
    JastrowArgs,
    EmbeddingArgs,
    LocalEnergy,
    LogAmplitude,
    Nuclei,
    ParameterizedWaveFunction,
    Parameters,
    PRNGKeyArray,
    SignedLogAmplitude,
    SlaterMatrices,
)
from sparse_wf.hamiltonian import make_local_energy, potential_energy
from sparse_wf.jax_utils import vectorize, fwd_lap
from sparse_wf.tree_utils import tree_add
from sparse_wf.model.jastrow import Jastrow
from sparse_wf.model.graph_utils import zeropad_jacobian, slogdet_with_sparse_fwd_lap
from sparse_wf.model.utils import (
    EfficientIsotropicEnvelopes,
    hf_orbitals_to_fulldet_orbitals,
    signed_logpsi_from_orbitals,
    swap_bottom_blocks,
)
from flax.struct import PyTreeNode
from sparse_wf.model.moon import MoonEmbedding, StaticInputMoon
from folx.api import FwdLaplArray


class MoonLikeParams(NamedTuple):
    embedding: Parameters
    to_orbitals: Parameters
    envelope: Parameters
    jastrow: Parameters


class MoonLikeWaveFunction(ParameterizedWaveFunction[Parameters, StaticInputMoon], PyTreeNode):
    # Molecule
    R: Nuclei
    Z: Charges
    n_electrons: int
    n_up: int

    # Hyperparams
    n_determinants: int
    n_envelopes: int

    # Submodules
    to_orbitals: nn.Dense
    envelope: EfficientIsotropicEnvelopes
    embedding: MoonEmbedding
    jastrow: Jastrow

    @property
    def n_nuclei(self):
        return len(self.R)

    def init(self, rng: PRNGKeyArray, electrons: Electrons) -> Parameters:  # type: ignore
        rngs = jax.random.split(rng, 7)
        dummy_embeddings = jnp.zeros([electrons.shape[-2], self.embedding.feature_dim])
        params = MoonLikeParams(
            embedding=self.embedding.init(rngs[0], electrons, self.get_static_input(electrons)),
            to_orbitals=self.to_orbitals.init(rngs[1], dummy_embeddings),
            envelope=self.envelope.init(rngs[2], jnp.zeros([self.n_nuclei])),
            jastrow=self.jastrow.init(rngs[3], electrons, dummy_embeddings),
        )
        return params

    @classmethod
    def create(
        cls,
        mol: pyscf.gto.Mole,
        embedding: EmbeddingArgs,
        jastrow: JastrowArgs,
        n_determinants: int,
        n_envelopes: int,
    ):
        R = np.asarray(mol.atom_coords(), dtype=jnp.float32)
        Z = np.asarray(mol.atom_charges())
        n_electrons = mol.nelectron
        n_up = mol.nelec[0]

        return cls(
            R=R,
            Z=Z,
            n_electrons=mol.nelectron,
            n_up=n_up,
            n_determinants=n_determinants,
            n_envelopes=n_envelopes,
            to_orbitals=nn.Dense(
                n_determinants * mol.nelectron,
                name="to_orbitals",
                bias_init=nn.initializers.truncated_normal(0.01, jnp.float32),
            ),
            envelope=EfficientIsotropicEnvelopes(n_determinants, n_electrons, n_envelopes),
            embedding=MoonEmbedding.create(R, Z, n_electrons, n_up, **embedding),
            jastrow=Jastrow(n_up, **jastrow),
        )

    def _orbitals(self, params: MoonLikeParams, electrons: Electrons, embeddings) -> SlaterMatrices:
        dist_en_full = jnp.linalg.norm(electrons[:, None, :] - self.R, axis=-1)
        orbitals = self.to_orbitals.apply(params.to_orbitals, embeddings)
        envelopes = jax.vmap(lambda d: self.envelope.apply(params.envelope, d))(dist_en_full)
        orbitals = einops.rearrange(orbitals * envelopes, "el (det orb) -> det el orb", det=self.n_determinants)  # type: ignore
        return (swap_bottom_blocks(orbitals, self.n_up),)

    def _orbitals_with_fwd_lap(self, params: MoonLikeParams, electrons: Electrons, embeddings) -> tuple[FwdLaplArray]:
        # vmap over electrons
        @functools.partial(jax.vmap, in_axes=(0, -2), out_axes=-2)
        def get_orbital_row(r_el, h):
            @fwd_lap
            def get_envelopes(r):
                distances = jnp.linalg.norm(r - self.R, axis=-1)
                return self.envelope.apply(params.envelope, distances)

            envelopes = get_envelopes(r_el)
            orbitals = fwd_lap(self.to_orbitals.apply, argnums=1)(params.to_orbitals, h)
            orbitals = zeropad_jacobian(orbitals, n_deps_out=embeddings.jacobian.data.shape[0])
            orbitals = fwd_lap(lambda o, e: (o * e).reshape(self.n_determinants, -1))(orbitals, envelopes)
            return orbitals

        orbitals = get_orbital_row(electrons, embeddings)
        orbitals = jtu.tree_map(lambda o: swap_bottom_blocks(o, self.n_up), orbitals)
        return orbitals

    def _logpsi_with_fwd_lap(self, params, electrons, static):
        embeddings, dependencies = self.embedding.apply_with_fwd_lap(params.embedding, electrons, static)
        orbitals = self._orbitals_with_fwd_lap(params, electrons, embeddings)

        # vmap over determinants
        signs, logdets = jax.vmap(lambda o: slogdet_with_sparse_fwd_lap(o, dependencies), in_axes=-3, out_axes=-1)(
            orbitals
        )
        logpsi = fwd_lap(lambda logdets_: jax.nn.logsumexp(logdets_, b=signs, return_sign=True)[0])(logdets)
        logpsi_jastrow = self.jastrow.apply_with_fwd_lap(params.jastrow, electrons, embeddings, dependencies)
        logpsi = tree_add(logpsi, logpsi_jastrow)
        return logpsi

    @vectorize(signature="(nel,dim)->(),()", excluded=(0, 1, 3))
    def signed(self, params: MoonLikeParams, electrons: Electrons, static: StaticInputMoon) -> SignedLogAmplitude:
        embeddings = self.embedding.apply(params.embedding, electrons, static)
        orbitals = self._orbitals(params, electrons, embeddings)
        signpsi, logpsi = signed_logpsi_from_orbitals(orbitals)
        logpsi += self.jastrow.apply(params.jastrow, electrons, embeddings)
        return signpsi, logpsi

    def orbitals(self, params: MoonLikeParams, electrons: Electrons, static: StaticInputMoon) -> SlaterMatrices:
        embeddings = self.embedding.apply(params.embedding, electrons, static)
        orbitals = self._orbitals(params, electrons, embeddings)
        return cast(SlaterMatrices, orbitals)

    def orbitals_with_fwd_lap(
        self, params: MoonLikeParams, electrons: Electrons, static: StaticInputMoon
    ) -> FwdLaplArray:
        embeddings, dependencies = self.embedding.apply_with_fwd_lap(params.embedding, electrons, static)
        orbitals = self._orbitals_with_fwd_lap(params, electrons, embeddings)
        return orbitals, dependencies

    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInputMoon) -> LogAmplitude:
        return self.signed(params, electrons, static)[1]

    def get_static_input(self, electrons: Electrons) -> StaticInputMoon:
        return self.embedding.get_static_input(electrons)

    def hf_transformation(self, hf_orbitals: HFOrbitals) -> SlaterMatrices:
        return hf_orbitals_to_fulldet_orbitals(hf_orbitals)

    def local_energy(self, params: Parameters, electrons: Electrons, static: StaticInputMoon) -> LocalEnergy:
        logpsi = self._logpsi_with_fwd_lap(params, electrons, static)
        kinetic_energy = -0.5 * (logpsi.laplacian.sum() + jnp.vdot(logpsi.jacobian.data, logpsi.jacobian.data))
        potential = potential_energy(electrons, self.R, self.Z)
        return kinetic_energy + potential

    def local_energy_dense(self, params: Parameters, electrons: Electrons, static: StaticInputMoon) -> LocalEnergy:
        return make_local_energy(self, self.R, self.Z)(params, electrons, static)
