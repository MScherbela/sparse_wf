import functools
from typing import Literal, NamedTuple, cast, overload, Optional, Generic, TypeVar
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pyscf
from flax.struct import PyTreeNode
from folx.api import FwdLaplArray

from sparse_wf.api import (
    Charges,
    ElectronEmb,
    ElectronIdx,
    Electrons,
    Embedding,
    EmbeddingArgs,
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
    EnvelopeArgs,
)
from sparse_wf.hamiltonian import make_local_energy, potential_energy
from sparse_wf.jax_utils import fwd_lap
from sparse_wf.model.envelopes import EfficientIsotropicEnvelopes, GLUEnvelopes, Envelope
from sparse_wf.model.new_model import NewEmbedding
from sparse_wf.model.new_sparse_model import NewSparseEmbedding
from sparse_wf.tree_utils import tree_add
from sparse_wf.model.graph_utils import slogdet_with_sparse_fwd_lap, zeropad_jacobian
from sparse_wf.model.sparse_fwd_lap import (
    NodeWithFwdLap,
    Linear,
    multiply_with_1el_fn,
    sparse_slogdet_with_fwd_lap,
    get_distinct_triplet_indices,
    to_slater_matrices,
)
from sparse_wf.model.jastrow import Jastrow, JastrowState
from sparse_wf.model.moon import MoonEmbedding
from sparse_wf.model.utils import (
    LogPsiState,
    hf_orbitals_to_fulldet_orbitals,
    signed_log_psi_from_orbitals_low_rank,
    signed_logpsi_from_orbitals,
    swap_bottom_blocks,
)


T = TypeVar("T")
S = TypeVar("S")
ES = TypeVar("ES")


class MoonLikeParams(NamedTuple, Generic[T]):
    embedding: T
    to_orbitals: Parameters
    envelope: Parameters
    jastrow: Parameters


class OrbitalState(NamedTuple):
    envelopes: jax.Array
    orbitals: jax.Array


class LowRankState(NamedTuple, Generic[T]):
    embedding: T
    orbitals: OrbitalState
    determinant: LogPsiState
    jastrow: JastrowState


class MoonLikeWaveFunction(ParameterizedWaveFunction[MoonLikeParams[T], S, LowRankState[ES]], PyTreeNode):
    # Molecule
    R: Nuclei
    Z: Charges
    n_electrons: int
    n_up: int

    # Hyperparams
    n_determinants: int

    # Submodules
    to_orbitals: nn.Dense | Linear
    envelope: Envelope
    embedding: Embedding[T, S, ES]
    jastrow: Jastrow

    _sparse_jacobian: bool = False

    @property
    def n_nuclei(self):
        return len(self.R)

    def init(self, rng: PRNGKeyArray, electrons: Electrons) -> Parameters:  # type: ignore
        rngs = jax.random.split(rng, 7)
        dummy_embeddings = jnp.zeros([electrons.shape[-2], self.embedding.feature_dim])
        static = jtu.tree_map(int, self.get_static_input(electrons))
        params = MoonLikeParams(
            embedding=self.embedding.init(rngs[0], electrons, static),
            to_orbitals=self.to_orbitals.init(rngs[1], dummy_embeddings),
            envelope=self.envelope.init(rngs[2], jnp.zeros([self.n_nuclei, 3])),
            jastrow=self.jastrow.init(rngs[3], electrons, dummy_embeddings),
        )
        return params

    @classmethod
    def create(
        cls,
        mol: pyscf.gto.Mole,
        embedding: EmbeddingArgs,
        jastrow: JastrowArgs,
        envelopes: EnvelopeArgs,
        n_determinants: int,
    ):
        R = np.asarray(mol.atom_coords(), dtype=jnp.float32)
        Z = np.asarray(mol.atom_charges())
        n_electrons = mol.nelectron
        n_up = mol.nelec[0]

        if envelopes["envelope"] == "isotropic":
            env = EfficientIsotropicEnvelopes(n_determinants, n_electrons, **envelopes["isotropic_args"])
        elif envelopes["envelope"] == "glu":
            env = GLUEnvelopes(Z, n_determinants, n_electrons, **envelopes["glu_args"])  # type: ignore
        else:
            raise ValueError(f"Unknown envelope type {envelopes['+envelope']}")

        to_orbitals: Linear | nn.Dense = nn.Dense(
            n_determinants * mol.nelectron,
            name="to_orbitals",
            bias_init=nn.initializers.truncated_normal(0.01, jnp.float32),
        )
        _sparse_jacobian = False
        emb_mod: MoonEmbedding | NewEmbedding
        match embedding["embedding"].lower():
            case "moon":
                emb_mod = MoonEmbedding.create(R, Z, n_electrons, n_up, **embedding["moon"])
            case "new":
                emb_mod = NewEmbedding.create(R, Z, n_electrons, n_up, **embedding["new"])
            case "new_sparse":
                emb_mod = NewSparseEmbedding.create(R, Z, n_electrons, n_up, **embedding["new"])
                to_orbitals = Linear(
                    n_determinants * mol.nelectron,
                    bias_init=nn.initializers.truncated_normal(0.01, jnp.float32),
                )
                _sparse_jacobian = True
            case _:
                raise ValueError(f"Unknown embedding type {embedding['embedding']}")

        return cls(
            R=R,
            Z=Z,
            n_electrons=mol.nelectron,
            n_up=n_up,
            n_determinants=n_determinants,
            to_orbitals=to_orbitals,
            envelope=env,  # type: ignore
            embedding=emb_mod,  # type: ignore
            jastrow=Jastrow(n_up, **jastrow),
            _sparse_jacobian=_sparse_jacobian,
        )

    @overload
    def _orbitals(
        self,
        params: MoonLikeParams[T],
        electrons: Electrons,
        embeddings: ElectronEmb,
        return_state: Literal[False] = False,
    ) -> SlaterMatrices: ...

    @overload
    def _orbitals(
        self,
        params: MoonLikeParams[T],
        electrons: Electrons,
        embeddings: ElectronEmb,
        return_state: Literal[True],
    ) -> tuple[SlaterMatrices, OrbitalState]: ...

    def _orbitals(
        self,
        params: MoonLikeParams[T],
        electrons: Electrons,
        embeddings: ElectronEmb,
        return_state: bool = False,
    ) -> SlaterMatrices | tuple[SlaterMatrices, OrbitalState]:
        diffs_en_full = electrons[:, None, :] - self.R
        envelopes = cast(jax.Array, jax.vmap(lambda d: self.envelope.apply(params.envelope, d))(diffs_en_full))
        raw_orbitals = cast(jax.Array, self.to_orbitals.apply(params.to_orbitals, embeddings))
        raw_orbitals *= envelopes
        orbitals = einops.rearrange(raw_orbitals, "el (det orb) -> det el orb", det=self.n_determinants)  # type: ignore
        result = (swap_bottom_blocks(orbitals, self.n_up),)
        if return_state:
            return result, OrbitalState(envelopes, raw_orbitals)
        return result

    def _orbitals_low_rank(
        self,
        params: MoonLikeParams[T],
        electrons: Electrons,
        embeddings: ElectronEmb,
        position_changed_electrons: ElectronIdx,
        changed_electrons: ElectronIdx,
        state: OrbitalState,
    ):
        # Compute envelopes only for the actually moved electrons
        diffs_en_full = electrons[position_changed_electrons][:, None, :] - self.R
        envelopes = cast(jax.Array, jax.vmap(lambda d: self.envelope.apply(params.envelope, d))(diffs_en_full))
        envelopes = state.envelopes.at[position_changed_electrons].set(envelopes)
        # Compute orbital update
        raw_orbitals = self.to_orbitals.apply(params.to_orbitals, embeddings[changed_electrons])
        raw_orbitals *= envelopes[changed_electrons]
        raw_orbitals = state.orbitals.at[changed_electrons].set(raw_orbitals)
        orbitals = einops.rearrange(raw_orbitals, "el (det orb) -> det el orb", det=self.n_determinants)  # type: ignore
        return (swap_bottom_blocks(orbitals, self.n_up),), OrbitalState(envelopes, raw_orbitals)

    def _orbitals_with_fwd_lap_folx(
        self, params: MoonLikeParams[T], electrons: Electrons, embeddings: ElectronEmb
    ) -> tuple[FwdLaplArray]:
        # vmap over electrons
        @functools.partial(jax.vmap, in_axes=(0, -2), out_axes=-2)
        def get_orbital_row(r_el, h):
            @fwd_lap
            def get_envelopes(r):
                diffs = r - self.R
                return self.envelope.apply(params.envelope, diffs)

            envelopes = get_envelopes(r_el)
            orbitals = fwd_lap(self.to_orbitals.apply, argnums=1)(params.to_orbitals, h)
            orbitals = zeropad_jacobian(orbitals, n_deps_out=embeddings.jacobian.data.shape[0])
            orbitals = fwd_lap(lambda o, e: (o * e).reshape(self.n_determinants, -1))(orbitals, envelopes)
            return orbitals

        orbitals = get_orbital_row(electrons, embeddings)
        orbitals = jtu.tree_map(lambda o: swap_bottom_blocks(o, self.n_up), orbitals)
        return orbitals

    def _orbitals_with_fwd_lap_sparse(
        self, params: MoonLikeParams[T], electrons: Electrons, embeddings: NodeWithFwdLap
    ):
        envelopes = jax.vmap(fwd_lap(lambda r: self.envelope.apply(params.envelope, r - self.R)))(electrons)
        orbitals = self.to_orbitals.apply_with_fwd_lap(params.to_orbitals, embeddings)
        orbitals = multiply_with_1el_fn(orbitals, envelopes)
        phi, jac_phi, lap_phi = to_slater_matrices(orbitals, self.n_electrons, self.n_up)
        return NodeWithFwdLap(phi, jac_phi, lap_phi, orbitals.idx_ctr, orbitals.idx_dep)

    def _logpsi_with_fwd_lap_folx(self, params: MoonLikeParams[T], electrons: Electrons, static: S):
        embeddings, dependencies = self.embedding.apply_with_fwd_lap(params.embedding, electrons, static)
        orbitals = self._orbitals_with_fwd_lap_folx(params, electrons, embeddings)

        # vmap over determinants
        signs, logdets = jax.vmap(lambda o: slogdet_with_sparse_fwd_lap(o, dependencies), in_axes=-3, out_axes=-1)(
            orbitals
        )
        logpsi = fwd_lap(lambda logdets_: jax.nn.logsumexp(logdets_, b=signs, return_sign=True)[0])(logdets)
        logpsi_jastrow = self.jastrow.apply_with_fwd_lap(params.jastrow, electrons, embeddings, dependencies)
        logpsi = tree_add(logpsi, logpsi_jastrow)
        return logpsi

    def _logpsi_with_fwd_lap_sparse(self, params: MoonLikeParams[T], electrons: Electrons, static: S):
        embeddings = cast(NodeWithFwdLap, self.embedding.apply_with_fwd_lap(params.embedding, electrons, static))
        orbitals = self._orbitals_with_fwd_lap_sparse(params, electrons, embeddings)

        triplet_indices = get_distinct_triplet_indices(electrons, self.embedding.cutoff, static.n_triplets)  # type: ignore
        signs, logdets = jax.vmap(
            lambda x, J, lap: sparse_slogdet_with_fwd_lap(
                NodeWithFwdLap(x, J, lap, orbitals.idx_ctr, orbitals.idx_dep), triplet_indices
            ),
            in_axes=0,
            out_axes=-1,
        )(orbitals.x, orbitals.jac, orbitals.lap)  # vmap over determinants
        logpsi = fwd_lap(lambda logdets_: jax.nn.logsumexp(logdets_, b=signs, return_sign=True)[0])(logdets)
        logpsi_jastrow = self.jastrow.apply_with_fwd_lap(params.jastrow, electrons, embeddings, None)
        logpsi = tree_add(logpsi, logpsi_jastrow)
        return logpsi

    def signed(self, params: MoonLikeParams[T], electrons: Electrons, static: S) -> SignedLogAmplitude:
        embeddings = self.embedding.apply(params.embedding, electrons, static)
        orbitals = self._orbitals(params, electrons, embeddings)
        signpsi, logpsi = signed_logpsi_from_orbitals(orbitals)
        sign_J, log_J = self.jastrow.apply(params.jastrow, electrons, embeddings)
        signpsi *= sign_J
        logpsi += log_J
        return signpsi, logpsi

    def orbitals(self, params: MoonLikeParams[T], electrons: Electrons, static: S) -> SlaterMatrices:
        embeddings = self.embedding.apply(params.embedding, electrons, static)
        orbitals = self._orbitals(params, electrons, embeddings)
        return cast(SlaterMatrices, orbitals)

    def orbitals_with_fwd_lap(self, params: MoonLikeParams[T], electrons: Electrons, static: S) -> FwdLaplArray:
        embeddings, dependencies = self.embedding.apply_with_fwd_lap(params.embedding, electrons, static)
        orbitals = self._orbitals_with_fwd_lap_folx(params, electrons, embeddings)
        return orbitals, dependencies

    def __call__(self, params: MoonLikeParams[T], electrons: Electrons, static: S) -> LogAmplitude:
        return self.signed(params, electrons, static)[1]

    def get_static_input(
        self, electrons: Electrons, electrons_new: Optional[Electrons] = None, idx_changed: Optional[ElectronIdx] = None
    ) -> S:
        return self.embedding.get_static_input(electrons, electrons_new, idx_changed)

    def hf_transformation(self, hf_orbitals: HFOrbitals) -> SlaterMatrices:
        return hf_orbitals_to_fulldet_orbitals(hf_orbitals)

    def local_energy(self, params: MoonLikeParams[T], electrons: Electrons, static: S) -> LocalEnergy:
        if self._sparse_jacobian:
            logpsi = self._logpsi_with_fwd_lap_sparse(params, electrons, static)
        else:
            logpsi = self._logpsi_with_fwd_lap_folx(params, electrons, static)
        kinetic_energy = -0.5 * (logpsi.laplacian.sum() + jnp.vdot(logpsi.jacobian.data, logpsi.jacobian.data))
        potential = potential_energy(electrons, self.R, self.Z)
        return kinetic_energy + potential

    def local_energy_dense(self, params: MoonLikeParams[T], electrons: Electrons, static: S) -> LocalEnergy:
        return make_local_energy(self, self.R, self.Z)(params, electrons, static)

    def log_psi_with_state(
        self, params: MoonLikeParams[T], electrons: Electrons, static: S
    ) -> tuple[SignedLogAmplitude, LowRankState]:
        embeddings, state = self.embedding.apply(
            params.embedding,
            electrons,
            static,
            return_scales=False,
            return_state=True,
        )
        orbitals, orbitals_state = self._orbitals(params, electrons, embeddings, return_state=True)
        (sign, logpsi), determinant_state = signed_logpsi_from_orbitals(orbitals, return_state=True)
        (sign_J, log_J), jastrow_state = self.jastrow.apply(params.jastrow, electrons, embeddings, return_state=True)
        logpsi += log_J
        sign *= sign_J
        return (sign, logpsi), LowRankState(
            embedding=state,
            orbitals=orbitals_state,
            determinant=determinant_state,
            jastrow=cast(JastrowState, jastrow_state),
        )

    def log_psi_low_rank_update(
        self,
        params: MoonLikeParams[T],
        electrons: Electrons,
        changed_electrons: ElectronIdx,
        static: S,
        state: LowRankState,
    ) -> tuple[SignedLogAmplitude, LowRankState]:
        embeddings, changed_embeddings, embedding_state = self.embedding.low_rank_update(
            params.embedding, electrons, changed_electrons, static, state.embedding
        )
        orbitals, orbitals_state = self._orbitals_low_rank(
            params, electrons, embeddings, changed_electrons, changed_embeddings, state.orbitals
        )
        (sign, logpsi), determinant_state = signed_log_psi_from_orbitals_low_rank(
            orbitals, changed_embeddings, state.determinant
        )
        (sign_J, log_J), jastrow_state = self.jastrow.apply(
            params.jastrow,
            electrons,
            embeddings,
            changed_electrons,
            changed_embeddings,
            state.jastrow,
            method=self.jastrow.low_rank_update,
        )
        sign *= sign_J
        logpsi += log_J
        return (sign, logpsi), LowRankState(
            embedding_state,
            orbitals_state,
            determinant_state,
            cast(JastrowState, jastrow_state),
        )
