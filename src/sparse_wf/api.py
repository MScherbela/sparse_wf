from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, NamedTuple, Optional, Protocol, Sequence, TypeAlias, TypedDict, TypeVar, overload
import jax
import numpy as np
import optax
from flax import struct
from flax.serialization import to_bytes, from_bytes
from jaxtyping import Array, ArrayLike, Float, Integer, PRNGKeyArray, PyTree
from pyscf.scf.hf import SCF
from typing import Literal

AnyArray = Array | list | np.ndarray
Int = Integer[Array, ""]


Position = Float[Array, "spatial=3"] | Float[np.ndarray, "spatial=3"] | tuple[float, float, float] | list[float]
Electrons = Float[Array, "*batch_dims n_electrons spatial=3"]
ElectronEmb = Float[Array, "*batch_dims n_electrons feature_dim"]
ElectronIdx = Integer[Array, "*batch_dims n_changed"]
Spins = Integer[Array, "*batch_dims n_electrons"]
Nuclei = Float[np.ndarray, "n_nuclei spatial=3"]
NucleiIdx = Integer[Array, "n_nuclei"]
Charges = Integer[np.ndarray, "n_nuclei"]
MeanField: TypeAlias = SCF

SlaterMatrix = Float[Array, "*batch_dims n_determinants n_electrons n_electrons"]
SlaterMatrices = tuple[SlaterMatrix, ...]
HFOrbitals = tuple[SlaterMatrix, SlaterMatrix]
OrbCoefficients = Float[Array, "n_determinants"]
Amplitude = Float[Array, "*batch_dims"]
LogAmplitude = Float[Array, "*batch_dims"]
WfSign = Integer[Array, "*batch_dims"]
SignedLogAmplitude = tuple[WfSign, LogAmplitude]

LocalEnergy = Float[Array, "*batch_dims"]
EnergyCotangent = Float[Array, "*batch_dims"]

Parameter = Float[Array, "..."]
Parameters: TypeAlias = PyTree[Parameter]

Width = Float[Array, ""]
PMove = Float[Array, ""]
PMoves = Float[PMove, "window_size"]

Damping = Float[Array, ""]
AuxData = PyTree[Float[Array, ""]]
Step = Integer[ArrayLike, ""]
ModelCache = PyTree[Any]


############################################################################
# Wave function
############################################################################
# Static input
S = TypeVar("S")
S_contra = TypeVar("S_contra", contravariant=True)
# Parameters
P = TypeVar("P", bound=Parameters)
P_contra = TypeVar("P_contra", bound=Parameters, contravariant=True)
# Model state
MS = TypeVar("MS")


class HFOrbitalFn(Protocol):
    def __call__(self, electrons: Electrons) -> HFOrbitals: ...


class ParameterizedWaveFunction(Protocol[P, S, MS]):
    def init(self, key: PRNGKeyArray, electrons: Electrons) -> P: ...
    def get_static_input(self, electrons: Electrons) -> S: ...
    def orbitals(self, params: P, electrons: Electrons, static: S) -> SlaterMatrices: ...
    def hf_transformation(self, hf_orbitals: HFOrbitals) -> SlaterMatrices: ...
    def local_energy(self, params: P, electrons: Electrons, static: S) -> LocalEnergy: ...
    def local_energy_dense(self, params: P, electrons: Electrons, static: S) -> LocalEnergy: ...
    def signed(self, params: P, electrons: Electrons, static: S) -> SignedLogAmplitude: ...

    @overload
    def __call__(
        self, params: P, electrons: Electrons, static: S, return_state: Literal[True]
    ) -> tuple[LogAmplitude, MS]: ...
    @overload
    def __call__(
        self, params: P, electrons: Electrons, static: S, return_state: Literal[False] = False
    ) -> LogAmplitude: ...

    def __call__(
        self, params: P, electrons: Electrons, static: S, return_state: bool = False
    ) -> LogAmplitude | tuple[LogAmplitude, MS]: ...

    def update_logpsi(
        self,
        params: P,
        electrons: Electrons,
        changed_electrons: ElectronIdx,
        static: S,
        state: MS,
    ) -> tuple[LogAmplitude, MS]: ...


############################################################################
# MCMC
############################################################################
class ClosedLogLikelihood(Protocol):
    def __call__(self, electrons: Electrons) -> LogAmplitude: ...


class MCStep(Protocol[P_contra, S_contra]):
    def __call__(
        self, key: PRNGKeyArray, params: P_contra, electrons: Electrons, static: S_contra, width: Width
    ) -> tuple[Electrons, PMove]: ...


class WidthSchedulerState(NamedTuple):
    width: Width
    pmoves: PMoves
    i: Int


class InitWidthState(Protocol):
    def __call__(self, init_width: Width) -> WidthSchedulerState: ...


class UpdateWidthState(Protocol):
    def __call__(self, state: WidthSchedulerState, pmove: PMove) -> WidthSchedulerState: ...


class WidthScheduler(NamedTuple):
    init: InitWidthState
    update: UpdateWidthState


############################################################################
# Natural Gradient
############################################################################
class PreconditionerState(NamedTuple, Generic[P]):
    last_grad: P


class PreconditionerInit(Protocol[P]):
    def __call__(self, params: P) -> PreconditionerState[P]: ...


class ApplyPreconditioner(Protocol[P, S_contra]):
    def __call__(
        self,
        params: P,
        electrons: Electrons,
        static: S_contra,
        dE_dlogpsi: EnergyCotangent,
        natgrad_state: PreconditionerState[P],
    ) -> tuple[P, PreconditionerState[P], AuxData]: ...


class Preconditioner(NamedTuple, Generic[P, S]):
    init: PreconditionerInit[P]
    precondition: ApplyPreconditioner[P, S]


############################################################################
# Optimizer
############################################################################
class OptState(NamedTuple, Generic[P]):
    opt: optax.OptState
    natgrad: PreconditionerState[P]


class TrainingState(Generic[P], struct.PyTreeNode):  # the order of inheritance is important here!
    key: PRNGKeyArray
    params: P
    electrons: Electrons
    opt_state: OptState[P]
    width_state: WidthSchedulerState

    def serialize(self):
        from sparse_wf.jax_utils import instance, pmap, pgather

        @pmap
        def gather_electrons(electrons):
            return pgather(electrons, axis=0, tiled=True)

        result = instance(self)  # only return a single copy of parameters, opt_state, etc.
        # include electrons from all devices
        result = result.replace(electrons=gather_electrons(self.electrons)[0])
        return to_bytes(result)

    def deserialize(self, data: bytes):
        from sparse_wf.jax_utils import replicate

        state_with_all_electrons = from_bytes(self, data)
        # Distribute electrons to devices
        electrons = state_with_all_electrons.electrons
        electrons = electrons.reshape(jax.process_count(), jax.local_device_count(), -1, *electrons.shape[1:])
        electrons = electrons[jax.process_index()]
        # We have to create new keys for all devices
        key = jax.random.split(state_with_all_electrons.key, (jax.process_count(), jax.local_device_count()))
        key = key[jax.process_index()]
        # Replicate the state to all devices
        result = replicate(state_with_all_electrons)
        result = result.replace(key=key, electrons=electrons)
        return result


class VMCStepFn(Protocol[P, S_contra]):
    def __call__(
        self,
        state: TrainingState[P],
        static: S_contra,
    ) -> tuple[TrainingState[P], LocalEnergy, AuxData]: ...


class SamplingStepFn(Protocol[P, S_contra]):
    def __call__(
        self,
        state: TrainingState[P],
        static: S_contra,
    ) -> tuple[TrainingState[P], AuxData]: ...


class InitTrainState(Protocol[P]):
    def __call__(
        self,
        key: PRNGKeyArray,
        params: P,
        electrons: Electrons,
        init_width: Width,
    ) -> TrainingState[P]: ...


@dataclass(frozen=True)
class Trainer(Generic[P, S, MS]):
    init: InitTrainState[P]
    step: VMCStepFn[P, S]
    sampling_step: SamplingStepFn[P, S]
    wave_function: ParameterizedWaveFunction[P, S, MS]
    mcmc: MCStep[P, S]
    width_scheduler: WidthScheduler
    optimizer: optax.GradientTransformation
    preconditioner: Preconditioner[P, S]


############################################################################
# Pretraining
############################################################################
class PretrainState(TrainingState[P]):
    pre_opt_state: optax.OptState

    def to_train_state(self):
        return TrainingState(
            key=self.key,
            params=self.params,
            electrons=self.electrons,
            opt_state=self.opt_state,
            width_state=self.width_state,
        )


Loss = Float[Array, ""]


class InitPretrainState(Protocol[P]):
    def __call__(
        self,
        training_state: TrainingState[P],
    ) -> PretrainState[P]: ...


class UpdatePretrainFn(Protocol[P, S_contra]):
    def __call__(self, state: PretrainState[P], static: S_contra) -> tuple[PretrainState[P], AuxData]: ...


class Pretrainer(NamedTuple, Generic[P, S]):
    init: InitPretrainState[P]
    step: UpdatePretrainFn[P, S]


############################################################################
# Logging
############################################################################


class Logger(Protocol):
    def __init__(self, config: dict) -> None: ...

    def log(self, data: dict) -> None: ...

    def log_config(self, config: dict) -> None: ...


############################################################################
# Arguments
############################################################################
class EmbeddingArgs(TypedDict):
    cutoff: float
    cutoff_1el: float
    feature_dim: int
    nuc_mlp_depth: int
    pair_mlp_widths: tuple[int, int]
    pair_n_envelopes: int


class JastrowArgs(TypedDict):
    e_e_cusps: Literal["none", "psiformer", "yukawa"]
    use_log_jastrow: bool
    use_mlp_jastrow: bool
    mlp_depth: int
    mlp_width: int


class IsotropicEnvelopeArgs(TypedDict):
    n_envelopes: int


class GLUEnvelopeArgs(TypedDict):
    n_envelopes: int
    width: int
    depth: int


class EnvelopeArgs(TypedDict):
    envelope: Literal["isotropic", "glu"]
    isotropic_args: IsotropicEnvelopeArgs
    glu_args: GLUEnvelopeArgs


class ModelArgs(TypedDict):
    embedding: EmbeddingArgs
    jastrow: JastrowArgs
    n_determinants: int
    envelopes: EnvelopeArgs


class SpringArgs(TypedDict):
    damping: float
    decay_factor: float


class SpringDenseArgs(SpringArgs):
    max_batch_size: int
    use_float64: bool


class CgArgs(TypedDict):
    damping: float
    maxiter: int


class SVDArgs(TypedDict):
    damping: float
    ema_natgrad: float
    ema_S: float
    history_length: int


class PreconditionerArgs(TypedDict):
    preconditioner: str
    spring_args: SpringArgs
    spring_dense_args: SpringDenseArgs
    cg_args: CgArgs
    svd_args: SVDArgs


class ClippingArgs(TypedDict):
    clip_local_energy: float
    stat: str


class WandBArgs(TypedDict):
    use: bool
    project: str
    entity: Optional[str]


class FileLoggingArgs(TypedDict):
    use: bool
    file_name: str


class PythonLoggingArgs(TypedDict):
    use: bool


class LoggingArgs(TypedDict):
    smoothing: int
    wandb: WandBArgs
    file: FileLoggingArgs
    python: PythonLoggingArgs
    name: str
    name_keys: Sequence[str] | None
    comment: str | None
    out_directory: str
    collection: str


class Schedule(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    HYPERBOLIC = "hyperbolic"


class TransformationArgs(TypedDict):
    name: str
    args: tuple
    kwargs: dict[str, Any]


class OptimizerArgs(TypedDict):
    lr_schedule: Schedule | str
    lr_schedule_args: dict[str, dict[str, Any]]
    transformations: Sequence[TransformationArgs]


class OptimizationArgs(TypedDict):
    steps: int
    burn_in: int
    optimizer_args: OptimizerArgs
    preconditioner_args: PreconditionerArgs
    clipping: ClippingArgs
    max_batch_size: int


class PretrainingArgs(TypedDict):
    steps: int
    optimizer_args: OptimizerArgs
    sample_from: Literal["hf", "wf"]


MCMC_proposal_type = Literal["all-electron", "single-electron"]


class MCMCArgs(TypedDict):
    proposal: MCMC_proposal_type
    steps: int
    init_width: float


class MoleculeDatabaseArgs(TypedDict):
    hash: str
    name: str
    comment: str


class MoleculeFromStrArgs(TypedDict):
    atom: str
    spin: int


class MoleculeChainArgs(TypedDict):
    element: str
    distance: float
    n: int


class MoleculeArgs(TypedDict):
    method: str
    from_str_args: MoleculeFromStrArgs
    chain_args: MoleculeChainArgs
    database_args: MoleculeDatabaseArgs
    basis: str


class StaticInput(NamedTuple):
    n_neighbours: dict
    n_deps: dict
