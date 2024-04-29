from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, NamedTuple, Optional, Protocol, Sequence, TypeAlias, TypedDict, TypeVar

import numpy as np
import optax
from flax import struct
from jaxtyping import Array, ArrayLike, Float, Integer, PRNGKeyArray, PyTree
from pyscf.scf.hf import SCF

AnyArray = Array | list | np.ndarray
Int = Integer[Array, ""]


Position = Float[Array, "spatial=3"] | Float[np.ndarray, "spatial=3"] | tuple[float, float, float] | list[float]
Electrons = Float[Array, "*batch_dims n_electrons spatial=3"]
Spins = Integer[Array, "*batch_dims n_electrons"]
Nuclei = Float[np.ndarray, "n_nuclei spatial=3"]
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


############################################################################
# Wave function
############################################################################
# Static input
S = TypeVar("S")
S_contra = TypeVar("S_contra", contravariant=True)
# Parameters
P = TypeVar("P", bound=Parameters)
P_contra = TypeVar("P_contra", bound=Parameters, contravariant=True)


class HFOrbitalFn(Protocol):
    def __call__(self, electrons: Electrons) -> HFOrbitals: ...


class ParameterizedWaveFunction(Protocol[P, S]):
    def init(self, key: PRNGKeyArray) -> P: ...
    def get_static_input(self, electrons: Electrons) -> S: ...
    def orbitals(self, params: P, electrons: Electrons, static: S) -> SlaterMatrices: ...
    def hf_transformation(self, hf_orbitals: HFOrbitals) -> SlaterMatrices: ...
    def local_energy(self, params: P, electrons: Electrons, static: S) -> LocalEnergy: ...
    def signed(self, params: P, electrons: Electrons, static: S) -> SignedLogAmplitude: ...
    def __call__(self, params: P, electrons: Electrons, static: S) -> LogAmplitude: ...


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


class VMCStepFn(Protocol[P, S_contra]):
    def __call__(
        self,
        state: TrainingState[P],
        static: S_contra,
    ) -> tuple[TrainingState[P], LocalEnergy, AuxData]: ...


class InitTrainState(Protocol[P]):
    def __call__(
        self,
        key: PRNGKeyArray,
        params: P,
        electrons: Electrons,
        init_width: Width,
    ) -> TrainingState[P]: ...


@dataclass(frozen=True)
class Trainer(Generic[P, S]):
    init: InitTrainState[P]
    step: VMCStepFn[P, S]
    wave_function: ParameterizedWaveFunction[P, S]
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


class ModelArgs(TypedDict):
    cutoff: float
    feature_dim: int
    nuc_mlp_depth: int
    pair_mlp_widths: Sequence[int]
    pair_n_envelopes: int
    n_determinants: int


class SpringArgs(TypedDict):
    damping: float
    decay_factor: float


class CgArgs(TypedDict):
    damping: float
    maxiter: int


class PreconditionerArgs(TypedDict):
    preconditioner: str
    spring_args: SpringArgs
    cg_args: CgArgs


class ClippingArgs(TypedDict):
    clip_local_energy: float
    stat: str


class WandBArgs(TypedDict):
    use: bool
    project: str
    entity: Optional[str]


class FileLoggingArgs(TypedDict):
    use: bool
    path: str


class LoggingArgs(TypedDict):
    smoothing: int
    wandb: WandBArgs
    file: FileLoggingArgs


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
    optimizer_args: OptimizerArgs
    preconditioner_args: PreconditionerArgs
    clipping: ClippingArgs


class PretrainingArgs(TypedDict):
    steps: int
    optimizer_args: OptimizerArgs
