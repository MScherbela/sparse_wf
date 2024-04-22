from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple, Protocol, Sequence, TypeAlias, Callable, TypedDict, Optional

import numpy as np
import optax
from flax import struct
from jaxtyping import Array, ArrayLike, Float, Integer, PRNGKeyArray, PyTree
from pyscf.scf.hf import SCF
from folx.api import FwdLaplArray

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
Gradients = Parameters

Width = Float[Array, ""]
PMove = Float[Array, ""]
PMoves = Float[PMove, "window_size"]

Damping = Float[Array, ""]
AuxData = PyTree[Float[Array, ""]]
Step = Integer[ArrayLike, ""]


############################################################################
# Wave function
############################################################################
ElectronElectronEdges = Integer[Array, "n_electrons n_nb_ee"]
ElectronNucleiEdges = Integer[Array, "n_electrons n_nb_en"]
NucleiElectronEdges = Integer[Array, "n_nuclei n_nb_ne"]
DistanceMatrix = Float[Array, "*batch_dims n1 n2"]


class NrOfNeighbours(NamedTuple):
    ee: int
    en: int
    ne: int


class NeighbourIndices(NamedTuple):
    ee: ElectronElectronEdges
    en: ElectronNucleiEdges
    ne: NucleiElectronEdges


NrOfDependencies = NamedTuple


class StaticInput(NamedTuple):
    n_neighbours: NrOfNeighbours
    n_deps: NrOfDependencies


Dependency = Integer[Array, "n_deps"]
Dependencies = Integer[Dependency, "*batch_dims"]
DependencyMap = Integer[Array, "*batch_dims n_center n_neighbour n_deps"]


class DynamicInput(NamedTuple):
    electrons: Electrons
    neighbours: NeighbourIndices


class DynamicInputWithDependencies(NamedTuple):
    electrons: Electrons
    neighbours: NeighbourIndices
    dependencies: tuple[Dependencies, ...]
    dep_maps: tuple[DependencyMap, ...]


class InputConstructor(Protocol):
    def get_static_input(self, electrons: Electrons) -> StaticInput: ...

    def get_dynamic_input(self, electrons: Electrons, static: StaticInput) -> DynamicInput: ...

    def get_dynamic_input_with_dependencies(
        self, electrons: Electrons, static: StaticInput
    ) -> DynamicInputWithDependencies: ...


class ClosedInputConstructor(Protocol):
    def __call__(self, electrons: Electrons) -> DynamicInput: ...


class Jastrow(Protocol):
    def __call__(self, electrons: Electrons, static: StaticInput) -> LogAmplitude: ...


class OrbitalFn(Protocol):
    def __call__(self, electrons: Electrons, static: StaticInput) -> SlaterMatrices: ...


class HFOrbitalFn(Protocol):
    def __call__(self, electrons: Electrons) -> HFOrbitals: ...


class HFOrbitalsToNNOrbitals(Protocol):
    def __call__(self, hf_orbitals: HFOrbitals) -> SlaterMatrices: ...


class OrbitalModel(Protocol):
    __call__: OrbitalFn
    transform_hf_orbitals: HFOrbitalsToNNOrbitals


class SLogPsi(Protocol):
    def __call__(self, electrons: Electrons, static: StaticInput) -> SignedLogAmplitude: ...


class LogPsi(Protocol):
    def __call__(self, electrons: Electrons, static: StaticInput) -> LogAmplitude: ...


class ParameterizedOrbitalFunction(Protocol):
    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput) -> SlaterMatrices: ...


class ParameterizedSLogPsi(Protocol):
    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput) -> SignedLogAmplitude: ...


class ParameterizedLogPsi(Protocol):
    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput) -> LogAmplitude: ...


class ParameterizedLocalEnergy(Protocol):
    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput) -> FwdLaplArray: ...


class ParameterizedWaveFunction(Protocol):
    init: Callable[[PRNGKeyArray], Parameters]
    input_constructor: InputConstructor
    orbitals: ParameterizedOrbitalFunction
    hf_transformation: HFOrbitalsToNNOrbitals
    signed: ParameterizedSLogPsi
    __call__: ParameterizedLogPsi
    local_energy: ParameterizedLocalEnergy


class EnergyFn(Protocol):
    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput) -> LocalEnergy: ...


############################################################################
# MCMC
############################################################################
class ClosedLogLikelihood(Protocol):
    def __call__(self, electrons: Electrons) -> LogAmplitude: ...


class MCStep(Protocol):
    def __call__(
        self, key: PRNGKeyArray, params: Parameters, electrons: Electrons, static: StaticInput, width: Width
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
class PreconditionerState(NamedTuple):
    last_grad: Parameters


class PreconditionerInit(Protocol):
    def __call__(self, params: Parameters) -> PreconditionerState: ...


class ApplyPreconditioner(Protocol):
    def __call__(
        self,
        params: Parameters,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        natgrad_state: PreconditionerState,
    ) -> tuple[Gradients, PreconditionerState, AuxData]: ...


class Preconditioner(NamedTuple):
    init: PreconditionerInit
    precondition: ApplyPreconditioner


############################################################################
# Optimizer
############################################################################
class OptState(NamedTuple):
    opt: optax.OptState
    natgrad: PreconditionerState


class TrainingState(struct.PyTreeNode):
    key: PRNGKeyArray
    params: Parameters
    electrons: Electrons
    opt_state: OptState
    width_state: WidthSchedulerState


class VMCStepFn(Protocol):
    def __call__(
        self,
        state: TrainingState,
        static: StaticInput,
    ) -> tuple[TrainingState, LocalEnergy, AuxData]: ...


class InitTrainState(Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        params: Parameters,
        electrons: Electrons,
        init_width: Width,
    ) -> TrainingState: ...


@dataclass(frozen=True)
class Trainer:
    init: InitTrainState
    step: VMCStepFn
    wave_function: ParameterizedWaveFunction
    mcmc: MCStep
    width_scheduler: WidthScheduler
    energy_fn: EnergyFn
    optimizer: optax.GradientTransformation
    preconditioner: Preconditioner


############################################################################
# Pretraining
############################################################################
class PretrainState(TrainingState):
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


class InitPretrainState(Protocol):
    def __call__(
        self,
        training_state: TrainingState,
    ) -> PretrainState: ...


class UpdatePretrainFn(Protocol):
    def __call__(self, state: PretrainState, static: StaticInput) -> tuple[PretrainState, AuxData]: ...


class Pretrainer(NamedTuple):
    init: InitPretrainState
    step: UpdatePretrainFn


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
