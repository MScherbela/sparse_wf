from dataclasses import dataclass
from typing import NamedTuple, Protocol, TypeAlias, Optional

import numpy as np
import optax
from flax import struct
from jaxtyping import Array, ArrayLike, Float, Integer, PRNGKeyArray, PyTree
from pyscf.scf.hf import SCF
from folx.api import FwdLaplArray

AnyArray = Array | list | np.ndarray
Int = Integer[Array, ""]

Electrons = Float[Array, "*batch_dims n_electrons spatial=3"]

Position = Float[Array, "spatial=3"] | Float[np.ndarray, "spatial=3"] | tuple[float, float, float] | list[float]
Nuclei = Float[Array, "n_nuclei spatial=3"]
Charges = Integer[Array, "n_nuclei"]
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

    def get_dynamic_input_with_dependencies(self, electrons: Electrons, static: StaticInput) -> DynamicInputWithDependencies: ...


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

class ParameterizedLogPsiWithFwdLap(Protocol):
    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput) -> Fwd

class ParameterizedWaveFunction(Protocol):
    construct_input: InputConstructor
    orbitals: ParameterizedOrbitalFunction
    hf_transformation: HFOrbitalsToNNOrbitals
    signed: ParameterizedSLogPsi
    __call__: ParameterizedLogPsi
    fwd_lap: ParameterizedLogPsiWithFwdLap


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
class NaturalGradientState(NamedTuple):
    last_grad: Parameters


class NaturalGradientInit(Protocol):
    def __call__(self, params: Parameters) -> NaturalGradientState: ...


class NaturalGradientPreconditioner(Protocol):
    def __call__(
        self,
        params: Parameters,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        natgrad_state: NaturalGradientState,
    ) -> tuple[Gradients, NaturalGradientState, AuxData]: ...


class NaturalGradient(NamedTuple):
    init: NaturalGradientInit
    precondition: NaturalGradientPreconditioner


############################################################################
# Optimizer
############################################################################
class NaturalGradientOptState(NamedTuple):
    opt: optax.OptState
    natgrad: NaturalGradientState | None


@struct.dataclass
class TrainingState:
    params: Parameters
    electrons: Electrons
    opt_state: NaturalGradientOptState
    width_state: WidthSchedulerState


class UpdateFn(Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
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
    update: UpdateFn
    wave_function: ParameterizedWaveFunction
    mcmc: MCStep
    width_scheduler: WidthScheduler
    energy_fn: EnergyFn
    optimizer: optax.GradientTransformation
    natgrad: NaturalGradient | None


############################################################################
# Pretraining
############################################################################
@struct.dataclass
class PretrainState(TrainingState):
    pre_opt_state: optax.OptState


Loss = Float[Array, ""]


class InitPretrainState(Protocol):
    def __call__(
        self,
        training_state: TrainingState,
    ) -> PretrainState: ...


class UpdatePretrainFn(Protocol):
    def __call__(
        self, key: PRNGKeyArray, state: PretrainState, static: StaticInput
    ) -> tuple[PretrainState, AuxData]: ...


class Pretrainer(NamedTuple):
    init: InitPretrainState
    step: UpdatePretrainFn
