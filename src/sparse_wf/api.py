from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, NamedTuple, Optional, Protocol, Sequence, TypeAlias, TypedDict, TypeVar, overload
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.serialization import to_bytes, from_bytes
from jaxtyping import Array, ArrayLike, Float, Integer, PRNGKeyArray, PyTree
from pyscf.scf.hf import SCF
from typing import Literal, Callable
import logging

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

ScalingParam = Float[Array, ""]

Dependant = Integer[Array, "n_dependants"]
Dependency = Integer[Array, "n_deps"]
DependencyMap = Integer[Array, "n_center n_neighbour n_deps"]

############################################################################
# Wave function
############################################################################
# Static input


# TODO: For some reason PyTreeNode and Generics don't work together
# Seems a bit ugly to use to PyTreeNodes for some stuff and dataclasses for others, but here we go...
@struct.dataclass
class StaticInput:
    def round_with_padding(self, padding_factor, n_el, n_up, n_nuc):
        return jax.tree_map(lambda x: x * padding_factor, self)

    def to_static(self):
        return jtu.tree_map(lambda x: int(jnp.max(x)), self)


@struct.dataclass
class StaticInputs:
    mcmc: StaticInput
    mcmc_jump: StaticInput | None
    pp: StaticInput | None

    def as_dict(self):
        return {"mcmc": self.mcmc, "mcmc_jump": self.mcmc_jump, "pp": self.pp}


S = TypeVar("S")
S_contra = TypeVar("S_contra", contravariant=True)
# Parameters
P = TypeVar("P", bound=Parameters)
P_contra = TypeVar("P_contra", bound=Parameters, contravariant=True)
# Model state
MS = TypeVar("MS")


class HFOrbitalFn(Protocol):
    def __call__(self, electrons: Electrons) -> HFOrbitals: ...


P2 = TypeVar("P2", bound=Parameters)
ES = TypeVar("ES")


class Embedding(Protocol[P2, ES]):
    feature_dim: int

    @overload
    def apply(
        self,
        params: P2,
        electrons: Electrons,
        static: StaticInput,
        return_scales: Literal[False] = False,
        return_state: Literal[False] = False,
    ) -> ElectronEmb: ...

    @overload
    def apply(
        self,
        params: P2,
        electrons: Electrons,
        static: StaticInput,
        return_scales: Literal[True],
        return_state: Literal[False] = False,
    ) -> tuple[ElectronEmb, PyTree[ScalingParam]]: ...

    @overload
    def apply(
        self,
        params: P2,
        electrons: Electrons,
        static: StaticInput,
        return_scales: Literal[False],
        return_state: Literal[True],
    ) -> tuple[ElectronEmb, ES]: ...

    def apply(
        self,
        params: P2,
        electrons: Electrons,
        static: StaticInput,
        return_scales: bool = False,
        return_state: bool = False,
    ): ...
    def init(self, key: PRNGKeyArray, electrons: Electrons, static: StaticInput) -> P2: ...
    def get_static_input(
        self, electrons: Electrons, electrons_new: Optional[Electrons] = None, idx_changed: Optional[ElectronIdx] = None
    ) -> StaticInput: ...
    def low_rank_update(
        self, params: P2, electrons: Electrons, changed_electrons: ElectronIdx, static: StaticInput, state: ES
    ) -> tuple[ElectronEmb, ElectronIdx, ES]: ...
    def apply_with_fwd_lap(
        self, params: P2, electrons: Electrons, static: StaticInput
    ) -> tuple[ElectronEmb, Dependency]: ...


class ParameterizedWaveFunction(Protocol[P, MS]):
    R: Nuclei
    Z: Charges
    n_up: int

    def init(self, key: PRNGKeyArray, electrons: Electrons) -> P: ...
    def get_static_input(
        self, electrons: Electrons, electrons_new: Optional[Electrons] = None, idx_changed: Optional[ElectronIdx] = None
    ) -> StaticInput: ...
    def orbitals(self, params: P, electrons: Electrons, static: StaticInput) -> SlaterMatrices: ...
    def hf_transformation(self, hf_orbitals: HFOrbitals) -> SlaterMatrices: ...
    def kinetic_energy(self, params: P, electrons: Electrons, static: StaticInput) -> LocalEnergy: ...
    def kinetic_energy_dense(self, params: P, electrons: Electrons, static: StaticInput) -> LocalEnergy: ...
    def signed(self, params: P, electrons: Electrons, static: StaticInput) -> SignedLogAmplitude: ...
    def __call__(self, params: P, electrons: Electrons, static: StaticInput) -> LogAmplitude: ...
    def log_psi_with_state(
        self, params: P, electrons: Electrons, static: StaticInput
    ) -> tuple[SignedLogAmplitude, MS]: ...
    def log_psi_low_rank_update(
        self, params: P, electrons: Electrons, changed_electrons: ElectronIdx, static: StaticInput, state: MS
    ) -> tuple[SignedLogAmplitude, MS]: ...


############################################################################
# MCMC
############################################################################


class MCMCStats(NamedTuple):
    pmove: jax.Array
    static_max: dict[str, StaticInput]
    logs: dict[str, jax.Array]


class ClosedLogLikelihood(Protocol):
    def __call__(self, electrons: Electrons) -> LogAmplitude: ...


# TODO: MyPy does not like the dict[str, S_contra] type hint here
class MCStep(Protocol[P_contra]):  # type: ignore
    def __call__(
        self,
        key: PRNGKeyArray,
        params: P_contra,
        electrons: Electrons,
        statics: StaticInputs,
        width: Width,
    ) -> tuple[Electrons, MCMCStats]: ...


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
    damping: Float[Array, ""]


class PreconditionerInit(Protocol[P]):
    def __call__(self, params: P) -> PreconditionerState[P]: ...


class ApplyPreconditioner(Protocol[P]):
    def __call__(
        self,
        params: P,
        electrons: Electrons,
        static: StaticInput,
        dE_dlogpsi: EnergyCotangent,
        aux_grad: P,
        natgrad_state: PreconditionerState[P],
    ) -> tuple[P, PreconditionerState[P], AuxData]: ...


class Preconditioner(NamedTuple, Generic[P]):
    init: PreconditionerInit[P]
    precondition: ApplyPreconditioner[P]


############################################################################
# Spin Operator
############################################################################
SS = TypeVar("SS")  # spin state
SpinValue = Float[Array, ""]


class SpinOperator(Protocol[P, SS]):
    def init_state(self) -> SS: ...
    def __call__(self, params: P, electrons: Electrons, static: StaticInput, state: SS) -> tuple[SpinValue, P, SS]: ...


############################################################################
# Optimizer
############################################################################
class OptState(NamedTuple, Generic[P]):
    opt: optax.OptState
    natgrad: PreconditionerState[P]


class TrainingState(Generic[P, SS], struct.PyTreeNode):  # the order of inheritance is important here!
    key: PRNGKeyArray
    params: P
    electrons: Electrons
    opt_state: OptState[P]
    width_state: WidthSchedulerState
    spin_state: SS
    step: Int

    def serialize(self):
        from sparse_wf.jax_utils import instance, pmap, pgather

        @pmap
        def gather_electrons(electrons):
            return pgather(electrons, axis=0, tiled=True)

        result = self.replace(electrons=gather_electrons(self.electrons))  # include electrons from all devices
        result = instance(result)  # only return a single copy of parameters, opt_state, etc.
        return to_bytes(result)

    def deserialize(self, data: bytes, batch_size=None):
        from sparse_wf.jax_utils import replicate

        state_with_all_electrons = from_bytes(self, data)
        # Distribute electrons to devices
        electrons = state_with_all_electrons.electrons
        if batch_size is not None:
            loaded_batch_size = electrons.shape[0]
            if batch_size < loaded_batch_size:
                logging.warning(
                    f"Batch size {batch_size} is smaller than the original batch size {loaded_batch_size}. Cropping."
                )
                electrons = electrons[:batch_size]
            elif batch_size > loaded_batch_size:
                logging.warning(
                    f"Batch size {batch_size} is larger than the original batch size {loaded_batch_size}. Using loaded batch-size."
                )

        electrons = electrons.reshape(jax.process_count(), jax.local_device_count(), -1, *electrons.shape[1:])
        electrons = electrons[jax.process_index()]
        # We have to create new keys for all devices
        key = jax.random.split(state_with_all_electrons.key, (jax.process_count(), jax.local_device_count()))
        key = key[jax.process_index()]
        # Replicate the state to all devices
        result = replicate(state_with_all_electrons)
        result = result.replace(key=key, electrons=electrons)
        return result


class VMCStepFn(Protocol[P, SS]):
    def __call__(
        self,
        state: TrainingState[P, SS],
        statics: StaticInputs,
    ) -> tuple[TrainingState[P, SS], LocalEnergy, AuxData, MCMCStats]: ...


class SamplingStepFn(Protocol[P, SS]):
    def __call__(
        self, state: TrainingState[P, SS], static: StaticInput, compute_energy: bool, overlap_fn: Callable | None
    ) -> tuple[TrainingState[P, SS], AuxData, MCMCStats]: ...


class InitTrainState(Protocol[P, SS]):
    def __call__(
        self,
        key: PRNGKeyArray,
        params: P,
        electrons: Electrons,
        init_width: Width,
    ) -> TrainingState[P, SS]: ...


@dataclass(frozen=True)
class Trainer(Generic[P, SS]):
    init: InitTrainState[P, SS]
    step: VMCStepFn[P, SS]
    sampling_step: SamplingStepFn[P, SS]
    wave_function: ParameterizedWaveFunction[P, MS]
    mcmc: MCStep[P]
    width_scheduler: WidthScheduler
    optimizer: optax.GradientTransformation
    preconditioner: Preconditioner[P]
    spin_operator: SpinOperator[P, SS]


############################################################################
# Pretraining
############################################################################
class PretrainState(TrainingState[P, SS]):
    pre_opt_state: optax.OptState

    def to_train_state(self):
        return TrainingState(
            key=self.key,
            params=self.params,
            electrons=self.electrons,
            opt_state=self.opt_state,
            width_state=self.width_state,
            spin_state=self.spin_state,
            step=self.step,
        )


Loss = Float[Array, ""]


class InitPretrainState(Protocol[P, SS]):
    def __call__(
        self,
        training_state: TrainingState[P, SS],
    ) -> PretrainState[P, SS]: ...


class UpdatePretrainFn(Protocol[P, SS]):
    def __call__(
        self, state: PretrainState[P, SS], statics: StaticInputs
    ) -> tuple[PretrainState[P, SS], AuxData, MCMCStats]: ...


class Pretrainer(NamedTuple, Generic[P, SS]):
    init: InitPretrainState[P, SS]
    step: UpdatePretrainFn[P, SS]


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
class NewEmbeddingArgs(TypedDict):
    cutoff: float
    cutoff_1el: float
    feature_dim: int
    pair_mlp_widths: tuple[int, int]
    pair_n_envelopes: int
    low_rank_buffer: int
    n_updates: int


class MoonEmbeddingArgs(TypedDict):
    cutoff: float
    cutoff_1el: float
    feature_dim: int
    nuc_mlp_depth: int
    pair_mlp_widths: tuple[int, int]
    pair_n_envelopes: int
    low_rank_buffer: int


class EmbeddingArgs(TypedDict):
    embedding: Literal["moon", "new", "new_sparse"]
    moon: MoonEmbeddingArgs
    new: NewEmbeddingArgs


class JastrowArgs(TypedDict):
    e_e_cusps: Literal["none", "psiformer", "yukawa"]
    use_e_e_mlp: bool
    use_log_jastrow: bool
    use_mlp_jastrow: bool
    mlp_depth: int
    mlp_width: int
    use_attention: bool
    attention_heads: int
    attention_dim: int


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
    spin_restricted: bool
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
    checkpoint_every: int
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


class SpinOperatorArgs(TypedDict):
    operator: str
    grad_scale: float


class OptimizationArgs(TypedDict):
    steps: int
    burn_in: int
    optimizer_args: OptimizerArgs
    preconditioner_args: PreconditionerArgs
    clipping: ClippingArgs
    max_batch_size: int
    spin_operator_args: SpinOperatorArgs
    energy_operator: Literal["dense", "sparse"]
    pp_grid_points: int  # number of spherical grid points for pseudopotential integration


class CASArgs(TypedDict):
    active_orbitals: int
    active_electrons: int
    det_threshold: float
    s2: float


class HFArgs(TypedDict):
    x2c: bool
    newton: bool
    smearing: float
    restricted: bool
    restart: bool
    cache_dir: str
    antiferromagnetic_broken_symmetry: bool  # This initializes the spin opposing for two irons.


class PretrainingArgs(TypedDict):
    steps: int
    optimizer_args: OptimizerArgs
    sample_from: Literal["hf", "wf"]
    reference: Literal["hf", "cas"]
    cas: CASArgs
    hf: HFArgs


class EvaluationArgs(TypedDict):
    steps: int
    compute_energy: bool
    compute_overlaps: bool
    overlap_states: list[list[int]]


MCMC_proposal_type = Literal["all-electron", "single-electron", "cluster-update"]


class MCMCProposalArg(TypedDict):
    init_width: int


class MCMCAllElectronArgs(MCMCProposalArg):
    steps: int


class MCMCSingleElectronArgs(MCMCProposalArg):
    sweeps: int


class MCMCClusterUpdateArgs(MCMCProposalArg):
    sweeps: float
    sweep_type: Literal["random", "sequential"]
    cluster_radius: float


class MCMCArgs(TypedDict):
    proposal: MCMC_proposal_type
    acceptance_target: float
    all_electron_args: MCMCAllElectronArgs
    single_electron_args: MCMCSingleElectronArgs
    cluster_update_args: MCMCClusterUpdateArgs
    jump_steps: int


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
    pseudopotentials: Sequence[str]


class MCMCStaticArgs(NamedTuple):
    max_cluster_size: int
