import functools
from typing import NamedTuple, Optional, Sequence, cast

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pyscf
from flax.struct import PyTreeNode, field
from folx.api import FwdLaplArray
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from sparse_wf.api import (
    Charges,
    Electrons,
    Int,
    Nuclei,
    ParameterizedWaveFunction,
    Parameters,
    SlaterMatrices,
    Spins,
    SignedLogAmplitude,
    JastrowArgs,
)
from sparse_wf.hamiltonian import make_local_energy, potential_energy
from sparse_wf.jax_utils import fwd_lap, jit, vectorize
from sparse_wf.model.graph_utils import (
    NO_NEIGHBOUR,
    Dependency,
    DependencyMap,
    DistanceMatrix,
    NeighbourIndices,
    NrOfNeighbours,
    densify_jacobian_by_zero_padding,
    get_dependency_map,
    get_full_distance_matrices,
    get_neighbour_indices,
    get_nr_of_neighbours,
    get_with_fill,
    merge_dependencies,
    pad_jacobian_to_output_deps,
    round_to_next_step,
    slogdet_with_sparse_fwd_lap,
)
from sparse_wf.model.utils import (
    MLP,
    Embedding,
    FilterKernel,
    IsotropicEnvelope,
    contract,
    cutoff_function,
    hf_orbitals_to_fulldet_orbitals,
    signed_logpsi_from_orbitals,
    swap_bottom_blocks,
    get_dist_same_diff,
)


class DependenciesMoon(NamedTuple):
    h_el_initial: Dependency
    H_nuc: Dependency
    h_el_out: Dependency


class NrOfDependencies(NamedTuple):
    h_el_initial: int
    H_nuc: int
    h_el_out: int


class DependencyMaps(NamedTuple):
    h0_to_Hnuc: DependencyMap
    Gamma_ne_to_Hnuc: DependencyMap
    Hnuc_to_hout: DependencyMap
    h0_to_hout: DependencyMap


class StaticInput(NamedTuple):
    n_neighbours: NrOfNeighbours
    n_deps: NrOfDependencies


@jit
def _get_max_nr_of_dependencies(dist_ee: DistanceMatrix, dist_ne: DistanceMatrix, cutoff: float):
    # Thest first electron message passing step can depend at most on electrons within 1 * cutoff
    n_deps_max_h0 = jnp.max(jnp.sum(dist_ee < cutoff, axis=-1))

    # The nuclear embeddings are computed with 2 message passing steps and can therefore depend at most on electrons within 2 * cutoff
    n_deps_max_H = jnp.max(jnp.sum(dist_ne < cutoff * 2, axis=-1))

    # The output electron embeddings are computed with 3 message passing step and can therefore depend at most on electrons within 3 * cutoff
    n_deps_max_h_out = jnp.max(jnp.sum(dist_ee < cutoff * 3, axis=-1))
    return n_deps_max_h0, n_deps_max_H, n_deps_max_h_out


@jit(static_argnames=("n_deps_max",))
def get_all_dependencies(idx_nb: NeighbourIndices, n_deps_max: NrOfDependencies):
    """Get the indices of electrons on which each embedding will depend on.

    Args:
        idx_nb: NeighbourIndices, named tuple containing the indices of the neighbours of each electron and nucleus.
        n_deps_max: maximum_nr_of electrons that each embedding can depend on.
            - n_deps_max[0]: maximum number of dependencies for the electron embeddings at the first step.
            - n_deps_max[1]: maximum number of dependencies for the nuclear embeddings.
            - n_deps_max[2]: maximum number of dependencies for the output electron embeddings.

    Returns:
        deps: tuple of jnp.ndarray, dependencies for the electron embeddings at each step.
            deps_h0: [n_el  x nr_of_deps_level_1]
            deps_H:  [n_nuc x nr_of deps_level_2]
            deps_hout: [n_el x nr_of_deps_level_3]
        dep_maps: tuple of jnp.ndarray, maps the dependencies between the levels:
            h0_to_Hnuc: [n_nuc x n_neighbouring_el x nr_of_deps_level_1]; values are in [0 ... deps_level_2]
            Gamma_ne_to_Hnuc: [n_nuc x n_neighbouring_el x 1]; values are in [0 ... deps_level_2]
            Hnuc_to_hout: [n_el x n_neighbouring_nuc x nr_of_deps_level_2]; values are in [0 ... deps_level_3]
            h0_to_hout: [n_el x n_neighbouring_el x nr_of_deps_level_1]; values are in [0 ... deps_level_3]
    """
    n_el = idx_nb.ee.shape[-2]
    batch_dims = idx_nb.ee.shape[:-2]
    self_dependency = jnp.arange(n_el)[:, None]
    self_dependency = jnp.tile(self_dependency, batch_dims + (1, 1))

    @functools.partial(jnp.vectorize, signature="(center1,deps),(center2,neigbours)->(center2,neigbours,deps)")
    def get_deps_nb(deps, idx_nb):
        return get_with_fill(deps, idx_nb, NO_NEIGHBOUR)

    get_dep_map_for_all_centers = jax.vmap(jax.vmap(get_dependency_map, in_axes=(0, None)))

    # Step 1: Initial electron embeddings depend on themselves and their neighbours
    deps_h0: Dependency = jnp.concatenate([self_dependency, idx_nb.ee], axis=-1)

    # Step 2: Nuclear embeddings depend on all dependencies of their neighbouring electrons
    deps_neighbours = get_deps_nb(deps_h0, idx_nb.ne)
    deps_H = merge_dependencies(deps_neighbours, idx_nb.ne, None, n_deps_max.H_nuc)
    dep_map_h0_to_H = get_dep_map_for_all_centers(deps_neighbours, deps_H)
    dep_map_Gamma_ne_to_H = get_dep_map_for_all_centers(idx_nb.ne[..., None], deps_H)

    # Step 3: Output electron embeddings depend on themselves, their neighbouring electrons and all dependencies of their neighbouring nuclei
    deps_neighbours = get_deps_nb(deps_H, idx_nb.en)
    deps_hout = merge_dependencies(deps_neighbours, deps_h0, jnp.arange(n_el)[:, None], n_deps_max.h_el_out)
    dep_map_H_to_hout = get_dep_map_for_all_centers(deps_neighbours, deps_hout)
    dep_map_h0_to_hout = jax.vmap(get_dependency_map)(deps_h0, deps_hout)

    return DependenciesMoon(deps_h0, deps_H, deps_hout), DependencyMaps(
        dep_map_h0_to_H, dep_map_Gamma_ne_to_H, dep_map_H_to_hout, dep_map_h0_to_hout
    )


def get_diff_features(
        r: Float[Array, "dim=3"],
        r_nb: Float[Array, "*neighbours dim=3"],
        s: Optional[Int] = None,
        s_nb: Optional[Integer[Array, " *neighbours"]] = None,
):
    diff = r - r_nb
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    features = [dist, diff]
    if (s is not None) and (s_nb is not None):
        # s_prod = s * s_nb
        # features.append(s_prod[..., None])
        n_neighbours = r_nb.shape[-2]
        s_tiled = jnp.tile(s[None], (n_neighbours,))  # => [n_neighbours]
        features += [s_tiled[:, None], s_nb[:, None]]  # => [n_neighbours x 1 (feature)]
    return jnp.concatenate(features, axis=-1)


class PairwiseFilter(nn.Module):
    cutoff: float
    directional_mlp_widths: Sequence[int]
    n_envelopes: int
    out_dim: int

    def scale_initializer(self, rng, shape, dtype):
        assert len(shape) == 1
        n_scales = shape[0]
        scale = jnp.linspace(0, self.cutoff, n_scales, dtype=dtype)
        scale *= 1 + 0.1 * jax.random.normal(rng, shape, dtype)
        return scale

    @nn.compact
    def __call__(self, dist_diff: Float[Array, "*batch_dims features_in"]) -> Float[Array, "*batch_dims features_out"]:
        """Compute the pairwise filters between two particles.

        Args:
            dist_diff: The distance, 3D difference and optional spin difference between two particles [n_el x n_nb x 4(5)].
                The 0th feature dimension must contain the distance, the remaining dimensions can contain arbitrary
                features that are used to compute the pairwise filters, e.g. product of spins.
        """
        # Direction- (and spin-) dependent MLP
        directional_features = MLP(widths=self.directional_mlp_widths)(dist_diff)

        # Distance-dependenet radial filters
        dist = dist_diff[..., 0]
        scales = self.param("scales", self.scale_initializer, (self.n_envelopes,), jnp.float32)
        scales = jax.nn.softplus(scales)
        envelopes = jnp.exp(-((dist[..., None] / scales) ** 2))
        envelopes *= cutoff_function(dist / self.cutoff)[..., None]
        envelopes = nn.Dense(directional_features.shape[-1], use_bias=False)(envelopes)
        beta = directional_features * envelopes
        return nn.Dense(self.out_dim, use_bias=False)(beta)


def _vmap_over_centers_and_neighbours(f):
    f_out = jax.vmap(f, in_axes=(None, None, 0), out_axes=-2)  # vmap over neighbours
    f_out = jax.vmap(f_out, in_axes=(None, 0, 0), out_axes=-3)  # vmap over centers
    return f_out


def get_neighbour_coordinates(electrons: Electrons, R: Nuclei, idx_nb: NeighbourIndices, spins: Spins):
    # [n_el  x n_neighbouring_electrons] - spin of each adjacent electron for each electron
    spin_nb_ee = get_with_fill(spins, idx_nb.ee, 0.0)
    # [n_el  x n_neighbouring_electrons x 3] - position of each adjacent electron for each electron
    r_nb_ee = get_with_fill(electrons, idx_nb.ee, NO_NEIGHBOUR)
    # [n_nuc x n_neighbouring_electrons x 3] - position of each adjacent electron for each nucleus
    r_nb_ne = get_with_fill(electrons, idx_nb.ne, NO_NEIGHBOUR)
    # [n_el  x n_neighbouring_nuclei    x 3] - position of each adjacent nuclei for each electron
    R_nb_en = get_with_fill(R, idx_nb.en, NO_NEIGHBOUR)
    return spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en


class ElElCusp(nn.Module):
    n_up: int

    @nn.compact
    def __call__(self, electrons: Electrons) -> Float[Array, " *batch_dims"]:
        dist_same, dist_diff = get_dist_same_diff(electrons, self.n_up)

        alpha_same = self.param("alpha_same", nn.initializers.ones, (), jnp.float32)
        alpha_diff = self.param("alpha_diff", nn.initializers.ones, (), jnp.float32)
        factor_same = self.param("factor_same", nn.initializers.constant(-0.25), (), jnp.float32)
        factor_diff = self.param("factor_diff", nn.initializers.constant(-0.5), (), jnp.float32)

        cusp_same = jnp.sum(alpha_same ** 2 / (alpha_same + dist_same), axis=-1)
        cusp_diff = jnp.sum(alpha_diff ** 2 / (alpha_diff + dist_diff), axis=-1)

        return factor_same * cusp_same + factor_diff * cusp_diff


class JastrowFactor(nn.Module):
    embedding_n_hidden: Sequence[int]
    soe_n_hidden: Sequence[int]
    init_with_zero: bool = False

    @nn.compact
    def __call__(self, embeddings):
        """
        There are three options here (i is the electron index):
        (1) J_i = MLP_[embedding_n_hidden, 1](embeddings) , J=sum(J_i)
        (2) J_i = MLP_[embedding_n_hidden](embeddings), J=MLP_[soe_n_hidden, 1](sum(J_i))
        (3) J=MLP_[soe_n_hidden, 1](sum(embeddings_i))
        """
        if self.embedding_n_hidden is None and self.soe_n_hidden is None:
            raise KeyError("Either embedding_n_hidden or soe_n_hidden must be specified when using mlp jastrow.")

        if self.embedding_n_hidden is not None:
            if self.soe_n_hidden is None:  # Option (1)
                jastrow = jnp.squeeze(
                    MLP(self.embedding_n_hidden + (1,), activate_final=False, residual=False, output_bias=False)(
                        embeddings),
                    axis=-1)
                jastrow = jnp.sum(jastrow, axis=-1)
            else:  # Option (2) part 1
                jastrow = MLP(self.embedding_n_hidden, activate_final=False, residual=False, output_bias=False)(embeddings)
        else:  # Option (3) part 2
            jastrow = embeddings

        if self.soe_n_hidden is not None:  # Option (2 or 3)
            jastrow = jnp.sum(jastrow, axis=-2)  # Sum over electrons.
            jastrow = jnp.squeeze(
                MLP(self.soe_n_hidden + (1,), activate_final=False, residual=False, output_bias=False)(jastrow), axis=-1)

        return jastrow


class MoonParams(PyTreeNode):
    ee_filter: Parameters
    ne_filter: Parameters
    en_filter: Parameters
    mlp_nuc: Parameters
    lin_h0: Parameters
    lin_orbitals: Parameters
    envelopes: Parameters
    el_el_cusp: Optional[Parameters]
    mlp_jastrow: Optional[Parameters]
    log_jastrow: Optional[Parameters]


class SparseMoonWavefunction(PyTreeNode, ParameterizedWaveFunction[MoonParams, StaticInput]):
    R: Nuclei
    Z: Charges
    n_electrons: int = field(pytree_node=False)
    n_up: int = field(pytree_node=False)
    n_determinants: int = field(pytree_node=False)
    cutoff: float
    ee_filter: PairwiseFilter
    ne_filter: PairwiseFilter
    en_filter: PairwiseFilter
    mlp_nuc: MLP
    lin_h0: nn.Dense
    lin_orbitals: nn.Dense
    envelopes: IsotropicEnvelope
    el_el_cusp: ElElCusp
    use_el_el_cusp: bool
    use_mlp_jastrow: bool
    mlp_jastrow: JastrowFactor
    use_log_jastrow: bool
    log_jastrow: JastrowFactor

    @property
    def n_dn(self):
        return self.n_electrons - self.n_up

    @property
    def spins(self):
        dtype = self.R.dtype
        return jnp.concatenate([jnp.ones(self.n_up, dtype), -jnp.ones(self.n_dn, dtype)])

    @classmethod
    def create(
            cls,
            mol: pyscf.gto.Mole,
            n_determinants: int,
            n_envelopes: int,
            cutoff: float,
            feature_dim: int,
            nuc_mlp_depth: int,
            pair_mlp_widths: Sequence[int],
            pair_n_envelopes: int,
            use_el_el_cusp: bool,
            mlp_jastrow: JastrowArgs,
            log_jastrow: JastrowArgs,
    ):
        n_up, n_dn = mol.nelec
        n_el = n_up + n_dn
        R = jnp.array(mol.atom_coords(), dtype=jnp.float32)
        Z = mol.atom_charges()
        return cls(
            R=R,
            Z=Z,
            n_electrons=int(n_el),
            n_up=int(n_up),
            n_determinants=n_determinants,
            cutoff=cutoff,
            ee_filter=PairwiseFilter(
                cutoff,
                pair_mlp_widths,
                pair_n_envelopes,
                feature_dim,
                name="Gamma_ee",
            ),
            ne_filter=PairwiseFilter(
                cutoff,
                pair_mlp_widths,
                pair_n_envelopes,
                feature_dim,
                name="Gamma_ne",
            ),
            en_filter=PairwiseFilter(
                cutoff,
                pair_mlp_widths,
                pair_n_envelopes,
                feature_dim,
                name="Gamma_en",
            ),
            mlp_nuc=MLP(widths=[feature_dim] * nuc_mlp_depth, name="mlp_nuc"),
            lin_h0=nn.Dense(feature_dim, use_bias=True, name="lin_h0"),
            lin_orbitals=nn.Dense(n_determinants * n_el, name="lin_orbitals"),
            envelopes=IsotropicEnvelope(n_determinants, n_el, n_envelopes),
            el_el_cusp=ElElCusp(n_up),
            use_el_el_cusp=use_el_el_cusp,
            use_mlp_jastrow=mlp_jastrow["use"],
            mlp_jastrow=JastrowFactor(mlp_jastrow["embedding_n_hidden"], mlp_jastrow["soe_n_hidden"]),
            use_log_jastrow=log_jastrow["use"],
            log_jastrow=JastrowFactor(log_jastrow["embedding_n_hidden"], log_jastrow["soe_n_hidden"]),
        )

    def get_static_input(self, electrons: Electrons):
        def round_fn(x):
            return int(round_to_next_step(x, 1.2, 1, self.n_electrons))

        dist_ee, dist_ne = get_full_distance_matrices(electrons, self.R)
        n_neighbours = get_nr_of_neighbours(dist_ee, dist_ne, self.cutoff, 1.2, 1)
        n_deps_h0, n_deps_H, n_deps_hout = _get_max_nr_of_dependencies(dist_ee, dist_ne, self.cutoff)

        n_deps_h0_padded = round_fn(n_deps_h0)
        n_deps_H_padded = round_fn(n_deps_H)
        n_deps_hout_padded = round_fn(n_deps_hout)
        n_deps = NrOfDependencies(n_deps_h0_padded, n_deps_H_padded, n_deps_hout_padded)
        return StaticInput(n_neighbours=n_neighbours, n_deps=n_deps)

    def hf_transformation(self, hf_orbitals):
        return hf_orbitals_to_fulldet_orbitals(hf_orbitals)

    def signed(self, params: MoonParams, electrons: Electrons, static: StaticInput) -> SignedLogAmplitude:
        embeddings = self._embedding(params, electrons, static)
        orbitals = self.orbitals(params, electrons, static, embeddings)
        signpsi, logpsi = signed_logpsi_from_orbitals(orbitals)
        if params.el_el_cusp is not None:
            logpsi += self.el_el_cusp.apply(params.el_el_cusp, electrons)
        if params.mlp_jastrow is not None:
            logpsi += self.mlp_jastrow.apply(params.mlp_jastrow, embeddings)
        if params.log_jastrow is not None:
            logpsi += jnp.log(jnp.abs(self.log_jastrow.apply(params.log_jastrow, embeddings)))
        return signpsi, logpsi

    def __call__(self, params: MoonParams, electrons: Electrons, static: StaticInput):
        return self.signed(params, electrons, static)[1]

    @vectorize(signature="(nel,dim)->()", excluded=(0, 1, 3))
    def local_energy(self, params: MoonParams, electrons: Electrons, static: StaticInput):
        raise NotImplementedError("Sparse energy not implemented for cusps")
        log_psi = self._logpsi_with_fwd_lap(params, electrons, static)
        E_kin = -0.5 * (log_psi.laplacian + jnp.vdot(log_psi.jacobian.data, log_psi.jacobian.data))
        E_pot = potential_energy(electrons, self.R, self.Z)
        return E_pot + E_kin

    @vectorize(signature="(nel,dim)->()", excluded=(0, 1, 3))
    def local_energy_dense(self, params: MoonParams, electrons: Electrons, static: StaticInput):
        return make_local_energy(self, self.R, self.Z)(params, electrons, static)

    @vectorize(signature="(nel,dim)->()", excluded=(0, 1, 3))
    def local_energy_dense_looped(self, params: MoonParams, electrons: Electrons, static: StaticInput):
        return make_local_energy(self, self.R, self.Z, use_fwd_lap=False)(params, electrons, static)

    def init(self, key: PRNGKeyArray):
        rngs = jax.random.split(key, 9)
        feature_dim = self.lin_h0.features
        n_atoms = self.R.shape[0]
        if self.use_el_el_cusp:  # Make sure to not include the el-el cusp parameters when it's not being used.
            el_el_cusp = self.el_el_cusp.init(rngs[7], np.zeros((self.n_electrons, 3)))
        else:
            el_el_cusp = None
        if self.use_mlp_jastrow:
            mlp_jastrow = self.mlp_jastrow.init(rngs[8], np.zeros((self.n_electrons, feature_dim)))
        else:
            mlp_jastrow = None
        if self.use_log_jastrow:
            log_jastrow = self.log_jastrow.init(rngs[8], np.zeros((self.n_electrons, feature_dim)))
        else:
            log_jastrow = None
        return MoonParams(
            ee_filter=self.ee_filter.init(rngs[0], np.zeros([6])),  # dist + 3 * diff + spin1 + spin2
            ne_filter=self.ne_filter.init(rngs[1], np.zeros([4])),  # dist + 3 * diff
            en_filter=self.en_filter.init(rngs[2], np.zeros([4])),  # dist + 3 * diff
            lin_h0=self.lin_h0.init(rngs[3], np.zeros([feature_dim])),
            mlp_nuc=self.mlp_nuc.init(rngs[4], np.zeros([feature_dim])),
            lin_orbitals=self.lin_orbitals.init(rngs[5], np.zeros([feature_dim])),
            envelopes=self.envelopes.init(rngs[6], np.zeros([n_atoms])),
            el_el_cusp=el_el_cusp,
            mlp_jastrow=mlp_jastrow,
            log_jastrow=log_jastrow,
        )

    def _get_h0(
            self,
            params: Parameters,
            r: Float[Array, "dim=3"],
            r_nb: Float[Array, "*neighbours dim=3"],
            s: Optional[Int] = None,
            s_nb: Optional[Integer[Array, " *neighbours"]] = None,
    ):
        # vmap over neighbours
        features_ee = get_diff_features(r, r_nb, s, s_nb)
        Gamma_ee = cast(FilterKernel, self.ee_filter.apply(params.ee_filter, features_ee))
        h0 = jnp.sum(Gamma_ee, axis=-2)  # sum over neighbours
        h0 = jax.nn.silu(h0)
        h0 = cast(Embedding, self.lin_h0.apply(params.lin_h0, h0))
        return h0

    def _get_Gamma_ne(self, params, R, r_nb_ne):
        features_ne = get_diff_features(R, r_nb_ne)
        return cast(FilterKernel, self.ne_filter.apply(params.ne_filter, features_ne))

    def _get_Gamma_en(self, params, r, R_nb_en_):
        features_en = get_diff_features(r, R_nb_en_)
        return cast(FilterKernel, self.en_filter.apply(params.en_filter, features_en))

    def _get_Gamma_ne_vmapped(self, params, R, r_nb_ne):
        _get_Gamma = jax.vmap(self._get_Gamma_ne, in_axes=(None, None, 0))  # vmap over neighbours
        _get_Gamma = jax.vmap(_get_Gamma, in_axes=(None, 0, 0))  # vmap over center
        return _get_Gamma(params, R, r_nb_ne)

    def _get_Gamma_en_vmapped(self, params, r, R_nb_en_):
        _get_Gamma = jax.vmap(self._get_Gamma_en, in_axes=(None, None, 0))  # vmap over neighbours
        _get_Gamma = jax.vmap(_get_Gamma, in_axes=(None, 0, 0))  # vmap over center
        return _get_Gamma(params, r, R_nb_en_)

    def _envelopes(self, params, r: Float[Array, "dim=3"]):
        dist_en_full = jnp.linalg.norm(r[None, :] - self.R, axis=-1)
        return cast(jax.Array, self.envelopes.apply(params.envelopes, dist_en_full))

    def _embedding_to_orbitals(self, params, r: Float[Array, "dim=3"], h: Embedding):
        orbitals = cast(jax.Array, self.lin_orbitals.apply(params.lin_orbitals, h))

        dist_en_full = jnp.linalg.norm(r[None, :] - self.R, axis=-1)
        envelopes = cast(jax.Array, self.envelopes.apply(params.envelopes, dist_en_full))
        orbitals *= envelopes
        orbitals = einops.rearrange(orbitals, "(det orb) -> det orb", det=self.n_determinants)
        return orbitals

    def _merge_orbitals_with_envelopes(self, orbitals, envelopes):
        orbitals = orbitals * envelopes
        orbitals = einops.rearrange(orbitals, "(det orb) -> det orb", det=self.n_determinants)
        return orbitals

    def _embedding(self, params: Parameters, electrons: Electrons, static: StaticInput):
        idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)

        # # Step 0: Get neighbours
        spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en = get_neighbour_coordinates(electrons, self.R, idx_nb, self.spins)

        # Step 1: Get h0
        h0 = jax.vmap(self._get_h0, in_axes=(None, 0, 0, 0, 0))(params, electrons, r_nb_ee, self.spins, spin_nb_ee)

        # # Step 2: Contract to nuclei + MLP
        Gamma_ne = self._get_Gamma_ne_vmapped(params, self.R, r_nb_ne)
        h0_nb_ne = get_with_fill(h0, idx_nb.ne, 0.0)
        H0 = contract(h0_nb_ne, Gamma_ne)
        H = cast(Embedding, self.mlp_nuc.apply(params.mlp_nuc, H0))

        # # Step 3: Contract back to electrons => Final electron embeddings h_out
        Gamma_en = self._get_Gamma_en_vmapped(params, electrons, R_nb_en)
        H_nb_en = get_with_fill(H, idx_nb.en, 0.0)
        h_out = contract(H_nb_en, Gamma_en, h0)
        return h_out

    @functools.partial(jnp.vectorize, excluded=(0, 1, 3, 4), signature="(el,dim)->(det,el,orb)")
    def orbitals(self, params: Parameters, electrons: Electrons, static: StaticInput,
                 embeddings: Optional[Embedding] = None) -> SlaterMatrices:
        if embeddings is None:
            embeddings = self._embedding(params, electrons, static)
        orbitals = cast(jax.Array, self.lin_orbitals.apply(params.lin_orbitals, embeddings))
        envelopes = jax.vmap(lambda r: self._envelopes(params, r))(electrons)
        orbitals = jax.vmap(self._merge_orbitals_with_envelopes, in_axes=0, out_axes=-2)(
            orbitals, envelopes
        )  # vmap over electrons
        orbitals = swap_bottom_blocks(orbitals, self.n_up)
        return (orbitals,)

    def _logpsi_with_fwd_lap(self, params, electrons, static):
        orbitals, dependencies = self._orbitals_with_fwd_lap(params, electrons, static)
        signs, logdets = jax.vmap(lambda o: slogdet_with_sparse_fwd_lap(o, dependencies), in_axes=-3, out_axes=-1)(
            orbitals
        )
        # We set return_sign=True and then ignore the sign (by only taking return value 0),
        # because otherwise logsumexp cannot deal with negative signs
        return fwd_lap(lambda logdets_: jax.nn.logsumexp(logdets_, b=signs, return_sign=True)[0])(logdets)

    def _orbitals_with_fwd_lap(
            self, params: Parameters, electrons: Electrons, static: StaticInput
    ) -> tuple[FwdLaplArray, Dependency]:
        h, dependencies = self._embedding_with_fwd_lap(params, electrons, static)
        # vmaps over electrons
        orbitals = jax.vmap(
            fwd_lap(lambda h_: self.lin_orbitals.apply(params.lin_orbitals, h_)), in_axes=-2, out_axes=-2
        )(h)
        envelopes = jax.vmap(fwd_lap(lambda r: self._envelopes(params, r)), in_axes=-2, out_axes=-2)(electrons)
        envelopes = densify_jacobian_by_zero_padding(envelopes, 3 * static.n_deps.h_el_out)  # type: ignore
        orbitals = jax.vmap(fwd_lap(lambda o, e: self._merge_orbitals_with_envelopes(o, e)), in_axes=-2, out_axes=-2)(
            orbitals, envelopes
        )
        orbitals = jtu.tree_map(lambda o: swap_bottom_blocks(o, self.n_up), orbitals)
        return orbitals, dependencies

    def _embedding_with_fwd_lap(
            self, params: MoonParams, electrons: Electrons, static: StaticInput
    ) -> tuple[FwdLaplArray, Dependency]:
        idx_nb = get_neighbour_indices(electrons, self.R, static.n_neighbours, self.cutoff)
        deps, dep_maps = get_all_dependencies(idx_nb, static.n_deps)

        # Step 0: Get neighbours
        spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en = get_neighbour_coordinates(electrons, self.R, idx_nb, self.spins)

        # Step 1: Contract ee to get electron embeedings h0
        get_h0 = fwd_lap(self._get_h0, argnums=(1, 2))
        get_h0 = jax.vmap(get_h0, in_axes=(None, 0, 0, 0, 0), out_axes=-2)  # vmap over center electrons
        # Shapes: h0: [nel x feature_dim]; h0.jac: [3*n_deps1 x n_el x feature_dim] (dense)
        h0 = get_h0(params, electrons, r_nb_ee, self.spins, spin_nb_ee)

        # Step 2: Contract to nuclei + MLP on nuclei => nuclear embedding H
        # 2a: Get the spatial filter between nuclei and neighbouring electrons
        get_Gamma_ne = fwd_lap(self._get_Gamma_ne, argnums=(2,))
        Gamma_ne = _vmap_over_centers_and_neighbours(get_Gamma_ne)(params, self.R, r_nb_ne)
        # Gamma_ne: [n_nuc x n_neighbour x feature_dim]; Gamma_ne.jac: [3 x n_nuc x n_neighbour x feature_dim]

        # 2b: Get the neighbouring electron embeddings
        h0_nb_ne = jtu.tree_map(lambda x: x.at[..., idx_nb.ne, :].get(mode="drop", fill_value=0.0), h0)

        # 2c: Pad/align all jacobians with the output dependencies
        _pad_jacobian = jax.vmap(pad_jacobian_to_output_deps, in_axes=(-2, -2, None), out_axes=-2)
        _pad_jacobian = jax.vmap(_pad_jacobian, in_axes=(-3, -3, None), out_axes=-3)
        h0_nb_ne = _pad_jacobian(h0_nb_ne, dep_maps.h0_to_Hnuc, static.n_deps.H_nuc)
        Gamma_ne = _pad_jacobian(Gamma_ne, dep_maps.Gamma_ne_to_Hnuc, static.n_deps.H_nuc)

        # 2d: Contract and apply the MLP
        @functools.partial(jax.vmap, in_axes=(-3, -3), out_axes=-2)  # vmap over centers (nuclei)
        @fwd_lap
        def contract_and_mlp(h0_nb_ne, Gamma_ne):
            H0 = contract(h0_nb_ne, Gamma_ne)
            return self.mlp_nuc.apply(params.mlp_nuc, H0)

        H = contract_and_mlp(h0_nb_ne, Gamma_ne)

        # Step 3: Contract back to electrons => Final electron embeddings h_out
        # 3a: Get Filters
        get_Gamma_en = fwd_lap(self._get_Gamma_en, argnums=(1,))
        Gamma_en = _vmap_over_centers_and_neighbours(get_Gamma_en)(params, electrons, R_nb_en)
        # Gamma_en: [n_el x n_neighbouring_nuc x feature_dim]; Gamma_en.jac: [3 x n_el x n_neighbouring_nuc x feature_dim]

        # 3b: Get the neighbouring nuclear embeddings
        H_nb_en = jtu.tree_map(lambda x: x.at[..., idx_nb.en, :].get(mode="drop", fill_value=0.0), H)

        # 3c: Pad/align all jacobians with the output dependencies
        H_nb_en = _pad_jacobian(H_nb_en, dep_maps.Hnuc_to_hout, static.n_deps.h_el_out)
        h0_residual = jax.vmap(pad_jacobian_to_output_deps, in_axes=(-2, -2, None), out_axes=-2)(
            h0, dep_maps.h0_to_hout, static.n_deps.h_el_out
        )
        Gamma_en = densify_jacobian_by_zero_padding(Gamma_en, 3 * static.n_deps.h_el_out)

        # 3d: Contract
        # vmap over centers (electrons)
        h_out = jax.vmap(fwd_lap(contract), in_axes=(-3, -3, -2), out_axes=-2)(H_nb_en, Gamma_en, h0_residual)

        return h_out, deps.h_el_out
