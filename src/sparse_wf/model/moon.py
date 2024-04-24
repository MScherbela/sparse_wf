import logging
from sparse_wf.api import (
    Dependencies,
    DependencyMap,
    Parameters,
    Electrons,
    Charges,
    Nuclei,
    StaticInput,
    DynamicInputWithDependencies,
    NeighbourIndices,
    DistanceMatrix,
    ParameterizedWaveFunction,
    SignedLogAmplitude,
    Int,
    SlaterMatrices,
)
import jax.numpy as jnp
from typing import NamedTuple, cast, Sequence, Optional
from jaxtyping import Float, Array, PRNGKeyArray, Integer
import jax
import numpy as np
from sparse_wf.model.utils import (
    cutoff_function,
    Embedding,
    FilterKernel,
    contract,
    MLP,
    signed_logpsi_from_orbitals,
    IsotropicEnvelope,
    hf_orbitals_to_fulldet_orbitals,
    swap_bottom_blocks,
)
from sparse_wf.hamiltonian import make_local_energy, potential_energy
from sparse_wf.jax_utils import jit, fwd_lap
from sparse_wf.model.graph_utils import (
    densify_jacobian_by_zero_padding,
    pad_jacobian_to_output_deps,
    merge_dependencies,
    get_dependency_map,
    NO_NEIGHBOUR,
    GenericInputConstructor,
    get_with_fill,
    slogdet_with_sparse_fwd_lap,
)
import flax.linen as nn
from flax.struct import PyTreeNode, field
import functools
from folx.api import FwdLaplArray
import pyscf
import einops
import jax.tree_util as jtu


class DependenciesMoon(NamedTuple):
    h_el_initial: Dependencies
    H_nuc: Dependencies
    h_el_out: Dependencies


class NrOfDependenciesMoon(NamedTuple):
    h_el_initial: int
    H_nuc: int
    h_el_out: int


class DependencyMapsMoon(NamedTuple):
    h0_to_Hnuc: DependencyMap
    Gamma_ne_to_Hnuc: DependencyMap
    Hnuc_to_hout: DependencyMap
    h0_to_hout: DependencyMap


class InputConstructorMoon(GenericInputConstructor):
    # This function cannot be jitted, because it returns a static tuple of integers
    def get_static_input(self, electrons: Electrons) -> StaticInput:
        dist_ee, dist_ne = self.get_full_distance_matrices(electrons)
        n_neighbours = self.get_nr_of_neighbours(dist_ee, dist_ne)
        n_deps_h0, n_deps_H, n_deps_hout = self._get_max_nr_of_dependencies(dist_ee, dist_ne)

        n_deps_h0_padded = self._round_to_next_step(n_deps_h0)
        n_deps_H_padded = self._round_to_next_step(n_deps_H)
        n_deps_hout_padded = self._round_to_next_step(n_deps_hout)
        n_deps = NrOfDependenciesMoon(n_deps_h0_padded, n_deps_H_padded, n_deps_hout_padded)
        return StaticInput(n_neighbours=n_neighbours, n_deps=n_deps)

    # @jit(static_argnames=("self", "static"))

    def get_dynamic_input_with_dependencies(
        self, electrons: Electrons, static: StaticInput
    ) -> DynamicInputWithDependencies:
        assert electrons.ndim == 2, "Use vmap to vectorize this function"

        logging.info(f"Compiling model for static: {static}")

        # Indices of neighbours
        dist_ee, dist_ne = self.get_full_distance_matrices(electrons)
        idx_nb = self.get_neighbour_indices(dist_ee, dist_ne, static.n_neighbours)

        # Dependencies of embedddings
        deps, dep_maps = self._get_all_dependencies(idx_nb, cast(NrOfDependenciesMoon, static.n_deps))
        return DynamicInputWithDependencies(
            electrons=electrons, neighbours=idx_nb, dependencies=deps, dep_maps=dep_maps
        )

    @jit(static_argnames="self")
    def _get_max_nr_of_dependencies(self, dist_ee: DistanceMatrix, dist_ne: DistanceMatrix):
        # Thest first electron message passing step can depend at most on electrons within 1 * cutoff
        n_deps_max_h0 = jnp.max(jnp.sum(dist_ee < self.cutoff, axis=-1))

        # The nuclear embeddings are computed with 2 message passing steps and can therefore depend at most on electrons within 2 * cutoff
        n_deps_max_H = jnp.max(jnp.sum(dist_ne < self.cutoff * 2, axis=-1))

        # The output electron embeddings are computed with 3 message passing step and can therefore depend at most on electrons within 3 * cutoff
        n_deps_max_h_out = jnp.max(jnp.sum(dist_ee < self.cutoff * 3, axis=-1))
        return n_deps_max_h0, n_deps_max_H, n_deps_max_h_out

    def _get_all_dependencies(
        self, idx_nb: NeighbourIndices, n_deps_max: NrOfDependenciesMoon
    ) -> tuple[tuple[Dependencies, ...], tuple[DependencyMap, ...]]:
        """Get the indices of electrons on which each embedding will depend on.

        Args:
            idx_nb: NeighbourIndices, named tuple containing the indices of the neighbours of each electron and nucleus.
            n_deps_max: maximum_nr_of electrons that each embedding can depend on.
                - n_deps_max[0]: maximum number of dependencies for the electron embeddings at the first step.
                - n_deps_max[1]: maximum number of dependencies for the nuclear embeddings.
                - n_deps_max[2]: maximum number of dependencies for the output electron embeddings.

        Returns:
            deps: tuple of jnp.ndarray, dependencies for the electron embeddings at each step.
                deps_h0: [batch_size x n_el  x nr_of_deps_level_1]
                deps_H:  [batch_size x n_nuc x nr_of deps_level_2]
                deps_hout: [batch_size x n_el x nr_of_deps_level_3]
            dep_maps: tuple of jnp.ndarray, maps the dependencies between the levels:
                dep_map_h0_to_H: [batch_size x n_nuc x n_neighbouring_el x nr_of_deps_level_1]; values are in [0 ... deps_level_2]
                dep_map_H_to_hout: [batch_size x n_el x n_neighbouring_nuc x nr_of_deps_level_2]; values are in [0 ... deps_level_3]
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
        deps_h0: Dependencies = jnp.concatenate([self_dependency, idx_nb.ee], axis=-1)

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

        return DependenciesMoon(deps_h0, deps_H, deps_hout), DependencyMapsMoon(
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
        s_prod = s * s_nb
        features.append(s_prod[..., None])
    return jnp.concatenate(features, axis=-1)


class PairwiseFilter(nn.Module):
    cutoff: float
    directional_mlp_widths: Sequence[int]
    n_envelopes: int
    out_dim: int

    def scale_initializer(self, rng, shape):
        relative_noise = 0.1 * jax.random.normal(rng, shape)
        mean_scale = 0.5 * self.cutoff
        return mean_scale * (1 + relative_noise)

    @nn.compact
    def __call__(self, dist_diff: Float[Array, "*batch_dims features_in"]) -> Float[Array, "*batch_dims features_out"]:
        """Compute the pairwise filters between two particles.

        Args:
            dist_diff: The distance, 3D difference and optionall spin difference between two particles [n_el x n_nb x 4(5)].
                The 0th feature dimension must contain the distance, the remaining dimensions can contain arbitrary
                features that are used to compute the pairwise filters, e.g. product of spins.
        """
        # Direction- (and spin-) dependent MLP
        directional_features = MLP(widths=self.directional_mlp_widths)(dist_diff)

        # Distance-dependenet radial filters
        dist = dist_diff[..., 0]
        scales = self.param("scales", self.scale_initializer, (self.n_envelopes,))
        scales = jax.nn.softplus(scales)
        envelopes = jnp.exp(-((dist[..., None] / scales) ** 2))
        envelopes = nn.Dense(directional_features.shape[-1], use_bias=False)(envelopes)
        envelopes *= cutoff_function(dist / self.cutoff)[..., None]
        beta = directional_features * envelopes
        return nn.Dense(self.out_dim, use_bias=False)(beta)


def _vmap_over_centers_and_neighbours(f):
    f_out = jax.vmap(f, in_axes=(None, None, 0), out_axes=-2)  # vmap over neighbours
    f_out = jax.vmap(f_out, in_axes=(None, 0, 0), out_axes=-3)  # vmap over centers
    return f_out


class SparseMoonParams(PyTreeNode):
    ee_filter: Parameters
    ne_filter: Parameters
    en_filter: Parameters
    mlp_nuc: Parameters
    lin_h0: Parameters
    lin_orbitals: Parameters
    envelopes: Parameters


class SparseMoonWavefunction(PyTreeNode, ParameterizedWaveFunction):
    R: Nuclei
    Z: Charges
    n_electrons: int = field(pytree_node=False)
    n_up: int = field(pytree_node=False)
    n_determinants: int = field(pytree_node=False)
    cutoff: float
    input_constructor: InputConstructorMoon
    ee_filter: PairwiseFilter
    ne_filter: PairwiseFilter
    en_filter: PairwiseFilter
    mlp_nuc: MLP
    lin_h0: nn.Dense
    lin_orbitals: nn.Dense
    envelopes: IsotropicEnvelope

    @property
    def n_dn(self):
        return self.n_electrons - self.n_up

    @property
    def spins(self):
        return jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_dn)])

    @classmethod
    def create(
        cls,
        mol: pyscf.gto.Mole,
        n_determinants: int,
        cutoff: float,
        feature_dim: int,
        nuc_mlp_depth: int,
        pair_mlp_widths: Sequence[int],
        pair_n_envelopes: int,
        model_name: str,
    ):
        n_up, n_dn = mol.nelec
        n_el = n_up + n_dn
        R = mol.atom_coords()
        Z = mol.atom_charges()
        return cls(
            R=R,
            Z=Z,
            n_electrons=int(n_el),
            n_up=int(n_up),
            n_determinants=n_determinants,
            cutoff=cutoff,
            input_constructor=InputConstructorMoon(R, Z, n_el, cutoff, padding_factor=1.2),
            ee_filter=PairwiseFilter(cutoff, pair_mlp_widths, pair_n_envelopes, feature_dim, name="Gamma_ee"),
            ne_filter=PairwiseFilter(cutoff, pair_mlp_widths, pair_n_envelopes, feature_dim, name="Gamma_ne"),
            en_filter=PairwiseFilter(cutoff, pair_mlp_widths, pair_n_envelopes, feature_dim, name="Gamma_en"),
            mlp_nuc=MLP(widths=[feature_dim] * nuc_mlp_depth, name="mlp_nuc"),
            lin_h0=nn.Dense(feature_dim, use_bias=True, name="lin_h0"),
            lin_orbitals=nn.Dense(n_determinants * n_el, name="lin_orbitals"),
            envelopes=IsotropicEnvelope(n_determinants * n_el),
        )

    def hf_transformation(self, hf_orbitals):
        return hf_orbitals_to_fulldet_orbitals(hf_orbitals)

    def signed(self, params: Parameters, electrons: Electrons, static: StaticInput) -> SignedLogAmplitude:
        orbitals = self.orbitals(params, electrons, static)
        return signed_logpsi_from_orbitals(orbitals)

    def __call__(self, params: Parameters, electrons: Electrons, static: StaticInput):
        return self.signed(params, electrons, static)[1]

    @functools.partial(jnp.vectorize, excluded=(0, 1, 3), signature="(nel,dim)->()")
    def local_energy(self, params: Parameters, electrons: Electrons, static: StaticInput):
        log_psi = self._logpsi_with_fwd_lap(params, electrons, static)
        E_kin = -0.5 * (log_psi.laplacian + jnp.vdot(log_psi.jacobian.data, log_psi.jacobian.data))
        E_pot = potential_energy(electrons, self.R, self.Z)
        return E_pot + E_kin

    @functools.partial(jnp.vectorize, excluded=(0, 1, 3), signature="(nel,dim)->()")
    def local_energy_dense(self, params: Parameters, electrons: Electrons, static: StaticInput):
        return make_local_energy(self, self.R, self.Z)(params, electrons, static)

    def init(self, rng: PRNGKeyArray):
        rngs = jax.random.split(rng, 7)
        feature_dim = self.lin_h0.features
        n_atoms = self.R.shape[0]
        params = SparseMoonParams(
            ee_filter=self.ee_filter.init(rngs[0], np.zeros([5])),  # dist + 3 * diff + spin
            ne_filter=self.ne_filter.init(rngs[1], np.zeros([4])),  # dist + 3 * diff
            en_filter=self.en_filter.init(rngs[2], np.zeros([4])),  # dist + 3 * diff
            lin_h0=self.lin_h0.init(rngs[3], np.zeros([feature_dim])),
            mlp_nuc=self.mlp_nuc.init(rngs[4], np.zeros([feature_dim])),
            lin_orbitals=self.lin_orbitals.init(rngs[5], np.zeros([feature_dim])),
            envelopes=self.envelopes.init(rngs[6], np.zeros([1, n_atoms])),
        )
        return params

    def get_neighbour_coordinates(self, electrons: Electrons, idx_nb: NeighbourIndices):
        spin_nb_ee = get_with_fill(self.spins, idx_nb.ee, 0.0)
        r_nb_ee = get_with_fill(electrons, idx_nb.ee, NO_NEIGHBOUR)  # [n_el  x n_neighbouring_electrons x 3]
        r_nb_ne = get_with_fill(electrons, idx_nb.ne, NO_NEIGHBOUR)  # [n_nuc x n_neighbouring_electrons x 3]
        R_nb_en = get_with_fill(self.R, idx_nb.en, NO_NEIGHBOUR)  # [n_el  x n_neighbouring_nuclei    x 3]
        return spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en

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
        params = cast(SparseMoonParams, params)
        idx_nb = self.input_constructor.get_dynamic_input(electrons, static).neighbours

        # # Step 0: Get neighbours
        spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en = self.get_neighbour_coordinates(electrons, idx_nb)

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

    @functools.partial(jnp.vectorize, excluded=(0, 1, 3), signature="(el,dim)->(det,el,orb)")
    def orbitals(self, params: Parameters, electrons: Electrons, static: StaticInput) -> SlaterMatrices:
        with jax.named_scope("Embedding"):
            h = self._embedding(params, electrons, static)

        with jax.named_scope("Orbitals"):
            orbitals = cast(jax.Array, self.lin_orbitals.apply(params.lin_orbitals, h))
            envelopes = jax.vmap(lambda r: self._envelopes(params, r))(electrons)
            orbitals = jax.vmap(self._merge_orbitals_with_envelopes, in_axes=0, out_axes=-2)(
                orbitals, envelopes
            )  # vmap over electrons
            orbitals = swap_bottom_blocks(orbitals, self.n_up)
        return (orbitals,)

    def _logpsi_with_fwd_lap(self, params, electrons, static):
        orbitals, dependencies = self._orbitals_with_fwd_lap(params, electrons, static)
        with jax.named_scope("LogPsi"):
            signs, logdets = jax.vmap(lambda o: slogdet_with_sparse_fwd_lap(o, dependencies), in_axes=-3, out_axes=-1)(
                orbitals
            )
        # We set return_sign=True and then ignore the sign (by only taking return value 0),
        # because otherwise logsumexp cannot deal with negative signs
        return fwd_lap(lambda logdets_: jax.nn.logsumexp(logdets_, b=signs, return_sign=True)[0])(logdets)

    def _orbitals_with_fwd_lap(
        self, params: Parameters, electrons: Electrons, static: StaticInput
    ) -> tuple[FwdLaplArray, Dependencies]:
        n_deps = cast(NrOfDependenciesMoon, static.n_deps)
        params = cast(SparseMoonParams, params)

        h, dependencies = self._embedding_with_fwd_lap(params, electrons, static)
        # vmaps over electrons
        orbitals = jax.vmap(
            fwd_lap(lambda h_: self.lin_orbitals.apply(params.lin_orbitals, h_)), in_axes=-2, out_axes=-2
        )(h)
        envelopes = jax.vmap(fwd_lap(lambda r: self._envelopes(params, r)), in_axes=-2, out_axes=-2)(electrons)
        envelopes = densify_jacobian_by_zero_padding(envelopes, 3 * n_deps.h_el_out)  # type: ignore
        orbitals = jax.vmap(fwd_lap(lambda o, e: self._merge_orbitals_with_envelopes(o, e)), in_axes=-2, out_axes=-2)(
            orbitals, envelopes
        )
        orbitals = jtu.tree_map(lambda o: swap_bottom_blocks(o, self.n_up), orbitals)
        return orbitals, dependencies

    def _embedding_with_fwd_lap(
        self, params: SparseMoonParams, electrons: Electrons, static_input: StaticInput
    ) -> tuple[FwdLaplArray, Dependencies]:
        r, idx_nb, deps, dep_maps = self.input_constructor.get_dynamic_input_with_dependencies(electrons, static_input)
        n_deps = cast(NrOfDependenciesMoon, static_input.n_deps)
        dep_maps = cast(DependencyMapsMoon, dep_maps)
        deps = cast(DependenciesMoon, deps)

        # Step 0: Get neighbours
        spin_nb_ee, r_nb_ee, r_nb_ne, R_nb_en = self.get_neighbour_coordinates(r, idx_nb)

        # Step 1: Contract ee to get electron embeedings h0
        get_h0 = fwd_lap(self._get_h0, argnums=(1, 2))
        get_h0 = jax.vmap(get_h0, in_axes=(None, 0, 0, 0, 0), out_axes=-2)  # vmap over center electrons
        # Shapes: h0: [nel x feature_dim]; h0.jac: [3*n_deps1 x n_el x feature_dim] (dense)
        h0 = get_h0(params, r, r_nb_ee, self.spins, spin_nb_ee)

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
        h0_nb_ne = _pad_jacobian(h0_nb_ne, dep_maps.h0_to_Hnuc, n_deps.H_nuc)
        Gamma_ne = _pad_jacobian(Gamma_ne, dep_maps.Gamma_ne_to_Hnuc, n_deps.H_nuc)

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
        Gamma_en = _vmap_over_centers_and_neighbours(get_Gamma_en)(params, r, R_nb_en)
        # Gamma_en: [n_el x n_neighbouring_nuc x feature_dim]; Gamma_en.jac: [3 x n_el x n_neighbouring_nuc x feature_dim]

        # 3b: Get the neighbouring nuclear embeddings
        H_nb_en = jtu.tree_map(lambda x: x.at[..., idx_nb.en, :].get(mode="drop", fill_value=0.0), H)

        # 3c: Pad/align all jacobians with the output dependencies
        H_nb_en = _pad_jacobian(H_nb_en, dep_maps.Hnuc_to_hout, n_deps.h_el_out)
        h0_residual = jax.vmap(pad_jacobian_to_output_deps, in_axes=(-2, -2, None), out_axes=-2)(
            h0, dep_maps.h0_to_hout, n_deps.h_el_out
        )
        Gamma_en = densify_jacobian_by_zero_padding(Gamma_en, 3 * n_deps.h_el_out)

        # 3d: Contract
        # vmap over centers (electrons)
        h_out = jax.vmap(fwd_lap(contract), in_axes=(-3, -3, -2), out_axes=-2)(H_nb_en, Gamma_en, h0_residual)

        return h_out, deps.h_el_out
