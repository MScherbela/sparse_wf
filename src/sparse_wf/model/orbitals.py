import flax.linen as nn
from sparse_wf.model.sparse_fwd_lap import Linear
from sparse_wf.model.envelopes import EfficientIsotropicEnvelopes, GLUEnvelopes
import jax.numpy as jnp
from sparse_wf.api import Charges, Electrons, ElectronEmb, EnvelopeArgs, Parameters, ElectronIdx, Nuclei
from sparse_wf.model.sparse_fwd_lap import NodeWithFwdLap, merge_up_down, multiply_with_1el_fn
from sparse_wf.jax_utils import fwd_lap
from folx.api import FwdLaplArray
import jax
from typing import NamedTuple
import einops
import jax.tree_util as jtu
import functools


class OrbitalState(NamedTuple):
    envelopes: jax.Array
    orbitals: jax.Array


def _swap_spin_updn(phi, n_electrons, n_up):
    # Input shape: [... x (dets*orbitals) ]"
    phi = einops.rearrange(phi, "... (det orb) -> ... det orb", orb=n_electrons)
    phi = jnp.concatenate([phi[..., n_up:], phi[..., :n_up]], axis=-1)  # swap spin up and down
    phi = einops.rearrange(phi, "... det orb -> ... (det orb)", orb=n_electrons)
    return phi


def _swap_spin_updn_with_fwd_lap(phi: NodeWithFwdLap, n_electrons, n_up):
    x = _swap_spin_updn(phi.x, n_electrons, n_up)
    lap = _swap_spin_updn(phi.lap, n_electrons, n_up)
    jac = einops.rearrange(phi.jac, "... (det orb) -> ... det orb", orb=n_electrons)
    jac_swapped = jnp.concatenate([jac[..., n_up:], jac[..., :n_up]], axis=-1)
    jac = jnp.where(
        (phi.idx_ctr < n_up)[:, None, None, None],
        jac,
        jac_swapped,
    )
    jac = einops.rearrange(jac, "... det orb -> ... (det orb)", orb=n_electrons)
    return NodeWithFwdLap(x, jac, lap, phi.idx_ctr, phi.idx_dep)


class Orbitals(nn.Module):
    n_electrons: int
    n_up: int
    n_determinants: int
    spin_restricted: bool
    Z: Charges
    R: Nuclei
    envelope_args: EnvelopeArgs

    def setup(self):
        self.to_orbitals_up = Linear(
            self.n_electrons * self.n_determinants,
            bias_init=nn.initializers.truncated_normal(0.01, jnp.float32),
            name="to_orbitals_up",
        )
        if not self.spin_restricted:
            self.to_orbitals_dn = Linear(
                self.n_electrons * self.n_determinants,
                bias_init=nn.initializers.truncated_normal(0.01, jnp.float32),
                name="to_orbitals_dn",
            )

        match self.envelope_args["envelope"]:
            case "isotropic":
                self.envelope_up = EfficientIsotropicEnvelopes(
                    self.n_determinants, self.n_electrons, **self.envelope_args["isotropic_args"], name="env_up"
                )
                if not self.spin_restricted:
                    self.envelope_dn = EfficientIsotropicEnvelopes(
                        self.n_determinants, self.n_electrons, **self.envelope_args["isotropic_args"], name="env_dn"
                    )
            case "glu":
                self.envelope_up = GLUEnvelopes(
                    self.Z, self.n_determinants, self.n_electrons, **self.envelope_args["glu_args"], name="env_up"
                )
                if not self.spin_restricted:
                    self.envelope_dn = GLUEnvelopes(
                        self.Z, self.n_determinants, self.n_electrons, **self.envelope_args["glu_args"], name="env_dn"
                    )
            case _:
                raise ValueError(f"Unknown envelope type {self.envelope_type}")

    def _get_orbitals_up(self, embeddings_up):
        return self.to_orbitals_up(embeddings_up)

    def _get_orbitals_dn(self, embeddings_dn):
        if self.spin_restricted:
            orbitals = self._get_orbitals_up(embeddings_dn)
            if isinstance(orbitals, NodeWithFwdLap):
                return _swap_spin_updn_with_fwd_lap(orbitals, self.n_electrons, self.n_up)
            else:
                return _swap_spin_updn(orbitals, self.n_electrons, self.n_up)
        else:
            return self.to_orbitals_dn(embeddings_dn)

    def _get_envelopes_up(self, electrons_up):
        diffs = electrons_up[..., None, :] - self.R
        return self.envelope_up(diffs)

    def _get_envelopes_dn(self, electrons_dn):
        diffs = electrons_dn[..., None, :] - self.R
        if self.spin_restricted:
            return _swap_spin_updn(self._get_envelopes_up(electrons_dn), self.n_electrons, self.n_up)
        else:
            return self.envelope_dn(diffs)

    def __call__(self, electrons, embeddings, return_state=False):
        r_up, r_dn = jnp.split(electrons, [self.n_up], axis=-2)
        h_up, h_dn = jnp.split(embeddings, [self.n_up], axis=-2)
        orb_raw_up, orb_raw_dn = self._get_orbitals_up(h_up), self._get_orbitals_dn(h_dn)
        env_up, env_dn = self._get_envelopes_up(r_up), self._get_envelopes_dn(r_dn)
        orb_raw = jnp.concatenate([orb_raw_up, orb_raw_dn], axis=-2)
        env = jnp.concatenate([env_up, env_dn], axis=-2)
        orbitals = einops.rearrange(orb_raw * env, "el (det orb) -> det el orb", det=self.n_determinants)  # type: ignore
        if return_state:
            return orbitals, OrbitalState(env, orb_raw)
        return orbitals

    def low_rank_update(
        self,
        params: Parameters,
        electrons: Electrons,
        embeddings: ElectronEmb,
        idx_changed_el: ElectronIdx,
        idx_changed_h: ElectronIdx,
        state: OrbitalState,
    ):
        # TODO: One could do better here by only recomputing the spins that have changed (instead of both and selecting with where)
        # Requires introduction of new static variables (n_changed_hout_up, n_chainged_hout_dn)
        envelope_update = jnp.where(
            (idx_changed_el < self.n_up)[:, None],
            self.apply(params, electrons[idx_changed_el], method=self._get_envelopes_up),  # type: ignore
            self.apply(params, electrons[idx_changed_el], method=self._get_envelopes_dn),  # type: ignore
        )
        env = state.envelopes.at[idx_changed_el].set(envelope_update)

        orb_update = jnp.where(
            (idx_changed_h < self.n_up)[:, None],
            self.apply(params, embeddings[idx_changed_h], method=self._get_orbitals_up),  # type: ignore
            self.apply(params, embeddings[idx_changed_h], method=self._get_orbitals_dn),  # type: ignore
        )
        orb_raw = state.orbitals.at[idx_changed_h].set(orb_update)
        orbitals = einops.rearrange(orb_raw * env, "el (det orb) -> det el orb", det=self.n_determinants)
        return orbitals, OrbitalState(env, orb_raw)

    def fwd_lap(self, params, electrons: Electrons, embeddings):
        r_up, r_dn = jnp.split(electrons, [self.n_up], axis=-2)
        env_up = jax.vmap(fwd_lap(lambda r: self.apply(params, r, method=self._get_envelopes_up)), out_axes=-2)(r_up)
        env_dn = jax.vmap(fwd_lap(lambda r: self.apply(params, r, method=self._get_envelopes_dn)), out_axes=-2)(r_dn)
        envelopes = jtu.tree_map(lambda u, d: jnp.concatenate([u, d], axis=-2), env_up, env_dn)

        if isinstance(embeddings, FwdLaplArray):
            # Use folx to compute the raw orbitals
            h_up, h_dn = fwd_lap(lambda h: jnp.split(h, [self.n_up], axis=-2))(embeddings)
            vmap_over_elec = functools.partial(jax.vmap, in_axes=-2, out_axes=-2)
            orb_raw_up = vmap_over_elec(fwd_lap(lambda h: self.apply(params, h, method=self._get_orbitals_up)))(h_up)
            orb_raw_dn = vmap_over_elec(fwd_lap(lambda h: self.apply(params, h, method=self._get_orbitals_dn)))(h_dn)

            orbitals = jtu.tree_map(lambda u, d: jnp.concatenate([u, d], axis=-2), orb_raw_up, orb_raw_dn)
            orbitals = vmap_over_elec(
                fwd_lap(lambda o, e: jnp.reshape(o * e, [self.n_determinants, self.n_electrons])),
            )(orbitals, envelopes)
        else:
            orb_raw_up = self.apply(params, embeddings, method=self._get_orbitals_up)
            orb_raw_dn = self.apply(params, embeddings, method=self._get_orbitals_dn)
            orb_raw = merge_up_down(orb_raw_up, orb_raw_dn, self.n_electrons, self.n_up)
            orbitals = multiply_with_1el_fn(orb_raw, envelopes)
            phi = einops.rearrange(orbitals.x, "el (det orb) -> det el orb", det=self.n_determinants)
            lap = einops.rearrange(orbitals.lap, "el (det orb) -> det el orb", det=self.n_determinants)
            jac = einops.rearrange(orbitals.jac, "pair dim (det orb) -> det pair dim orb", det=self.n_determinants)
            orbitals = NodeWithFwdLap(phi, jac, lap, orbitals.idx_ctr, orbitals.idx_dep)
        return orbitals
