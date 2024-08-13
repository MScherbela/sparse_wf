import flax.linen as nn
from sparse_wf.model.sparse_fwd_lap import Linear
from sparse_wf.model.envelopes import EfficientIsotropicEnvelopes, GLUEnvelopes
import jax.numpy as jnp
from sparse_wf.api import Charges, Electrons, ElectronEmb, EnvelopeArgs, Parameters, ElectronIdx, Nuclei
import jax
from typing import NamedTuple
import einops


class OrbitalState(NamedTuple):
    envelopes: jax.Array
    orbitals: jax.Array


def _swap_spin_updn(phi, n_electrons, n_up):
    # Input shape: [... x (dets*orbitals) ]"
    phi = phi.reshape([*phi.shape[-1], -1, n_electrons])  # split off det axis
    phi = jnp.concatenate([phi[..., n_up:], phi[..., :n_up]], axis=-1)  # swap spin up and down
    return phi.reshape([*phi.shape[:-2], -1])  # flatten det axis


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
            self.n_orbitals * self.n_determinants,
            bias_init=nn.initializers.truncated_normal(0.01, jnp.float32),
            name="to_orbitals_up",
        )
        if not self.spin_restricted:
            self.to_orbitals_dn = Linear(
                self.n_orbitals * self.n_determinants,
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
            return _swap_spin_updn(self._get_orbitals_up(embeddings_dn), self.n_electrons, self.n_up)
        else:
            return self.to_orbitals_dn(embeddings_dn)

    def _get_envelopes_up(self, electrons_up):
        diffs = electrons_up[:, None, :] - self.R[None, :, :]
        return self.envelope_up(diffs)

    def _get_envelopes_dn(self, electrons_dn):
        diffs = electrons_dn[:, None, :] - self.R[None, :, :]
        if self.spin_restricted:
            return _swap_spin_updn(self._get_envelopes_up(electrons_dn), self.n_electrons, self.n_up)
        else:
            return self.envelope_dn(diffs)

    def __call__(self, electrons, embeddings, return_state=False):
        r_up, r_dn = jnp.split(electrons, [self.n_up], axis=-2)
        h_up, h_dn = jnp.split(embeddings, [self.n_up], axis=-2)
        h_dn = embeddings[..., self.n_up :, :]
        orb_raw_up, orb_raw_dn = self._get_orbitals_up(h_up), self._get_orbitals_dn(h_dn)
        env_up, env_dn = self._get_envelopes_up(r_up), self._get_envelopes_dn(r_dn)
        orb_raw = jnp.concatenate([orb_raw_up, orb_raw_dn], axis=-1)
        env = jnp.concatenate([env_up, env_dn], axis=-1)
        orbitals = einops.rearrange(orb_raw * env, "el (det orb) -> det el orb", det=self.n_determinants)  # type: ignore
        if return_state:
            return orbitals, OrbitalState(env, orb_raw)

    def low_rank_update(
        self,
        params: Parameters,
        electrons: Electrons,
        embeddings: ElectronEmb,
        idx_changed_el: ElectronIdx,
        idx_changed_h: ElectronIdx,
        state: OrbitalState,
    ):
        # One could do better here by only recomputing the spins that have changed (instead of both and selecting with where)
        envelope_update = jnp.where(
            idx_changed_el < self.n_up,
            self.apply(params, electrons[idx_changed_el], method=self._get_envelopes_dn),  # type: ignore
            self.apply(params, electrons[idx_changed_el], method=self._get_envelopes_up),  # type: ignore
        )
        env = state.envelopes.at[idx_changed_el].set(envelope_update)

        orb_update = jnp.where(
            idx_changed_h < self.n_up,
            self.apply(params, embeddings[idx_changed_h], method=self._get_orbitals_up),  # type: ignore
            self.apply(params, embeddings[idx_changed_h], method=self._get_orbitals_dn),  # type: ignore
        )
        orb_raw = state.orbitals.at[idx_changed_h].set(orb_update)
        orbitals = einops.rearrange(orb_raw * env, "el (det orb) -> det el orb", det=self.n_determinants)
        return orbitals, OrbitalState(env, orb_raw)

    # def fwd_lap(self, params, electrons: Electrons, embeddings, static: S):
    #     embeddings, dependencies = self.embedding.apply_with_fwd_lap(params.embedding, electrons, static)
    #     orbitals = self._orbitals_with_fwd_lap_folx(params, electrons, embeddings)

    #     # vmap over determinants
    #     signs, logdets = jax.vmap(lambda o: slogdet_with_sparse_fwd_lap(o, dependencies), in_axes=-3, out_axes=-1)(
    #         orbitals
    #     )
    #     logpsi = fwd_lap(lambda logdets_: jax.nn.logsumexp(logdets_, b=signs, return_sign=True)[0])(logdets)
    #     logpsi_jastrow = self.jastrow.apply_with_fwd_lap(params.jastrow, electrons, embeddings, dependencies)
    #     logpsi = tree_add(logpsi, logpsi_jastrow)
    #     return logpsi

    # def fwd_lap_sparse()
