from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .gaussian import Gaussian
from .factors import PoseFactor, DynamicsFactor


@dataclass
class Var2FacMessages:
    poses: jnp.array
    dynamics: jnp.array


@dataclass
class Fac2VarMessages:
    poses: jnp.array
    dynamics: jnp.array


@dataclass
class Factors:
    poses: Gaussian
    dynamics: Gaussian


class FactorGraph:
    def __init__(
        self,
        states: jnp.array,
        target_states: jnp.array,
        delta_t: float,
        var2fac_msgs: Var2FacMessages = None,
        fac2var_msgs: Fac2VarMessages = None,
    ) -> None:
        self._states = states
        self._target_states = target_states
        self._delta_t = delta_t

        self._outer_idx = jnp.array([0, -1])
        self.var2fac_msgs = var2fac_msgs
        self.fac2var_msgs = fac2var_msgs

    def run_gbp(self, steps: int = 1) -> Tuple:
        pass
        # if not self.var2fac_msgs:
        #     updated_var2fac_msgs = self._init_var2fac_msgs()
        # else:
        #     updated_var2fac_msgs = self._update_var_to_factor_messages(
        #         self.fac2var_msgs
        #     )
        # updated_factor_likelihoods = self._update_factor_likelihoods(self._states)
        # updated_fac2var_msgs = self._update_factor_to_var_messages(self.var2fac_msgs)
        # marginals = self._update_marginal_beliefs(updated_fac2var_msgs)
        # return FactorGraph(
        #     marginals.info,
        #     self._target_states,
        #     self._delta_t,
        #     updated_var2fac_msgs,
        #     updated_fac2var_msgs,
        # )

    def _update_factor_to_var_messages(
        self, var2fac_msgs: Var2FacMessages
    ) -> Fac2VarMessages:
        return Fac2VarMessages(None, None)

    def _update_var_to_factor_messages(
        self, fac2var_msgs: Fac2VarMessages
    ) -> Var2FacMessages:
        def batched_update_var_to_factor_messages(agent_fac2var_msgs: Fac2VarMessages):
            fac2var_pose_msgs = fac2var_msgs.poses
            fac2var_dynamic_msgs = fac2var_msgs.dynamics

            poses = fac2var_dynamic_msgs[self._outer_idx]
            return Var2FacMessages(poses)

        return jax.vmap(batched_update_var_to_factor_messages)(fac2var_msgs)

    def _update_factor_likelihoods(self, states: jnp.array) -> Factors:
        # shapes check out when run
        def batch_update_factor_likelihoods(agent_states, end_pos):
            pose_combos = jnp.stack((agent_states[0], end_pos))  # [2,4]
            poses = jax.vmap(lambda x: PoseFactor(x).calculate_likelihood())(
                pose_combos
            )  #
            dynamic_combos = jnp.hstack(
                (agent_states[0:-1], agent_states[1:])
            )  # [time_horizon - 1, 8]
            dynamics = jax.vmap(
                lambda x: DynamicsFactor(x, self._delta_t).calculate_likelihood()
            )(dynamic_combos)
            return Factors(poses, dynamics)

        return jax.vmap(batch_update_factor_likelihoods)(states, self._target_states)

    def _update_marginal_beliefs(self, fac2var_msgs: Fac2VarMessages) -> Gaussian:
        # works in notebook
        def batched_update_marginal_beliefs(
            agent_fac2var_msgs: Fac2VarMessages,
        ) -> Gaussian:
            pose = agent_fac2var_msgs.poses
            dynamics = agent_fac2var_msgs.dynamics

            outer_info = pose.info[self._outer_idx] + dynamics.info[self._outer_idx]
            outer_precision = (
                pose.precision[self._outer_idx] + dynamics.precision[self._outer_idx]
            )

            inner_info = dynamics.info[1:-1:2] + dynamics.info[2:-1:2]
            inner_precision = dynamics.precision[1:-1:2] + dynamics.precision[2:-1:2]

            return Gaussian(
                info=jnp.concat((outer_info[0:1], inner_info, outer_info[-1:])),
                precision=jnp.concat(
                    (outer_precision[0:1], inner_precision, outer_precision[-1:])
                ),
                dims=None
            )

        return jax.vmap(batched_update_marginal_beliefs)(fac2var_msgs)

    def _init_var2fac_msgs(self) -> Var2FacMessages:
        n_agents = self._states.shape[0]
        time_horizon = self._states.shape[1]
        pose_msgs = jax.vmap(jax.vmap(lambda _: Gaussian.identity()))(
            jnp.zeros((n_agents, 2)), jnp.array([[0, time_horizon - 1]], 2, axis=0)
        )
        def create_dynamics_axes(carry: jnp.ndarray, _: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            return carry + 1, carry
        _, dynamics_axes = jax.lax.scan(create_dynamics_axes, jnp.array([1,1]), length=time_horizon-2) 
        dynamics_axes = jnp.concat((jnp.array([[0]]), dynamics_axes.reshape((1, -1)),jnp.array([[time_horizon - 1]])), axis=1)
        dynamics_msgs = jax.vmap(jax.vmap(lambda _: Gaussian.identity()))(
            jnp.zeros((n_agents, (time_horizon - 1) * 2)), jnp.repeat(dynamics_axes, n_agents, axis=0)
        )
        return Var2FacMessages(poses=pose_msgs, dynamics=dynamics_msgs)
    
    @property
    def states(self):
        return self._states

    @staticmethod
    def from_graph(other: "FactorGraph") -> "FactorGraph":
        pass
