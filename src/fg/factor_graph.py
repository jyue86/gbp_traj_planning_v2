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
    def __init__(self, states: jnp.array, target_states: jnp.array, delta_t: float) -> None:
        self._states = states
        self._target_states = target_states
        self._delta_t = delta_t

        self._outer_idx = jnp.array([0, -1])
        # self.factor_to_var_msgs = None
        # self.var_to_factor_msgs = None
        # self.pose_likelihoods = None
        # self.dynamic_likelihoods = None

        # init variable to factor messages first

    def run_gbp(current_states: jnp.array, steps: int=1) -> None:
        """
        1. Create new factor graph using current_states (and end_pos for pose factors)
        2. variable to factor messages set to uninformative
        """
        pass

    def _update_factor_to_var_messages(self, var2fac_msgs: Var2FacMessages) -> Fac2VarMessages:
        return Fac2VarMessages()

    def _update_var_to_factor_messages(self, fac2var_msgs: Fac2VarMessages) -> Var2FacMessages:
        def batched_update_var_to_factor_messages():
            return Var2FacMessages()
        return jax.vmap(batched_update_var_to_factor_messages)(fac2var_msgs)

    def _update_factor_likelihoods(self, states: jnp.array) -> Factors:
        # shapes check out when run
        def batch_update_factor_likelihoods(agent_states, end_pos):
            pose_combos = jnp.stack((agent_states[0], end_pos)) # [2,4]
            poses = jax.vmap(lambda x: PoseFactor(x).calculate_likelihood())(pose_combos) # 
            dynamic_combos = jnp.hstack((agent_states[0:-1], agent_states[1:])) # [time_horizon - 1, 8]
            dynamics = jax.vmap(lambda x: DynamicsFactor(x, self._delta_t).calculate_likelihood())(dynamic_combos)
            return Factors(poses, dynamics)
        return jax.vmap(batch_update_factor_likelihoods)(states, self._target_states)

    def _update_marginal_beliefs(self, fac2var_msgs: Fac2VarMessages) -> Gaussian:
        # works in notebook
        def batched_update_marginal_beliefs(agent_fac2var_msgs: Fac2VarMessages) -> Gaussian:
            pose = agent_fac2var_msgs.pose
            dynamics = agent_fac2var_msgs.dynamics

            outer_info = pose.info[self._outer_idx] + dynamics.info[self._outer_idx]
            outer_precision = pose.precision[self._outer_idx] + dynamics.precision[self._outer_idx]

            inner_info = dynamics.info[1:-1:2] + dynamics.info[2:-1:2]
            inner_precision = dynamics.precision[1:-1:2] + dynamics.precision[2:-1:2]

            return Gaussian(
                info=jnp.concat((outer_info[0:1], inner_info, outer_info[-1:])),
                precision=jnp.concat((outer_precision[0:1], inner_precision, outer_precision[-1:])),
            )
        return jax.vmap(batched_update_marginal_beliefs)(fac2var_msgs)

    def _init_var2fac_msgs(self) -> Var2FacMessages:
        n_agents = self._states.shape[0]
        time_horizon = self._states.shape[1]
        pose_msgs = jax.vmap(jax.vmap(lambda _: Gaussian.identity()))(
            jnp.zeros((n_agents, 2))
        )
        dynamics_msgs = jax.vmap(jax.vmap(lambda _: Gaussian.identity()))(
            jnp.zeros((n_agents, (time_horizon - 1) * 2))
        )
        return Var2FacMessages(pose=pose_msgs, dynamics=dynamics_msgs)

    @staticmethod
    def from_graph(other: "FactorGraph") -> "FactorGraph":
        pass
