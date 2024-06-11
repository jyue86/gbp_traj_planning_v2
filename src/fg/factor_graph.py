from typing import Tuple, Dict

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .gaussian import Gaussian
from .factors import PoseFactor, DynamicsFactor


def init_var2fac_neighbors(iters: int) -> Dict:
    def create_dynamic_dims(carry, _):
        return carry + 2, carry
    _, dynamic_dims = jax.lax.scan(create_dynamic_dims, jnp.array([2,1]), length=iters)
    dynamic_dims = dynamic_dims.flatten()

    return {
        "dynamics": dynamic_dims
    }

def init_fac2var_neighbors(iters: int) -> Dict:
    def create_dynamic_dims(carry, _):
        return carry + 2, carry
    _, dynamic_dims = jax.lax.scan(create_dynamic_dims, jnp.array([2,1]), length=iters)
    dynamic_dims = dynamic_dims.flatten()

    def create_factor_dims(carry, _):
        return carry + 1, carry
    _, factor_dims = jax.lax.scan(create_factor_dims, jnp.array([0,0]), length=iters)
    factor_dims = factor_dims.flatten()

    return {
        "dynamics": dynamic_dims,
        "factors": factor_dims
    }

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
        n_agents,
        time_horizon: int,
        target_states: jnp.array,
        delta_t: float,
    ) -> None:
        self._n_agents = n_agents
        self._time_horizon = time_horizon
        self._target_states = target_states
        self._delta_t = delta_t

        self._outer_idx = jnp.array([0, -1])
        self._var2fac_neighbors = init_var2fac_neighbors(time_horizon - 2)
        self._fac2var_neighbors = init_fac2var_neighbors(time_horizon - 1)

    def run_gbp(self, states: jnp.ndarray, var2fac_msgs: Var2FacMessages, fac2var_msgs: Fac2VarMessages=None, init: bool=False) -> Dict:
        if init:
            updated_var2fac_msgs = var2fac_msgs
        else:
            updated_var2fac_msgs = self._update_var_to_factor_messages(fac2var_msgs, self._var2fac_neighbors)
        # updated_factor_likelihoods = self._update_factor_likelihoods(states)
        # updated_fac2var_msgs = self._update_factor_to_var_messages(var2fac_msgs, updated_factor_likelihoods, self._fac2var_neighbors)
        updated_fac2var_msgs = None
        marginals = self._update_marginal_beliefs(updated_var2fac_msgs)
        return {
            "var2fac": updated_var2fac_msgs,
            "fac2var": updated_fac2var_msgs,
            "marginals": marginals
        }

    def _update_factor_to_var_messages(
        self,
        var2fac_msgs: Var2FacMessages,
        factors: Factors,
        var2fac_neighbors: Dict,
    ) -> Fac2VarMessages:
        def batched_update_factor_to_var_messages(agent_var2fac_msgs: Var2FacMessages, factors: Factors, neighbors: Dict) -> Fac2VarMessages:
            # poses = agent_var2fac_msgs.poses
            dynamics = agent_var2fac_msgs.dynamics

            updated_poses = factors.poses 
            f_likelihoods = factors.dynamics[neighbors["factors"]]
            jax.debug.print("likelihood shape: {}", f_likelihoods.shape)
            # updated_dynamics = dynamics[neighbors["dynamics"]]

            # updated_dynamics = jax.tree_util.tree_map(lambda x, y: x * y, f_likelihoods, dynamics[neighbors["dynamics"]])
            def multiply_gaussians(g1, g2):
                return g1 * g2
            updated_dynamics = jax.vmap(multiply_gaussians)(f_likelihoods, dynamics[neighbors["dynamics"]])

            return Fac2VarMessages(updated_poses, updated_dynamics)
        return jax.vmap(batched_update_factor_to_var_messages, in_axes=(0, 0, None))(var2fac_msgs, factors, var2fac_neighbors)

    def _update_var_to_factor_messages(
        self,
        fac2var_msgs: Fac2VarMessages,
        fac2var_neighbors: Dict
    ) -> Var2FacMessages:
        def batched_update_var_to_factor_messages(agent_fac2var_msgs: Fac2VarMessages, neighbors: Dict):
            poses = agent_fac2var_msgs.poses
            dynamics = agent_fac2var_msgs.dynamics

            updated_poses = dynamics[self._outer_idx]
            outer_dynamics = poses 
            inner_dynamics = dynamics[neighbors["dynamics"]]
            updated_dynamics = jax.tree_util.tree_map(lambda x, y, z: jnp.concatenate((x, y, z)), outer_dynamics[0:1], inner_dynamics, outer_dynamics[1:])
            return Var2FacMessages(updated_poses, updated_dynamics)

        return jax.vmap(batched_update_var_to_factor_messages, in_axes=(0, None))(fac2var_msgs, fac2var_neighbors)

    def _update_factor_likelihoods(self, states: jnp.array) -> Factors:
        def batch_update_factor_likelihoods(agent_states, end_pos):
            pose_combos = jnp.stack((agent_states[0], end_pos))  # [2,4]
            pose_dims = jnp.stack([jnp.zeros(4,), jnp.ones(4,) * (self._time_horizon - 1)])
            poses = jax.vmap(lambda x, y: PoseFactor(x, y).calculate_likelihood())(
                pose_combos, pose_dims
            )  #
            dynamic_dims = jnp.arange(0, self._time_horizon - 1).reshape(-1,1) * jnp.ones((self._time_horizon - 1, 4))
            dynamic_dims = jnp.hstack((dynamic_dims, dynamic_dims + 1))
            dynamic_combos = jnp.hstack(
                (agent_states[0:-1], agent_states[1:])
            )  # [time_horizon - 1, 8]
            dynamics = jax.vmap(
                lambda x, y: DynamicsFactor(x, self._delta_t, y).calculate_likelihood()
            )(dynamic_combos, dynamic_dims)
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
                dims=jnp.zeros(0,)
            )

        return jax.vmap(batched_update_marginal_beliefs)(fac2var_msgs)

    def init_var2fac_msgs(self) -> Var2FacMessages:
        pose_msgs = jax.vmap(jax.vmap(lambda _, var: Gaussian.identity(var)))(
            jnp.zeros((self._n_agents, 2)), jnp.repeat(jnp.array([[0, self._time_horizon - 1]]), self._n_agents, axis=0)
        )
        def create_dynamics_axes(carry: jnp.ndarray, _: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            return carry + 1, carry
        _, dynamics_axes = jax.lax.scan(create_dynamics_axes, jnp.array([1,1]), length=self._time_horizon-2) 
        dynamics_axes = jnp.concat((jnp.array([[0]]), dynamics_axes.reshape((1, -1)),jnp.array([[self._time_horizon - 1]])), axis=1)
        dynamics_msgs = jax.vmap(jax.vmap(lambda _, var: Gaussian.identity(var)))(
            jnp.zeros((self._n_agents, (self._time_horizon - 1) * 2)), jnp.repeat(dynamics_axes, self._n_agents, axis=0)
        )
        return Var2FacMessages(poses=pose_msgs, dynamics=dynamics_msgs)
    
    @property
    def states(self):
        return self._states
