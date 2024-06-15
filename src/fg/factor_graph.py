from typing import Tuple, Dict

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .gaussian import Gaussian
from .factors import PoseFactor, DynamicsFactor, InterRobotFactor


def init_var2fac_neighbors(time_horizon: int) -> Dict:
    def create_dynamic_dims(carry, _):
        return carry + 2, carry

    _, dynamic_dims = jax.lax.scan(
        create_dynamic_dims, jnp.array([2,1]), length=time_horizon - 2
    )
    dynamic_dims = dynamic_dims.flatten()

    return {"dynamics": dynamic_dims}


def init_fac2var_neighbors(time_horizon: int) -> Dict:
    def create_dynamic_dims(carry, _):
        return carry + 2, carry

    _, dynamic_dims = jax.lax.scan(
        create_dynamic_dims, jnp.array([1,0]), length=time_horizon - 1
    )
    dynamic_dims = dynamic_dims.flatten()
    

    def create_factor_dims(carry, _):
        return carry + 1, carry

    _, factor_dims = jax.lax.scan(
        create_factor_dims, jnp.array([0, 0]), length=time_horizon - 1
    )
    factor_dims = factor_dims.flatten()

    def create_marginalize_order(carry, _):
        return carry + 1, carry

    _, marg_order = jax.lax.scan(
        create_marginalize_order,
        jnp.array([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
        length=time_horizon - 1,
    )
    marg_order = jnp.stack(jnp.split(marg_order.flatten(), (time_horizon - 1 ) * 2))

    return {"dynamics": dynamic_dims, "factors": factor_dims, "margs": marg_order}


@dataclass
class InterRobotVar2FacMessages:
    robot: Gaussian
    other_robot: Gaussian


@dataclass
class InterRobotFac2VarMessages:
    robot: Gaussian
    other_robot: Gaussian


@dataclass
class InterRobotFactors:
    robots: Gaussian


@dataclass
class Var2FacMessages:
    poses: Gaussian
    dynamics: Gaussian
    ir: Gaussian


@dataclass
class Fac2VarMessages:
    poses: Gaussian
    dynamics: Gaussian
    ir: Gaussian = None


@dataclass
class Factors:
    poses: Gaussian
    dynamics: Gaussian
    ir: Gaussian = None


class FactorGraph:
    def __init__(
        self,
        n_agents,
        time_horizon: int,
        agent_radius: float,
        crit_distance: float,
        target_states: jnp.array,
        delta_t: float,
    ) -> None:
        self._n_agents = n_agents
        self._time_horizon = time_horizon
        self._agent_radius = agent_radius
        self._crit_distance = crit_distance
        self._target_states = target_states
        self._delta_t = delta_t

        self._outer_idx = jnp.array([0, -1])
        self._var2fac_neighbors = init_var2fac_neighbors(time_horizon)
        self._fac2var_neighbors = init_fac2var_neighbors(time_horizon)

    def get_energy(self, state):
        def agent_energies(agent_state):

            def pose_energy(state):
                fac = PoseFactor(state, jnp.ones(4))
                hX = 0-fac._calc_measurement(state)
                prec = fac._calc_precision(state, fac._state_precision)
                return hX.T @ prec @ hX

            def dynamics_energy(current_state, next_state):
                state = jnp.concatenate((current_state, next_state))
                fac = DynamicsFactor(state, self._delta_t, jnp.ones(8))
                h1 = 0-fac._calc_measurement(state)
                h2 = 0-fac._calc_measurement(state)
                hX = jnp.concatenate((h1, h2))
                prec = fac._calc_precision(state, fac._state_precision)
                return hX.T @ prec @ hX
            
            pose = jax.vmap(pose_energy)(agent_state)
            dynamics = jax.vmap(dynamics_energy)(agent_state[:-1], agent_state[1:])

            return pose.sum() + dynamics.sum()
        
        return 0.5 * jax.vmap(agent_energies)(state)

    def run_inter_robot_gbp_init(
        self,
        states: jnp.ndarray,
        init_var2fac_msgs: InterRobotFac2VarMessages,
        time: jnp.ndarray,
    ) -> Dict:
        updated_var2fac_msgs = init_var2fac_msgs
        closest_robots = FactorGraph.find_closest_robot(states)
        updated_factor_likelihoods = self._update_inter_robot_factor_likelihoods(
            states, closest_robots, time
        )
        updated_fac2var_msgs = self._update_inter_robot_factor_to_var_messages(
            updated_var2fac_msgs, updated_factor_likelihoods
        )
        return {"var2fac": updated_var2fac_msgs, "fac2var": updated_fac2var_msgs}

    def run_inter_robot_gbp(
        self,
        states: jnp.ndarray,
        var2fac_msgs: Var2FacMessages,
        fac2var_msgs: Fac2VarMessages,
        time: jnp.ndarray,
    ) -> Dict:
        updated_var2fac_msgs = self._update_inter_robot_var_to_factor_messages(
            fac2var_msgs
        )
        closest_robots = FactorGraph.find_closest_robot(states)
        updated_factor_likelihoods = self._update_inter_robot_factor_likelihoods(
            states, closest_robots, time
        )
        updated_fac2var_msgs = self._update_inter_robot_factor_to_var_messages(
            var2fac_msgs, updated_factor_likelihoods
        )
        return {"var2fac": updated_var2fac_msgs, "fac2var": updated_fac2var_msgs}

    def run_gbp_init(
        self,
        states: jnp.ndarray,
        init_var2fac_msgs: Var2FacMessages,
        time: jnp.ndarray,
        # inter_robot_fac2var_msgs: InterRobotFac2VarMessages,
    ) -> Dict:
        updated_factor_likelihoods = self._update_factor_likelihoods(states)

        closest_robots = FactorGraph.find_closest_robot(states)
        ir_factor_likelihoods = self._update_inter_robot_factor_likelihoods(
            states, closest_robots, time
        )
        updated_fac2var_msgs = self._update_factor_to_var_messages(
            init_var2fac_msgs, updated_factor_likelihoods, self._fac2var_neighbors
        ).replace(ir=self._inter_robot_factor_to_var_messages(init_var2fac_msgs.ir, ir_factor_likelihoods))
        
        marginals = self._update_marginal_beliefs(
            updated_fac2var_msgs, # inter_robot_fac2var_msgs
        )
        return {
            "var2fac": init_var2fac_msgs,
            "fac2var": updated_fac2var_msgs,
            "marginals": marginals,
        }

    def run_gbp(
        self,
        states: jnp.ndarray,
        var2fac_msgs: Var2FacMessages,
        fac2var_msgs: Fac2VarMessages,
        time: jnp.ndarray,
        # inter_robot_fac2var_msgs: InterRobotFac2VarMessages,
    ) -> Dict:
        updated_var2fac_msgs = self._update_var_to_factor_messages(
            fac2var_msgs, self._var2fac_neighbors
        )
        updated_factor_likelihoods = self._update_factor_likelihoods(states)
        closest_robots = FactorGraph.find_closest_robot(states)
        ir_factor_likelihoods = self._update_inter_robot_factor_likelihoods(
            states, closest_robots, time
        )
        updated_fac2var_msgs = self._update_factor_to_var_messages(
            var2fac_msgs, updated_factor_likelihoods, self._fac2var_neighbors
        ).replace(ir=self._inter_robot_factor_to_var_messages(var2fac_msgs.ir, ir_factor_likelihoods))
        marginals = self._update_marginal_beliefs(
            updated_fac2var_msgs, # inter_robot_fac2var_msgs
        )
        return {
            "var2fac": updated_var2fac_msgs,
            "fac2var": updated_fac2var_msgs,
            "marginals": marginals,
        }

    @jax.vmap
    def _inter_robot_factor_to_var_messages(
            self,
            agent_var2fac_msgs: Var2FacMessages,
            agent_factors: Factors,
        ):
            robot_msgs = agent_var2fac_msgs # Message from closest robot at time t to robot i for index [i, t, ...]

            def multiply_gaussians(factor_likelihood, msg):
                return factor_likelihood * msg

            sum_product_fn = lambda g0, g1, marginal_order: multiply_gaussians(g0, g1).marginalize(marginal_order)

            updated_robot_marginalize_dims = jnp.full((self._time_horizon, 4), 100.0)

            updated_robot_msgs = jax.vmap(sum_product_fn)(
                agent_factors, robot_msgs, updated_robot_marginalize_dims
            )

            return updated_robot_msgs


    def _update_factor_to_var_messages(
        self,
        var2fac_msgs: Var2FacMessages,
        factors: Factors,
        var2fac_neighbors: Dict,
    ) -> Fac2VarMessages:
        def batched_update_factor_to_var_messages(
            agent_var2fac_msgs: Var2FacMessages, factors: Factors, neighbors: Dict
        ) -> Fac2VarMessages:
            """FACTORS class ARE LIKELIHOODS"""
            dynamics = agent_var2fac_msgs.dynamics[neighbors["dynamics"]]

            updated_poses = factors.poses
            f_likelihoods = factors.dynamics[neighbors["factors"]]
            marginalize_order = neighbors["margs"]
            

            def multiply_gaussians(g1, g2):
                return g1 * g2

            def fn(i):
                mult_result = multiply_gaussians(
                    f_likelihoods[i], dynamics[i]
                )
                marginal_result =  mult_result.marginalize(marginalize_order[i])
                # jax.debug.breakpoint()
                return marginal_result

            updated_dynamics = jax.vmap(fn)(jnp.arange(f_likelihoods.info.shape[0]))
            # jax.debug.breakpoint()

            return Fac2VarMessages(updated_poses, updated_dynamics)

        return jax.vmap(batched_update_factor_to_var_messages, in_axes=(0, 0, None))(
            var2fac_msgs,
            factors,
            var2fac_neighbors,
        )

    def _update_var_to_factor_messages(
        self, fac2var_msgs: Fac2VarMessages, fac2var_neighbors: Dict
    ) -> Var2FacMessages:
        def batched_update_var_to_factor_messages(
            agent_fac2var_msgs: Fac2VarMessages, neighbors: Dict
        ):
            poses = agent_fac2var_msgs.poses
            dynamics = agent_fac2var_msgs.dynamics
            ir = agent_fac2var_msgs.ir

            def multiply_fn(x, y):
                # jax.debug.breakpoint()
                # jax.debug.print("y: {}", y)
                x = x.replace(dims= jnp.full(4, 1))
                y = y.replace(dims= jnp.full(4, 100.0))
                # return x * y
                return Gaussian.identity(1.0)
                        
            updated_poses = jax.vmap(multiply_fn)(dynamics[self._outer_idx], ir[self._outer_idx])
            outer_dynamics = jax.vmap(multiply_fn)(poses, ir[self._outer_idx])
            inner_ir = jax.tree_util.tree_map(lambda x: jnp.repeat(x, 2, axis=0), ir[1:-1])
            inner_dynamics = jax.vmap(multiply_fn)(dynamics[neighbors["dynamics"]], inner_ir)
            updated_dynamics = jax.tree_util.tree_map(
                lambda x, y, z: jnp.concatenate((x, y, z)),
                outer_dynamics[0:1],
                inner_dynamics,
                outer_dynamics[1:],
            )

            outer_ir = jax.vmap(multiply_fn)(dynamics[self._outer_idx], poses)
            inner_ir = jax.vmap(multiply_fn)(dynamics[1:-1:2], dynamics[2:-1:2])

            updated_ir = jax.tree_util.tree_map(
                lambda x, y, z: jnp.concatenate((x, y, z)),
                outer_ir[0:1],
                inner_ir,
                outer_ir[1:],
            )
            return Var2FacMessages(updated_poses, updated_dynamics, updated_ir)

        return jax.vmap(batched_update_var_to_factor_messages, in_axes=(0, None))(
            fac2var_msgs, fac2var_neighbors
        )
    
    @staticmethod
    def find_closest_robot(states: jnp.ndarray):
        def find_closest_robot_across_horizon(robot, other_robots):
            closest_index = jnp.argmin(jnp.linalg.norm(robot[0:2] - other_robots[:,0:2], axis=1))
            return other_robots[closest_index]

        def find_batched_closest_robot(batch_states, i):
            modified_states = batch_states.at[i].set(jnp.inf)
            return jax.vmap(find_closest_robot_across_horizon, in_axes=(0, 1))(batch_states[i], modified_states)
        
        return jax.vmap(find_batched_closest_robot, in_axes=(None, 0))(states, jnp.arange(states.shape[0]))

    def _update_inter_robot_factor_likelihoods(
        self, states: jnp.ndarray, other_states: jnp.ndarray, time: jnp.ndarray
    ) -> InterRobotFactors:
        def batch_update_factor_likelihoods(
            agent_states: jnp.ndarray,
            other_agent_states: jnp.ndarray,
            time: jnp.ndarray,
        ):
            batch_calc_likelihoods = jax.vmap(
                lambda x, y: InterRobotFactor(
                    x, self._agent_radius, self._crit_distance, time, y
                ).calculate_likelihood()
            )
            dims = jnp.ones((self._time_horizon, 4)) * jnp.arange(
                1, self._time_horizon + 1
            ).reshape(-1, 1)
            other_dims = jnp.full((self._time_horizon, 4), 100.0)

            states_combos = jnp.hstack((agent_states, other_agent_states))
            inter_robot_likelihoods = batch_calc_likelihoods(
                states_combos, jnp.hstack((dims, other_dims))
            )
            return inter_robot_likelihoods

        return jax.vmap(batch_update_factor_likelihoods, in_axes=(0, 0, None))(
            states, other_states, time,
        )

    def _update_factor_likelihoods(self, states: jnp.array) -> Factors:
        def batch_update_factor_likelihoods(agent_states, end_pos):
            updated_vel = jnp.clip(-end_pos[0:2] - agent_states[0,0:2], -0.2, 0.2)
            start_state = jnp.hstack((agent_states[0,0:2], updated_vel))
            pose_combos = jnp.stack((start_state, -end_pos))  # [2,4]
            # pose_combos = jnp.stack((agent_states[0], -end_pos))  # [2,4]
            pose_dims = jnp.stack(
                [
                    jnp.ones(
                        4,
                    ),
                    jnp.ones(
                        4,
                    )
                    * (self._time_horizon),
                ]
            )
            poses = jax.vmap(lambda x, y: PoseFactor(x, y).calculate_likelihood())(
                pose_combos, pose_dims
            )  #
            dynamic_dims = jnp.arange(1, self._time_horizon).reshape(-1, 1) * jnp.ones(
                (self._time_horizon - 1, 4)
            )
            dynamic_dims = jnp.hstack((dynamic_dims, dynamic_dims + 1))
            dynamic_combos = jnp.hstack(
                (agent_states[0:-1], agent_states[1:])
            )  # [time_horizon - 1, 8]
            dynamics = jax.vmap(
                lambda x, y: DynamicsFactor(x, self._delta_t, y).calculate_likelihood()
            )(dynamic_combos, dynamic_dims)
            return Factors(poses, dynamics)

        return jax.vmap(batch_update_factor_likelihoods)(states, self._target_states)

    def _update_marginal_beliefs(
        self,
        fac2var_msgs: Fac2VarMessages,
        # inter_fac2var_msgs: InterRobotFac2VarMessages,
    ) -> Gaussian:
        def batched_update_marginal_beliefs(
            agent_fac2var_msgs: Fac2VarMessages, # agent_inter_fac2var_msgs
        ) -> Gaussian:
            pose = agent_fac2var_msgs.poses
            dynamics = agent_fac2var_msgs.dynamics

            outer_info = pose.info[self._outer_idx] + dynamics.info[self._outer_idx]
            outer_precision = (
                pose.precision[self._outer_idx] + dynamics.precision[self._outer_idx]
            )

            inner_info = dynamics.info[1:-1:2] + dynamics.info[2:-1:2]
            inner_precision = dynamics.precision[1:-1:2] + dynamics.precision[2:-1:2]

            # robot_info = jnp.zeros((4,4))
            # robot_info = robot_info.at[0].set(jnp.array([0, -0.5, 0, 0]))
            intra_info = jnp.concat((outer_info[0:1], inner_info, outer_info[-1:]))
            marginal_info = intra_info + agent_fac2var_msgs.ir.info
            intra_precision = jnp.concat(
                (outer_precision[0:1], inner_precision, outer_precision[-1:])
            )
            marginal_precision = intra_precision + agent_fac2var_msgs.ir.precision

            return Gaussian(
                info=marginal_info,
                precision=marginal_precision,
                dims=None,
            )

        return jax.vmap(batched_update_marginal_beliefs)(
            fac2var_msgs# , inter_fac2var_msgs
        )

    def init_inter_robot_var2fac_msgs(self) -> InterRobotVar2FacMessages:
        n_agents = self._n_agents
        time_horizon = self._time_horizon
        dummy = jnp.zeros((n_agents, time_horizon))
        robot_msgs = jax.vmap(jax.vmap(lambda _, var: Gaussian.identity(var)))(
            dummy,
            jnp.repeat(
                jnp.arange(1, time_horizon + 1, dtype=jnp.float32)[jnp.newaxis, :], n_agents, axis=0
            ),
        )
        return robot_msgs

    def init_var2fac_msgs(self) -> Var2FacMessages:
        pose_msgs = jax.vmap(jax.vmap(lambda _, var: Gaussian.identity(var)))(
            jnp.zeros((self._n_agents, 2)),
            jnp.repeat(
                jnp.array([[1., self._time_horizon]]),
                self._n_agents,
                axis=0,
            ),
        )

        def create_dynamics_axes(
            carry: jnp.ndarray, _: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            return carry + 1, carry

        _, dynamics_axes = jax.lax.scan(
            create_dynamics_axes,
            jnp.array([2., 2.]),
            length=self._time_horizon - 2,
        )
        dynamics_axes = jnp.concat(
            (
                jnp.array([[1.]]),
                dynamics_axes.reshape((1, -1)),
                jnp.array([[self._time_horizon]]),
            ),
            axis=1,
        )
        dynamics_msgs = jax.vmap(jax.vmap(lambda _, var: Gaussian.identity(var)))(
            jnp.zeros((self._n_agents, (self._time_horizon - 1) * 2)),
            jnp.repeat(dynamics_axes, self._n_agents, axis=0),
        )
        return Var2FacMessages(poses=pose_msgs, dynamics=dynamics_msgs, ir=self.init_inter_robot_var2fac_msgs())

    @property
    def states(self):
        return self._states

    def _tree_flatten(self):
        children = (self._target_states,)
        aux_data = {
            "n_agents": self._n_agents,
            "time_horizon": self._time_horizon,
            "delta_t": self._delta_t,
            "agent_radius": self._agent_radius,
            "crit_distance": self._crit_distance,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(
            aux_data["n_agents"],
            aux_data["time_horizon"],
            aux_data["agent_radius"],
            aux_data["crit_distance"],
            children[0],
            aux_data["delta_t"],
        )


jax.tree_util.register_pytree_node(
    FactorGraph, FactorGraph._tree_flatten, FactorGraph._tree_unflatten
)
