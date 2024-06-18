from typing import Tuple, Dict

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .gaussian import Gaussian
from .factors import (
    PoseFactor,
    DynamicsFactor, 
    InterRobotFactor, 
    ObstacleFactor
)


def init_var2fac_neighbors(time_horizon: int) -> Dict:
    def create_dynamic_dims(carry, _):
        return carry + 2, carry

    _, dynamic_dims = jax.lax.scan(
        create_dynamic_dims, jnp.array([2,1]), None, length=time_horizon - 2
    )
    dynamic_dims = dynamic_dims.flatten()

    return {"dynamics": dynamic_dims}


def init_fac2var_neighbors(time_horizon: int) -> Dict:
    def create_dynamic_dims(carry, _):
        return carry + 2, carry

    _, dynamic_dims = jax.lax.scan(
        create_dynamic_dims, jnp.array([1,0]), None, length=time_horizon - 1
    )
    dynamic_dims = dynamic_dims.flatten()
    

    def create_factor_dims(carry, _):
        return carry + 1, carry

    _, factor_dims = jax.lax.scan(
        create_factor_dims, jnp.array([0, 0]), None, length=time_horizon - 1
    )
    factor_dims = factor_dims.flatten()

    def create_marginalize_order(carry, _):
        return carry + 1, carry

    _, marg_order = jax.lax.scan(
        create_marginalize_order,
        jnp.array([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
        None,
        length=time_horizon - 1,
    )
    marg_order = jnp.stack(jnp.split(marg_order.flatten(), (time_horizon - 1 ) * 2))

    return {"dynamics": dynamic_dims, "factors": factor_dims, "margs": marg_order}

@dataclass
class Var2FacMessages:
    poses: Gaussian
    dynamics: Gaussian
    obstacle: Gaussian
    ir: Gaussian


@dataclass
class Fac2VarMessages:
    poses: Gaussian
    dynamics: Gaussian
    obstacle: Gaussian
    ir: Gaussian = None


@dataclass
class Factors:
    poses: Gaussian
    dynamics: Gaussian
    obstacle: Gaussian
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
        obstacles
    ) -> None:
        self._n_agents = n_agents
        self._time_horizon = time_horizon
        self._agent_radius = agent_radius
        self._crit_distance = crit_distance
        self._target_states = target_states
        self._delta_t = delta_t

        self._obstacles = obstacles

        self._outer_idx = jnp.array([0, -1])
        self._var2fac_neighbors = init_var2fac_neighbors(time_horizon)
        self._fac2var_neighbors = init_fac2var_neighbors(time_horizon)

    # def get_energy(self, state):

        # def agent_energies(self, state):
        # def agent_energies(agent_state, idx, state):
            
        #     def pose_energy(state):
        #         fac = PoseFactor(state, jnp.ones(4))
        #         hX = state -fac._calc_measurement(state)
        #         return hX.T @ fac._state_precision @ hX

        #     def dynamics_energy(current_state, next_state):
        #         state = jnp.concatenate((current_state, next_state))
        #         fac = DynamicsFactor(state, self._delta_t, jnp.ones(8))
        #         hX = current_state-fac._calc_measurement(state)
        #         return hX.T @ fac._state_precision @ hX
            
        #     closest = FactorGraph.find_closest_robot(state)[idx]
            
        #     def ir_energy(ind_state, closest, state):
        #         both_state = jnp.concatenate((ind_state, closest))
        #         fac = InterRobotFactor(both_state, self._crit_distance, 1, jnp.ones(4))
        #         hX = ind_state - fac._calc_measurement(both_state)
                
        #         return hX.T @ fac._energy_precision @ hX
            
        #     closest_obstacle = FactorGraph.find_closest_obstacle(state, self._obstacles)[idx]

        #     def obstacle_energy(ind_state, closest):
        #         fac = ObstacleFactor(ind_state, closest, self._crit_distance, self._agent_radius, jnp.ones(4))
        #         hX = ind_state - fac._calc_measurement(ind_state)
        #         # prec = fac._calc_precision(fac._state_precision)
        #         return hX.T @ fac._energy_precision @ hX
            
        #     pose = jax.vmap(pose_energy)(agent_state)
        #     dynamics = jax.vmap(dynamics_energy)(agent_state[:-1], agent_state[1:])
        #     ir = jax.vmap(ir_energy, in_axes=(0, 0, None))(agent_state, closest, state)
        #     obstacle = jax.vmap(obstacle_energy)(agent_state, closest_obstacle)

        #     return pose.sum() + dynamics.sum() + ir.sum() + obstacle.sum()
        
        # return 0.5 * jax.vmap(agent_energies, in_axes=(0, 0, None))(state, jnp.arange(state.shape[0]), state)

    def get_energy(self, marginals, state):
        
        def agent_energy(marginals, state):
            
            def time_energy(marginal, state):
                return marginal(state)

            return jax.vmap(time_energy)(marginals, state).sum()
        
        return jax.vmap(agent_energy)(marginals, state)

    def run_inter_robot_gbp_init(
        self,
        states: jnp.ndarray,
        init_var2fac_msgs: Gaussian,
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
            updated_var2fac_msgs, updated_factor_likelihoods, self._fac2var_neighbors
        ).replace(ir=self._inter_robot_factor_to_var_messages(var2fac_msgs.ir, ir_factor_likelihoods))

        BETA = 1
        damp_fn = lambda x, y: BETA * x + (1 - BETA) * y

        updated_fac2var_msgs = jax.tree_map(damp_fn, updated_fac2var_msgs, fac2var_msgs)
        
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
                return factor_likelihood.mul(msg, None)
            
            def sum_product_fn(g0, g1, marginal_order):
                # jax.debug.print("factor likelihoods: {}", g0)
                mult_result = multiply_gaussians(g0, g1)
                # jax.debug.print("mult result: {}", mult_result)
                marginalize_result = mult_result.marginalize(marginal_order)
                # jax.debug.breakpoint()
                # jax.debug.print("marginalized result: {}", marginalize_result.info)
                # jax.debug.print("marginalized result's mean: {}", marginalize_result.mean)
                return marginalize_result

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
            updated_obstacle = factors.obstacle
            f_likelihoods = factors.dynamics[neighbors["factors"]]
            marginalize_order = neighbors["margs"]

            def multiply_gaussians(g1, g2):
                return g1.mul(g2, None)

            def fn(i):
                mult_result = multiply_gaussians(
                    f_likelihoods[i], dynamics[i]
                )
                marginal_result =  mult_result.marginalize(marginalize_order[i])
                return marginal_result

            updated_dynamics = jax.vmap(fn)(jnp.arange(f_likelihoods.info.shape[0]))
            # jax.debug.breakpoint()

            return Fac2VarMessages(updated_poses, updated_dynamics, updated_obstacle)

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
            obstacle = agent_fac2var_msgs.obstacle
            ir = agent_fac2var_msgs.ir
            # jax.debug.breakpoint()
            
            def multiply_combine(g1, g2):
                return g1.mul(g2, False)
                        
            updated_poses = jax.vmap(multiply_combine)(dynamics[self._outer_idx], ir[self._outer_idx])
            updated_poses = jax.vmap(multiply_combine)(updated_poses, obstacle[self._outer_idx])
            
            outer_dynamics = jax.vmap(multiply_combine)(poses, ir[self._outer_idx])
            outer_dynamics = jax.vmap(multiply_combine)(outer_dynamics, obstacle[self._outer_idx])

            repeat_fn = lambda x: jnp.repeat(x, 2, axis=0)  

            inner_ir = jax.tree_util.tree_map(repeat_fn, ir[1:-1])
            inner_dynamics = jax.vmap(multiply_combine)(dynamics[neighbors["dynamics"]], inner_ir)
            inner_obstacle = jax.tree_util.tree_map(repeat_fn, obstacle[1:-1])
            inner_dynamics = jax.vmap(multiply_combine)(inner_dynamics, inner_obstacle)

            updated_dynamics = jax.tree_util.tree_map(
                lambda x, y, z: jnp.concatenate((x, y, z)),
                outer_dynamics[0:1],
                inner_dynamics,
                outer_dynamics[1:],
            )

            outer_obstacle = jax.vmap(multiply_combine)(dynamics[self._outer_idx], ir[self._outer_idx])
            outer_obstacle = jax.vmap(multiply_combine)(outer_obstacle, poses[self._outer_idx])
            
            inner_obstacle = jax.vmap(multiply_combine)(dynamics[1:-1:2], dynamics[2:-1:2])
            inner_obstacle = jax.vmap(multiply_combine)(inner_obstacle, ir[1:-1])

            updated_obstacle = jax.tree_util.tree_map(
                lambda x, y, z: jnp.concatenate((x, y, z)),
                outer_obstacle[0:1],
                inner_obstacle,
                outer_obstacle[1:],
            )

            outer_ir = jax.vmap(multiply_combine)(dynamics[self._outer_idx], poses)
            outer_ir = jax.vmap(multiply_combine)(outer_ir, obstacle[self._outer_idx])

            inner_ir = jax.vmap(multiply_combine)(dynamics[1:-1:2], dynamics[2:-1:2])
            inner_ir = jax.vmap(multiply_combine)(inner_ir, obstacle[1:-1])

            updated_ir = jax.tree_util.tree_map(
                lambda x, y, z: jnp.concatenate((x, y, z)),
                outer_ir[0:1],
                inner_ir,
                outer_ir[1:],
            )
            
            return Var2FacMessages(updated_poses, updated_dynamics, updated_obstacle, updated_ir)

        return jax.vmap(batched_update_var_to_factor_messages, in_axes=(0, None))(
            fac2var_msgs, fac2var_neighbors
        )
    
    @staticmethod
    def find_closest_robot(states: jnp.ndarray):
        def find_closest_robot_across_horizon(robot, other_robots):
            closest_index = jnp.argmin(jnp.linalg.norm(robot[0:2] - other_robots[:,0:2], axis=1))
            return other_robots[closest_index]

        def find_batched_closest_robot(batch_states, i):
            modified_states = batch_states.at[i].set(10000)
            return jax.vmap(find_closest_robot_across_horizon, in_axes=(0, 1))(batch_states[i], modified_states)
        
        return jax.vmap(find_batched_closest_robot, in_axes=(None, 0))(states, jnp.arange(states.shape[0]))
    
    @staticmethod
    def find_closest_obstacle(states: jnp.ndarray, obstacles: jnp.ndarray) -> jnp.ndarray:
        """
        obstacles is Nx2 array where N is the number of obstacles

        Returns a AxKx2 array where A is the number of agents, K is the number of time horizon
        """
        def find_closest_obstacle_for_pt(state_t, obstacles):
            closest_obstacle_idx = jnp.argmin(jnp.linalg.norm(state_t[0:2] - obstacles, axis=1))
            return obstacles[closest_obstacle_idx]
        def batch_find_closest_obstacle(agent_states, obstacles):
            return jax.vmap(find_closest_obstacle_for_pt, in_axes=(0, None))(agent_states, obstacles)
        return jax.vmap(batch_find_closest_obstacle, in_axes=(0, None))(states, obstacles)

    def _update_inter_robot_factor_likelihoods(
        self, states: jnp.ndarray, other_states: jnp.ndarray, time: jnp.ndarray
    ) -> Gaussian:
        def batch_update_factor_likelihoods(
            agent_states: jnp.ndarray,
            other_agent_states: jnp.ndarray,
            time: jnp.ndarray,
        ):
            batch_calc_likelihoods = jax.vmap(
                lambda x, y: InterRobotFactor(
                    x, self._crit_distance, time, y
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
        def batch_update_factor_likelihoods(agent_states, end_pos, closest_obstacle):
            updated_vel = jnp.clip(-end_pos[0:2] - agent_states[0,0:2], -0.5, 0.5)
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
            obstacle = jax.vmap(
                lambda x, y, idx: ObstacleFactor(x, y, self._crit_distance, self._agent_radius, jnp.full(4, idx, dtype=jnp.float32)).calculate_likelihood()
            )(agent_states, closest_obstacle, jnp.arange(1, self._time_horizon+1, dtype=jnp.float32))

            return Factors(poses, dynamics, obstacle)
        return jax.vmap(batch_update_factor_likelihoods)(states, self._target_states, FactorGraph.find_closest_obstacle(states, self._obstacles))

    def _update_marginal_beliefs(
        self,
        fac2var_msgs: Fac2VarMessages,
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

            # jax.debug.print("{}", Gaussian(marginal_info[0], marginal_precision[0], jnp.ones(4)).mean)
            # jax.debug.breakpoint()

            return Gaussian(
                info=marginal_info,
                precision=marginal_precision,
                dims=None,
            )

        return jax.vmap(batched_update_marginal_beliefs)(
            fac2var_msgs# , inter_fac2var_msgs
        )

    def init_inter_robot_var2fac_msgs(self) -> Gaussian:
        n_agents = self._n_agents
        time_horizon = self._time_horizon
        dummy = jnp.zeros((n_agents, time_horizon))
        robot_msgs = jax.vmap(jax.vmap(lambda _, var: Gaussian.identity(var)))(
            dummy,
            jnp.repeat(
                jnp.arange(1, time_horizon + 1)[jnp.newaxis, :], n_agents, axis=0
            ).astype(float),
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

        dynamics_axes = jnp.repeat(jnp.arange(2, self._time_horizon), 2)

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


        obstacle_msgs = jax.vmap(jax.vmap(lambda var: Gaussian.identity(var)))(
            jnp.tile(jnp.arange(1, self._time_horizon + 1, dtype=jnp.float32), (self._n_agents, 1))
        )

        return Var2FacMessages(
            poses=pose_msgs, 
            dynamics=dynamics_msgs,
            obstacle=obstacle_msgs,
            ir=self.init_inter_robot_var2fac_msgs()
        )

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
            "obstacles": self._obstacles,
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
            aux_data["obstacles"],
        )


jax.tree_util.register_pytree_node(
    FactorGraph, FactorGraph._tree_flatten, FactorGraph._tree_unflatten
)
