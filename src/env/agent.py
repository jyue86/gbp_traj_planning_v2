"""
Implementation of the online algorithm from:
https://arxiv.org/pdf/2203.11618.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from fg import FactorGraph
from fg.gaussian import Gaussian


class Agent:
    def __init__(
        self,
        start_state: jnp.ndarray,
        end_pos: jnp.ndarray,
        agent_radius: float,
        crit_distance: float,
        delta_t: float,
        time_horizon: float = 10,
    ):
        self._start_state = start_state
        self._end_pos = end_pos
        self._n_agents = self._start_state.shape[0]
        self._agent_radius = agent_radius
        self._crit_distance = crit_distance
        self._delta_t = delta_t
        self._time_horizon = time_horizon

        self._state_transition = jnp.eye(4)
        self._state_transition = self._state_transition.at[:2, 2:].set(
            jnp.eye(2) * self._delta_t
        )
        self._update_marginals = jax.jit(
            jax.vmap(
                lambda horizon_states: (self._state_transition @ horizon_states.T).T
            )
        )

        self._factor_graph = FactorGraph(
            self._n_agents,
            self._time_horizon + 1,
            self._agent_radius,
            self._crit_distance,
            self._end_pos,
            self._delta_t,
        )

    def get_energy(self, beliefs, factor_likeliood):
        def energy_helper(info, precision, belief):
            return Gaussian(info, precision, jnp.ones(4))(belief)

        temp = jax.vmap(jax.vmap(energy_helper))(
            beliefs.info, beliefs.precision, factor_likeliood
        )
        return -temp.sum(axis=1)

    @jax.jit
    def run(self, states: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        mean = states.copy()

        # code for inter robot
        # inter_robot_var2fac_msgs = self._factor_graph.init_inter_robot_var2fac_msgs()
        # inter_gbp_results = self._factor_graph.run_inter_robot_gbp_init(states, inter_robot_var2fac_msgs, time)
        # inter_var2fac_msgs = inter_gbp_results["var2fac"]
        # inter_fac2var_msgs = inter_gbp_results["fac2var"]

        # def run_inter_gbp(carry, _):
        #     current_mean = carry[0]
        #     var2fac_msgs = carry[1]
        #     fac2var_msgs = carry[2]
            
        #     inter_gbp_results = self._factor_graph.run_inter_robot_gbp(current_mean, var2fac_msgs, fac2var_msgs, time)
        #     # marginals = gbp_results["marginals"]
        #     updated_var2fac_msgs = inter_gbp_results["var2fac"]
        #     updated_fac2var_msgs = inter_gbp_results["fac2var"]
        #     # updated_marginal_mean = self._extract_mean(marginals.info, marginals.precision) 

        #     # return (updated_marginal_mean, updated_var2fac_msgs, updated_fac2var_msgs)
        #     return (current_mean, updated_var2fac_msgs, updated_fac2var_msgs), _
        # inter_gbp_results, _ = jax.lax.scan(run_inter_gbp, (mean, inter_var2fac_msgs, inter_fac2var_msgs), length=10)
        # inter_fac2var_msgs = inter_gbp_results[2]
        # end for inter robot

        var2fac_msgs = self._factor_graph.init_var2fac_msgs()
        # gbp_results = self._factor_graph.run_gbp_init(mean, var2fac_msgs, inter_fac2var_msgs)
        gbp_results = self._factor_graph.run_gbp_init(mean, var2fac_msgs, time)
        var2fac_msgs = gbp_results["var2fac"]
        fac2var_msgs = gbp_results["fac2var"]

        def run_gbp(carry, _):
            current_mean = carry[0]
            var2fac_msgs = carry[1]
            fac2var_msgs = carry[2]

            # gbp_results = self._factor_graph.run_gbp(
            #     current_mean, var2fac_msgs, fac2var_msgs, inter_fac2var_msgs
            # )
            gbp_results = self._factor_graph.run_gbp(
                current_mean, var2fac_msgs, fac2var_msgs, time
            )
            marginals = gbp_results["marginals"]
            updated_var2fac_msgs = gbp_results["var2fac"]
            updated_fac2var_msgs = gbp_results["fac2var"]
            updated_mean = self._extract_mean(
                marginals.info, marginals.precision
            )
            return (
                updated_mean,
                updated_var2fac_msgs,
                updated_fac2var_msgs,
            ), self._factor_graph.get_energy(updated_mean)

        gbp_results, energies = jax.lax.scan(
            run_gbp, (mean, var2fac_msgs, fac2var_msgs), length=200
        )
        mean = gbp_results[0]
        next_states = self._update_marginals(mean)
        return next_states, energies

    @jax.jit
    def _init_traj(self) -> jnp.ndarray:
        key = jax.random.PRNGKey(0)
        random_noise = jax.random.normal(key, (self._time_horizon, *self._start_state.T.shape))
        # def update_state(carry: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.array, int]:
        #     next_state = carry + noise
        #     next_state = next_state.at[2:,:].multiply(self._delta_t)
        #     return carry, next_state.T
        def update_state(carry: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
            next_state = (self._state_transition @ carry) # + noise
            return next_state, carry.T

        _, states = jax.lax.scan(
            update_state, self._start_state.T, random_noise, length=self._time_horizon
        )
        initial_states = jnp.swapaxes(states, 0, 1)
        initial_states = jnp.concat((initial_states, self._end_pos[:,None,:]), axis=1)
        return initial_states

    def _extract_mean(self, info: jnp.ndarray, precision: jnp.ndarray) -> jnp.ndarray:
        def batched_extract_mean(state_info: jnp.ndarray, state_precision: jnp.ndarray):
            moments_mean = (jnp.linalg.inv(state_precision) @ state_info.reshape(-1, 1)).flatten()
            return moments_mean

        return jax.vmap(jax.vmap(batched_extract_mean))(info, precision)

    @jax.jit
    def _transition_to_next_state(self, current_state: jnp.ndarray) -> jnp.ndarray:
        def update_fn(state):
            x = self._state_transition @ state.reshape((4, 1))
            return x

        return jax.vmap(update_fn)(current_state)

    @property
    def initial_state(self) -> jnp.ndarray:
        return self._init_traj()

    @property
    def agent_radius(self) -> float:
        return self._agent_radius

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def end_pos(self) -> jnp.ndarray:
        return self._end_pos

    def _tree_flatten(self):
        children = (self._start_state, self._end_pos)
        aux_data = {
            "agent_radius": self._agent_radius,
            "crit_distance": self._crit_distance,
            "delta_t": self._delta_t,
            "time_horizon": self._time_horizon,
        }
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(Agent, Agent._tree_flatten, Agent._tree_unflatten)
