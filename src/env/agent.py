"""
Implementation of the online algorithm from:
https://arxiv.org/pdf/2203.11618.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from fg import FactorGraph, Gaussian

class Agent:
    def __init__(
        self,
        start_state: jnp.ndarray,
        end_pos: jnp.ndarray,
        agent_radius: float,
        crit_distance: float,
        delta_t: float,
        time_horizon: float=10,
    ):
        self._end_pos = end_pos
        self._n_agents = start_state.shape[0]
        self._agent_radius = agent_radius
        self._crit_distance = crit_distance
        self._delta_t = delta_t
        self._time_horizon = time_horizon

        self._state_transition = jnp.eye(4)
        self._state_transition = self._state_transition.at[:2,2:].set(jnp.eye(2) * self._delta_t)
        self._initial_state = self._init_traj(start_state)
        self._update_marginals = jax.vmap(lambda horizon_states: (self._state_transition @ horizon_states.T).T)

        self._factor_graph = FactorGraph(self._n_agents, self._time_horizon, self._end_pos, self._delta_t)
        self._var2fac_msgs = self._factor_graph.init_var2fac_msgs()
        self._init = True
        self._fac2var_msgs = None
    
    def run(self, states: jnp.ndarray) -> jnp.ndarray:
        ### START REPLACE
        marginal_belief = states.copy()
        for _ in range(10):
            gbp_results = self._factor_graph.run_gbp(marginal_belief, self._var2fac_msgs, self._fac2var_msgs, self._init)
            marginals = gbp_results["marginals"]
            self._var2fac_msgs = gbp_results["var2fac"]
            self._fac2var_msgs = gbp_results["fac2var"]
            if self._init:
                self._init = False
            # then extract marginals
            marginal_belief = self._extract_mean(marginals.info, marginals.precision) 

        # Backup line, can replace when ready
        # marginal_belief = states
        ### END OF REPLACE

        next_states = self._update_marginals(marginal_belief)
        return next_states

    def _init_traj(self, start_state: jnp.ndarray) -> jnp.ndarray:
        # def update_state(carry: jnp.array, _: int) -> Tuple[jnp.array, int]:
        #     next_state = self._state_transition @ carry
        #     return next_state, carry.T
        key = jax.random.PRNGKey(0)
        random_noise = jax.random.normal(key, (self._time_horizon, *start_state.T.shape))
        def update_state(carry: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.array, int]:
            next_state = carry + noise 
            next_state = next_state.at[2:,:].multiply(self._delta_t)
            return carry, next_state.T
        _, states = jax.lax.scan(update_state, start_state.T, random_noise, length=self._time_horizon)
        initial_states = jnp.swapaxes(states, 0, 1)
        return initial_states
    
    def _extract_mean(self, info: jnp.ndarray, precision: jnp.ndarray) -> jnp.ndarray:
        def batched_extract_mean(state_info: jnp.ndarray, state_precision: jnp.ndarray):
            return (jnp.linalg.inv(state_precision)  @ state_info.reshape(-1,1)).flatten()
        return jax.vmap(jax.vmap(batched_extract_mean))(info, precision)

    def _transition_to_next_state(self, current_state: jnp.ndarray) -> jnp.ndarray:
        def update_fn(state):
            x = (self._state_transition @ state.reshape((4,1)))
            return x
        return jax.vmap(update_fn)(current_state)
    
    @property
    def initial_state(self) -> jnp.ndarray:
        return self._initial_state
    
    @property
    def agent_radius(self) -> float:
        return self._agent_radius

    @property
    def n_agents(self) -> int:
        return self._n_agents 

    @property
    def end_pos(self) -> jnp.ndarray:
        return self._end_pos