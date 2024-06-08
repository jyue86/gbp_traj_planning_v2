"""
Implementation of the online algorithm from:
https://arxiv.org/pdf/2203.11618.
"""

import jax
import jax.numpy as jnp

from fg import FactorGraph

class Agent:
    def __init__(
        self,
        start_state: jnp.array,
        end_pos: jnp.array,
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
        self._current_state = self._init_traj(start_state)
        self._update_marginals = jax.vmap(lambda horizon_states: (self._state_transition @ horizon_states.T).T)

        self._factor_graph = FactorGraph(self._current_state, self._end_pos)
    
    def run(self, states: jnp.array):
        ### START REPLACE
        marginal_belief = states
        ### END OF REPLACE

        next_states = self._update_marginals(marginal_belief)
        return next_states

    def _init_traj(self, start_state):
        def update_state(carry: jnp.array, _: int):
            carry = self._state_transition @ carry
            return carry, carry.T
        _, states = jax.lax.scan(update_state, start_state.T, length=self._time_horizon)
        initial_states = jnp.swapaxes(states, 0, 1)
        return initial_states

    def _transition_to_next_state(self, current_state):
        def update_fn(state):
            x = (self.factor_graph.dynamics_factors.state_transition @ state.reshape((4,1)))
            return x
        return jax.vmap(update_fn)(current_state)
    
    @property
    def current_state(self):
        return self._current_state
    
    @property
    def agent_radius(self):
        return self._agent_radius

    @property
    def n_agents(self):
        return self._n_agents 

    @property
    def end_pos(self):
        return self._end_pos