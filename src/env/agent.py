"""
Implementation of the online algorithm from:
https://arxiv.org/pdf/2203.11618.
"""

import jax
import jax.numpy as jnp

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
        self.start_state = start_state
        self.end_pos = end_pos
        self.n_agents = start_state.shape[0]
        self.agent_radius = agent_radius
        self.crit_distance = crit_distance
        self.delta_t = delta_t
        self.time_horizon = time_horizon

        self.state_transition = jnp.eye(4)
        self.state_transition = self.state_transition.at[:2,2:].set(jnp.eye(2) * self.delta_t)
        self.update_marginals = jax.vmap(lambda horizon_states: (self.state_transition @ horizon_states.T).T)
    
    def run(self, states: jnp.array):
        marginal_belief = states
        next_states = self.update_marginals(marginal_belief)
        return next_states

    def init_traj(self):
        def update_state(carry: jnp.array, _: int):
            carry = self.state_transition @ carry
            return carry, carry.T
        _, states = jax.lax.scan(update_state, self.start_state.T, length=self.time_horizon)        
        initial_states = jnp.swapaxes(states, 0, 1)
        return initial_states

    def _transition_to_next_state(self, current_state):
        def update_fn(state):
            x = (self.factor_graph.dynamics_factors.state_transition @ state.reshape((4,1)))
            return x
        return jax.vmap(update_fn)(current_state)
    
    def get_agent_radius(self):
        return self.agent_radius

    def get_n_agents(self):
        return self.n_agents 

    def get_end_pos(self):
        return self.end_pos