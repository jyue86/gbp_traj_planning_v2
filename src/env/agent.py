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
        self.n_agents = start_state.shape[0]
        self.agent_radius = agent_radius
        self.crit_distance = crit_distance
        self.time_horizon = time_horizon
    
    def run(self, state: jnp.array):
        pass

    def init_states(self):
        def inner_loop(state, _):
            updated_state = self._transition_to_next_state(state)
            return updated_state[:, -4:].squeeze(), updated_state
        
        _, states = jax.lax.scan(inner_loop, self.start_state, length=self.time_horizon)        
        initial_states = jnp.swapaxes(states, 0, 1).squeeze()
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
    
    def get_waypoints_and_trajectory(self):
        # Get current state (state at x0), extract position and velocity
        position = jnp.array([1.0, 0.0]) # Test position, replace later
        velocity = jnp.array([0.1, -0.2])

        waypoints = [position.copy()]
        trajectory = [jnp.add(position, velocity)]

        return jnp.stack(waypoints), jnp.stack(trajectory)

