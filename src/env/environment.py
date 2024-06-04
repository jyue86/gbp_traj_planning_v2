import jax.numpy as jnp
from .agent import Agent
from viz import Visualizer


class Environment:
    def __init__(self, agent: Agent, save_gif_path: str=None, max_timesteps: int=1000) -> None:
        self.agent = agent
        self.save_fig_path = save_gif_path
        self.timesteps = 0
        self.max_timesteps = max_timesteps 
        self.waypoints = {f"agent{i}": [] for i in range(agent.get_n_agents())} # T x 2, T being number of timestesp
        self.planned_trajs = {f"agent{i}": [] for i in range(agent.get_n_agents())} # T X K X 2, K is time horizon
        self.states = agent.init_states()

    def step(self):
        self.states = self.agent.run(self.states)
        self.timesteps += 1

    def render(self) -> None:
        if not self.save_fig_path:
            return 
        for key in self.waypoints:
            self.waypoints[key] = jnp.stack(self.waypoints[key])
        for key in self.planned_trajs:
            self.planned_trajs[key] = jnp.stack(self.planned_trajs[key])
        
        visualizer = Visualizer(
            self.timesteps, self.agent.get_agent_radius, self.waypoints, self.planned_trajs
        )
        visualizer.animate(save_fname=self.save_gif_path)
    
    def count_collisions(self):
        pass

    def initialize_with_dummy_data(self) -> None:
        pass

    def is_done(self) -> bool:
        return False
    
    def is_truncated(self) -> bool:
        return self.timesteps == self.max_timesteps
