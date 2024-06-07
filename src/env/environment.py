import jax.numpy as jnp
from .agent import Agent
from .obstacle import Obstacle
from viz import Visualizer


class Environment:
    def __init__(self, agent: Agent, obstacle: Obstacle, max_timesteps: int=1000, save_gif_path: str=None) -> None:
        self.agent = agent
        self.max_timesteps = max_timesteps 
        self.obstacle = obstacle
        self.save_gif_path = save_gif_path
        self.timesteps = 0

        self.waypoints = {f"agent{i}": [] for i in range(agent.get_n_agents())} # T x 2, T being number of timestesp
        self.planned_trajs = {f"agent{i}": [] for i in range(agent.get_n_agents())} # T X K X 2, K is time horizon
        self.states = agent.init_traj()

    def step(self):
        self.states = self.agent.run(self.states)
        for i in range(self.states.shape[0]):
            self.waypoints[f"agent{i}"].append(self.states[i,0,0:2])
            self.planned_trajs[f"agent{i}"].append(self.states[i,:,0:2])
        self.timesteps += 1
    
    def _get_closest_obstacles(self):
        pass

    def render(self) -> None:
        if not self.save_gif_path:
            return 
        for key in self.waypoints:
            self.waypoints[key] = jnp.stack(self.waypoints[key])
        for key in self.planned_trajs:
            self.planned_trajs[key] = jnp.stack(self.planned_trajs[key])
        
        visualizer = Visualizer(
            self.timesteps, self.agent.get_agent_radius(), self.waypoints, self.planned_trajs
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
