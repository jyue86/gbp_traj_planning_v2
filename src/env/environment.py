import jax.numpy as jnp
from viz import Visualizer

from .agent import Agent
from .obstacle import Obstacle


class Environment:
    def __init__(self, agent: Agent, obstacle: Obstacle, max_timesteps: int=1000, save_gif_path: str=None) -> None:
        self.agent = agent
        self.max_timesteps = max_timesteps 
        self.obstacle = obstacle
        self.save_gif_path = save_gif_path
        self.timesteps = jnp.ones((1,))

        self.states = agent.initial_state
        self.waypoints = {f"agent{i}": [self.states[i,0,0:2]] for i in range(agent.n_agents)} # T x 2, T being number of timestesp
        self.planned_trajs = {f"agent{i}": [self.states[i,:,0:2]] for i in range(agent.n_agents)} # T X K X 2, K is time horizon
        self.energies = []

    def step(self) -> None:
        self.states, energies = self.agent.run(self.states, self.timesteps)
        self.energies.append(energies)
        for i in range(self.states.shape[0]):
            self.waypoints[f"agent{i}"].append(self.states[i,0,0:2])
            self.planned_trajs[f"agent{i}"].append(self.states[i,:,0:2])
        self.timesteps = self.timesteps.at[0].add(1)
    
    def _get_closest_obstacles(self) -> None:
        pass

    def render(self) -> None:
        if not self.save_gif_path:
            return 
        for key in self.waypoints:
            self.waypoints[key] = jnp.stack(self.waypoints[key])
        for key in self.planned_trajs:
            self.planned_trajs[key] = jnp.stack(self.planned_trajs[key])
        
        visualizer = Visualizer(
            int(self.timesteps[0]), self.agent.agent_radius, self.waypoints, self.planned_trajs
        )
        visualizer.animate(save_fname=self.save_gif_path)
    
    def count_collisions(self) -> None:
        pass

    def is_done(self) -> bool:
        end_pos = self.agent.end_pos[:,0:2]
        current_pos = self.states[:,0,0:2]
        return jnp.linalg.norm(end_pos - current_pos) <= 0.2
    
    def is_truncated(self) -> bool:
        return self.timesteps == self.max_timesteps
