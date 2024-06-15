import random
from typing import Dict

import jax.numpy as jnp
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


class Visualizer:
    def __init__(
        self,
        n_timesteps: int,
        agent_radius: float,
        waypoints: Dict[str, jnp.array],
        trajs: Dict[str, jnp.array] = None,
        obstacles: jnp.array = None
    ) -> None:
        """
        waypoints: {agent_name: pt_data}, pt_data's shape is (timestep,2)
        trajs: {agent_name: traj_data}, traj_data's shape is (timestep,K,2), K is time horizon
        """
        self.waypoints_data = waypoints
        self.trajs = trajs
        self.agent_radius = agent_radius
        self.obstacles = obstacles
        # self.colors = [
        #     Visualizer._generate_random_color() for _ in range(len(waypoints))
        # ]
        self.colors = ["red", "blue", "orange", "black"]
        self.n_timesteps = n_timesteps
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

    def animate(self, save_fname: str = "test.gif", view: bool = True):
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.n_timesteps,
            init_func=self.init_sim,
            blit=False,
            interval=200,
        )

        if save_fname is not None:
            ani.save(save_fname)

        if view:
            plt.show(block=True)
            plt.close()

    def init_sim(self):
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)

    def update(self, t):
        for p in self.ax.patches:
            p.remove()
        for line in self.ax.get_lines():
            line.remove()
        for i, agent in enumerate(self.waypoints_data):
            waypoints = self.waypoints_data[agent]
            trajs = self.trajs[agent]
            circle = Circle(
                (waypoints[t, 0], waypoints[t, 1]),
                radius=self.agent_radius,
                color=self.colors[i],
            )
            self.ax.add_patch(circle)
            self.ax.plot(trajs[t, :, 0], trajs[t, :, 1], c=self.colors[i])
        if self.obstacles != None:
            for x, y in self.obstacles:
                obs_circle = Circle(
                    (x, y),
                    radius=self.agent_radius*5,
                    color="green",
                )
                self.ax.add_patch(obs_circle)

    def plot_whole_trajectories(self):
        for i, agent in enumerate(self.waypoints_data):
            waypoints = self.waypoints_data[agent]
            plt.scatter(waypoints[:, 0], waypoints[:, 1], c=self.colors[i])
        plt.show()

    @staticmethod
    def _generate_random_color():
        hex_digits = "0123456789abcdef"
        n_digits = len(hex_digits)
        return "#" + "".join(
            [hex_digits[random.randint(0, n_digits - 1)] for _ in range(6)]
        )


if __name__ == "__main__":
    agent1 = jnp.array([1.0, 0.0])
    agent1_waypoints = [agent1.copy()]
    agent2 = jnp.array([5.0, 1.0])
    agent2_waypoints = [agent2.copy()]
    trajs = {"agent1": [], "agent2": []}
    velocity = 0.1

    velocity1 = jnp.array([0.1, -0.2])
    velocity2 = jnp.array([-0.2, -0.1])
    acceleration = jnp.array([-0.001, 0.005])

    # def create_trajectory(pos: jnp.array, idx: int, change: float, steps: int = 10):
    #     traj = [pos]
    #     for _ in range(10):
    #         pos = pos.at[idx].add(change)
    #         traj.append(pos)
    #     return jnp.stack(traj)

    def create_trajectory(pos: jnp.array, velocity: jnp.array, acceleration: jnp.array, steps: int = 10):
        traj = [pos]
        for _ in range(10):
            pos = jnp.add(pos, velocity)
            velocity = jnp.add(velocity, acceleration)
            traj.append(pos)
        return jnp.stack(traj)

    # for i in range(1000):
    #     agent1 = agent1.at[1].add(velocity)
    #     agent1_waypoints.append(agent1.copy())
    #     trajs["agent1"].append(create_trajectory(agent1, 1, velocity))
    #     agent2 = agent2.at[0].add(-velocity)
    #     agent2_waypoints.append(agent2.copy())
    #     trajs["agent2"].append(create_trajectory(agent2, 0, -velocity))

    for i in range(1000):
        agent1 = jnp.add(agent1, velocity1)
        velocity1 = jnp.add(velocity1, acceleration)
        agent1_waypoints.append(agent1.copy())
        trajs["agent1"].append(create_trajectory(agent1, velocity1, acceleration))

        agent2 = jnp.add(agent2, velocity2)
        velocity2 = jnp.add(velocity1, acceleration)
        agent2_waypoints.append(agent2.copy())
        trajs["agent2"].append(create_trajectory(agent2, velocity2, acceleration))

    agent1_waypoints = jnp.stack(agent1_waypoints)
    agent2_waypoints = jnp.stack(agent2_waypoints)
    trajs["agent1"] = jnp.stack(trajs["agent1"])
    trajs["agent2"] = jnp.stack(trajs["agent2"])

    obstacles = jnp.array([
        [0, 2], [-3, -2]
    ])

    viz = Visualizer(
        100, 0.2, {"agent1": agent1_waypoints, "agent2": agent2_waypoints}, trajs, obstacles
    )
    viz.animate(view=True)