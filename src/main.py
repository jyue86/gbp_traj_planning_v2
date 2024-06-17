from argparse import ArgumentParser
from typing import Dict

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from env import Agent, Environment, Obstacle
from utils import load_json
from metrics import total_dist_travelled, log_dimensionless_jerk


def init(scenario_config: Dict) -> Dict:
    states = []
    target_states = []
    for agent_name in scenario_config["agents"]:
        state = jnp.array(scenario_config["agents"][agent_name]["current_state"])
        states.append(state)
        target_states.append(
            jnp.array(scenario_config["agents"][agent_name]["target_state"])
        )
    crit_distance = (
        2 * scenario_config["agent_radius"] + scenario_config["safety_distance"]
    )
    jax.debug.print("Crit distance: {}", crit_distance)
    # obstacle_radius = scenario_config["obstacle_radius"]
    obstacle_pos = jnp.array(scenario_config["obstacle_pos"])
    obstacle = Obstacle(obstacle_pos)
    delta_t = scenario_config["delta_t"]
    agent = Agent(
        jnp.stack(states),
        jnp.stack(target_states),
        scenario_config["agent_radius"],
        crit_distance,
        delta_t,
        obstacles=obstacle_pos,
        time_horizon=4,
    )

    return {
        "agent": agent,
        "max_timesteps": scenario_config["max_timesteps"],
        "obstacle": obstacle_pos,
        "save_gif_path": scenario_config["save_gif_path"],
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/scenario1.json")
    args = parser.parse_args()

    config = load_json(args.config)
    env_data = init(config)
    env = Environment(
        env_data["agent"],
        obstacle=env_data["obstacle"],
        max_timesteps=env_data["max_timesteps"],
        save_gif_path=env_data["save_gif_path"],
    )

    i = 0
    while True:
        env.step()
        if env.is_done():
            print("Done!")
            break
        elif env.is_truncated():
            print("Truncated!")
            break
        i += 1
    print("Iterations:", i)
    
    maybe_plot_energies = True
    if maybe_plot_energies:
        eng = jnp.stack(env.energies)
        colors = ["red", "blue", "orange", "black"]
        # print(eng.shape)
        _, _, agents = eng.shape
        # print(agents)
        for agent in range(agents):
            plt.plot(eng[-1, :, agent], color=colors[agent], label=f"Agent {agent}")
            # plt.plot(eng[1:,1:, agent], color=colors[agent])
        # print(eng.shape)
        # plt.plot(eng[0, 0, 0], color="b", label="Agent 1")
        # plt.plot(eng[1:,1:, 0], color="b")
        # plt.plot(eng[0, 0, 1], color="r", label="Agent 2")
        # plt.plot(eng[1:, 1:, 1], color="r")
        
            plt.title("Expected Energy vs. GBP Iterations")
            plt.ylabel("Energy")
            plt.xlabel("Iterations")
            plt.legend()# tuple(p1 + p2), ("Agent 1", "Agent 2"), handler_map={tuple: HandlerTuple(ndivide=None)})
            plt.show(block=True)

    plot_metrics = True
    if plot_metrics:
        total_dist = 0
        total_ldj = 0
        num_agents = len(env.waypoints)
        for agent, waypoints in env.waypoints.items():
            dist_travelled = total_dist_travelled(jnp.array(waypoints))
            ldj = log_dimensionless_jerk(jnp.array(waypoints), 0.2*len(waypoints))

            total_dist += dist_travelled
            total_ldj += ldj

            print(f"{agent} Statistics")
            print(f"Total Distance Travlled: {dist_travelled}")
            print(f"Log Dimensionless Jerk: {ldj}")

        print(f"Avg Statistics Across All Agents")
        print(f"Avg Total Distance Travlled: {total_dist/num_agents}")
        print(f"Avg Log Dimensionless Jerk: {total_ldj/num_agents}")
    env.render()


if __name__ == "__main__":
    main()
