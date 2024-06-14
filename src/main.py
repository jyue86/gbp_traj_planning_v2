from argparse import ArgumentParser
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

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
        time_horizon=4
    )

    return {
        "agent": agent,
        "max_timesteps": scenario_config["max_timesteps"],
        "obstacle": obstacle,
        "save_gif_path": scenario_config["save_gif_path"],
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    env_data = init(config)
    env = Environment(
        env_data["agent"],
        obstacle=env_data["obstacle"],
        max_timesteps=env_data["max_timesteps"],
        save_gif_path=env_data["save_gif_path"]
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
        plt.plot(eng[0, 0, 0], color="b", label="Agent 1")
        plt.plot(eng[1:,1:, 0], color="b")
        plt.plot(eng[0, 0, 1], color="r", label="Agent 2")
        plt.plot(eng[1:, 1:, 1], color="r")
        
        plt.title("Expected Energy vs. Timesteps")
        plt.ylabel("Energy")
        plt.xlabel("Timesteps")
        plt.legend()# tuple(p1 + p2), ("Agent 1", "Agent 2"), handler_map={tuple: HandlerTuple(ndivide=None)})
        plt.show(block=True)

    plot_metrics = True
    if plot_metrics:
        for agent, waypoints in env.waypoints.items():
            print(f"{agent} Statistics")
            print(f"Total Distance Travlled: {total_dist_travelled(jnp.array(waypoints))}")
            print(f"Log Dimensionless Jerk: {log_dimensionless_jerk(jnp.array(waypoints), 0.2)}")
    env.render()


if __name__ == "__main__":
    main()
