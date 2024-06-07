from argparse import ArgumentParser
from typing import Dict

import jax.numpy as jnp
from env import Agent, Environment, Obstacle
from utils import load_json


def init(scenario_config: Dict) -> Dict:
    states = []
    target_pos = []
    for agent_name in scenario_config["agents"]:
        pos = jnp.array(scenario_config["agents"][agent_name]["current_pos"])
        vel = jnp.array(scenario_config["agents"][agent_name]["vel"])
        states.append(jnp.concat((pos, vel)))
        target_pos.append(
            jnp.array(scenario_config["agents"][agent_name]["target_pos"])
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
        jnp.stack(target_pos),
        scenario_config["agent_radius"],
        crit_distance,
        delta_t,
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

    while True:
        env.step()
        if env.is_done():
            print("Done!")
            break
        elif env.is_truncated():
            print("Truncated!")
            break
    env.render()


if __name__ == "__main__":
    main()
