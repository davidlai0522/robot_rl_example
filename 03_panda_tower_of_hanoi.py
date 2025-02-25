import robosuite
from robosuite.wrappers import GymWrapper
import mujoco
import stable_baselines3
from environment.tower_of_hanoi import TowerOfHanoi
import numpy as np
from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

def main():
    env = TowerOfHanoi(
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )
    env = GymWrapper(env)
    obs, _ = env.reset()


    console = Console()

    table = Table(title="Environment Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Available Actions", f"{env.action_space}")
    table.add_row("Available Observations", f"{env.observation_space}")
    table.add_row("Pole Body IDs", f"{list(env.pole_body_ids.values())}")
    table.add_row("Disk Body IDs", f"{list(env.disk_body_ids.values())}")

    console.print(table)

    pole_positions = []
    disk_positions = []

    console.print("[bold]Object Placement:[/bold]")
    for id in env.pole_body_ids.values():
        pole_positions.append(f"{env.sim.data.body_xpos[id]}")
        console.print(f"Pole {id} Positions:", env.sim.data.body_xpos[id], sep="\t")
        console.print(f"Pole {id} Orientation:", env.sim.data.body_xquat[id], sep="\t")
        console.print("--" * 20)
    for id in env.disk_body_ids.values():
        disk_positions.append(f"{env.sim.data.body_xpos[id]}")
        console.print(f"Disk {id} Positions:", env.sim.data.body_xpos[id], sep="\t")
        console.print(f"Disk {id} Orientation:", env.sim.data.body_xquat[id], sep="\t")
        console.print("--" * 20)

    # for _ in range(10000):
    #     # Sample a random action
    #     # random_action = env.action_space.sample()
    #     random_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    #     obs, reward, done, _, info = env.step(random_action)

    #     env.render()
    #     # print(obs)

    # env.close()
    # exit()


if __name__ == "__main__":
    main()