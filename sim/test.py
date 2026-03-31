import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.world import World


def main() -> None:
    world = World(seed=1, enable_camera=True)
    observation = world.get_observation()
    print(observation)
    step_count = 0
    trajectory = []

    while True:
        throttle, steering = world.heuristic_action()
        world.move_car(throttle=throttle, steering=steering)
        next_observation = world.step()
        step_count += 1
        reached_goal = world.goal_reached()
        hit_obstacle = world.hit_obstacle()
        timed_out = step_count >= 1000
        done = reached_goal or hit_obstacle or timed_out
        trajectory.append(
            {
                "observation": observation,
                "action": {"throttle": throttle, "steering": steering},
                "next_observation": next_observation,
                "done": done,
            }
        )
        observation = next_observation
        if reached_goal:
            print(f"goal reached at step {step_count}")
            break
        if hit_obstacle:
            print(f"hit obstacle at step {step_count}")
            break
        if timed_out:
            print(f"timeout at step {step_count}")
            break

    print(f"collected {len(trajectory)} samples")


if __name__ == "__main__":
    main()
