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

    while True:
        throttle, steering = world.heuristic_action()
        world.move_car(throttle=throttle, steering=steering)
        observation = world.step()
        step_count += 1
        if world.goal_reached():
            print(f"goal reached at step {step_count}")
            break
        if world.hit_obstacle():
            print(f"hit obstacle at step {step_count}")
            break
        if step_count >= 1000:
            print(f"timeout at step {step_count}")
            break


if __name__ == "__main__":
    main()
