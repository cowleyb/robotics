from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.world import World


def main() -> None:
    world = World(seed=1)
    observation = world.get_observation()
    print(observation)
    step_count = 0

    while True:
        world.move_car(throttle=10.0, steering=0.0)
        observation = world.step()
        step_count += 1
        if world.goal_reached():
            print("goal reached")
            break
        if world.hit_obstacle():
            print("hit obstacle")
            break
        if step_count >= 1000:
            print("timeout")
            break


if __name__ == "__main__":
    main()
